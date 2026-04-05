import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import nn

from .actions import LocalWilsonLoopAction
from .baselines import tree_level_coarse_beta
from .blocking import (
    FixedGaugeCovariantBlocker,
    LearnableGaugeCovariantBlocker,
    SpatialGaugeCovariantBlocker,
)
from .hmc import HMCU1Sampler
from .lattice import plaquette_angles, rectangle_x_angles, rectangle_y_angles, topological_charge, wilson_loop_angles


@dataclass
class RGTrainingConfig:
    fine_lattice_size: int = 8
    fine_beta: float = 4.0
    coarse_beta_init: float | None = None
    n_fine_samples: int = 24
    n_model_samples: int = 24
    sampler_burn_in: int = 48
    sampler_thin: int = 4
    hmc_steps: int = 8
    hmc_step_size: float = 0.15
    epochs: int = 40
    learning_rate: float = 5e-2
    distribution_loss_weight: float = 10.0
    mean_loss_weight: float = 2.0
    path_sparsity_weight: float = 1e-2
    coefficient_l2: float = 1e-3
    gradient_clip: float = 1.0
    device: str = "cpu"
    seed: int = 7
    basis: tuple[str, ...] = ("plaquette", "rectangle_x", "rectangle_y")
    measurement_set: tuple[str, ...] = ("plaquette", "rectangle_x", "rectangle_y", "wilson_2x2")
    evaluation_measurement_set: tuple[str, ...] = (
        "plaquette",
        "rectangle_x",
        "rectangle_y",
        "wilson_2x2",
        "topological_charge",
    )
    mmd_bandwidth: float = 0.5
    blocker_type: str = "spatial"
    n_test_samples: int = 0
    spatial_hidden_dim: int = 32
    spatial_kernel_size: int = 3

    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path


@dataclass
class RGTrainingResult:
    baseline_mismatch: float
    final_mismatch: float
    measurement_names: tuple[str, ...]
    evaluation_measurement_names: tuple[str, ...]
    baseline_observables: dict[str, float]
    final_data_observables: dict[str, float]
    final_model_observables: dict[str, float]
    blocker_summary: dict
    learned_action_coefficients: dict[str, float]
    final_distribution_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    test_distribution_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    history: list[dict[str, float]] = field(default_factory=list)
    final_blocked_ensemble: torch.Tensor | None = None
    final_model_ensemble: torch.Tensor | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("final_blocked_ensemble", None)
        d.pop("final_model_ensemble", None)
        return d

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _observables_dict(basis: tuple[str, ...], tensor: torch.Tensor) -> dict[str, float]:
    return {name: float(value.detach().cpu()) for name, value in zip(basis, tensor)}


def _create_blocker(config: RGTrainingConfig) -> nn.Module:
    if config.blocker_type == "spatial":
        return SpatialGaugeCovariantBlocker(
            hidden_dim=config.spatial_hidden_dim,
            kernel_size=config.spatial_kernel_size,
        )
    if config.blocker_type == "global":
        return LearnableGaugeCovariantBlocker()
    if config.blocker_type == "fixed":
        return FixedGaugeCovariantBlocker()
    raise ValueError(f"Unknown blocker_type: {config.blocker_type!r}")


def generate_fine_ensemble(config: RGTrainingConfig, n_samples: int | None = None,
                           seed: int | None = None) -> torch.Tensor:
    _set_seed(seed if seed is not None else config.seed)
    action = LocalWilsonLoopAction.wilson(config.fine_beta, basis=config.basis).to(config.device)
    sampler = HMCU1Sampler(
        lattice_size=config.fine_lattice_size,
        action=action,
        n_steps=config.hmc_steps,
        step_size=config.hmc_step_size,
        device=config.device,
    )
    samples, _, _ = sampler.sample(
        n_samples=n_samples if n_samples is not None else config.n_fine_samples,
        burn_in=config.sampler_burn_in,
        thin=config.sampler_thin,
    )
    return samples


def _sample_model_ensemble(
    action: nn.Module,
    coarse_lattice_size: int,
    config: RGTrainingConfig,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float, torch.Tensor]:
    sampler = HMCU1Sampler(
        lattice_size=coarse_lattice_size,
        action=action,
        n_steps=config.hmc_steps,
        step_size=config.hmc_step_size,
        device=config.device,
    )
    return sampler.sample(
        n_samples=config.n_model_samples,
        burn_in=config.sampler_burn_in,
        thin=config.sampler_thin,
        initial_state=initial_state,
    )


def _observable_mismatch(data_obs: torch.Tensor, model_obs: torch.Tensor) -> torch.Tensor:
    return torch.mean((data_obs - model_obs) ** 2)


def _loop_mean_per_configuration(field: torch.Tensor, measurement_name: str) -> torch.Tensor:
    if measurement_name == "plaquette":
        angles = plaquette_angles(field)
    elif measurement_name == "rectangle_x":
        angles = rectangle_x_angles(field)
    elif measurement_name == "rectangle_y":
        angles = rectangle_y_angles(field)
    elif measurement_name.startswith("wilson_"):
        _, extents = measurement_name.split("_", maxsplit=1)
        extent_x, extent_y = (int(value) for value in extents.split("x"))
        angles = wilson_loop_angles(field, extent_x=extent_x, extent_y=extent_y)
    else:
        raise ValueError(f"Unknown loop measurement: {measurement_name}")
    return torch.cos(angles).mean(dim=(-2, -1))


def _measurement_features(field: torch.Tensor, measurement_names: tuple[str, ...]) -> torch.Tensor:
    if field.dim() == 3:
        field = field.unsqueeze(0)
    features = []
    for measurement_name in measurement_names:
        if measurement_name == "topological_charge":
            feature = topological_charge(field).float()
        else:
            feature = _loop_mean_per_configuration(field, measurement_name)
        features.append(feature)
    return torch.stack(features, dim=-1)


def measurement_distribution_mmd(
    blocked_field: torch.Tensor,
    coarse_field: torch.Tensor,
    measurement_names: tuple[str, ...],
    bandwidth: float,
) -> torch.Tensor:
    blocked_features = _measurement_features(blocked_field, measurement_names)
    coarse_features = _measurement_features(coarse_field, measurement_names)
    return _gaussian_mmd(blocked_features, coarse_features, bandwidth=bandwidth)


def _gaussian_mmd(data_features: torch.Tensor, model_features: torch.Tensor, bandwidth: float) -> torch.Tensor:
    gamma = 1.0 / max(2.0 * bandwidth * bandwidth, 1e-8)
    xx = torch.cdist(data_features, data_features).pow(2)
    yy = torch.cdist(model_features, model_features).pow(2)
    xy = torch.cdist(data_features, model_features).pow(2)
    return (
        torch.exp(-gamma * xx).mean()
        + torch.exp(-gamma * yy).mean()
        - 2.0 * torch.exp(-gamma * xy).mean()
    )


def _energy_distance_1d(x: torch.Tensor, y: torch.Tensor) -> float:
    """One-dimensional energy distance between two sample vectors."""
    if x.numel() == 0 or y.numel() == 0:
        return 0.0
    x = x.float().flatten()
    y = y.float().flatten()
    xy = torch.cdist(x.unsqueeze(-1), y.unsqueeze(-1), p=1).mean()
    xx = torch.cdist(x.unsqueeze(-1), x.unsqueeze(-1), p=1).mean()
    yy = torch.cdist(y.unsqueeze(-1), y.unsqueeze(-1), p=1).mean()
    return float((2 * xy - xx - yy).cpu())


def _compute_distribution_metrics(
    blocked: torch.Tensor,
    model: torch.Tensor,
    measurement_names: tuple[str, ...],
    bandwidth: float,
) -> dict[str, dict[str, float]]:
    """Per-observable distribution metrics between blocked-fine and model ensembles."""
    metrics: dict[str, dict[str, float]] = {}
    for name in measurement_names:
        obs_mmd = float(measurement_distribution_mmd(
            blocked, model, measurement_names=(name,), bandwidth=bandwidth,
        ).cpu())
        blocked_feat = _measurement_features(blocked, (name,)).squeeze(-1)
        model_feat = _measurement_features(model, (name,)).squeeze(-1)
        metrics[name] = {
            "mmd": obs_mmd,
            "energy_distance": _energy_distance_1d(blocked_feat, model_feat),
            "blocked_mean": float(blocked_feat.mean()),
            "model_mean": float(model_feat.mean()),
            "blocked_std": float(blocked_feat.std()),
            "model_std": float(model_feat.std()),
        }
    return metrics


def train_learned_rg(
    fine_configs: torch.Tensor | None = None,
    config: RGTrainingConfig | None = None,
    blocker: nn.Module | None = None,
) -> RGTrainingResult:
    config = config or RGTrainingConfig()
    _set_seed(config.seed)
    device = torch.device(config.device)
    coarse_beta_init = config.coarse_beta_init
    if coarse_beta_init is None:
        coarse_beta_init = tree_level_coarse_beta(config.fine_beta)

    if fine_configs is None:
        fine_configs = generate_fine_ensemble(config)
    fine_configs = fine_configs.to(device)
    coarse_lattice_size = fine_configs.shape[-1] // 2

    test_configs: torch.Tensor | None = None
    if config.n_test_samples > 0:
        test_configs = generate_fine_ensemble(
            config, n_samples=config.n_test_samples, seed=config.seed + 1000,
        ).to(device)

    fixed_blocker = FixedGaugeCovariantBlocker().to(device)
    learnable_blocker = (blocker if blocker is not None else _create_blocker(config)).to(device)
    coarse_action = LocalWilsonLoopAction.wilson(coarse_beta_init, basis=config.basis).to(device)

    with torch.no_grad():
        baseline_data = fixed_blocker(fine_configs)
        baseline_data_obs = coarse_action.observable_vector(baseline_data)
        baseline_model_action = LocalWilsonLoopAction.wilson(coarse_beta_init, basis=config.basis).to(device)
        baseline_model, _, model_state = _sample_model_ensemble(
            baseline_model_action,
            coarse_lattice_size=coarse_lattice_size,
            config=config,
        )
        baseline_model_obs = coarse_action.observable_vector(baseline_model)
        baseline_mismatch = float(
            measurement_distribution_mmd(
                baseline_data,
                baseline_model,
                measurement_names=config.evaluation_measurement_set,
                bandwidth=config.mmd_bandwidth,
            ).cpu()
        )

    optimizer = torch.optim.Adam(
        list(learnable_blocker.parameters()) + list(coarse_action.parameters()),
        lr=config.learning_rate,
    )
    history: list[dict[str, float]] = []

    for epoch in range(config.epochs):
        optimizer.zero_grad()
        blocked = learnable_blocker(fine_configs)
        data_obs = coarse_action.observable_vector(blocked)

        with torch.no_grad():
            model_samples, acceptance_rate, model_state = _sample_model_ensemble(
                coarse_action,
                coarse_lattice_size=coarse_lattice_size,
                config=config,
                initial_state=model_state,
            )
            model_obs = coarse_action.observable_vector(model_samples)

        mean_mismatch = _observable_mismatch(data_obs, model_obs.detach())
        distribution_mismatch = measurement_distribution_mmd(
            blocked,
            model_samples.detach(),
            measurement_names=config.measurement_set,
            bandwidth=config.mmd_bandwidth,
        )
        contrastive_loss = torch.dot(coarse_action.coefficients, model_obs.detach() - data_obs)

        if hasattr(learnable_blocker, "regularization_loss"):
            sparsity_term = learnable_blocker.regularization_loss()
        else:
            sparsity_term = torch.tensor(0.0, device=device)

        coefficient_penalty = coarse_action.coefficients[1:].square().sum()
        loss = (
            contrastive_loss
            + config.distribution_loss_weight * distribution_mismatch
            + config.mean_loss_weight * mean_mismatch
            + config.path_sparsity_weight * sparsity_term
            + config.coefficient_l2 * coefficient_penalty
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(learnable_blocker.parameters()) + list(coarse_action.parameters()),
            config.gradient_clip,
        )
        optimizer.step()

        history.append(
            {
                "epoch": float(epoch),
                "loss": float(loss.detach().cpu()),
                "distribution_mismatch": float(distribution_mismatch.detach().cpu()),
                "mean_mismatch": float(mean_mismatch.detach().cpu()),
                "acceptance_rate": float(acceptance_rate),
            }
        )

    with torch.no_grad():
        final_blocked = learnable_blocker(fine_configs)
        final_data_obs = coarse_action.observable_vector(final_blocked)
        final_model, _, _ = _sample_model_ensemble(
            coarse_action,
            coarse_lattice_size=coarse_lattice_size,
            config=config,
            initial_state=model_state,
        )
        final_model_obs = coarse_action.observable_vector(final_model)
        final_mismatch = float(
            measurement_distribution_mmd(
                final_blocked,
                final_model,
                measurement_names=config.evaluation_measurement_set,
                bandwidth=config.mmd_bandwidth,
            ).cpu()
        )
        final_distribution_metrics = _compute_distribution_metrics(
            final_blocked,
            final_model,
            measurement_names=config.evaluation_measurement_set,
            bandwidth=config.mmd_bandwidth,
        )

        test_distribution_metrics: dict[str, dict[str, float]] = {}
        if test_configs is not None:
            test_blocked = learnable_blocker(test_configs)
            test_model, _, _ = _sample_model_ensemble(
                coarse_action,
                coarse_lattice_size=coarse_lattice_size,
                config=config,
            )
            test_distribution_metrics = _compute_distribution_metrics(
                test_blocked,
                test_model,
                measurement_names=config.evaluation_measurement_set,
                bandwidth=config.mmd_bandwidth,
            )

    if hasattr(learnable_blocker, "summary"):
        blocker_summary = learnable_blocker.summary()
    else:
        blocker_summary = {"type": type(learnable_blocker).__name__}

    return RGTrainingResult(
        baseline_mismatch=baseline_mismatch,
        final_mismatch=final_mismatch,
        measurement_names=config.measurement_set,
        evaluation_measurement_names=config.evaluation_measurement_set,
        baseline_observables=_observables_dict(config.basis, baseline_data_obs),
        final_data_observables=_observables_dict(config.basis, final_data_obs),
        final_model_observables=_observables_dict(config.basis, final_model_obs),
        blocker_summary=blocker_summary,
        learned_action_coefficients=coarse_action.coefficient_dict(),
        final_distribution_metrics=final_distribution_metrics,
        test_distribution_metrics=test_distribution_metrics,
        history=history,
        final_blocked_ensemble=final_blocked.detach().cpu(),
        final_model_ensemble=final_model.detach().cpu(),
    )
