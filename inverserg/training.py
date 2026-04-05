from dataclasses import dataclass, field

import torch
from torch import nn

from .actions import LocalWilsonLoopAction
from .baselines import tree_level_coarse_beta
from .blocking import FixedGaugeCovariantBlocker, LearnableGaugeCovariantBlocker
from .hmc import HMCU1Sampler


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
    blocker_loss_weight: float = 10.0
    path_sparsity_weight: float = 1e-2
    coefficient_l2: float = 1e-3
    gradient_clip: float = 1.0
    device: str = "cpu"
    seed: int = 7
    basis: tuple[str, ...] = ("plaquette", "rectangle_x", "rectangle_y")


@dataclass
class RGTrainingResult:
    baseline_mismatch: float
    final_mismatch: float
    baseline_observables: dict[str, float]
    final_data_observables: dict[str, float]
    final_model_observables: dict[str, float]
    learned_path_weights: dict[str, list[float]]
    learned_action_coefficients: dict[str, float]
    history: list[dict[str, float]] = field(default_factory=list)


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _observables_dict(basis: tuple[str, ...], tensor: torch.Tensor) -> dict[str, float]:
    return {name: float(value.detach().cpu()) for name, value in zip(basis, tensor)}


def generate_fine_ensemble(config: RGTrainingConfig) -> torch.Tensor:
    _set_seed(config.seed)
    action = LocalWilsonLoopAction.wilson(config.fine_beta, basis=config.basis).to(config.device)
    sampler = HMCU1Sampler(
        lattice_size=config.fine_lattice_size,
        action=action,
        n_steps=config.hmc_steps,
        step_size=config.hmc_step_size,
        device=config.device,
    )
    samples, _, _ = sampler.sample(
        n_samples=config.n_fine_samples,
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


def train_learned_rg(
    fine_configs: torch.Tensor | None = None,
    config: RGTrainingConfig | None = None,
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

    fixed_blocker = FixedGaugeCovariantBlocker().to(device)
    learnable_blocker = LearnableGaugeCovariantBlocker().to(device)
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
        baseline_mismatch = float(_observable_mismatch(baseline_data_obs, baseline_model_obs).cpu())

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

        mismatch = _observable_mismatch(data_obs, model_obs.detach())
        contrastive_loss = torch.dot(coarse_action.coefficients, model_obs.detach() - data_obs)
        probs = learnable_blocker.path_probabilities()
        # This negative-entropy term intentionally encourages sparse path selection.
        sparsity_term = torch.sum(probs * torch.log(probs + 1e-8))
        coefficient_penalty = coarse_action.coefficients[1:].square().sum()
        loss = (
            contrastive_loss
            + config.blocker_loss_weight * mismatch
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
                "mismatch": float(mismatch.detach().cpu()),
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
        final_mismatch = float(_observable_mismatch(final_data_obs, final_model_obs).cpu())
        probs = learnable_blocker.path_probabilities().detach().cpu()

    return RGTrainingResult(
        baseline_mismatch=baseline_mismatch,
        final_mismatch=final_mismatch,
        baseline_observables=_observables_dict(config.basis, baseline_data_obs),
        final_data_observables=_observables_dict(config.basis, final_data_obs),
        final_model_observables=_observables_dict(config.basis, final_model_obs),
        learned_path_weights={
            "x_links": [float(x) for x in probs[0]],
            "y_links": [float(x) for x in probs[1]],
        },
        learned_action_coefficients=coarse_action.coefficient_dict(),
        history=history,
    )
