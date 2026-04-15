import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from .baselines import tree_level_coarse_beta
from .blocking import ConditionedSpatialGaugeCovariantBlocker
from .hmc import HMCU1Sampler
from .lattice import loop_observables
from .training import RGTrainingConfig, generate_fine_ensemble, measurement_distribution_mmd


def _as_batch_couplings(J_fine: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if J_fine.dim() == 1:
        return J_fine.unsqueeze(0), True
    if J_fine.dim() != 2:
        raise ValueError(f"Expected [d] or [B, d] couplings, got {tuple(J_fine.shape)}")
    return J_fine, False


class _FunctionalWilsonLoopAction(nn.Module):
    def __init__(self, basis: tuple[str, ...], coefficients: torch.Tensor) -> None:
        super().__init__()
        self.basis = basis
        self.coefficients = coefficients.detach().clone().float()

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        if field.dim() == 3:
            field = field.unsqueeze(0)
        contributions = []
        for coefficient, obs in zip(self.coefficients, self._loop_values(field)):
            contributions.append(-coefficient * obs.sum(dim=(-2, -1)))
        return torch.stack(contributions, dim=0).sum(dim=0).sum()

    def _loop_values(self, field: torch.Tensor) -> list[torch.Tensor]:
        from .actions import LocalWilsonLoopAction

        temp = LocalWilsonLoopAction(basis=self.basis, initial_coefficients=self.coefficients)
        return temp.loop_values(field)


@dataclass
class ForwardRGConfig:
    fine_lattice_size: int = 8
    n_fine_samples: int = 24
    n_model_samples: int = 24
    sampler_burn_in: int = 48
    sampler_thin: int = 4
    hmc_steps: int = 8
    hmc_step_size: float = 0.15
    epochs: int = 200
    learning_rate: float = 3e-3
    device: str = "cpu"
    seed: int = 17
    basis: tuple[str, ...] = ("plaquette", "rectangle_x", "rectangle_y")
    measurement_set: tuple[str, ...] = ("plaquette", "rectangle_x", "rectangle_y", "wilson_2x2")
    evaluation_measurement_set: tuple[str, ...] = (
        "plaquette",
        "rectangle_x",
        "rectangle_y",
        "wilson_2x2",
        "topological_charge",
    )
    distribution_loss_weight: float = 10.0
    mean_loss_weight: float = 2.0
    path_sparsity_weight: float = 1e-2
    coefficient_l2: float = 1e-3
    gradient_clip: float = 1.0
    mmd_bandwidth: float = 0.5
    hidden_dim: int = 64
    z_phi_dim: int = 16
    blocker_hidden_dim: int = 32
    blocker_kernel_size: int = 3

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ForwardRGResult:
    history: list[dict[str, float]]
    beta_values: list[float]
    basis: tuple[str, ...]
    predicted_J_coarse: list[list[float]]
    latent_codes: list[list[float]]
    blocker_summary: dict

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ForwardRGResult":
        data = json.loads(Path(path).read_text())
        data["basis"] = tuple(data["basis"])
        return cls(**data)


class ForwardRGHypernetwork(nn.Module):
    def __init__(self, coupling_dim: int = 3, hidden_dim: int = 64, z_phi_dim: int = 16) -> None:
        super().__init__()
        self.coupling_dim = coupling_dim
        self.hidden_dim = hidden_dim
        self.z_phi_dim = z_phi_dim
        self.net = nn.Sequential(
            nn.Linear(coupling_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.coarse_head = nn.Linear(hidden_dim, coupling_dim)
        self.latent_head = nn.Linear(hidden_dim, z_phi_dim)
        nn.init.zeros_(self.coarse_head.weight)
        nn.init.zeros_(self.coarse_head.bias)
        nn.init.zeros_(self.latent_head.bias)

    def forward(self, J_fine: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        J_fine, squeezed = _as_batch_couplings(J_fine)
        hidden = self.net(J_fine)
        coarse_delta = self.coarse_head(hidden)
        base = torch.zeros_like(J_fine)
        base[:, 0] = 0.25 * J_fine[:, 0]
        J_coarse = base + coarse_delta
        z_phi = self.latent_head(hidden)
        if squeezed:
            return J_coarse.squeeze(0), z_phi.squeeze(0)
        return J_coarse, z_phi

    def predict_forward_rg(self, J_fine: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(J_fine)


def predict_forward_rg(
    model: ForwardRGHypernetwork,
    J_fine: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return model.predict_forward_rg(J_fine)


def _sample_model_ensemble(
    coefficients: torch.Tensor,
    basis: tuple[str, ...],
    coarse_lattice_size: int,
    config: ForwardRGConfig,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float, torch.Tensor]:
    action = _FunctionalWilsonLoopAction(basis=basis, coefficients=coefficients.to(config.device))
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


def _wilson_coupling(beta: float, basis: tuple[str, ...], device: torch.device) -> torch.Tensor:
    J_fine = torch.zeros(len(basis), device=device)
    J_fine[basis.index("plaquette")] = beta
    return J_fine


def train_forward_rg(
    beta_values: list[float],
    config: ForwardRGConfig | None = None,
    fine_ensembles: dict[float, torch.Tensor] | None = None,
    verbose: bool = True,
) -> tuple[ForwardRGHypernetwork, ConditionedSpatialGaugeCovariantBlocker, ForwardRGResult]:
    config = config or ForwardRGConfig()
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    basis = config.basis

    if fine_ensembles is None:
        fine_ensembles = {}
        for beta in beta_values:
            fine_cfg = RGTrainingConfig(
                fine_lattice_size=config.fine_lattice_size,
                fine_beta=beta,
                coarse_beta_init=tree_level_coarse_beta(beta),
                n_fine_samples=config.n_fine_samples,
                n_model_samples=config.n_model_samples,
                sampler_burn_in=config.sampler_burn_in,
                sampler_thin=config.sampler_thin,
                hmc_steps=config.hmc_steps,
                hmc_step_size=config.hmc_step_size,
                device=config.device,
                seed=config.seed + int(10 * beta),
                basis=basis,
                measurement_set=config.measurement_set,
                evaluation_measurement_set=config.evaluation_measurement_set,
                blocker_type="fixed",
            )
            fine_ensembles[beta] = generate_fine_ensemble(fine_cfg).to(device)
    else:
        fine_ensembles = {beta: ensemble.to(device) for beta, ensemble in fine_ensembles.items()}

    coarse_lattice_size = next(iter(fine_ensembles.values())).shape[-1] // 2
    hyper = ForwardRGHypernetwork(
        coupling_dim=len(basis),
        hidden_dim=config.hidden_dim,
        z_phi_dim=config.z_phi_dim,
    ).to(device)
    blocker = ConditionedSpatialGaugeCovariantBlocker(
        hidden_dim=config.blocker_hidden_dim,
        kernel_size=config.blocker_kernel_size,
        context_dim=config.z_phi_dim,
    ).to(device)
    optimizer = torch.optim.Adam(
        list(hyper.parameters()) + list(blocker.parameters()),
        lr=config.learning_rate,
    )
    model_states: dict[float, torch.Tensor | None] = {beta: None for beta in beta_values}
    history: list[dict[str, float]] = []

    for epoch in range(config.epochs):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device)
        total_mmd = torch.tensor(0.0, device=device)
        total_mean = torch.tensor(0.0, device=device)
        total_contrastive = torch.tensor(0.0, device=device)
        total_acceptance = 0.0

        for beta in beta_values:
            J_fine = _wilson_coupling(beta, basis, device)
            J_coarse, z_phi = hyper.predict_forward_rg(J_fine)
            blocked = blocker(fine_ensembles[beta], z_phi)
            data_obs = loop_observables(blocked, basis)

            with torch.no_grad():
                model_samples, acceptance_rate, model_states[beta] = _sample_model_ensemble(
                    coefficients=J_coarse.detach(),
                    basis=basis,
                    coarse_lattice_size=coarse_lattice_size,
                    config=config,
                    initial_state=model_states[beta],
                )
                model_obs = loop_observables(model_samples, basis)

            distribution_mismatch = measurement_distribution_mmd(
                blocked,
                model_samples.detach(),
                measurement_names=config.measurement_set,
                bandwidth=config.mmd_bandwidth,
            )
            mean_mismatch = torch.mean((data_obs - model_obs.detach()) ** 2)
            contrastive_loss = torch.dot(J_coarse, model_obs.detach() - data_obs)
            coeff_penalty = J_coarse[1:].square().sum()
            loss = (
                contrastive_loss
                + config.distribution_loss_weight * distribution_mismatch
                + config.mean_loss_weight * mean_mismatch
                + config.path_sparsity_weight * blocker.regularization_loss()
                + config.coefficient_l2 * coeff_penalty
            )
            total_loss = total_loss + loss
            total_mmd = total_mmd + distribution_mismatch
            total_mean = total_mean + mean_mismatch
            total_contrastive = total_contrastive + contrastive_loss
            total_acceptance += acceptance_rate

        total_loss = total_loss / max(len(beta_values), 1)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(hyper.parameters()) + list(blocker.parameters()),
            config.gradient_clip,
        )
        optimizer.step()

        record = {
            "epoch": float(epoch),
            "loss": float(total_loss.detach().cpu()),
            "distribution_mismatch": float((total_mmd / max(len(beta_values), 1)).detach().cpu()),
            "mean_mismatch": float((total_mean / max(len(beta_values), 1)).detach().cpu()),
            "contrastive_loss": float((total_contrastive / max(len(beta_values), 1)).detach().cpu()),
            "acceptance_rate": float(total_acceptance / max(len(beta_values), 1)),
        }
        history.append(record)
        if verbose and (epoch % 25 == 0 or epoch == config.epochs - 1):
            print(
                f"epoch {epoch:4d}  loss={record['loss']:.4f}  "
                f"mmd={record['distribution_mismatch']:.4f}  "
                f"mean={record['mean_mismatch']:.4f}"
            )

    with torch.no_grad():
        predicted = []
        latents = []
        for beta in beta_values:
            J_fine = _wilson_coupling(beta, basis, device)
            J_coarse, z_phi = hyper.predict_forward_rg(J_fine)
            predicted.append(J_coarse.detach().cpu().tolist())
            latents.append(z_phi.detach().cpu().tolist())

    result = ForwardRGResult(
        history=history,
        beta_values=beta_values,
        basis=basis,
        predicted_J_coarse=predicted,
        latent_codes=latents,
        blocker_summary=blocker.summary(),
    )
    return hyper, blocker, result


def save_forward_rg_checkpoint(
    path: str | Path,
    hypernetwork: ForwardRGHypernetwork,
    blocker: ConditionedSpatialGaugeCovariantBlocker,
    config: ForwardRGConfig,
    result: ForwardRGResult | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": config.to_dict(),
            "hypernetwork_state": hypernetwork.state_dict(),
            "blocker_state": blocker.state_dict(),
            "result": asdict(result) if result is not None else None,
        },
        path,
    )
    return path


def load_forward_rg_checkpoint(
    path: str | Path,
    device: str = "cpu",
) -> tuple[ForwardRGHypernetwork, ConditionedSpatialGaugeCovariantBlocker, ForwardRGConfig, ForwardRGResult | None]:
    payload = torch.load(Path(path), map_location=device)
    config_dict = payload["config"]
    config_dict["basis"] = tuple(config_dict["basis"])
    config_dict["measurement_set"] = tuple(config_dict["measurement_set"])
    config_dict["evaluation_measurement_set"] = tuple(config_dict["evaluation_measurement_set"])
    config = ForwardRGConfig(**config_dict)
    hyper = ForwardRGHypernetwork(
        coupling_dim=len(config.basis),
        hidden_dim=config.hidden_dim,
        z_phi_dim=config.z_phi_dim,
    ).to(device)
    blocker = ConditionedSpatialGaugeCovariantBlocker(
        hidden_dim=config.blocker_hidden_dim,
        kernel_size=config.blocker_kernel_size,
        context_dim=config.z_phi_dim,
    ).to(device)
    hyper.load_state_dict(payload["hypernetwork_state"])
    blocker.load_state_dict(payload["blocker_state"])
    result_data = payload.get("result")
    result = None
    if result_data is not None:
        result_data["basis"] = tuple(result_data["basis"])
        result = ForwardRGResult(**result_data)
    return hyper, blocker, config, result
