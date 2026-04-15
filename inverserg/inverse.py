import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .actions import LocalWilsonLoopAction
from .blocking import ConditionedSpatialGaugeCovariantBlocker
from .forward_rg import ForwardRGHypernetwork
from .lattice import (
    plaquette_angles,
    rectangle_x_angles,
    rectangle_y_angles,
    regularize,
)
from .training import RGTrainingConfig, generate_fine_ensemble, measurement_distribution_mmd


def _as_batched_field(field: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if field.dim() == 3:
        return field.unsqueeze(0), True
    if field.dim() != 4:
        raise ValueError(f"Expected [2, L, L] or [B, 2, L, L], got {tuple(field.shape)}")
    return field, False


def gauge_transform(field: torch.Tensor, site_angles: torch.Tensor) -> torch.Tensor:
    field, squeezed = _as_batched_field(field)
    if site_angles.dim() == 2:
        site_angles = site_angles.unsqueeze(0)
    if site_angles.shape[0] == 1 and field.shape[0] > 1:
        site_angles = site_angles.expand(field.shape[0], -1, -1)
    ux = regularize(field[:, 0] + site_angles - torch.roll(site_angles, shifts=-1, dims=-2))
    uy = regularize(field[:, 1] + site_angles - torch.roll(site_angles, shifts=-1, dims=-1))
    transformed = torch.stack([ux, uy], dim=1)
    return transformed.squeeze(0) if squeezed else transformed


def prolong_site_gauge(site_angles: torch.Tensor) -> torch.Tensor:
    if site_angles.dim() == 2:
        site_angles = site_angles.unsqueeze(0)
    return site_angles.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)


def canonical_prolongation(coarse_field: torch.Tensor) -> torch.Tensor:
    coarse_field, squeezed = _as_batched_field(coarse_field)
    batch_size, _, Lc, _ = coarse_field.shape
    fine = torch.zeros((batch_size, 2, 2 * Lc, 2 * Lc), device=coarse_field.device, dtype=coarse_field.dtype)
    fine[:, 0, 1::2, :] = coarse_field[:, 0].repeat_interleave(2, dim=-1)
    fine[:, 1, :, 1::2] = coarse_field[:, 1].repeat_interleave(2, dim=-2)
    fine = regularize(fine)
    return fine.squeeze(0) if squeezed else fine


def _embed_sublattice(coefficients: torch.Tensor, x_offset: int, y_offset: int) -> torch.Tensor:
    batch_size, Lc, _ = coefficients.shape
    full = torch.zeros((batch_size, 2 * Lc, 2 * Lc), device=coefficients.device, dtype=coefficients.dtype)
    full[:, x_offset::2, y_offset::2] = coefficients
    return full


def _plaquette_loop_field(coefficients: torch.Tensor, x_offset: int, y_offset: int) -> torch.Tensor:
    potential = _embed_sublattice(coefficients, x_offset, y_offset)
    ux = potential - torch.roll(potential, shifts=1, dims=-1)
    uy = -potential + torch.roll(potential, shifts=1, dims=-2)
    return torch.stack([ux, uy], dim=1)


def _rectangle_x_loop_field(coefficients: torch.Tensor) -> torch.Tensor:
    potential = _embed_sublattice(coefficients, 0, 0)
    ux = (
        potential
        + torch.roll(potential, shifts=1, dims=-2)
        - torch.roll(potential, shifts=1, dims=-1)
        - torch.roll(potential, shifts=(1, 1), dims=(-2, -1))
    )
    uy = -potential + torch.roll(potential, shifts=2, dims=-2)
    return torch.stack([ux, uy], dim=1)


def _rectangle_y_loop_field(coefficients: torch.Tensor) -> torch.Tensor:
    potential = _embed_sublattice(coefficients, 0, 0)
    ux = potential - torch.roll(potential, shifts=2, dims=-1)
    uy = (
        -potential
        - torch.roll(potential, shifts=1, dims=-1)
        + torch.roll(potential, shifts=1, dims=-2)
        + torch.roll(potential, shifts=(1, 1), dims=(-2, -1))
    )
    return torch.stack([ux, uy], dim=1)


def closed_loop_residual_field(residual_coefficients: torch.Tensor) -> torch.Tensor:
    if residual_coefficients.dim() != 4 or residual_coefficients.shape[1] != 6:
        raise ValueError(f"Expected [B, 6, Lc, Lc], got {tuple(residual_coefficients.shape)}")
    p00, p10, p01, p11, rect_x, rect_y = residual_coefficients.unbind(dim=1)
    fine = (
        _plaquette_loop_field(p00, 0, 0)
        + _plaquette_loop_field(p10, 1, 0)
        + _plaquette_loop_field(p01, 0, 1)
        + _plaquette_loop_field(p11, 1, 1)
        + _rectangle_x_loop_field(rect_x)
        + _rectangle_y_loop_field(rect_y)
    )
    return regularize(fine)


def build_fine_proposal(coarse_field: torch.Tensor, residual_coefficients: torch.Tensor) -> torch.Tensor:
    coarse_field, squeezed = _as_batched_field(coarse_field)
    proposal = regularize(canonical_prolongation(coarse_field) + closed_loop_residual_field(residual_coefficients))
    return proposal.squeeze(0) if squeezed else proposal


def _coarse_gauge_invariant_features(coarse_field: torch.Tensor) -> torch.Tensor:
    coarse_field, _ = _as_batched_field(coarse_field)
    plaquette = plaquette_angles(coarse_field)
    rect_x = rectangle_x_angles(coarse_field)
    rect_y = rectangle_y_angles(coarse_field)
    return torch.stack(
        [
            torch.cos(plaquette),
            torch.sin(plaquette),
            torch.cos(rect_x),
            torch.sin(rect_x),
            torch.cos(rect_y),
            torch.sin(rect_y),
        ],
        dim=1,
    )


def _broadcast_vector(vector: torch.Tensor, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if vector.dim() == 1:
        vector = vector.unsqueeze(0)
    if vector.shape[0] == 1 and batch_size > 1:
        vector = vector.expand(batch_size, -1)
    if vector.shape[0] != batch_size:
        raise ValueError(f"Expected batch {batch_size}, got {vector.shape[0]}")
    return vector.to(device=device, dtype=dtype)


@dataclass
class InverseRGConfig:
    basis: tuple[str, ...] = ("plaquette", "rectangle_x", "rectangle_y")
    fine_measurement_set: tuple[str, ...] = (
        "plaquette",
        "rectangle_x",
        "rectangle_y",
        "wilson_2x2",
        "topological_charge",
    )
    coarse_measurement_set: tuple[str, ...] = ("plaquette", "rectangle_x", "rectangle_y")
    hidden_dim: int = 32
    noise_channels: int = 6
    residual_channels: int = 6
    context_dim: int = 22
    refinement_steps: int = 5
    refinement_step_size: float = 0.05
    lambda_block: float = 10.0
    lambda_fine: float = 1.0
    lambda_obs: float = 1.0
    train_roundtrip_weight: float = 10.0
    train_fine_action_weight: float = 1.0
    train_mmd_weight: float = 2.0
    mmd_bandwidth: float = 0.5
    epochs: int = 200
    learning_rate: float = 1e-3
    fine_lattice_size: int = 8
    n_fine_samples: int = 24
    sampler_burn_in: int = 48
    sampler_thin: int = 4
    hmc_steps: int = 8
    hmc_step_size: float = 0.15
    device: str = "cpu"
    seed: int = 29

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class InverseRGResult:
    history: list[dict[str, float]]
    predicted_J_fine: list[float]
    roundtrip_loss: float
    fine_action: float
    fine_mmd: float
    generated_fine_ensemble: torch.Tensor | None = None
    blocked_generated_ensemble: torch.Tensor | None = None
    target_coarse_ensemble: torch.Tensor | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        for key in ("generated_fine_ensemble", "blocked_generated_ensemble", "target_coarse_ensemble"):
            data.pop(key, None)
        return data

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path


class EquivariantInverseProposalNet(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 32,
        noise_channels: int = 6,
        residual_channels: int = 6,
        context_dim: int = 22,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.noise_channels = noise_channels
        self.residual_channels = residual_channels
        self.context_dim = context_dim
        self.conv1 = nn.Conv2d(6 + noise_channels, hidden_dim, 3, padding=0)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.conv_out = nn.Conv2d(hidden_dim, residual_channels, 1)
        self.film1 = nn.Linear(context_dim, 2 * hidden_dim)
        self.film2 = nn.Linear(context_dim, 2 * hidden_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)
        nn.init.normal_(self.film1.weight, mean=0.0, std=2e-2)
        nn.init.zeros_(self.film1.bias)
        nn.init.normal_(self.film2.weight, mean=0.0, std=2e-2)
        nn.init.zeros_(self.film2.bias)

    def _apply_film(self, x: torch.Tensor, context: torch.Tensor, layer: nn.Linear) -> torch.Tensor:
        gamma, beta = layer(context).chunk(2, dim=-1)
        return x * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]

    def sample_noise(
        self,
        batch_size: int,
        lattice_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.randn((batch_size, self.noise_channels, lattice_size, lattice_size), device=device, dtype=dtype)

    def forward(
        self,
        coarse_field: torch.Tensor,
        J_coarse: torch.Tensor,
        J_fine: torch.Tensor,
        z_phi: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        coarse_field, _ = _as_batched_field(coarse_field)
        batch_size = coarse_field.shape[0]
        invariant_features = _coarse_gauge_invariant_features(coarse_field)
        if noise is None:
            noise = self.sample_noise(
                batch_size=batch_size,
                lattice_size=coarse_field.shape[-1],
                device=coarse_field.device,
                dtype=coarse_field.dtype,
            )
        context = torch.cat(
            [
                _broadcast_vector(J_coarse, batch_size, coarse_field.device, coarse_field.dtype),
                _broadcast_vector(J_fine, batch_size, coarse_field.device, coarse_field.dtype),
                _broadcast_vector(z_phi, batch_size, coarse_field.device, coarse_field.dtype),
            ],
            dim=-1,
        )
        x = torch.cat([invariant_features, noise], dim=1)
        x = F.pad(x, [1] * 4, mode="circular")
        x = self._apply_film(F.relu(self.conv1(x)), context, self.film1)
        x = self._apply_film(F.relu(self.conv2(F.relu(x))), context, self.film2)
        return self.conv_out(F.relu(x))

    def proposal(
        self,
        coarse_field: torch.Tensor,
        J_coarse: torch.Tensor,
        J_fine: torch.Tensor,
        z_phi: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residuals = self.forward(coarse_field, J_coarse, J_fine, z_phi, noise=noise)
        return build_fine_proposal(coarse_field, residuals), residuals


class ConditionedFineAction(nn.Module):
    def __init__(
        self,
        J_fine: torch.Tensor,
        coarse_target: torch.Tensor,
        blocker: ConditionedSpatialGaugeCovariantBlocker,
        z_phi: torch.Tensor,
        config: InverseRGConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.coarse_target = coarse_target
        self.blocker = blocker
        self.z_phi = z_phi
        coeffs = J_fine.detach().cpu()
        self.fine_action = LocalWilsonLoopAction(basis=config.basis, initial_coefficients=coeffs).to(coarse_target.device)

    def fine_action_value(self, fine_field: torch.Tensor) -> torch.Tensor:
        return self.fine_action.per_configuration_action(fine_field).mean()

    def block_loss(self, fine_field: torch.Tensor) -> torch.Tensor:
        blocked = self.blocker(fine_field, self.z_phi)
        return torch.mean(1.0 - torch.cos(blocked - self.coarse_target))

    def observable_loss(self, fine_field: torch.Tensor) -> torch.Tensor:
        blocked = self.blocker(fine_field, self.z_phi)
        return measurement_distribution_mmd(
            blocked,
            self.coarse_target,
            measurement_names=self.config.coarse_measurement_set,
            bandwidth=self.config.mmd_bandwidth,
        )

    def total_energy(self, fine_field: torch.Tensor) -> torch.Tensor:
        return (
            self.config.lambda_block * self.block_loss(fine_field)
            + self.config.lambda_fine * self.fine_action_value(fine_field)
            + self.config.lambda_obs * self.observable_loss(fine_field)
        )

    def differentiable_refinement_energy(self, fine_field: torch.Tensor) -> torch.Tensor:
        return (
            self.config.lambda_block * self.block_loss(fine_field)
            + self.config.lambda_fine * self.fine_action_value(fine_field)
        )

    def forward(self, fine_field: torch.Tensor) -> torch.Tensor:
        return self.total_energy(fine_field)


def equivariant_refinement(
    theta_fine: torch.Tensor,
    energy: ConditionedFineAction,
    steps: int,
    step_size: float,
    differentiable: bool = False,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    theta = theta_fine
    history: list[dict[str, float]] = []
    if differentiable:
        for step in range(steps):
            theta_var = theta.requires_grad_(True)
            total_energy = energy.differentiable_refinement_energy(theta_var)
            (grad,) = torch.autograd.grad(total_energy, theta_var, create_graph=True)
            theta = regularize(theta_var - step_size * grad)
            history.append(
                {
                    "step": float(step),
                    "block_loss": float(energy.block_loss(theta).detach().cpu()),
                    "total_energy": float(total_energy.detach().cpu()),
                }
            )
        return theta, history

    theta = theta.detach()
    for step in range(steps):
        current_block = energy.block_loss(theta).detach()
        theta_var = theta.clone().requires_grad_(True)
        total_energy = energy.total_energy(theta_var)
        (grad,) = torch.autograd.grad(total_energy, theta_var)
        local_step = step_size
        candidate = theta
        for _ in range(8):
            trial = regularize(theta - local_step * grad.detach())
            if energy.block_loss(trial).detach() <= current_block + 1e-7:
                candidate = trial
                break
            local_step *= 0.5
        theta = candidate.detach()
        history.append(
            {
                "step": float(step),
                "block_loss": float(energy.block_loss(theta).detach().cpu()),
                "total_energy": float(total_energy.detach().cpu()),
            }
        )
    return theta, history


def inverse_rg_step(
    coarse_field: torch.Tensor,
    J_coarse: torch.Tensor,
    J_fine: torch.Tensor,
    z_phi: torch.Tensor,
    blocker: ConditionedSpatialGaugeCovariantBlocker,
    proposal_net: EquivariantInverseProposalNet,
    config: InverseRGConfig | None = None,
    noise: torch.Tensor | None = None,
) -> InverseRGResult:
    config = config or InverseRGConfig()
    coarse_field, _ = _as_batched_field(coarse_field)
    proposal, _ = proposal_net.proposal(coarse_field, J_coarse, J_fine, z_phi, noise=noise)
    energy = ConditionedFineAction(
        J_fine=J_fine,
        coarse_target=coarse_field,
        blocker=blocker,
        z_phi=z_phi,
        config=config,
    )
    refined, history = equivariant_refinement(
        proposal,
        energy,
        steps=config.refinement_steps,
        step_size=config.refinement_step_size,
        differentiable=False,
    )
    blocked = blocker(refined, z_phi)
    roundtrip = torch.mean(1.0 - torch.cos(blocked - coarse_field))
    fine_action = energy.fine_action_value(refined)
    fine_mmd = measurement_distribution_mmd(
        refined,
        canonical_prolongation(coarse_field),
        measurement_names=config.fine_measurement_set,
        bandwidth=config.mmd_bandwidth,
    )
    return InverseRGResult(
        history=history,
        predicted_J_fine=J_fine.detach().cpu().tolist(),
        roundtrip_loss=float(roundtrip.detach().cpu()),
        fine_action=float(fine_action.detach().cpu()),
        fine_mmd=float(fine_mmd.detach().cpu()),
        generated_fine_ensemble=refined.detach().cpu(),
        blocked_generated_ensemble=blocked.detach().cpu(),
        target_coarse_ensemble=coarse_field.detach().cpu(),
    )


def _wilson_coupling(beta: float, basis: tuple[str, ...], device: torch.device) -> torch.Tensor:
    couplings = torch.zeros(len(basis), device=device)
    couplings[basis.index("plaquette")] = beta
    return couplings


def train_inverse_rg(
    beta_values: list[float],
    forward_model: ForwardRGHypernetwork,
    blocker: ConditionedSpatialGaugeCovariantBlocker,
    config: InverseRGConfig | None = None,
    fine_ensembles: dict[float, torch.Tensor] | None = None,
    verbose: bool = True,
) -> tuple[EquivariantInverseProposalNet, InverseRGResult]:
    config = config or InverseRGConfig()
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    forward_model = forward_model.to(device).eval()
    blocker = blocker.to(device).eval()
    for param in forward_model.parameters():
        param.requires_grad_(False)
    for param in blocker.parameters():
        param.requires_grad_(False)

    if fine_ensembles is None:
        fine_ensembles = {}
        for beta in beta_values:
            fine_cfg = RGTrainingConfig(
                fine_lattice_size=config.fine_lattice_size,
                fine_beta=beta,
                n_fine_samples=config.n_fine_samples,
                n_model_samples=config.n_fine_samples,
                sampler_burn_in=config.sampler_burn_in,
                sampler_thin=config.sampler_thin,
                hmc_steps=config.hmc_steps,
                hmc_step_size=config.hmc_step_size,
                device=config.device,
                seed=config.seed + int(10 * beta),
                basis=config.basis,
            )
            fine_ensembles[beta] = generate_fine_ensemble(fine_cfg).to(device)
    else:
        fine_ensembles = {beta: ensemble.to(device) for beta, ensemble in fine_ensembles.items()}

    proposal_net = EquivariantInverseProposalNet(
        hidden_dim=config.hidden_dim,
        noise_channels=config.noise_channels,
        residual_channels=config.residual_channels,
        context_dim=config.context_dim,
    ).to(device)
    optimizer = torch.optim.Adam(proposal_net.parameters(), lr=config.learning_rate)
    history: list[dict[str, float]] = []
    last_generated = None
    last_blocked = None
    last_coarse = None
    last_J_fine = None

    for epoch in range(config.epochs):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device)
        total_roundtrip = torch.tensor(0.0, device=device)
        total_fine_action = torch.tensor(0.0, device=device)
        total_mmd = torch.tensor(0.0, device=device)

        for beta in beta_values:
            fine_batch = fine_ensembles[beta]
            J_fine = _wilson_coupling(beta, config.basis, device)
            with torch.no_grad():
                J_coarse, z_phi = forward_model.predict_forward_rg(J_fine)
                coarse_batch = blocker(fine_batch, z_phi)

            proposal, _ = proposal_net.proposal(coarse_batch, J_coarse, J_fine, z_phi)
            energy = ConditionedFineAction(
                J_fine=J_fine,
                coarse_target=coarse_batch,
                blocker=blocker,
                z_phi=z_phi,
                config=config,
            )
            refined, _ = equivariant_refinement(
                proposal,
                energy,
                steps=config.refinement_steps,
                step_size=config.refinement_step_size,
                differentiable=True,
            )
            blocked_refined = blocker(refined, z_phi)
            roundtrip = torch.mean(1.0 - torch.cos(blocked_refined - coarse_batch))
            fine_action = energy.fine_action_value(refined)
            fine_mmd = measurement_distribution_mmd(
                refined,
                fine_batch,
                measurement_names=config.fine_measurement_set,
                bandwidth=config.mmd_bandwidth,
            )
            loss = (
                config.train_roundtrip_weight * roundtrip
                + config.train_fine_action_weight * fine_action
                + config.train_mmd_weight * fine_mmd
            )
            total_loss = total_loss + loss
            total_roundtrip = total_roundtrip + roundtrip
            total_fine_action = total_fine_action + fine_action
            total_mmd = total_mmd + fine_mmd
            last_generated = refined.detach().cpu()
            last_blocked = blocked_refined.detach().cpu()
            last_coarse = coarse_batch.detach().cpu()
            last_J_fine = J_fine.detach().cpu().tolist()

        total_loss = total_loss / max(len(beta_values), 1)
        total_loss.backward()
        optimizer.step()

        record = {
            "epoch": float(epoch),
            "loss": float(total_loss.detach().cpu()),
            "roundtrip_loss": float((total_roundtrip / max(len(beta_values), 1)).detach().cpu()),
            "fine_action": float((total_fine_action / max(len(beta_values), 1)).detach().cpu()),
            "fine_mmd": float((total_mmd / max(len(beta_values), 1)).detach().cpu()),
        }
        history.append(record)
        if verbose and (epoch % 25 == 0 or epoch == config.epochs - 1):
            print(
                f"epoch {epoch:4d}  loss={record['loss']:.4f}  "
                f"roundtrip={record['roundtrip_loss']:.4f}  "
                f"fine_mmd={record['fine_mmd']:.4f}"
            )

    result = InverseRGResult(
        history=history,
        predicted_J_fine=last_J_fine or [0.0, 0.0, 0.0],
        roundtrip_loss=history[-1]["roundtrip_loss"] if history else 0.0,
        fine_action=history[-1]["fine_action"] if history else 0.0,
        fine_mmd=history[-1]["fine_mmd"] if history else 0.0,
        generated_fine_ensemble=last_generated,
        blocked_generated_ensemble=last_blocked,
        target_coarse_ensemble=last_coarse,
    )
    return proposal_net, result
