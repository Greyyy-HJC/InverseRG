import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import torch
from torch import nn

from .baselines import tree_level_coarse_beta
from .training import RGTrainingConfig, train_learned_rg


class RGMonotone(nn.Module):
    """Scalar function C: R^d -> R over coupling space.

    The RG flow is defined as dJ/dl = -grad_J C(J).
    Integrating from l=0 to l=1 maps J_fine to J_coarse.
    """

    def __init__(self, coupling_dim: int = 3, hidden_dim: int = 64, n_layers: int = 3) -> None:
        super().__init__()
        self.coupling_dim = coupling_dim
        layers: list[nn.Module] = []
        in_dim = coupling_dim
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, J: torch.Tensor) -> torch.Tensor:
        """Evaluate C(J).  Input: [..., d], Output: [...]."""
        return self.net(J).squeeze(-1)

    def beta_function(self, J: torch.Tensor) -> torch.Tensor:
        """Compute -grad_J C(J) (the RG beta function / flow velocity)."""
        with torch.enable_grad():
            J_in = J.detach().clone().requires_grad_(True)
            C = self.forward(J_in)
            C_scalar = C.sum() if C.dim() > 0 else C
            (grad_C,) = torch.autograd.grad(C_scalar, J_in)
        return -grad_C.detach()


def rg_flow_step(
    monotone: RGMonotone,
    J_fine: torch.Tensor,
    n_steps: int = 10,
    create_graph: bool = True,
) -> torch.Tensor:
    """Euler integration of dJ/dl = -grad C(J) from l=0 to l=1.

    When *create_graph* is True the computation graph is retained so
    that the loss can back-propagate through the integration to the
    monotone network parameters.
    """
    dt = 1.0 / n_steps
    with torch.enable_grad():
        J = J_fine.detach().clone().requires_grad_(True)
        for _ in range(n_steps):
            C = monotone(J)
            C_scalar = C.sum() if C.dim() > 0 else C
            (grad_C,) = torch.autograd.grad(
                C_scalar, J, create_graph=create_graph, retain_graph=create_graph,
            )
            J = J - dt * grad_C
    return J


# ---------------------------------------------------------------------------
# Multi-beta data collection (Stage 1)
# ---------------------------------------------------------------------------


@dataclass
class CollectedRGData:
    """(J_fine, J_coarse) pairs collected across a grid of beta values."""

    beta_values: list[float]
    J_fine: torch.Tensor
    J_coarse: torch.Tensor
    basis: tuple[str, ...]
    metrics: list[dict] = field(default_factory=list)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "beta_values": self.beta_values,
            "J_fine": self.J_fine.tolist(),
            "J_coarse": self.J_coarse.tolist(),
            "basis": list(self.basis),
            "metrics": self.metrics,
        }
        path.write_text(json.dumps(data, indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "CollectedRGData":
        data = json.loads(Path(path).read_text())
        return cls(
            beta_values=data["beta_values"],
            J_fine=torch.tensor(data["J_fine"], dtype=torch.float32),
            J_coarse=torch.tensor(data["J_coarse"], dtype=torch.float32),
            basis=tuple(data["basis"]),
            metrics=data.get("metrics", []),
        )


def collect_multi_beta_data(
    beta_values: list[float],
    config_template: RGTrainingConfig | None = None,
    cache_path: str | Path | None = None,
    verbose: bool = True,
) -> CollectedRGData:
    """Run Phase 1 training at each beta and collect (J_fine, J_coarse) pairs.

    If *cache_path* points to an existing file the data is loaded from disk.
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            if verbose:
                print(f"Loading cached data from {cache_path}")
            return CollectedRGData.load(cache_path)

    config_template = config_template or RGTrainingConfig()
    basis = config_template.basis

    J_fine_list: list[torch.Tensor] = []
    J_coarse_list: list[torch.Tensor] = []
    metrics_list: list[dict] = []

    for beta in beta_values:
        if verbose:
            print(f"Training at beta={beta:.2f} ...")

        cfg = replace(
            config_template,
            fine_beta=beta,
            coarse_beta_init=tree_level_coarse_beta(beta),
        )
        result = train_learned_rg(config=cfg)

        j_fine = torch.zeros(len(basis), dtype=torch.float32)
        j_fine[list(basis).index("plaquette")] = beta

        j_coarse = torch.tensor(
            [result.learned_action_coefficients[n] for n in basis],
            dtype=torch.float32,
        )
        J_fine_list.append(j_fine)
        J_coarse_list.append(j_coarse)

        entry = {
            "beta": beta,
            "baseline_mismatch": result.baseline_mismatch,
            "final_mismatch": result.final_mismatch,
            "learned_coefficients": result.learned_action_coefficients,
        }
        metrics_list.append(entry)

        if verbose:
            coeff_str = ", ".join(
                f"{k}={v:.4f}" for k, v in result.learned_action_coefficients.items()
            )
            print(f"  -> {coeff_str}")

    collected = CollectedRGData(
        beta_values=beta_values,
        J_fine=torch.stack(J_fine_list),
        J_coarse=torch.stack(J_coarse_list),
        basis=basis,
        metrics=metrics_list,
    )

    if cache_path is not None:
        collected.save(cache_path)
        if verbose:
            print(f"Saved to {cache_path}")

    return collected


# ---------------------------------------------------------------------------
# Monotone training (Stage 2)
# ---------------------------------------------------------------------------


@dataclass
class MonotoneTrainingConfig:
    hidden_dim: int = 64
    n_layers: int = 3
    n_euler_steps: int = 20
    learning_rate: float = 1e-3
    epochs: int = 500
    seed: int = 42
    flow_loss_weight: float = 1.0
    monotone_reg_weight: float = 0.01
    weight_decay: float = 1e-4

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MonotoneTrainingResult:
    history: list[dict[str, float]]
    predicted_J_coarse: list[list[float]]
    actual_J_coarse: list[list[float]]
    tree_level_J_coarse: list[list[float]]
    beta_values: list[float]
    basis: tuple[str, ...]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "MonotoneTrainingResult":
        data = json.loads(Path(path).read_text())
        data["basis"] = tuple(data["basis"])
        return cls(**data)


def train_rg_monotone(
    collected: CollectedRGData,
    config: MonotoneTrainingConfig | None = None,
    verbose: bool = True,
) -> tuple[RGMonotone, MonotoneTrainingResult]:
    """Fit C_theta so that its gradient flow maps J_fine -> J_coarse."""
    config = config or MonotoneTrainingConfig()
    torch.manual_seed(config.seed)

    coupling_dim = collected.J_fine.shape[1]
    monotone = RGMonotone(
        coupling_dim=coupling_dim,
        hidden_dim=config.hidden_dim,
        n_layers=config.n_layers,
    )

    optimizer = torch.optim.Adam(
        monotone.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    J_fine = collected.J_fine
    J_target = collected.J_coarse

    plaq_idx = list(collected.basis).index("plaquette")
    tree_level = torch.zeros_like(J_target)
    for i, beta in enumerate(collected.beta_values):
        tree_level[i, plaq_idx] = tree_level_coarse_beta(beta)

    history: list[dict[str, float]] = []

    for epoch in range(config.epochs):
        optimizer.zero_grad()

        J_pred = rg_flow_step(
            monotone, J_fine, n_steps=config.n_euler_steps, create_graph=True,
        )
        flow_loss = (J_pred - J_target).pow(2).mean()

        C_fine = monotone(J_fine.detach())
        C_coarse = monotone(J_pred.detach())
        monotone_violation = torch.relu(C_coarse - C_fine).mean()

        loss = config.flow_loss_weight * flow_loss + config.monotone_reg_weight * monotone_violation

        loss.backward()
        optimizer.step()

        record = {
            "epoch": epoch,
            "loss": float(loss.detach()),
            "flow_loss": float(flow_loss.detach()),
            "monotone_violation": float(monotone_violation.detach()),
        }
        history.append(record)

        if verbose and (epoch % 100 == 0 or epoch == config.epochs - 1):
            print(
                f"  epoch {epoch:4d}  loss={record['loss']:.6f}  "
                f"flow={record['flow_loss']:.6f}  mono_viol={record['monotone_violation']:.6f}"
            )

    with torch.no_grad():
        J_pred_final = rg_flow_step(
            monotone, J_fine, n_steps=config.n_euler_steps, create_graph=False,
        )

    result = MonotoneTrainingResult(
        history=history,
        predicted_J_coarse=J_pred_final.detach().tolist(),
        actual_J_coarse=J_target.tolist(),
        tree_level_J_coarse=tree_level.tolist(),
        beta_values=collected.beta_values,
        basis=collected.basis,
    )

    return monotone, result
