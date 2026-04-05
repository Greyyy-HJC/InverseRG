import torch

from .lattice import mean_plaquette, plaquette_angles, topological_charge, wilson_loop_angles


def mean_wilson_loop(field: torch.Tensor, extent_x: int, extent_y: int) -> torch.Tensor:
    return torch.cos(wilson_loop_angles(field, extent_x=extent_x, extent_y=extent_y)).mean()


def summarize_observables(
    field: torch.Tensor,
    loop_sizes: tuple[tuple[int, int], ...] = ((1, 1), (1, 2), (2, 2)),
) -> dict[str, float]:
    summary = {
        "plaquette": float(mean_plaquette(field).detach().cpu()),
        "topological_charge": float(topological_charge(field).float().mean().detach().cpu()),
        "plaquette_angle_mean": float(plaquette_angles(field).mean().detach().cpu()),
    }
    for extent_x, extent_y in loop_sizes:
        key = f"wilson_{extent_x}x{extent_y}"
        summary[key] = float(mean_wilson_loop(field, extent_x, extent_y).detach().cpu())
    return summary
