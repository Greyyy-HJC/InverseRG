import torch

from .lattice import (
    mean_plaquette,
    plaquette_angles,
    rectangle_x_angles,
    rectangle_y_angles,
    topological_charge,
    wilson_loop_angles,
)


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


def measurement_samples(field: torch.Tensor, measurement_names: tuple[str, ...]) -> dict[str, torch.Tensor]:
    if field.dim() == 3:
        field = field.unsqueeze(0)
    samples: dict[str, torch.Tensor] = {}
    for measurement_name in measurement_names:
        if measurement_name == "plaquette":
            values = torch.cos(plaquette_angles(field)).mean(dim=(-2, -1))
        elif measurement_name == "rectangle_x":
            values = torch.cos(rectangle_x_angles(field)).mean(dim=(-2, -1))
        elif measurement_name == "rectangle_y":
            values = torch.cos(rectangle_y_angles(field)).mean(dim=(-2, -1))
        elif measurement_name == "topological_charge":
            values = topological_charge(field).float()
        elif measurement_name.startswith("wilson_"):
            _, extents = measurement_name.split("_", maxsplit=1)
            extent_x, extent_y = (int(value) for value in extents.split("x"))
            values = torch.cos(wilson_loop_angles(field, extent_x=extent_x, extent_y=extent_y)).mean(dim=(-2, -1))
        else:
            raise ValueError(f"Unknown measurement: {measurement_name}")
        samples[measurement_name] = values.detach().cpu()
    return samples
