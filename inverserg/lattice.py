import math

import torch


def regularize(theta: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(theta), torch.cos(theta))


def _as_batched(field: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if field.dim() == 3:
        return field.unsqueeze(0), True
    if field.dim() != 4:
        raise ValueError(f"Expected [2, L, L] or [B, 2, L, L], got shape {tuple(field.shape)}")
    return field, False


def plaquette_angles(field: torch.Tensor) -> torch.Tensor:
    field, squeezed = _as_batched(field)
    ux = field[:, 0]
    uy = field[:, 1]
    plaquettes = ux - uy - torch.roll(ux, shifts=-1, dims=-1) + torch.roll(uy, shifts=-1, dims=-2)
    plaquettes = regularize(plaquettes)
    return plaquettes.squeeze(0) if squeezed else plaquettes


def rectangle_x_angles(field: torch.Tensor) -> torch.Tensor:
    field, squeezed = _as_batched(field)
    ux = field[:, 0]
    uy = field[:, 1]
    rect = (
        ux
        + torch.roll(ux, shifts=-1, dims=-2)
        + torch.roll(uy, shifts=-2, dims=-2)
        - torch.roll(ux, shifts=(-1, -1), dims=(-2, -1))
        - torch.roll(ux, shifts=-1, dims=-1)
        - uy
    )
    rect = regularize(rect)
    return rect.squeeze(0) if squeezed else rect


def rectangle_y_angles(field: torch.Tensor) -> torch.Tensor:
    field, squeezed = _as_batched(field)
    ux = field[:, 0]
    uy = field[:, 1]
    rect = (
        ux
        + torch.roll(uy, shifts=-1, dims=-2)
        + torch.roll(uy, shifts=(-1, -1), dims=(-2, -1))
        - torch.roll(ux, shifts=-2, dims=-1)
        - torch.roll(uy, shifts=-1, dims=-1)
        - uy
    )
    rect = regularize(rect)
    return rect.squeeze(0) if squeezed else rect


def wilson_loop_angles(field: torch.Tensor, extent_x: int, extent_y: int) -> torch.Tensor:
    field, squeezed = _as_batched(field)
    ux = field[:, 0]
    uy = field[:, 1]

    loop = torch.zeros_like(ux)
    for step in range(extent_x):
        loop = loop + torch.roll(ux, shifts=-step, dims=-2)
    for step in range(extent_y):
        loop = loop + torch.roll(uy, shifts=(-extent_x, -step), dims=(-2, -1))
    for step in range(extent_x):
        loop = loop - torch.roll(ux, shifts=(-step, -extent_y), dims=(-2, -1))
    for step in range(extent_y):
        loop = loop - torch.roll(uy, shifts=(0, -step), dims=(-2, -1))

    loop = regularize(loop)
    return loop.squeeze(0) if squeezed else loop


def mean_plaquette(field: torch.Tensor) -> torch.Tensor:
    return torch.cos(plaquette_angles(field)).mean()


def topological_charge(field: torch.Tensor) -> torch.Tensor:
    plaquettes = plaquette_angles(field)
    return torch.round(plaquettes.sum(dim=(-2, -1)) / (2 * math.pi))


def loop_observables(field: torch.Tensor, basis: tuple[str, ...]) -> torch.Tensor:
    obs = []
    for name in basis:
        if name == "plaquette":
            angles = plaquette_angles(field)
        elif name == "rectangle_x":
            angles = rectangle_x_angles(field)
        elif name == "rectangle_y":
            angles = rectangle_y_angles(field)
        else:
            raise ValueError(f"Unknown loop basis element: {name}")
        obs.append(torch.cos(angles).mean())
    return torch.stack(obs)


def mean_observables(field: torch.Tensor, basis: tuple[str, ...] = ("plaquette", "rectangle_x", "rectangle_y")) -> dict[str, float]:
    values = loop_observables(field, basis)
    return {name: float(value.detach().cpu()) for name, value in zip(basis, values)}
