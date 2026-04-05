import torch
from torch import nn

from .lattice import regularize


def _circular_average(paths: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    sin_sum = (weights[:, :, None, None] * torch.sin(paths)).sum(dim=1)
    cos_sum = (weights[:, :, None, None] * torch.cos(paths)).sum(dim=1)
    return torch.atan2(sin_sum, cos_sum)


def _subsample_even_even(tensor: torch.Tensor) -> torch.Tensor:
    return tensor[..., 0::2, 0::2]


def _x_paths(field: torch.Tensor) -> torch.Tensor:
    ux = field[:, 0]
    uy = field[:, 1]
    straight = ux + torch.roll(ux, shifts=-1, dims=-2)
    up = uy + torch.roll(ux, shifts=-1, dims=-1) + torch.roll(ux, shifts=(-1, -1), dims=(-2, -1)) - torch.roll(uy, shifts=-2, dims=-2)
    down = -torch.roll(uy, shifts=1, dims=-1) + torch.roll(ux, shifts=1, dims=-1) + torch.roll(ux, shifts=(-1, 1), dims=(-2, -1)) + torch.roll(torch.roll(uy, shifts=1, dims=-1), shifts=-2, dims=-2)
    return regularize(torch.stack([_subsample_even_even(straight), _subsample_even_even(up), _subsample_even_even(down)], dim=1))


def _y_paths(field: torch.Tensor) -> torch.Tensor:
    ux = field[:, 0]
    uy = field[:, 1]
    straight = uy + torch.roll(uy, shifts=-1, dims=-1)
    right = ux + torch.roll(uy, shifts=-1, dims=-2) + torch.roll(uy, shifts=(-1, -1), dims=(-2, -1)) - torch.roll(ux, shifts=-2, dims=-1)
    left = -torch.roll(ux, shifts=1, dims=-2) + torch.roll(uy, shifts=1, dims=-2) + torch.roll(uy, shifts=(1, -1), dims=(-2, -1)) + torch.roll(torch.roll(ux, shifts=1, dims=-2), shifts=-2, dims=-1)
    return regularize(torch.stack([_subsample_even_even(straight), _subsample_even_even(right), _subsample_even_even(left)], dim=1))


class LearnableGaugeCovariantBlocker(nn.Module):
    def __init__(self, path_logits: torch.Tensor | None = None) -> None:
        super().__init__()
        if path_logits is None:
            path_logits = torch.tensor([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float32)
        self.path_logits = nn.Parameter(path_logits.clone().detach().float())

    def path_probabilities(self) -> torch.Tensor:
        return torch.softmax(self.path_logits, dim=-1)

    def forward(self, fine_field: torch.Tensor) -> torch.Tensor:
        if fine_field.dim() == 3:
            fine_field = fine_field.unsqueeze(0)
        if fine_field.shape[-1] % 2 != 0 or fine_field.shape[-2] % 2 != 0:
            raise ValueError("2x2 blocking requires even lattice dimensions.")
        weights = self.path_probabilities()
        coarse_x = _circular_average(_x_paths(fine_field), weights[0:1])
        coarse_y = _circular_average(_y_paths(fine_field), weights[1:2])
        coarse = torch.stack([coarse_x, coarse_y], dim=1)
        return regularize(coarse)


class FixedGaugeCovariantBlocker(LearnableGaugeCovariantBlocker):
    def __init__(self) -> None:
        super().__init__(path_logits=torch.tensor([[12.0, -12.0, -12.0], [12.0, -12.0, -12.0]], dtype=torch.float32))
        self.path_logits.requires_grad_(False)
