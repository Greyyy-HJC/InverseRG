import torch
import torch.nn.functional as F
from torch import nn

from .lattice import plaquette_angles, rectangle_x_angles, rectangle_y_angles, regularize


def _circular_average(paths: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    sin_sum = (weights[:, :, None, None] * torch.sin(paths)).sum(dim=1)
    cos_sum = (weights[:, :, None, None] * torch.cos(paths)).sum(dim=1)
    return torch.atan2(sin_sum, cos_sum)


def _spatial_circular_average(paths: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Circular average with per-site weights.

    paths:   [B, n_paths, L_c, L_c]
    weights: [B, n_paths, L_c, L_c]
    """
    sin_sum = (weights * torch.sin(paths)).sum(dim=1)
    cos_sum = (weights * torch.cos(paths)).sum(dim=1)
    return torch.atan2(sin_sum, cos_sum)


def _block_plaquette_features(field: torch.Tensor) -> torch.Tensor:
    """Gauge-invariant cos(plaquette) features for each 2x2 blocking cell."""
    plaq = plaquette_angles(field)
    if plaq.dim() == 2:
        plaq = plaq.unsqueeze(0)
    return torch.stack([
        torch.cos(plaq[:, 0::2, 0::2]),
        torch.cos(plaq[:, 1::2, 0::2]),
        torch.cos(plaq[:, 0::2, 1::2]),
        torch.cos(plaq[:, 1::2, 1::2]),
    ], dim=1)


def _block_rectangle_features(field: torch.Tensor) -> torch.Tensor:
    """Gauge-invariant cos(rectangle) features for each 2x2 blocking cell.

    Returns [B, 8, L_c, L_c]: 4 channels from rectangle_x, 4 from rectangle_y.
    """
    rect_x = rectangle_x_angles(field)
    rect_y = rectangle_y_angles(field)
    if rect_x.dim() == 2:
        rect_x = rect_x.unsqueeze(0)
        rect_y = rect_y.unsqueeze(0)
    return torch.stack([
        torch.cos(rect_x[:, 0::2, 0::2]),
        torch.cos(rect_x[:, 1::2, 0::2]),
        torch.cos(rect_x[:, 0::2, 1::2]),
        torch.cos(rect_x[:, 1::2, 1::2]),
        torch.cos(rect_y[:, 0::2, 0::2]),
        torch.cos(rect_y[:, 1::2, 0::2]),
        torch.cos(rect_y[:, 0::2, 1::2]),
        torch.cos(rect_y[:, 1::2, 1::2]),
    ], dim=1)


def _block_gauge_invariant_features(field: torch.Tensor) -> torch.Tensor:
    """Combined plaquette + rectangle features: [B, 12, L_c, L_c]."""
    return torch.cat([_block_plaquette_features(field),
                      _block_rectangle_features(field)], dim=1)


def _subsample_even_even(tensor: torch.Tensor) -> torch.Tensor:
    return tensor[..., 0::2, 0::2]


def _x_paths(field: torch.Tensor) -> torch.Tensor:
    """All non-backtracking paths from (2i,2j) to (2i+2,2j) within |y|<=1.

    Returns [B, 7, L_c, L_c] with paths:
      0  straight      x, x
      1  up             y, x, x, -y
      2  down          -y, x, x,  y
      3  staircase_up   y, x, -y, x
      4  staircase_dn  -y, x,  y, x
      5  mid_up         x, y,  x, -y
      6  mid_down       x, -y, x,  y
    """
    ux = field[:, 0]
    uy = field[:, 1]
    straight = ux + torch.roll(ux, shifts=-1, dims=-2)
    up = (uy + torch.roll(ux, shifts=-1, dims=-1)
          + torch.roll(ux, shifts=(-1, -1), dims=(-2, -1))
          - torch.roll(uy, shifts=-2, dims=-2))
    down = (-torch.roll(uy, shifts=1, dims=-1)
            + torch.roll(ux, shifts=1, dims=-1)
            + torch.roll(ux, shifts=(-1, 1), dims=(-2, -1))
            + torch.roll(torch.roll(uy, shifts=1, dims=-1), shifts=-2, dims=-2))
    staircase_up = (uy + torch.roll(ux, shifts=-1, dims=-1)
                    - torch.roll(uy, shifts=-1, dims=-2)
                    + torch.roll(ux, shifts=-1, dims=-2))
    staircase_down = (-torch.roll(uy, shifts=1, dims=-1)
                      + torch.roll(ux, shifts=1, dims=-1)
                      + torch.roll(uy, shifts=(-1, 1), dims=(-2, -1))
                      + torch.roll(ux, shifts=-1, dims=-2))
    mid_up = (ux + torch.roll(uy, shifts=-1, dims=-2)
              + torch.roll(ux, shifts=(-1, -1), dims=(-2, -1))
              - torch.roll(uy, shifts=-2, dims=-2))
    mid_down = (ux - torch.roll(uy, shifts=(-1, 1), dims=(-2, -1))
                + torch.roll(ux, shifts=(-1, 1), dims=(-2, -1))
                + torch.roll(uy, shifts=(-2, 1), dims=(-2, -1)))
    paths = [straight, up, down, staircase_up, staircase_down, mid_up, mid_down]
    return regularize(torch.stack([_subsample_even_even(p) for p in paths], dim=1))


def _y_paths(field: torch.Tensor) -> torch.Tensor:
    """All non-backtracking paths from (2i,2j) to (2i,2j+2) within |x|<=1.

    Returns [B, 7, L_c, L_c] with paths:
      0  straight        y, y
      1  right           x, y, y, -x
      2  left           -x, y, y,  x
      3  staircase_rt    x, y, -x, y
      4  staircase_lt   -x, y,  x, y
      5  mid_right       y, x,  y, -x
      6  mid_left        y, -x, y,  x
    """
    ux = field[:, 0]
    uy = field[:, 1]
    straight = uy + torch.roll(uy, shifts=-1, dims=-1)
    right = (ux + torch.roll(uy, shifts=-1, dims=-2)
             + torch.roll(uy, shifts=(-1, -1), dims=(-2, -1))
             - torch.roll(ux, shifts=-2, dims=-1))
    left = (-torch.roll(ux, shifts=1, dims=-2)
            + torch.roll(uy, shifts=1, dims=-2)
            + torch.roll(uy, shifts=(1, -1), dims=(-2, -1))
            + torch.roll(torch.roll(ux, shifts=1, dims=-2), shifts=-2, dims=-1))
    staircase_right = (ux + torch.roll(uy, shifts=-1, dims=-2)
                       - torch.roll(ux, shifts=-1, dims=-1)
                       + torch.roll(uy, shifts=-1, dims=-1))
    staircase_left = (-torch.roll(ux, shifts=1, dims=-2)
                      + torch.roll(uy, shifts=1, dims=-2)
                      + torch.roll(ux, shifts=(1, -1), dims=(-2, -1))
                      + torch.roll(uy, shifts=-1, dims=-1))
    mid_right = (uy + torch.roll(ux, shifts=-1, dims=-1)
                 + torch.roll(uy, shifts=(-1, -1), dims=(-2, -1))
                 - torch.roll(ux, shifts=-2, dims=-1))
    mid_left = (uy - torch.roll(ux, shifts=(1, -1), dims=(-2, -1))
                + torch.roll(uy, shifts=(1, -1), dims=(-2, -1))
                + torch.roll(ux, shifts=(1, -2), dims=(-2, -1)))
    paths = [straight, right, left, staircase_right, staircase_left, mid_right, mid_left]
    return regularize(torch.stack([_subsample_even_even(p) for p in paths], dim=1))


class LearnableGaugeCovariantBlocker(nn.Module):
    def __init__(self, path_logits: torch.Tensor | None = None) -> None:
        super().__init__()
        if path_logits is None:
            path_logits = torch.zeros(2, 7, dtype=torch.float32)
            path_logits[:, 0] = 2.0
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

    def regularization_loss(self) -> torch.Tensor:
        probs = self.path_probabilities()
        return torch.sum(probs * torch.log(probs + 1e-8))

    def summary(self) -> dict:
        probs = self.path_probabilities().detach().cpu()
        return {
            "type": "LearnableGaugeCovariantBlocker",
            "x_links": [float(x) for x in probs[0]],
            "y_links": [float(x) for x in probs[1]],
        }


class FixedGaugeCovariantBlocker(LearnableGaugeCovariantBlocker):
    def __init__(self) -> None:
        logits = torch.full((2, 7), -12.0, dtype=torch.float32)
        logits[:, 0] = 12.0
        super().__init__(path_logits=logits)
        self.path_logits.requires_grad_(False)

    def regularization_loss(self) -> torch.Tensor:
        return torch.tensor(0.0)

    def summary(self) -> dict:
        return {"type": "FixedGaugeCovariantBlocker"}


class SpatialGaugeCovariantBlocker(nn.Module):
    """Gauge-covariant blocker with spatially varying path weights.

    A convolutional network predicts per-site path logits from local
    gauge-invariant features (plaquette + rectangle cosines within each 2x2
    blocking cell).  At initialization the network favours the straight path,
    matching the behaviour of :class:`LearnableGaugeCovariantBlocker`.
    """

    N_PATHS = 7
    N_INPUT_FEATURES = 12

    def __init__(self, hidden_dim: int = 32, kernel_size: int = 3) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self._pad = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(self.N_INPUT_FEATURES, hidden_dim, kernel_size, padding=0)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.conv_out = nn.Conv2d(hidden_dim, 2 * self.N_PATHS, 1)
        self._init_output_bias()

    def _init_output_bias(self) -> None:
        nn.init.zeros_(self.conv_out.weight)
        with torch.no_grad():
            self.conv_out.bias.zero_()
            self.conv_out.bias[0] = 2.0
            self.conv_out.bias[self.N_PATHS] = 2.0

    def _predict_logits(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.pad(features, [self._pad] * 4, mode="circular") if self._pad > 0 else features
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv_out(x)
        return x[:, :self.N_PATHS], x[:, self.N_PATHS:]

    def forward(self, fine_field: torch.Tensor) -> torch.Tensor:
        if fine_field.dim() == 3:
            fine_field = fine_field.unsqueeze(0)
        if fine_field.shape[-1] % 2 != 0 or fine_field.shape[-2] % 2 != 0:
            raise ValueError("2x2 blocking requires even lattice dimensions.")
        features = _block_gauge_invariant_features(fine_field)
        x_logits, y_logits = self._predict_logits(features)
        x_weights = torch.softmax(x_logits, dim=1)
        y_weights = torch.softmax(y_logits, dim=1)
        coarse_x = _spatial_circular_average(_x_paths(fine_field), x_weights)
        coarse_y = _spatial_circular_average(_y_paths(fine_field), y_weights)
        return regularize(torch.stack([coarse_x, coarse_y], dim=1))

    def regularization_loss(self) -> torch.Tensor:
        loss = torch.tensor(0.0)
        for p in self.parameters():
            loss = loss + p.square().sum()
        return loss

    def summary(self) -> dict:
        return {
            "type": "SpatialGaugeCovariantBlocker",
            "n_parameters": sum(p.numel() for p in self.parameters()),
            "n_paths": self.N_PATHS,
            "hidden_dim": self.hidden_dim,
            "kernel_size": self.kernel_size,
        }


class NaiveBlocker(nn.Module):
    def forward(self, fine_field: torch.Tensor) -> torch.Tensor:
        if fine_field.dim() == 3:
            fine_field = fine_field.unsqueeze(0)
        if fine_field.shape[-1] % 2 != 0 or fine_field.shape[-2] % 2 != 0:
            raise ValueError("2x2 blocking requires even lattice dimensions.")
        ux = fine_field[:, 0]
        uy = fine_field[:, 1]
        coarse_x = regularize(ux[:, 0::2, 0::2] + ux[:, 1::2, 0::2])
        coarse_y = regularize(uy[:, 0::2, 0::2] + uy[:, 0::2, 1::2])
        return torch.stack([coarse_x, coarse_y], dim=1)
