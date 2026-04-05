import torch
from torch import nn

from .lattice import loop_observables, plaquette_angles, rectangle_x_angles, rectangle_y_angles


class LocalWilsonLoopAction(nn.Module):
    def __init__(
        self,
        basis: tuple[str, ...] = ("plaquette", "rectangle_x", "rectangle_y"),
        initial_coefficients: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.basis = basis
        if initial_coefficients is None:
            initial_coefficients = torch.zeros(len(basis), dtype=torch.float32)
        self.coefficients = nn.Parameter(initial_coefficients.clone().detach().float())

    @classmethod
    def wilson(
        cls,
        beta: float,
        basis: tuple[str, ...] = ("plaquette", "rectangle_x", "rectangle_y"),
    ) -> "LocalWilsonLoopAction":
        coefficients = torch.zeros(len(basis), dtype=torch.float32)
        if "plaquette" not in basis:
            raise ValueError("Wilson initialization requires a plaquette term in the loop basis.")
        coefficients[basis.index("plaquette")] = beta
        return cls(basis=basis, initial_coefficients=coefficients)

    def loop_values(self, field: torch.Tensor) -> list[torch.Tensor]:
        values = []
        for name in self.basis:
            if name == "plaquette":
                angles = plaquette_angles(field)
            elif name == "rectangle_x":
                angles = rectangle_x_angles(field)
            elif name == "rectangle_y":
                angles = rectangle_y_angles(field)
            else:
                raise ValueError(f"Unknown loop basis element: {name}")
            values.append(torch.cos(angles))
        return values

    def per_configuration_action(self, field: torch.Tensor) -> torch.Tensor:
        if field.dim() == 3:
            field = field.unsqueeze(0)
        contributions = []
        for coefficient, loop_values in zip(self.coefficients, self.loop_values(field)):
            contributions.append(-coefficient * loop_values.sum(dim=(-2, -1)))
        return torch.stack(contributions, dim=0).sum(dim=0)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        return self.per_configuration_action(field).sum()

    def observable_vector(self, field: torch.Tensor) -> torch.Tensor:
        return loop_observables(field, self.basis)

    def coefficient_dict(self) -> dict[str, float]:
        return {
            name: float(value.detach().cpu())
            for name, value in zip(self.basis, self.coefficients)
        }
