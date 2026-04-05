from dataclasses import dataclass, field

import torch

from .lattice import mean_plaquette, regularize, topological_charge


@dataclass
class HMCDiagnostics:
    plaquette_history: list[float] = field(default_factory=list)
    hamiltonian_history: list[float] = field(default_factory=list)
    topological_charge_history: list[float] = field(default_factory=list)
    acceptance_history: list[bool] = field(default_factory=list)
    burn_in_length: int = 0


class HMCU1Sampler:
    def __init__(
        self,
        lattice_size: int,
        action,
        n_steps: int = 8,
        step_size: float = 0.15,
        device: str = "cpu",
    ) -> None:
        self.lattice_size = lattice_size
        self.action = action
        self.n_steps = n_steps
        self.step_size = step_size
        self.device = torch.device(device)

    def initialize(self) -> torch.Tensor:
        return torch.zeros((2, self.lattice_size, self.lattice_size), device=self.device)

    def force(self, theta: torch.Tensor) -> torch.Tensor:
        # The sampler may be invoked from evaluation code that wraps model
        # generation in `torch.no_grad()`, but HMC still needs field gradients.
        with torch.enable_grad():
            theta = theta.detach().clone().requires_grad_(True)
            action_value = self.action(theta)
            (grad,) = torch.autograd.grad(action_value, theta)
        return grad.detach()

    def omelyan(self, theta: torch.Tensor, pi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lam = 0.1931833
        dt = self.step_size
        theta_new = theta
        pi_new = pi - lam * dt * self.force(theta_new)
        for step in range(self.n_steps):
            theta_new = theta_new + 0.5 * dt * pi_new
            pi_new = pi_new - (1.0 - 2.0 * lam) * dt * self.force(theta_new)
            theta_new = theta_new + 0.5 * dt * pi_new
            if step != self.n_steps - 1:
                pi_new = pi_new - 2.0 * lam * dt * self.force(theta_new)
        pi_new = pi_new - lam * dt * self.force(theta_new)
        return regularize(theta_new), pi_new

    def metropolis_step(self, theta: torch.Tensor) -> tuple[torch.Tensor, bool]:
        pi = torch.randn_like(theta, device=self.device)
        old_h = self.action(theta) + 0.5 * torch.sum(pi.square())
        theta_new, pi_new = self.omelyan(theta.clone(), pi.clone())
        new_h = self.action(theta_new) + 0.5 * torch.sum(pi_new.square())
        delta_h = new_h - old_h
        accepted = torch.rand((), device=self.device) < torch.exp(-delta_h)
        return (theta_new if accepted else theta), bool(accepted.detach().cpu())

    def sample(
        self,
        n_samples: int,
        burn_in: int = 64,
        thin: int = 4,
        initial_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, float, torch.Tensor]:
        theta = self.initialize() if initial_state is None else initial_state.clone().to(self.device)
        accepted = 0
        total = 0
        samples = []

        for _ in range(burn_in):
            theta, did_accept = self.metropolis_step(theta)
            total += 1
            accepted += int(did_accept)

        while len(samples) < n_samples:
            for _ in range(thin):
                theta, did_accept = self.metropolis_step(theta)
                total += 1
                accepted += int(did_accept)
            samples.append(theta.clone())

        acceptance_rate = accepted / max(total, 1)
        return torch.stack(samples, dim=0), acceptance_rate, theta

    def sample_with_diagnostics(
        self,
        n_samples: int,
        burn_in: int = 64,
        thin: int = 4,
        initial_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, float, torch.Tensor, HMCDiagnostics]:
        diagnostics = HMCDiagnostics()
        diagnostics.burn_in_length = burn_in
        theta = self.initialize() if initial_state is None else initial_state.clone().to(self.device)
        accepted = 0
        total = 0
        samples = []

        def record(theta_step: torch.Tensor, did_accept: bool) -> None:
            with torch.no_grad():
                diagnostics.plaquette_history.append(float(mean_plaquette(theta_step).detach().cpu()))
                diagnostics.hamiltonian_history.append(float(self.action(theta_step).detach().cpu()))
                diagnostics.topological_charge_history.append(
                    float(topological_charge(theta_step).detach().cpu())
                )
                diagnostics.acceptance_history.append(did_accept)

        for _ in range(burn_in):
            theta, did_accept = self.metropolis_step(theta)
            total += 1
            accepted += int(did_accept)
            record(theta, did_accept)

        while len(samples) < n_samples:
            for _ in range(thin):
                theta, did_accept = self.metropolis_step(theta)
                total += 1
                accepted += int(did_accept)
                record(theta, did_accept)
            samples.append(theta.clone())

        acceptance_rate = accepted / max(total, 1)
        return torch.stack(samples, dim=0), acceptance_rate, theta, diagnostics
