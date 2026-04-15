import tempfile
from pathlib import Path

import torch

from inverserg.blocking import ConditionedSpatialGaugeCovariantBlocker
from inverserg.forward_rg import (
    ForwardRGConfig,
    ForwardRGHypernetwork,
    load_forward_rg_checkpoint,
    save_forward_rg_checkpoint,
    train_forward_rg,
)
from inverserg.inverse import (
    ConditionedFineAction,
    EquivariantInverseProposalNet,
    InverseRGConfig,
    build_fine_proposal,
    canonical_prolongation,
    equivariant_refinement,
    gauge_transform,
    prolong_site_gauge,
    train_inverse_rg,
)


def test_forward_hypernetwork_output_shapes() -> None:
    model = ForwardRGHypernetwork(coupling_dim=3, hidden_dim=16, z_phi_dim=16)
    J_coarse, z_phi = model.predict_forward_rg(torch.tensor([4.0, 0.0, 0.0]))
    assert J_coarse.shape == (3,)
    assert z_phi.shape == (16,)


def test_conditioned_blocker_changes_with_context() -> None:
    torch.manual_seed(0)
    field = torch.randn(2, 2, 8, 8) * 0.3
    blocker = ConditionedSpatialGaugeCovariantBlocker(hidden_dim=8, kernel_size=1, context_dim=4)
    out_zero = blocker(field, torch.zeros(4))
    out_one = blocker(field, torch.ones(4))
    assert out_zero.shape == (2, 2, 4, 4)
    assert not torch.allclose(out_zero, out_one)


def test_canonical_prolongation_is_gauge_covariant() -> None:
    torch.manual_seed(1)
    coarse = torch.randn(2, 2, 4, 4) * 0.1
    alpha = torch.randn(2, 4, 4) * 0.2
    lhs = canonical_prolongation(gauge_transform(coarse, alpha))
    rhs = gauge_transform(canonical_prolongation(coarse), prolong_site_gauge(alpha))
    assert torch.allclose(lhs, rhs, atol=1e-6)


def test_equivariant_inverse_proposal_is_gauge_covariant() -> None:
    torch.manual_seed(2)
    proposal_net = EquivariantInverseProposalNet(hidden_dim=8, noise_channels=6, residual_channels=6, context_dim=22)
    coarse = torch.randn(2, 2, 4, 4) * 0.1
    alpha = torch.randn(2, 4, 4) * 0.1
    coarse_g = gauge_transform(coarse, alpha)
    J_coarse = torch.tensor([1.0, 0.1, -0.1])
    J_fine = torch.tensor([4.0, 0.0, 0.0])
    z_phi = torch.randn(16)
    noise = torch.randn(2, 6, 4, 4)
    residuals = proposal_net(coarse, J_coarse, J_fine, z_phi, noise=noise)
    residuals_g = proposal_net(coarse_g, J_coarse, J_fine, z_phi, noise=noise)
    assert torch.allclose(residuals, residuals_g, atol=1e-6)
    proposal = build_fine_proposal(coarse, residuals)
    proposal_g = build_fine_proposal(coarse_g, residuals_g)
    rhs = gauge_transform(proposal, prolong_site_gauge(alpha))
    assert torch.allclose(proposal_g, rhs, atol=1e-5)


def test_refinement_does_not_increase_block_loss() -> None:
    torch.manual_seed(3)
    coarse = torch.randn(2, 2, 4, 4) * 0.1
    blocker = ConditionedSpatialGaugeCovariantBlocker(hidden_dim=8, kernel_size=1, context_dim=16)
    proposal_net = EquivariantInverseProposalNet(hidden_dim=8, noise_channels=6, residual_channels=6, context_dim=22)
    J_coarse = torch.tensor([1.0, 0.0, 0.0])
    J_fine = torch.tensor([4.0, 0.0, 0.0])
    z_phi = torch.zeros(16)
    proposal, _ = proposal_net.proposal(coarse, J_coarse, J_fine, z_phi, noise=torch.zeros(2, 6, 4, 4))
    energy = ConditionedFineAction(
        J_fine=J_fine,
        coarse_target=coarse,
        blocker=blocker,
        z_phi=z_phi,
        config=InverseRGConfig(refinement_steps=3),
    )
    _, history = equivariant_refinement(proposal, energy, steps=3, step_size=0.05, differentiable=False)
    losses = [entry["block_loss"] for entry in history]
    assert all(next_loss <= prev_loss + 1e-7 for prev_loss, next_loss in zip(losses, losses[1:]))


def test_forward_rg_training_and_checkpoint_smoke() -> None:
    config = ForwardRGConfig(
        fine_lattice_size=4,
        n_fine_samples=2,
        n_model_samples=2,
        sampler_burn_in=2,
        sampler_thin=1,
        hmc_steps=2,
        hmc_step_size=0.2,
        epochs=2,
        hidden_dim=16,
        blocker_hidden_dim=8,
        blocker_kernel_size=1,
        z_phi_dim=16,
    )
    model, blocker, result = train_forward_rg([2.0, 3.0], config=config, verbose=False)
    assert len(result.history) == 2
    assert len(result.predicted_J_coarse) == 2
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "forward_rg.pt"
        save_forward_rg_checkpoint(path, model, blocker, config, result)
        loaded_model, loaded_blocker, loaded_config, loaded_result = load_forward_rg_checkpoint(path)
    assert loaded_config.basis == config.basis
    assert loaded_result is not None
    test_field = torch.randn(1, 2, 4, 4) * 0.1
    J_coarse, z_phi = loaded_model.predict_forward_rg(torch.tensor([2.5, 0.0, 0.0]))
    blocked = loaded_blocker(test_field, z_phi)
    assert J_coarse.shape == (3,)
    assert blocked.shape == (1, 2, 2, 2)


def test_inverse_rg_training_smoke() -> None:
    forward_config = ForwardRGConfig(
        fine_lattice_size=4,
        n_fine_samples=2,
        n_model_samples=2,
        sampler_burn_in=2,
        sampler_thin=1,
        hmc_steps=2,
        hmc_step_size=0.2,
        epochs=1,
        hidden_dim=16,
        blocker_hidden_dim=8,
        blocker_kernel_size=1,
        z_phi_dim=16,
    )
    forward_model, blocker, _ = train_forward_rg([2.0], config=forward_config, verbose=False)
    inverse_config = InverseRGConfig(
        hidden_dim=8,
        refinement_steps=2,
        epochs=1,
        fine_lattice_size=4,
        n_fine_samples=2,
        sampler_burn_in=2,
        sampler_thin=1,
        hmc_steps=2,
        hmc_step_size=0.2,
    )
    proposal_net, result = train_inverse_rg(
        [2.0],
        forward_model=forward_model,
        blocker=blocker,
        config=inverse_config,
        verbose=False,
    )
    assert len(result.history) == 1
    assert result.generated_fine_ensemble is not None
    assert result.blocked_generated_ensemble is not None
    assert proposal_net.hidden_dim == 8
