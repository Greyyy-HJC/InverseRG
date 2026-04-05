import math

import torch

from inverserg.diagnostics import analyze_distribution_consistency
from inverserg.actions import LocalWilsonLoopAction
from inverserg.hmc import HMCU1Sampler
from inverserg.lattice import mean_plaquette, plaquette_angles, regularize, topological_charge, wilson_loop_angles
from inverserg.measurements import summarize_observables
from inverserg.training import RGTrainingConfig, measurement_distribution_mmd


def test_regularize_wraps_to_principal_branch() -> None:
    angles = torch.tensor([-4 * math.pi, -math.pi, 0.0, math.pi, 3 * math.pi])
    wrapped = regularize(angles)
    assert torch.all(wrapped <= math.pi)
    assert torch.all(wrapped >= -math.pi)
    assert torch.allclose(torch.sin(wrapped), torch.sin(angles), atol=1e-6)
    assert torch.allclose(torch.cos(wrapped), torch.cos(angles), atol=1e-6)


def test_zero_field_has_unit_plaquette_and_zero_topology() -> None:
    field = torch.zeros((2, 4, 4))
    assert torch.allclose(plaquette_angles(field), torch.zeros((4, 4)))
    assert torch.isclose(mean_plaquette(field), torch.tensor(1.0))
    assert torch.equal(topological_charge(field), torch.tensor(0.0))


def test_topological_charge_supports_batched_inputs() -> None:
    fields = torch.zeros((3, 2, 4, 4))
    charges = topological_charge(fields)
    assert charges.shape == (3,)
    assert torch.equal(charges, torch.zeros(3))


def test_wilson_loop_angles_match_zero_field() -> None:
    field = torch.zeros((2, 6, 6))
    loop = wilson_loop_angles(field, extent_x=2, extent_y=2)
    assert torch.allclose(loop, torch.zeros((6, 6)))


def test_summarize_observables_exposes_expected_keys() -> None:
    summary = summarize_observables(torch.zeros((2, 4, 4)))
    assert {"plaquette", "topological_charge", "plaquette_angle_mean", "wilson_1x1", "wilson_1x2", "wilson_2x2"} <= set(summary)


def test_hmc_sampler_returns_regularized_shapes_for_small_run() -> None:
    action = LocalWilsonLoopAction.wilson(beta=1.0)
    sampler = HMCU1Sampler(lattice_size=4, action=action, n_steps=1, step_size=0.05)
    samples, acceptance_rate, final_state = sampler.sample(n_samples=2, burn_in=1, thin=1)

    assert samples.shape == (2, 2, 4, 4)
    assert final_state.shape == (2, 4, 4)
    assert 0.0 <= acceptance_rate <= 1.0
    assert torch.all(samples <= math.pi)
    assert torch.all(samples >= -math.pi)


def test_distribution_diagnostics_marks_identical_ensembles_consistent() -> None:
    fields = torch.zeros((4, 2, 4, 4))
    diagnostics, merged = analyze_distribution_consistency(
        fields,
        fields.clone(),
        measurement_names=("plaquette", "topological_charge"),
        ks_alpha=0.05,
    )
    assert set(merged) == {"blocked_plaquette", "blocked_topological_charge", "coarse_plaquette", "coarse_topological_charge"}
    assert all(item.consistent for item in diagnostics)


def test_distribution_mmd_vanishes_for_identical_measurement_samples() -> None:
    fields = torch.zeros((4, 2, 4, 4))
    config = RGTrainingConfig()
    mmd = measurement_distribution_mmd(
        fields,
        fields.clone(),
        measurement_names=config.measurement_set,
        bandwidth=config.mmd_bandwidth,
    )
    assert torch.isclose(mmd, torch.tensor(0.0), atol=1e-6)


def test_default_optimized_measurements_exclude_topological_charge() -> None:
    config = RGTrainingConfig()
    assert "topological_charge" not in config.measurement_set
    assert "topological_charge" in config.evaluation_measurement_set
