import math

import torch

from inverserg.blocking import (
    FixedGaugeCovariantBlocker,
    LearnableGaugeCovariantBlocker,
    NaiveBlocker,
    SpatialGaugeCovariantBlocker,
    _block_gauge_invariant_features,
    _block_plaquette_features,
    _block_rectangle_features,
    _x_paths,
    _y_paths,
)
from inverserg.lattice import plaquette_angles, regularize
from inverserg.training import RGTrainingConfig, _create_blocker


def test_x_paths_returns_seven_paths() -> None:
    field = torch.zeros(2, 2, 4, 4)
    paths = _x_paths(field)
    assert paths.shape == (2, 7, 2, 2)


def test_y_paths_returns_seven_paths() -> None:
    field = torch.zeros(2, 2, 4, 4)
    paths = _y_paths(field)
    assert paths.shape == (2, 7, 2, 2)


def test_all_paths_vanish_on_zero_field() -> None:
    field = torch.zeros(1, 2, 6, 6)
    x_paths = _x_paths(field)
    y_paths = _y_paths(field)
    assert torch.allclose(x_paths, torch.zeros_like(x_paths), atol=1e-7)
    assert torch.allclose(y_paths, torch.zeros_like(y_paths), atol=1e-7)


def test_straight_path_matches_naive_blocker() -> None:
    torch.manual_seed(42)
    field = torch.randn(3, 2, 8, 8) * 0.5
    x_paths = _x_paths(field)
    y_paths = _y_paths(field)
    naive = NaiveBlocker()(field)
    assert torch.allclose(regularize(x_paths[:, 0]), naive[:, 0], atol=1e-5)
    assert torch.allclose(regularize(y_paths[:, 0]), naive[:, 1], atol=1e-5)


def test_staircase_up_differs_from_straight_by_plaquette() -> None:
    """staircase_up - straight = -plaquette at each coarse site.

    The staircase path wraps around the plaquette in the opposite sense.
    """
    torch.manual_seed(42)
    field = torch.randn(2, 2, 4, 4) * 0.3
    x_paths = _x_paths(field)
    straight = x_paths[:, 0]
    staircase_up = x_paths[:, 3]
    plaq = plaquette_angles(field)
    expected_diff = -plaq[:, 0::2, 0::2]
    actual_diff = regularize(staircase_up - straight)
    assert torch.allclose(actual_diff, expected_diff, atol=1e-5)


def test_staircase_right_differs_from_straight_by_plaquette_y() -> None:
    """staircase_right - straight = +plaquette at each coarse site for y-paths."""
    torch.manual_seed(42)
    field = torch.randn(2, 2, 4, 4) * 0.3
    y_paths = _y_paths(field)
    straight = y_paths[:, 0]
    staircase_right = y_paths[:, 3]
    plaq = plaquette_angles(field)
    expected_diff = plaq[:, 0::2, 0::2]
    actual_diff = regularize(staircase_right - straight)
    assert torch.allclose(actual_diff, expected_diff, atol=1e-5)


def test_paths_are_in_principal_branch() -> None:
    torch.manual_seed(42)
    field = torch.randn(2, 2, 6, 6)
    x_paths = _x_paths(field)
    y_paths = _y_paths(field)
    assert torch.all(x_paths >= -math.pi) and torch.all(x_paths <= math.pi)
    assert torch.all(y_paths >= -math.pi) and torch.all(y_paths <= math.pi)


def test_block_plaquette_features_shape() -> None:
    field = torch.zeros(2, 2, 8, 8)
    features = _block_plaquette_features(field)
    assert features.shape == (2, 4, 4, 4)


def test_block_rectangle_features_shape() -> None:
    field = torch.zeros(2, 2, 8, 8)
    features = _block_rectangle_features(field)
    assert features.shape == (2, 8, 4, 4)


def test_block_gauge_invariant_features_shape() -> None:
    field = torch.zeros(2, 2, 8, 8)
    features = _block_gauge_invariant_features(field)
    assert features.shape == (2, 12, 4, 4)


def test_learnable_blocker_produces_valid_output() -> None:
    field = torch.randn(2, 2, 8, 8) * 0.5
    blocker = LearnableGaugeCovariantBlocker()
    coarse = blocker(field)
    assert coarse.shape == (2, 2, 4, 4)
    assert torch.all(coarse >= -math.pi) and torch.all(coarse <= math.pi)


def test_fixed_blocker_produces_valid_output() -> None:
    field = torch.randn(2, 2, 8, 8) * 0.5
    blocker = FixedGaugeCovariantBlocker()
    coarse = blocker(field)
    assert coarse.shape == (2, 2, 4, 4)
    assert torch.all(coarse >= -math.pi) and torch.all(coarse <= math.pi)


def test_spatial_blocker_produces_valid_output() -> None:
    field = torch.randn(2, 2, 8, 8) * 0.5
    blocker = SpatialGaugeCovariantBlocker(hidden_dim=8, kernel_size=1)
    coarse = blocker(field)
    assert coarse.shape == (2, 2, 4, 4)
    assert torch.all(coarse >= -math.pi) and torch.all(coarse <= math.pi)


def test_spatial_blocker_with_kernel3() -> None:
    field = torch.randn(2, 2, 8, 8) * 0.5
    blocker = SpatialGaugeCovariantBlocker(hidden_dim=8, kernel_size=3)
    coarse = blocker(field)
    assert coarse.shape == (2, 2, 4, 4)


def test_spatial_blocker_is_differentiable() -> None:
    field = torch.randn(1, 2, 4, 4) * 0.3
    blocker = SpatialGaugeCovariantBlocker(hidden_dim=4, kernel_size=1)
    coarse = blocker(field)
    loss = coarse.sum()
    loss.backward()
    for p in blocker.parameters():
        assert p.grad is not None


def test_create_blocker_spatial() -> None:
    config = RGTrainingConfig(blocker_type="spatial", spatial_hidden_dim=8, spatial_kernel_size=1)
    blocker = _create_blocker(config)
    assert isinstance(blocker, SpatialGaugeCovariantBlocker)


def test_create_blocker_global() -> None:
    config = RGTrainingConfig(blocker_type="global")
    blocker = _create_blocker(config)
    assert isinstance(blocker, LearnableGaugeCovariantBlocker)


def test_create_blocker_fixed() -> None:
    config = RGTrainingConfig(blocker_type="fixed")
    blocker = _create_blocker(config)
    assert isinstance(blocker, FixedGaugeCovariantBlocker)


def test_config_has_test_samples_field() -> None:
    config = RGTrainingConfig(n_test_samples=16)
    assert config.n_test_samples == 16
    d = config.to_dict()
    assert d["n_test_samples"] == 16


def test_fixed_blocker_matches_straight_path() -> None:
    """FixedGaugeCovariantBlocker with dominant straight logits should match NaiveBlocker."""
    torch.manual_seed(42)
    field = torch.randn(3, 2, 8, 8) * 0.3
    fixed = FixedGaugeCovariantBlocker()(field)
    naive = NaiveBlocker()(field)
    assert torch.allclose(fixed, naive, atol=1e-4)
