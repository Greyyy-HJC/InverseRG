import json
import tempfile
from pathlib import Path

import torch

from inverserg.monotone import (
    CollectedRGData,
    MonotoneTrainingConfig,
    RGMonotone,
    rg_flow_step,
    train_rg_monotone,
)


def test_monotone_forward_scalar() -> None:
    m = RGMonotone(coupling_dim=3, hidden_dim=8, n_layers=2)
    J = torch.tensor([1.0, 0.0, 0.0])
    C = m(J)
    assert C.shape == ()


def test_monotone_forward_batch() -> None:
    m = RGMonotone(coupling_dim=3, hidden_dim=8, n_layers=2)
    J = torch.randn(5, 3)
    C = m(J)
    assert C.shape == (5,)


def test_beta_function_shape() -> None:
    m = RGMonotone(coupling_dim=3, hidden_dim=8, n_layers=2)
    J = torch.tensor([2.0, 0.0, 0.0])
    bf = m.beta_function(J)
    assert bf.shape == (3,)


def test_beta_function_batch() -> None:
    m = RGMonotone(coupling_dim=3, hidden_dim=8, n_layers=2)
    J = torch.randn(4, 3)
    bf = m.beta_function(J)
    assert bf.shape == (4, 3)


def test_rg_flow_step_shape() -> None:
    m = RGMonotone(coupling_dim=3, hidden_dim=8, n_layers=2)
    J_fine = torch.tensor([4.0, 0.0, 0.0])
    J_coarse = rg_flow_step(m, J_fine, n_steps=5, create_graph=False)
    assert J_coarse.shape == (3,)


def test_rg_flow_step_batch() -> None:
    m = RGMonotone(coupling_dim=3, hidden_dim=8, n_layers=2)
    J_fine = torch.randn(3, 3)
    J_coarse = rg_flow_step(m, J_fine, n_steps=5, create_graph=False)
    assert J_coarse.shape == (3, 3)


def test_rg_flow_step_differentiable() -> None:
    """Gradient of a loss on J_coarse back-propagates to monotone params."""
    m = RGMonotone(coupling_dim=3, hidden_dim=8, n_layers=2)
    J_fine = torch.tensor([4.0, 0.0, 0.0])
    J_coarse = rg_flow_step(m, J_fine, n_steps=5, create_graph=True)
    loss = J_coarse.pow(2).sum()
    loss.backward()
    grad_nonzero = False
    for p in m.parameters():
        if p.grad is not None and p.grad.abs().max() > 0:
            grad_nonzero = True
            break
    assert grad_nonzero, "No gradient reached monotone parameters"


def test_rg_flow_step_moves_J() -> None:
    """Flow should change J (it would be extremely unlikely to stay the same)."""
    torch.manual_seed(99)
    m = RGMonotone(coupling_dim=3, hidden_dim=16, n_layers=2)
    J_fine = torch.tensor([4.0, 0.0, 0.0])
    J_coarse = rg_flow_step(m, J_fine, n_steps=10, create_graph=False)
    assert not torch.allclose(J_fine, J_coarse, atol=1e-6)


def test_collected_rg_data_save_load() -> None:
    data = CollectedRGData(
        beta_values=[2.0, 4.0],
        J_fine=torch.tensor([[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
        J_coarse=torch.tensor([[0.5, 0.01, 0.01], [1.0, 0.02, 0.02]]),
        basis=("plaquette", "rectangle_x", "rectangle_y"),
        metrics=[{"beta": 2.0}, {"beta": 4.0}],
    )
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "test_pairs.json"
        data.save(path)
        loaded = CollectedRGData.load(path)
    assert loaded.beta_values == [2.0, 4.0]
    assert loaded.basis == ("plaquette", "rectangle_x", "rectangle_y")
    assert torch.allclose(loaded.J_fine, data.J_fine)
    assert torch.allclose(loaded.J_coarse, data.J_coarse)


def _make_synthetic_collected(n: int = 6) -> CollectedRGData:
    """Synthetic data where J_coarse = (beta/4, 0, 0) (tree-level)."""
    betas = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0][:n]
    J_fine = torch.zeros(n, 3)
    J_coarse = torch.zeros(n, 3)
    for i, b in enumerate(betas):
        J_fine[i, 0] = b
        J_coarse[i, 0] = b / 4.0
    return CollectedRGData(
        beta_values=betas,
        J_fine=J_fine,
        J_coarse=J_coarse,
        basis=("plaquette", "rectangle_x", "rectangle_y"),
    )


def test_train_rg_monotone_runs() -> None:
    collected = _make_synthetic_collected(4)
    cfg = MonotoneTrainingConfig(
        hidden_dim=16, n_layers=2, n_euler_steps=5,
        epochs=20, learning_rate=1e-2,
    )
    monotone, result = train_rg_monotone(collected, config=cfg, verbose=False)
    assert len(result.history) == 20
    assert result.history[-1]["loss"] < result.history[0]["loss"]


def test_train_rg_monotone_converges_on_tree_level() -> None:
    """With enough capacity the monotone should learn the tree-level flow."""
    collected = _make_synthetic_collected(6)
    cfg = MonotoneTrainingConfig(
        hidden_dim=32, n_layers=3, n_euler_steps=10,
        epochs=600, learning_rate=5e-3,
    )
    monotone, result = train_rg_monotone(collected, config=cfg, verbose=False)
    pred = torch.tensor(result.predicted_J_coarse)
    target = torch.tensor(result.actual_J_coarse)
    plaq_err = (pred[:, 0] - target[:, 0]).abs().max().item()
    assert plaq_err < 0.5, f"Plaquette prediction error too large: {plaq_err:.4f}"
