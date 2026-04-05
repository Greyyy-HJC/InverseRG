# InverseRG

Minimal research code for 2D U(1) inverse RG experiments:

- Hybrid Monte Carlo sampling for the fine and coarse ensembles.
- Fixed and learnable 2x2 gauge-covariant blocking maps.
- Local generalized Wilson coarse actions built from small Wilson loops.
- A contrastive local training loop that jointly tunes blocking weights and coarse-action couplings.
- Measurement helpers for plaquettes, topology, and small Wilson loops.

## Quick start

```bash
python examples/train_learned_rg.py
```

The example runs a small smoke-test experiment and prints the observable mismatch before and after training.

For the fixed Wilson coarse baseline, the current tree-level starting hypothesis is `beta_c ~= beta_f / 4` for 2D U(1) with 2x2 blocking. This is exposed as a helper in `inverserg.baselines.tree_level_coarse_beta`, but it should still be checked numerically against blocked-fine observables.
