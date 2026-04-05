# InverseRG

Research code for 2D U(1) inverse renormalization group experiments.

## What This Does

1. **HMC Sampling**: Hybrid Monte Carlo with Omelyan integrator for 2D compact U(1) lattice gauge theory.
2. **MCRG Blocking**: Naive 2x2 blocking that sums consecutive link phases to produce coarse configurations.
3. **Ensemble Comparison**: Distribution-level comparison of blocked-fine vs independent coarse-HMC ensembles using KS tests, histograms, and CDFs.
4. **Theoretical References**: Exact plaquette expectation values, topological susceptibility, and autocorrelation functions.

## Quick Start

A virtual environment is pre-configured at `.venv/` (Python 3.11) with all dependencies installed:

```bash
source .venv/bin/activate
pip install -e .          # if not already installed
jupyter notebook presentation.ipynb
```

Or without activating:

```bash
.venv/bin/jupyter notebook presentation.ipynb
```

The presentation notebook runs the full pipeline:
- Fine ensemble: L=32, beta=4.0, 1000 configurations
- Naive 2x2 blocking to L=16
- Independent coarse ensemble: L=16, beta=1.0, 1000 configurations
- HMC diagnostics (plaquette, Hamiltonian, topology, autocorrelation)
- Blocked-fine vs coarse-HMC distribution comparison

## Baseline

Tree-level coarse coupling: `beta_c = beta_f / 4` for 2D U(1) with 2x2 blocking. This is exposed as `inverserg.baselines.tree_level_coarse_beta`.

## Project Status

See `presentation.ipynb` for the current visual summary. See `SPEC.md` for the full specification and phase plan.
