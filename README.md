# InverseRG

Research code for inverse renormalization group in 2D compact U(1) lattice gauge theory.

## Overview

This project implements a machine-learning approach to the Monte Carlo renormalization group (MCRG) for 2D compact U(1) lattice gauge theory. The core idea: given a fine-lattice ensemble sampled by HMC, learn both a gauge-covariant blocking map and a local coarse effective action such that the blocked-fine and independently-sampled coarse ensembles agree at the distribution level.

The approach is inspired by the MLRG framework (Hou & You, 2023), which uses teacher-student RBMs and an RG monotone network to learn renormalization group flows for Ising models. We adapt this to continuous gauge theories, where gauge covariance of the blocking map is a hard constraint.

## Phase Roadmap

### Phase 0: Naive Pipeline (complete)

Validates the full pipeline without neural networks.

- **HMC sampling**: Omelyan integrator for 2D compact U(1) with full diagnostics (plaquette, Hamiltonian, topological charge, acceptance rate, autocorrelation)
- **Naive blocking**: 2x2 blocking by summing two consecutive link phases along each direction
- **Baseline comparison**: blocked-fine (L=32, beta=4.0) vs independent coarse HMC (L=16, beta=1.0) with tree-level coupling `beta_c = beta_f / 4`
- **Evaluation**: histogram, CDF, and KS-test comparison of per-configuration observables
- **Notebook**: `presentation/phase0-naive-pipeline.ipynb`

### Phase 1: Learned Blocking and Coarse Action (complete)

Introduces neural network components to improve both the blocking map and the coarse action.

- **Gauge-covariant path blocker**: 7 non-backtracking paths per coarse-link direction within |transverse| <= 1. Paths combined via circular (vector) average with softmax weights.
- **Spatial blocker** (`SpatialGaugeCovariantBlocker`): A CNN predicts per-site path combination weights from 12-channel gauge-invariant features (plaquette + rectangle cosines within each 2x2 blocking cell).
- **Coarse action** (`LocalWilsonLoopAction`): Parameterized as a linear combination of Wilson loop basis elements (plaquette 1x1, rectangles 2x1, 1x2) with learnable coefficients.
- **Training**: Joint optimization of blocker and coarse action using MMD (maximum mean discrepancy) + contrastive + mean-mismatch loss. Train/test split for evaluation.
- **Configurations**: Two coupling points: beta=4.0 (L=32 -> 16) and beta=6.0 (L=32 -> 16), 1000 configurations each.
- **Notebooks**: `presentation/phase1-learned-blocking-beta4.ipynb`, `presentation/phase1-learned-blocking-beta6.ipynb`

### Phase 2: RG Monotone and Coupling Flow (current)

Learns the RG flow across the full coupling space, eliminating the need to retrain at each beta value.

**Motivation**: In Phase 1, the blocker and coarse action are trained independently at each beta. Phase 2 introduces an RG monotone network C(J) -- a scalar function over the coupling space J = (betaplaq, betarectx, betarecty) -- whose gradient flow dJ/dl = -grad C(J) predicts the optimal coarse coupling for any fine coupling. This is the continuous-gauge-theory analogue of the "moderator" network in MLRG.

**Two-stage approach**:

1. **Data collection**: Run Phase 1 `train_learned_rg` at a grid of beta values (e.g. beta = 2, 3, 4, 5, 6, 8) to collect (Jfine, Jcoarseoptimal) pairs. This validates the coupling landscape and provides training targets.
2. **Monotone fitting**: Train a small MLP Ctheta: R^3 -> R such that Euler integration of dJ/dl = -grad C(J) from Jfine maps to Jcoarse. Validate against the collected pairs and the tree-level baseline betac = betaf / 4.

**Key physics note**: 2D compact U(1) has no phase transition (unlike 2D Ising), so there are no unstable fixed points to locate. The value of the monotone here is: (a) one-shot coupling prediction for any beta, (b) quantifying deviations from tree-level, (c) enabling multi-step RG chains (L -> L/2 -> L/4 -> ...), (d) smoothing out per-beta statistical noise.

## Architecture

```
Coupling space: J = (beta_plaq, beta_rect_x, beta_rect_y)

Fine lattice                          Coarse lattice
  theta[2, L, L]                        Theta[2, L/2, L/2]
       |                                      |
  HMC(S_fine(J_fine))                    HMC(S_coarse(J_coarse))
       |                                      |
  fine ensemble                          coarse ensemble
       |                                      |
  blocker(fine) -----> blocked ensemble       |
                             |                |
                       distributional comparison
                       (MMD, KS, energy distance)

Phase 2 addition:
  J_fine ----> RG monotone C(J) ----> dJ/dl = -grad C ----> J_coarse
               (MLP, R^d -> R)        (Euler integration)
```

## Quick Start

A virtual environment is pre-configured at `.venv/` (Python 3.11) with all dependencies installed:

```bash
source .venv/bin/activate
pip install -e .
```

Run the presentation notebooks:

```bash
# Phase 0: naive pipeline
jupyter notebook presentation/phase0-naive-pipeline.ipynb

# Phase 1: learned blocking
jupyter notebook presentation/phase1-learned-blocking-beta4.ipynb
jupyter notebook presentation/phase1-learned-blocking-beta6.ipynb
```

Run tests:

```bash
pytest tests/ -q
```

## Project Layout

```
inverserg/
  hmc.py          -- HMC sampler (Omelyan integrator, diagnostics)
  lattice.py      -- loop geometry, regularization, topology
  measurements.py -- observable extraction, theoretical references
  blocking.py     -- naive and gauge-covariant blockers (7-path family, NN blocker)
  actions.py      -- local Wilson-loop coarse actions
  baselines.py    -- tree-level coupling relations
  diagnostics.py  -- KS tests, distribution plots
  training.py     -- learned RG training (Phase 1)
  monotone.py     -- RG monotone network, flow integration (Phase 2)
examples/         -- runnable scripts
tests/            -- pytest tests
presentation/     -- Jupyter notebooks (one per phase)
```

## References

- W. Hou and Y.-Z. You, "Machine learning renormalization group for statistical physics," *Mach. Learn.: Sci. Technol.* **4** 045010 (2023). [arXiv:2310.xxxxx](https://doi.org/10.1088/2632-2153/ad0101)
- M. Creutz, *Quarks, Gluons and Lattices*, Cambridge University Press (1983).

