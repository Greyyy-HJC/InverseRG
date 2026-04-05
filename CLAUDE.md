# CLAUDE.md

Project conventions for AI coding agents (Claude, Codex, etc.) working on InverseRG.

## Project Overview
2D compact U(1) lattice gauge theory with inverse renormalization group.
Currently in Phase 1: learned gauge-covariant blocking with neural networks.

## Key Conventions

### Physics
- Link angles `theta[mu, x, y]` on periodic 2D square lattice, shape `[2, L, L]` or `[B, 2, L, L]`
- `field[:, 0]` = x-links (mu=0), `field[:, 1]` = y-links (mu=1)
- Index convention: `field[batch, mu, x, y]`
- Wilson action: `S = -beta * sum cos(plaquette_angles)`
- Angles always regularized to `[-pi, pi]` via `atan2(sin, cos)`

### Blocking
- Naive 2x2 blocking: sum two consecutive link phases along same direction, then regularize
- Gauge-covariant path blocking: 7 non-backtracking paths per direction within |transverse| <= 1
- Path combination via circular (vector) average with softmax weights
- Blocker types: `NaiveBlocker`, `FixedGaugeCovariantBlocker`, `LearnableGaugeCovariantBlocker`, `SpatialGaugeCovariantBlocker`
- `SpatialGaugeCovariantBlocker`: CNN predicts per-site path logits from gauge-invariant features (12 channels: plaquette + rectangle cosines)
- Coarse lattice is `L/2` from fine lattice `L`
- Tree-level coupling: `beta_c = beta_f / 4` for 2D

### Code Style
- No comments that merely narrate what code does
- All tensor operations should handle both single `[2, L, L]` and batched `[B, 2, L, L]` inputs
- Use `torch.no_grad()` for measurement computations when not inside training
- Preserve backward compatibility when adding new methods

### Testing
- `pytest tests/` for unit tests
- Example scripts in `examples/` for integration-level checks
- Presentation notebooks in `presentation/` for visual validation

## File Layout
```
inverserg/
  hmc.py          -- HMC sampler (Omelyan integrator, diagnostics)
  lattice.py      -- loop geometry, regularization, topology
  measurements.py -- observable extraction, theoretical references
  blocking.py     -- naive and gauge-covariant blockers (7-path family, NN blocker)
  actions.py      -- local Wilson-loop coarse actions
  baselines.py    -- tree-level coupling relations
  diagnostics.py  -- KS tests, distribution plots
  training.py     -- learned RG training (Phase 1: train/test split, blocker_type config)
examples/         -- runnable scripts
tests/            -- pytest tests
presentation/     -- human-facing progress presentations (one notebook per phase)
  phase0-naive-pipeline.ipynb           -- Phase 0 naive pipeline presentation
  phase1-learned-blocking-beta4.ipynb  -- Phase 1 learned blocking (beta=4.0)
  phase1-learned-blocking-beta6.ipynb  -- Phase 1 learned blocking (beta=6.0)
```

## Virtual Environment

A project-local virtualenv is already set up at `.venv/` (Python 3.11, gitignored).
All dependencies (torch, numpy, scipy, matplotlib, jupyter, pytest) are installed.

**Always use the venv Python** -- never the system `/usr/bin/python3` (which lacks pip and packages):

```bash
# Activate (if running interactively)
source .venv/bin/activate

# Or invoke directly
.venv/bin/python -m pytest tests/ -q
.venv/bin/jupyter notebook presentation/phase0-naive-pipeline.ipynb
```

The Jupyter kernel `inverserg` is registered for the notebook.

If you need to install additional packages:
```bash
.venv/bin/pip install <package>
```

## Running
```bash
source .venv/bin/activate
pip install -e .
jupyter notebook presentation/phase0-naive-pipeline.ipynb
pytest tests/ -q
```
