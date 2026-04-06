# AGENTS

This file defines how agents should collaborate on `InverseRG`.

## Roles

- Planner
  - owns problem decomposition, specifications, acceptance criteria, and task boundaries
  - records decisions that affect multiple phases
- Builder
  - owns implementation, local verification, and concrete code delivery
  - reports blockers with exact missing dependencies or failed checks
- Reviewer
  - owns skeptical review of correctness, regressions, missing files, and mismatches between code and spec
  - should prioritize actionable findings over general commentary

## Work Rules

- Claim task ownership before starting implementation.
- Report progress in the task thread that owns the work.
- Do not treat `beta_c = beta_f / 4` as a proof; treat it as the approved working baseline that still requires numerical validation.
- Keep gauge covariance explicit in both code and documentation when defining blocked links.
- Prefer small local Wilson-loop bases until there is evidence that a larger basis is necessary.
- Treat distribution-level agreement of measurements across coarse configurations as the long-term target; mean-only matching is a staging proxy, not the final success criterion.

## Phase 0 Deliverables (complete)

- HMC sampler with per-step diagnostics (plaquette, Hamiltonian, topological charge, acceptance)
- Naive 2x2 blocker (sum two consecutive link phases + regularize)
- Theoretical reference functions (exact plaquette, topological susceptibility, autocorrelation)
- Presentation notebook: `presentation/phase0-naive-pipeline.ipynb`

## Phase 1 Deliverables (complete)

- Learnable gauge-covariant path-weight blocker (`SpatialGaugeCovariantBlocker`)
- Generalized local coarse action with multiple loop terms (`LocalWilsonLoopAction`)
- Training loop with distribution-matching loss (MMD + contrastive + mean-mismatch)
- Train/test split evaluation
- Comparison against Phase 0 naive baseline at beta=4.0 and beta=6.0
- Presentation notebooks: `phase1-learned-blocking-beta4.ipynb`, `phase1-learned-blocking-beta6.ipynb`

## Phase 2 Deliverables (current)

- Multi-beta data collection: run Phase 1 training at a grid of beta values, collect `(J_fine, J_coarse_optimal)` pairs
- RG monotone network in `inverserg/monotone.py`: MLP `C_theta: R^d -> R`
- Gradient flow integration with backprop support (Euler with `create_graph=True`)
- Validation: monotone-predicted `J_coarse` vs independently trained `J_coarse` across the beta grid
- Validation: comparison with tree-level baseline `beta_c = beta_f / 4`
- Presentation notebook: `presentation/phase2-rg-monotone.ipynb`

## Verification Expectations

- Syntax or import checks are the minimum bar, not the full bar.
- If runtime validation is blocked by missing dependencies, say so explicitly and name the dependency.
- Broken package exports, missing documented modules, and examples that contradict agreed project assumptions should be treated as review blockers.

## Documentation Expectations

- `SPEC.md` is the scientific and architectural source of truth for agents (high density, constraints, acceptance criteria).
- `README.md` is the human-facing project description (phase details, motivation, architecture, how to run).
- `CLAUDE.md` is the agent coding conventions file (physics conventions, code style, file layout, environment).
- Example scripts should use defaults consistent with the current project baseline unless a deviation is intentionally being tested and is documented inline.
- Status reports for humans should say exactly which lattice sizes, couplings, observables, and comparison criteria were actually run.

## Escalation

- If a phase discovers that the current observable set is insufficient, raise that as a planning question instead of silently expanding scope.
- If learned blocking behavior depends on an entropy or sparsity regularizer, document the intended direction explicitly so the loss sign cannot be misread.