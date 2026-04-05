# AGNETS

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
- If one branch absorbs multiple planned phases because the repo was initially empty, the assignee should also claim the affected tasks so the board remains accurate.
- Do not treat `beta_c = beta_f / 4` as a proof; treat it as the approved working baseline that still requires numerical validation.
- Keep gauge covariance explicit in both code and documentation when defining blocked links.
- Prefer small local Wilson-loop bases until there is evidence that a larger basis is necessary.
- Treat distribution-level agreement of measurements across coarse configurations as the long-term target; mean-only matching is a staging proxy, not the final success criterion.

## Phase 0 Deliverables (Naive Pipeline -- current focus)
- HMC sampler with per-step diagnostics (plaquette, Hamiltonian, topological charge, acceptance)
- Naive 2x2 blocker (sum two consecutive link phases + regularize)
- Theoretical reference functions (exact plaquette, topological susceptibility, autocorrelation)
- Presentation notebook (`presentation/phase0-naive-pipeline.ipynb`) with:
  - Fine HMC diagnostics (4-panel plot matching reference style)
  - Naive MCRG blocking explanation and visualization
  - Independent coarse HMC diagnostics
  - Blocked-fine vs coarse-HMC distribution comparison (histograms, CDFs, KS tests)
  - Summary and next steps

## Phase 1 Deliverables (Learned RG -- future)
- Learnable gauge-covariant path-weight blocker
- Generalized local coarse action with multiple loop terms
- Training loop with distribution-matching loss (MMD + contrastive)
- Comparison against Phase 0 naive baseline

## Verification Expectations
- Syntax or import checks are the minimum bar, not the full bar.
- If runtime validation is blocked by missing dependencies, say so explicitly and name the dependency.
- Broken package exports, missing documented modules, and examples that contradict agreed project assumptions should be treated as review blockers.

## Documentation Expectations
- `SPEC.md` is the scientific and architectural source of truth.
- `README.md` should stay consistent with the actual import surface.
- Example scripts should use defaults consistent with the current project baseline unless a deviation is intentionally being tested and is documented inline.
- Status reports for humans should say exactly which lattice sizes, couplings, observables, and comparison criteria were actually run.
- The presentation notebook is the primary human-facing deliverable for Phase 0.

## Escalation
- If a phase discovers that the current observable set is insufficient, raise that as a planning question instead of silently expanding scope.
- If learned blocking behavior depends on an entropy or sparsity regularizer, document the intended direction explicitly so the loss sign cannot be misread.
