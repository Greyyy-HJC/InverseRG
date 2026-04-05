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
- Do not treat `beta_c ~= beta_f / 4` as a proof; treat it as the approved working baseline that still requires Phase 2 numerical validation.
- Keep gauge covariance explicit in both code and documentation when defining blocked links.
- Prefer small local Wilson-loop bases until there is evidence that a larger basis is necessary.

## Required Deliverables By Phase
- Phase 1
  - package scaffold
  - HMC baseline
  - measurement utilities
  - runnable baseline example
- Phase 2
  - fixed `2x2` gauge-covariant blocker
  - coarse Wilson baseline
  - explicit numerical validation plan or result for the working `beta_c` relation
- Phase 3
  - learnable path-weight blocker
  - generalized local coarse action
  - training loop and clear loss definition

## Verification Expectations
- Syntax or import checks are the minimum bar, not the full bar.
- If runtime validation is blocked by missing dependencies, say so explicitly and name the dependency.
- Broken package exports, missing documented modules, and examples that contradict agreed project assumptions should be treated as review blockers.

## Documentation Expectations
- `SPEC.md` is the scientific and architectural source of truth.
- `README.md` should stay consistent with the actual import surface.
- Example scripts should use defaults consistent with the current project baseline unless a deviation is intentionally being tested and is documented inline.

## Escalation
- If a phase discovers that the current observable set is insufficient, raise that as a planning question instead of silently expanding scope.
- If learned blocking behavior depends on an entropy or sparsity regularizer, document the intended direction explicitly so the loss sign cannot be misread.
