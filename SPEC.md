# InverseRG Specification

## Objective
Build a research codebase for inverse renormalization in 2D compact U(1) lattice gauge theory.

The project has two linked goals:
- simulate the fine theory with Hybrid Monte Carlo (HMC)
- learn a gauge-covariant Monte Carlo renormalization group (MCRG) map and a local coarse effective action that reproduce fine-lattice measurements after blocking

## Consistency Target
- The target is not just to make a few ensemble means numerically close.
- The intended notion of consistency is:
  - start from a fine-lattice ensemble
  - apply the MCRG blocking map to obtain a blocked coarse ensemble
  - separately sample a coarse ensemble with the learned coarse action
  - compare measurement distributions across configurations between those two coarse ensembles
- Therefore, future evaluation should treat per-measurement distributions as the primary object, with mean summaries used only as a first-pass proxy.

## Baseline Physics Model
- Fine degrees of freedom are link angles `theta[mu, x, y]` on a periodic 2D square lattice.
- Fine link variables are `U_mu(x) = exp(i theta_mu(x))`.
- The fine baseline action is the Wilson plaquette action:
  - `S_f[theta] = -beta_f * sum_p cos(theta_p)`
- The first coarse baseline action is also Wilson:
  - `S_c[Theta] = -beta_c * sum_P cos(Theta_P)`

## Blocking Model
- Default blocking is `2x2`, so the coarse lattice spacing is doubled.
- Coarse links must be built from gauge-covariant fine paths with the same endpoints.
- A safe baseline construction is:
  - enumerate a small local path family for each coarse-link direction
  - compute the path holonomy for each candidate path
  - combine paths with weights `w_p`
  - project the weighted sum back to U(1)
- The learned blocker should therefore operate over path families, not arbitrary link combinations, so gauge covariance is preserved by construction.

## Coarse Action Model
- The coarse action should remain local.
- Start with a small generalized Wilson basis:
  - plaquette `1x1`
  - rectangles `2x1` and `1x2`
  - add larger loops only if there is a clear need
- The action is parameterized by coefficients multiplying loop cosines.

## Coupling Baseline
- Working hypothesis for 2D U(1): `beta_c ~= beta_f / 4` under `2x2` blocking.
- This was accepted by @Greyyy on 2026-04-05 as the project baseline.
- It is still a calibration target, not a proof, so the blocked-fine and coarse ensembles must be compared numerically before the coarse Wilson baseline is treated as validated.

## Observable Targets
At minimum, the project should track:
- average plaquette
- topological-charge statistic
- at least one small Wilson loop beyond the plaquette

Additional observables can be added later if the learned coarse model matches the baseline set but still misses relevant structure.

For the current implementation stage:
- Phase 2 runtime reporting uses summary measurements on coarse configurations:
  - `plaquette`
  - `topological_charge`
  - `plaquette_angle_mean`
  - `wilson_1x1`
  - `wilson_1x2`
  - `wilson_2x2`
- Phase 3 training currently optimizes a smaller proxy basis:
  - `plaquette`
  - `rectangle_x`
  - `rectangle_y`

This means the present code is a valid first implementation pass, but not yet the final distribution-matching formulation.

## Project Stages
1. Fine-theory HMC for 2D compact U(1).
2. Measurement utilities and reproducible baseline runs.
3. Fixed `2x2` gauge-covariant blocking.
4. Coarse Wilson baseline with the working `beta_c` hypothesis.
5. Learn blocking path weights with the coarse action fixed or lightly constrained.
6. Jointly learn blocking weights and the local coarse action.

## Current Module Targets
The current implementation direction reported by @Builder uses a flat package layout:
- `inverserg/hmc.py`
- `inverserg/lattice.py`
- `inverserg/measurements.py`
- `inverserg/blocking.py`
- `inverserg/actions.py`
- `inverserg/baselines.py`
- `inverserg/training.py`
- `examples/fixed_coarse_baseline.py`
- `examples/train_learned_rg.py`

## Acceptance Criteria
- HMC runs with stable acceptance behavior and reasonable Hamiltonian conservation.
- Measurement utilities produce reproducible outputs across seeds.
- Fixed blocking returns valid U(1) coarse links and preserves gauge covariance numerically.
- Phase 2 includes an explicit blocked-fine vs coarse-baseline comparison for the working `beta_c` hypothesis.
- The learned blocker and coarse action improve observable matching relative to the fixed baseline.
- The next evaluation milestone is to compare measurement distributions across coarse configurations, not only summary means.

## Non-Goals For The First Pass
- exhaustive loop bases
- aggressive model complexity before the baseline is numerically stable
- claiming a validated RG map before the coarse Wilson baseline has been checked
