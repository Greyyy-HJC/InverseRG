# InverseRG Specification

## Objective

- Simulate 2D compact U(1) lattice gauge theory with HMC
- Learn a gauge-covariant MCRG blocking map and local coarse effective action that reproduce fine-lattice measurement distributions after blocking

## Consistency Target

- Primary criterion: distributional agreement of per-configuration observables between blocked-fine and coarse-model ensembles
- Mean matching is a first-pass proxy, not the success criterion

## Baseline Physics Model

- Fine degrees of freedom: link angles `theta[mu, x, y]` on periodic 2D square lattice
- Fine action: Wilson plaquette `S_f = -beta_f * sum_p cos(theta_p)`
- Coarse action: `LocalWilsonLoopAction` with plaquette + rectangle basis

## Blocking Model

- 2x2 blocking (coarse lattice spacing = 2 * fine)
- Naive: sum two consecutive link phases along same direction, regularize to `[-pi, pi]`
- Learned: 7 non-backtracking paths per direction within |transverse| <= 1, combined via circular average with softmax weights from gauge-invariant features

## Coupling Baseline

- Tree-level: `beta_c = beta_f / 4` (2D, factor-2 blocking)
- Calibration target only; must be validated numerically

## Observable Targets

- Average plaquette (exact reference: `I1(beta)/I0(beta)`)
- Topological charge and susceptibility
- Wilson loops: 1x2, 2x1, 2x2
- Per-configuration distributions, not just ensemble means

## Phase Overview

| Phase | Status | Focus |
|-------|--------|-------|
| 0 | complete | Naive pipeline: HMC, naive blocking, baseline comparison |
| 1 | complete | Learned blocking: 7-path CNN blocker, Wilson-loop coarse action, MMD+contrastive training |
| 2 | current | RG monotone: multi-beta coupling flow C(J), beta function |

## Phase 2 Constraints

- Coupling space: `J = (beta_plaq, beta_rect_x, beta_rect_y)`, dim = len(action.basis)
- RG monotone: MLP `C_theta: R^d -> R`; RG flow `dJ/dl = -grad_J C(J)`
- Flow integration: Euler method with `create_graph=True` for backprop through the ODE
- Two-stage approach:
  1. Data collection: run Phase 1 `train_learned_rg` at a grid of beta values to collect `(J_fine, J_coarse_optimal)` pairs
  2. Monotone fitting: train `C_theta` so its gradient flow maps each `J_fine` to the corresponding `J_coarse`
- Blocker: shared across beta values (gauge-covariant blocking is geometry, not coupling-dependent)
- Validation: predicted `J_coarse` vs tree-level baseline and vs Stage 1 collected pairs
- 2D compact U(1) has no phase transition; monotone should decrease monotonically along the flow direction

## Acceptance Criteria

- HMC runs with stable acceptance and reasonable Hamiltonian conservation
- Measurement utilities produce reproducible outputs across seeds
- Blocking preserves gauge covariance
- Distribution-level comparison presented with KS tests, MMD, energy distance
- Presentation notebooks are self-contained and runnable
- Phase 2: monotone-predicted `J_coarse` agrees with independently trained `J_coarse` across the beta grid
