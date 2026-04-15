# InverseRG Specification

## Objective

- Simulate 2D compact U(1) lattice gauge theory with HMC
- Learn a gauge-covariant MCRG blocking map and local coarse effective action that reproduce fine-lattice measurement distributions after blocking
- Learn a coarse-to-fine inverse RG transformation that lifts coarse configurations back to fine lattices while preserving gauge covariance and fine-theory consistency

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
| 2 | complete | RG monotone: multi-beta coupling flow C(J), beta function |
| 3 | current | Forward/inverse RG: `J_f -> (J_c, z_phi)` and coarse-to-fine lifting |

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

## Phase 3 Constraints

- Scope: single-step `2x2` inverse lift only
- Fine theory: Wilson-only in v1, but the coupling interface stays `J = (beta_plaq, beta_rect_x, beta_rect_y)`
- Forward RG must be a continuous model over coupling space:
  - input `J_f`
  - output coarse couplings `J_c`
  - output blocker conditioning code `z_phi`
- Blocker must be a shared gauge-covariant spatial backbone conditioned by `z_phi`, not a separate per-beta artifact bank
- Inverse RG must be stochastic: input `(theta_c, J_c, J_f, z_phi, noise)` and output `theta_f`
- Inverse proposal must be strictly gauge-equivariant by construction
- The coarse-to-fine lift must factor into:
  1. gauge-covariant canonical prolongation
  2. gauge-invariant closed-loop residual modes
  3. equivariant refinement with blocker-consistency and fine-action terms
- Round-trip consistency target:
  - block generated `theta_f` with the frozen forward blocker
  - recover the input coarse configuration and its coarse observable distributions
- The monotone is optional in Phase 3 and is not the primary inverse-RG training path

## Acceptance Criteria

- HMC runs with stable acceptance and reasonable Hamiltonian conservation
- Measurement utilities produce reproducible outputs across seeds
- Blocking preserves gauge covariance
- Distribution-level comparison presented with KS tests, MMD, energy distance
- Presentation notebooks are self-contained and runnable
- Phase 2: monotone-predicted `J_coarse` agrees with independently trained `J_coarse` across the beta grid
- Phase 3:
  - forward RG predicts stable `(J_c, z_phi)` across the Wilson beta grid
  - inverse RG generates fine configurations whose blocked images match the input coarse configurations
  - inverse-generated fine ensembles move toward direct fine HMC ensembles in distributional metrics
