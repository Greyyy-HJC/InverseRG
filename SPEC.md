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
- Therefore, evaluation should treat per-measurement distributions as the primary object, with mean summaries used only as a first-pass proxy.

## Baseline Physics Model
- Fine degrees of freedom are link angles `theta[mu, x, y]` on a periodic 2D square lattice.
- Fine link variables are `U_mu(x) = exp(i theta_mu(x))`.
- The fine baseline action is the Wilson plaquette action:
  - `S_f[theta] = -beta_f * sum_p cos(theta_p)`
- The first coarse baseline action is also Wilson:
  - `S_c[Theta] = -beta_c * sum_P cos(Theta_P)`

## Blocking Model

### Naive Blocking (current baseline)
- Default blocking is `2x2`, so the coarse lattice spacing is doubled.
- Coarse links are formed by summing the phases of two consecutive fine links along the same direction, then regularizing to `[-pi, pi]`:
  - x-direction: `Theta_x(i,j) = regularize(theta_x(2i,2j) + theta_x(2i+1,2j))`
  - y-direction: `Theta_y(i,j) = regularize(theta_y(2i,2j) + theta_y(2i,2j+1))`
- This preserves gauge covariance because the sum of link phases along a path equals the phase of the product of link variables.

### Gauge-Covariant Path Blocking (implemented)
- Enumerate all 7 non-backtracking paths (up to 4 links) per coarse-link direction within transverse offset <= 1.
- Compute the path holonomy (sum of link phases) for each candidate path.
- Combine paths via circular (vector) average with softmax weights.
- Weights predicted per-site by a CNN from gauge-invariant features (plaquette + rectangle cosines).
- Gauge covariance is preserved by construction: path holonomies transform covariantly, weights depend only on gauge-invariant inputs.

## Coarse Action Model
- The coarse action should remain local.
- Start with the pure Wilson plaquette action at `beta_c = beta_f / 4`.
- Generalized Wilson basis: plaquette `1x1`, rectangles `2x1` and `1x2` (implemented as `LocalWilsonLoopAction`).

## Coupling Baseline
- Working hypothesis for 2D U(1): `beta_c = beta_f / 4` under `2x2` blocking.
- This corresponds to doubling the lattice spacing (factor 2 blocking in 2D).
- It is a calibration target, not a proof, so the blocked-fine and coarse ensembles must be compared numerically.

## Observable Targets
At minimum, the project should track:
- average plaquette (with exact reference `I1(beta)/I0(beta)`)
- topological charge and susceptibility
- small Wilson loops beyond the plaquette (1x2, 2x2)
- rectangle observables (2x1, 1x2)

## Project Phases

### Phase 0: Naive Pipeline (complete)
Validate the full pipeline without neural networks:
1. HMC for fine 2D U(1) ensembles (L=32, beta=4.0, 1000 configs)
2. Naive 2x2 blocking to produce coarse ensembles (L=16)
3. Independent coarse HMC (L=16, beta=1.0, 1000 configs)
4. Distribution-level comparison of blocked-fine vs coarse-HMC ensembles
5. Presentation notebook with HMC diagnostics, blocking visualization, and ensemble comparison

### Phase 1: Learned Blocking and Coarse Action (current)
- 7-path gauge-covariant path family per direction (all non-backtracking paths within |transverse| <= 1)
- `SpatialGaugeCovariantBlocker`: CNN predicts per-site path combination weights from 12-channel gauge-invariant features (plaquette + rectangle cosines)
- `LocalWilsonLoopAction` with plaquette + rectangle basis for the coarse action
- Training loop with MMD + contrastive + mean-mismatch loss
- Train/test split for evaluation (configurable via `n_test_samples`)
- Blocker type selection via `blocker_type` config: `"spatial"`, `"global"`, `"fixed"`
- Compare against Phase 0 naive baseline
- Lattice size: L=32→16 (matching Phase 0), 1000 configurations
- Two coupling points: β=4.0→1.0 and β=6.0→1.5
- Presentation notebooks: `phase1-learned-blocking-beta4.ipynb`, `phase1-learned-blocking-beta6.ipynb`

### Phase 2: Scaling and Validation
- Larger lattices and multiple beta values
- Distributional agreement as primary success criterion
- Systematic error analysis

## Current Module Targets
- `inverserg/hmc.py` -- HMC sampler with Omelyan integrator and diagnostics
- `inverserg/lattice.py` -- geometric helpers for loops, topology, regularization
- `inverserg/measurements.py` -- observable extraction and theoretical references
- `inverserg/blocking.py` -- naive blocker, 7-path family, NN gauge-covariant blockers
- `inverserg/actions.py` -- local Wilson-loop coarse actions
- `inverserg/baselines.py` -- tree-level coupling relations
- `inverserg/diagnostics.py` -- KS tests, distribution plots, reports
- `presentation/phase0-naive-pipeline.ipynb` -- Phase 0 naive pipeline presentation
- `presentation/phase1-learned-blocking-beta4.ipynb` -- Phase 1 learned blocking (β=4.0, L=32→16)
- `presentation/phase1-learned-blocking-beta6.ipynb` -- Phase 1 learned blocking (β=6.0, L=32→16)

## Acceptance Criteria
- HMC runs with stable acceptance and reasonable Hamiltonian conservation.
- Measurement utilities produce reproducible outputs across seeds.
- Naive blocking returns valid U(1) coarse links.
- Distribution-level comparison of blocked-fine vs coarse-HMC ensembles is presented with KS tests.
- The presentation notebook is self-contained and runnable.

## Non-Goals For Phase 0
- Neural network training
- Learned blocking weights
- Aggressive model complexity
- Claiming a validated RG map before the naive baseline is checked
