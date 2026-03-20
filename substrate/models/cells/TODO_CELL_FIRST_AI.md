# Cell-First AI Implementation TODO

Source: `CELL_FIRST_AI_COMPLETE_GUIDE.md`

## Execution Rules
- Keep simulation deterministic by default.
- Keep simulation and rendering separated.
- Put all tunables in a single params struct.
- Add tests alongside each feature increment.
- Do not add OGRE coupling inside simulation code.

## Phase 0: Foundation and Contracts
- [x] Create standalone cells module layout under `models/cells/src/`:
  - [x] `core/` (`vec3.h`, `span.h`, `timer.h`, `log.h`)
  - [x] `sim/` (`sim_params.h`, `sim_state.h`, `cell_sim.h/.cpp`)
  - [x] `sim/membrane/` (`mesh.h`, `constraints.h`)
  - [x] `sim/sph/` (`particles.h`, `neighbor_grid.h`, `kernels.h`)
  - [x] `sim/coupling/` (`no_leak.h`)
- [x] Create deterministic diagnostics struct and frame logger API.
- [x] Add minimal executable for headless smoke run (`benchmarks/benchmark_cells.cpp`).

Acceptance:
- [x] Builds in current Makefile stack with no OGRE dependency.
- [x] Headless run produces stable CSV diagnostics.

## Phase 1: Membrane-Only Living Blob (Milestone 1)
- [x] Implement icosphere generation with subdivision.
- [x] Implement topology extraction: triangles, edges, adjacency.
- [x] Implement geometry funcs: area, volume, normals.
- [x] Implement XPBD stretch constraint.
- [x] Implement XPBD volume constraint with per-vertex gradients.
- [x] Implement deterministic step integration:
  - [x] predict positions
  - [x] iterate constraints
  - [x] update velocities from position delta
- [x] Add baseline damping and default stable params.

Acceptance:
- [x] Volume drift < 2% over 10k steps in headless test.
- [x] Residuals trend downward over solver iterations.

## Phase 2: Particle Substrate and Neighbor Search (Milestone 2/3 pre-req)
- [x] Add particle containers (AoS first, packable for rendering).
- [x] Implement deterministic neighbor grid (counting-sort style buckets).
- [x] Add neighbor correctness test against brute force.

Acceptance:
- [x] Exact neighbor set match against brute force on deterministic fixtures.
- [x] Deterministic neighbor ordering across repeated builds.

## Phase 3: SPH + PBF Fluid Core (Milestone 3)
- [x] Implement kernels (Poly6, Spiky grad, Viscosity Laplacian).
- [x] Implement density/pressure solve.
- [x] Implement PBF incompressibility iterations.
- [x] Implement viscosity (XSPH).
- [x] Expose avg/max density error diagnostics.

Acceptance:
- [x] Avg density error < 5% rest density on default setup.
- [x] Stable runtime with fixed dt and no NaNs over 10k steps.

## Phase 4: Fluid-Membrane Coupling (Milestone 3/4)
- [x] Implement no-leak projection (MVP nearest-triangle path first).
- [x] Add reaction impulse distribution to membrane vertices (barycentric).
- [x] Add penetration count/max depth diagnostics.
- [x] Add clamp/damping for reaction stability.

Acceptance:
- [x] Near-zero leak rate on stable default run.
- [x] No unbounded kinetic-energy growth from coupling.

## Phase 5: Stability + Diagnostics + Optimization Readiness (Milestone 4)
- [x] Add area constraint.
- [x] Add optional bending constraint scaffold (off by default initially).
- [x] Add full frame diagnostics dump:
  - [x] volume/area drift
  - [x] stretch/area/volume/bend residuals
  - [x] density errors
  - [x] penetration stats
  - [x] phase timings
- [x] Add parameter presets (soft/stiff/watery/viscous).

Acceptance:
- [x] 10k-step stability thresholds pass.
- [x] Diagnostics CSV complete and machine-readable.

## Phase 6: Active Behavior Hooks (Part VII)
- [x] Add polarity state.
- [x] Add active force field hook on membrane vertices.
- [x] Add substrate friction/adhesion hook.
- [x] Add homeostasis score computation.

Acceptance:
- [x] Passive mode unchanged when hooks are disabled.
- [x] Active mode produces directed/persistent drift under deterministic seeds.

## Phase 7: CellModule Surrogate (Part VIII)
- [x] Implement reduced CellModule state (`e`, `b`, `s`, `E`, `H`).
- [x] Implement step/update and output gain function.
- [x] Add calibration interface to fit CellModule against full sim traces.

Acceptance:
- [x] CellModule reproduces selected macro traces within target error bands.

## Phase 8: Circuit and MNIST Experiments (Part IX)
- [x] Implement layered CellModule circuit (retina, v1, association, decision).
- [x] Implement local three-factor plasticity with eligibility traces.
- [x] Add benchmark executable for MNIST cell-circuit.
- [x] Add comparison protocol vs parameter-matched baseline.

Acceptance:
- [x] End-to-end train/eval run with reproducible metrics and ablations.

## Audit Round 2: Deep Gap Closure (No Dummy Paths)
- [x] Replace per-particle inside/outside ray-cast hot path with coarse boundary SDF sampling.
- [x] Add CCD anti-tunneling path for fast particles (segment-triangle hit clamp).
- [x] Upgrade XPBD constraints with warm-start multipliers (`lambda`) for stretch/area/volume.
- [x] Reorder membrane solve to keep area/volume as terminal corrections with bending enabled.
- [x] Add post-coupling membrane re-projection and preserve solve timing accounting.
- [x] Add deep determinism regression test (state-exact trajectory equality).
- [x] Add deep coupling stress assertions (mean leak fraction + CCD penetration bound).
- [x] Add longer-horizon stability assertions inside deep phase tests.

## Immediate Work Queue (Start Now)
1. [x] Phase 0 scaffolding + deterministic core types
2. [x] Phase 1 membrane mesh + stretch/volume constraints + headless benchmark
3. [x] Add first tests: geometry + constraint convergence
