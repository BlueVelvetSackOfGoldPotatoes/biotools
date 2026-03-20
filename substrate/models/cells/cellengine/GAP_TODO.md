# CellEngine Gap Todo

Date: 2026-03-14

This file turns the current Deep Dive / ODD review into an implementation backlog.

## 1. Documentation Sync

- [x] Update [DEEP_DIVE.md](/home/pessegueirolunar/Documents/MNIST/models/cells/cellengine/DEEP_DIVE.md) so it no longer says morphology is fixed.
- [x] Update `DEEP_DIVE.md` to reflect that replay no longer retrains RL on the UI path.
- [x] Update `DEEP_DIVE.md` so the task-general interface is described explicitly, not as mostly cartpole.
- [x] Add a short "implemented vs planned" matrix shared by `README.md`, `DEEP_DIVE.md`, and `ODD_GOLD_STANDARD.md`.

## 2. ODD Gold-Standard Telemetry

### 2.1 Per-cell current decomposition

- [x] Log per tick and per cell:
  - [x] `I_elec`
  - [x] `I_chem`
  - [x] `I_mech`
  - [x] self-chemical term
  - [x] `I_total`

### 2.2 Stress decomposition

- [x] Log per tick and per cell:
  - [x] hinge-moment contribution
  - [x] lateral-force contribution
  - [x] gravity-load contribution
  - [x] cart-velocity contribution
  - [x] relaxed neighbor contribution
  - [x] final smoothed `stress`

### 2.3 Mechanical contribution decomposition

- [x] Log per tick and per cell:
  - [x] `a_gate`
  - [x] `mech_contract`
  - [x] contractility prefactor
  - [x] final per-cell force contribution

### 2.4 Gap plasticity decomposition

- [x] Log per tick and per edge:
  - [x] `corr_ij`
  - [x] Hebbian increment
  - [x] decay term
  - [x] old `gap`
  - [x] new `gap`

### 2.5 Trial context variables

- [x] Log per trial:
  - [x] teacher active yes/no
  - [x] damage active yes/no
  - [x] tissue snapshot before reset
  - [x] tissue snapshot after reset

Current state: replay mode emits the legacy unprefixed files, and benchmark/report writes now emit
named bundles:
- `clean_teacher_trial_context.json`
- `clean_autonomous_trial_context.json`
- `damaged_autonomous_trial_context.json`
with corresponding pre/post reset cell snapshots.

### 2.6 Decoded parameter dump

- [x] Persist once per replay/run:
  - [x] all decoded scalar parameters
  - [x] all developmental `W[g][f]`
  - [x] all developmental `bias[g]`

## 3. ODD Intervention Suite

### 3.1 Single-variable replay ablations

- [x] remove `I_elec`
- [x] remove `I_chem`
- [x] remove `I_mech`
- [x] remove self-chemical term
- [x] freeze `sigma`
- [x] freeze `gap`
- [x] freeze `G` via `no_slow_drift`
- [x] zero teacher

### 3.2 Gene-channel ablations

- [x] for each `G[k]`, support:
  - [x] zero
  - [x] freeze
  - [x] small perturbation

### 3.3 Per-cell mechanical importance

- [x] zero only one cell's mechanical output
- [x] measure delta in survival / task metrics
- [x] expose ranking in report and UI

### 3.4 Edge importance

- [x] zero one edge or edge family
- [x] measure delta in survival / task metrics
- [x] expose ranking in report and UI

## 4. Benchmark Maturity

- [ ] Bring `mass_spring_balance` to the same solve/tuning quality as cartpole.
- [ ] Bring `worm_drag_race` to the same solve/tuning quality as cartpole.
- [ ] Bring `pong_return` to the same solve/tuning quality as cartpole.
- [ ] Add per-task benchmark baselines and target thresholds that are explicit and justified.
- [ ] Add per-task ODD outputs, not only the cartpole-style interpretation.

## 5. Biological Model Gaps

- [ ] Add a second connectivity layer for neurite / axon-like growth.
- [ ] Add long-range targeted wiring instead of only local contact adjacency.
- [ ] Add conduction delay / transmission lag for long-range edges.
- [ ] Add richer GRN-style dynamics beyond the 8-channel slow phenotype vector.
- [ ] Add cell migration / remeshing during development or task runtime.
- [ ] Add explicit cell division / proliferation during development with lineage tracking.

## 6. Task-General Interface Gaps

- [ ] Generalize beyond force-control tasks to proper multi-action decision tasks.
- [ ] Introduce a clean task API so symbolic/discrete tasks do not need force-control hacks.
- [ ] Revisit tic-tac-toe only after the controller/output interface is generalized.

## 7. Baselines And Optimization

- [ ] Keep cell-state fully resident on GPU for whole episodes.
- [ ] Batch multiple organisms/trials on GPU.
- [ ] Parallelize refinement/local search, not only population evaluation.
- [ ] Vectorize RL env rollout.
- [ ] Add stronger RL baselines:
  - [ ] PPO
  - [ ] stronger actor-critic implementation
- [ ] Add apples-to-apples damage baselines where possible.

## 8. UI / Analysis Gaps

- [x] Surface the new ODD decomposition telemetry directly in the React UI.
- [ ] Add counterfactual intervention viewers, not just replay telemetry viewers.
- [x] Add per-cell and per-edge causal ranking panels from intervention results.
- [x] Add explicit telemetry-quality badges:
  - [x] measured
  - [x] restored
  - [x] fitness-only
- [ ] Add population-wide morphology / causal distribution views tied to benchmark outputs.

## 9. Data Integrity And Testing

- [x] Enforce that fixed `grown3d` cannot silently realize `extent_z = 0`.
- [x] Derive body depth from actual occupied cells, not padded diffusion grid.
- [ ] Add invariant tests for all body modes and discoverability combinations.
- [x] Add regression tests for the new ODD decomposition CSV columns.
- [ ] Add replay/run consistency tests so summary metadata always matches emitted body cells.

## 10. Suggested Execution Order

1. ODD telemetry
2. ODD interventions
3. Docs sync
4. Task maturity
5. Biological connectivity/development expansion
6. Performance optimization
7. UI causal analysis polish
