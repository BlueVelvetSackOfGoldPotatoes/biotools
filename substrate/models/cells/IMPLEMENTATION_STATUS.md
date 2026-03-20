# Cell-First AI Implementation Status

Canonical guide: `models/cells/CELL_FIRST_AI_COMPLETE_GUIDE.md`

This file is the reality check for the cells roadmap. Status values are:
- `implemented`
- `partial`
- `not implemented`

## Status Ledger

| Guide area | Status | Notes |
| --- | --- | --- |
| Canonical guide migration | implemented | The former top-level `cells/README.md` has been promoted to `models/cells/CELL_FIRST_AI_COMPLETE_GUIDE.md`, and the placeholder top-level `cells/` folder has been removed. |
| Deterministic 3D cell substrate | implemented | `models/cells/src/cell_models.h`, `models/cells/src/cell_models.cpp`, `benchmarks/benchmark_cells.cpp`, `tests/test_cells_deep.cpp`. |
| 3D active-force hooks and homeostasis | implemented | Existing hooks plus public per-step control input via `CellControlInput` and `CellSim::step(control, dt)`. |
| 3D chemotaxis / shift / damage task harness | implemented | `models/cells/src/cell_tasks.h`, `models/cells/src/cell_tasks.cpp`, `benchmarks/benchmark_cell_tasks.cpp`. |
| Lightweight recurrent baseline for 3D tasks | implemented | `RecurrentBaselineController` in `models/cells/src/cell_tasks.*`. |
| 2D tissue / morphogenesis engine | implemented | `models/cells/src/cell_tissue2d.h`, `models/cells/src/cell_tissue2d.cpp`, `benchmarks/benchmark_leaf_cells.cpp`. |
| Guidance / region / attention / teacher / novelty fields | implemented | `FieldSample2D` and `LeafField2D` now expose these channels; developmental MNIST field uses them directly. |
| Contact-timed self-wiring tissue graph | implemented | `models/cells/src/cell_developmental.h`, `models/cells/src/cell_developmental.cpp`. |
| Edge pruning / eligibility traces / compatibility-based wiring | implemented | Developmental graph updates are local and deterministic; pruning is exercised in `tests/test_leaf_cells.cpp`. |
| Fixed-graph MNIST cell circuit baseline | implemented | `models/cells/src/cell_circuit.*`, `benchmarks/benchmark_cell_circuit.cpp`, `benchmarks/benchmark_cell_circuit_compare.cpp`. |
| Developmental MNIST benchmark | implemented | `benchmarks/benchmark_cell_developmental.cpp`. |
| Developmental vs fixed-graph comparison harness | implemented | Same benchmark logs clean accuracy, sensor-damage robustness, shift robustness, energy/homeostasis, and developmental internal-graph damage. |
| Optional OGRE/CMake sidecar | implemented | Added under `models/cells/standalone_ogre/`; kept outside the Makefile path with dedicated CMake targets (`cell_sim`, `cell_render_ogre`, `cell_app`) and a smoke check (`make test_cells_ogre_smoke`). |
| Manual OGRE viewer app | partial | `cell_app` scaffold exists with fixed-step stepping and debug controls. Runtime validation still depends on a local OGRE install. |
| Internal-teacher developmental curriculum | implemented | The developmental classifier exposes a switchable internal-teacher mode with blend control (`CELL_DEV_INTERNAL_BLEND`) and epoch scheduling (`CELL_DEV_INTERNAL_AFTER_EPOCH`). |
| Broad parameter-matched GRU/LSTM/Transformer comparison suite | implemented | `benchmark_cell_tasks` now supports `rnn`, `gru`, `lstm`, and `transformer` baselines (`CELL_TASK_CONTROLLERS=...`) alongside the homeostatic cell controller. |
| OGRE smoke test in default repo tests | implemented | `make test_cells_ogre_smoke` is available and integrated into `make test_cells_suite` as an optional/skip-aware check (`REQUIRE_OGRE=1` for strict mode). |

## Tests Covering The New Work

- `make test_cells_deep`
  - default-control path parity for `CellSim`
  - deterministic task harness regression (homeostatic + rnn/gru/lstm/transformer baselines)
- `make test_leaf_cells`
  - deterministic developmental field sampling
  - deterministic self-wiring edge formation
  - pruning via developmental damage
  - developmental MNIST smoke test above the random floor
- `make test_cells_ogre_smoke`
  - configure/build smoke test for the standalone OGRE sidecar
  - skip-aware when OGRE packages are unavailable; strict mode with `REQUIRE_OGRE=1`

## Practical Entry Points

- Raw simulator diagnostics: `./bin/benchmark_cells`
- 3D task harness: `./bin/benchmark_cell_tasks`
- Fixed-graph MNIST baseline: `./bin/benchmark_cell_circuit`
- Developmental MNIST compare: `./bin/benchmark_cell_developmental`
- 2D morphogenesis benchmark: `./bin/benchmark_leaf_cells`
- Optional OGRE viewer: `models/cells/standalone_ogre/`
