# Substrate Repository Specification

## 1. Purpose and Scope

This repository is a C++17 machine learning benchmark suite centered on MNIST classification plus discrete-control tasks and biologically-inspired cell simulation, with:

- A shared core runtime (`core/`) for tensor math, modules, optimization, losses, metrics, data loading, and logging.
- 17 model families (`models/`) and 26 benchmark entrypoints (`benchmarks/`).
- A common run artifact contract (`runs/<run_id>/...`) consumed by Python plotting, Streamlit analytics, and a React + Node dashboard/API.
- A substantial cells/biological simulation subsystem including a standalone physics-based evolutionary cell engine (`cellengine`), developmental tissue benchmarks, task-driven cell controllers, and optional 3D OGRE visualization.

The maintained implementation path is:

- `core/`
- `models/`
- `benchmarks/`

Legacy code exists under `legacy/standalone_mlp_app/` and is not the primary development target.

## 2. Repository Layout

### 2.1 Source of Truth (maintained code)

- `core/`
  - `tensor/`: tensor type, math ops, CPU/CUDA matmul routing (`tensor.h|.cpp`, `cuda_backend.h|.cpp`).
  - `nn/`: layers/modules — linear, activations, normalization, conv/pooling, embedding, sequential (`module.h|.cpp`).
  - `optim/`: SGD/Adam/RMSProp + LR schedulers (`optimizer.h|.cpp`).
  - `losses/`: CE, MSE, BCE, NLL, KLDiv, Huber + softmax helpers (`losses.h|.cpp`).
  - `metrics/`: classification/regression/clustering metrics (`metrics.h|.cpp`).
  - `data/`: MNIST IDX loader, generic batching, sequence batching (`dataloader.h|.cpp`).
  - `io/`: `RunLogger` and run artifact writer (`run_logger.h|.cpp`).
  - `online/`: `TrainableModel` contract + continuous active/sleep runtime (`trainable_model.h`, `continuous_runtime.h|.cpp`).
  - `benchmark/`: discrete env interface, bit bridge/codec, Q runtime, game envs (`discrete_env.h`, `discrete_q_runtime.h|.cpp`, `bit_codec.h|.cpp`, `bit_bridge_model.h|.cpp`, `piecewise_bit_bridge_model.h|.cpp`, `tictactoe_env.h|.cpp`, `game_envs.h|.cpp` plus per-game implementations for connect four, battleship, go-lite, chess-lite, and cartpole).
  - `bio/`: biology runtime and parameter dynamics instrumentation (`bio_runtime.h|.cpp`, split across `bio_runtime_core.cpp` and `bio_runtime_runtime.cpp`, plus `bio_runtime_internal.h`, `bio_affine.h`).
  - `interpret/`: placeholder (empty).
  - `training/`: placeholder (empty).
  - `utils/`: placeholder (empty).
- `models/`
  - Family implementations in `models/<family>/src/*.h|*.cpp`.
  - 17 families present: `actor_critic`, `cells`, `clustering`, `cnn`, `diffusion`, `forests`, `forward_forward`, `gnn`, `hebbian`, `hybrid`, `lstm`, `markov`, `reinforcement`, `rnn`, `transformer`, `trees`, `vit`.
  - `models/hybrid/src/model_registry.*` maps adapter-supported families to the shared `TrainableModel` contract.
  - `models/cells/src/` includes the cell simulation stack:
    - Core models: `cell_models.h|.cpp`, `cell_models_internal.h`, `cell_models_membrane.cpp`, `cell_models_sim.cpp` (with `.inc` setup/runtime split).
    - Cell circuit: `cell_circuit.h|.cpp` — fixed-graph MNIST cell circuit.
    - Cell module: `cell_module.h|.cpp` — individual cell dynamics.
    - Cell tasks: `cell_tasks.h|.cpp` — task-driven controllers (chemotaxis, distribution-shift, damage-recovery) with baseline comparison matrix.
    - Cell tissue 2D: `cell_tissue2d.h|.cpp` — 2D morphogenesis/tissue simulation.
    - Cell developmental: `cell_developmental.h|.cpp` — migrating self-wiring developmental tissue with internal-teacher curriculum.
    - Sub-simulators: `sim/` (cell_sim, SPH neighbor grid, particle dynamics) and `core/` (coupling, membrane constraints).
  - `models/cells/cellengine/` — standalone physics-based evolutionary cell engine:
    - Core: `cellengine.h|.cpp`, `cellengine_internal.h`.
    - Split compilation units: `cellengine_primitives.cpp`, `cellengine_cuda.cpp`, `cellengine_env_body.cpp`, `cellengine_population.cpp`, `cellengine_trace_io.cpp`, `cellengine_evolution_odd.cpp`, `cellengine_rl_replay.cpp`.
    - Implementation includes (`.inc` files): primitives, CUDA, env body, population, trace I/O, evolution ODD, RL replay, public API.
    - Documentation: `DEEP_DIVE.md`, `ODD_GOLD_STANDARD.md`, `README.md`, `RESULTS_2026-03-09.md`, `GAP_TODO.md`.
  - `models/cells/standalone_ogre/`: standalone CMake + OGRE 3D visualization project (`CMakeLists.txt`, `vcpkg.json`, `app/`, `render/`, `media/`, `resources.cfg`).
  - `models/cells/` documentation: `CELL_FIRST_AI_COMPLETE_GUIDE.md`, `IMPLEMENTATION_STATUS.md`, `TODO_CELL_FIRST_AI.md`, plus report MDs.
  - `models/cellspec.md`: full specification document for the CellEngine benchmark — scientific basis, cell model derivation, evolutionary algorithm, and evaluation methodology.
- `benchmarks/`
  - 26 benchmark executables plus shared utilities (`hpo_utils.h`, `logging_utils.h`).
- `analytics/`
  - Python plot generation, per-family plot modules, and Streamlit dashboard.
- `scripts/`
  - Random search, bio sweep queue, run pruning, smoke test, cells OGRE smoke test, cells production check.
- `tests/`
  - Native C++ tests: `test_core_smoke.cpp`, `test_cells_deep.cpp`, `test_leaf_cells.cpp`, `test_cellengine.cpp`.
- `ui/`
  - React frontend (`ui/src/`) with component library, hooks, and utilities.
  - Express API (`ui/api/server.mjs`).
  - CellEngine support scripts (`ui/scripts/`).

### 2.2 Data / Docs / Legacy

- `data/`
  - Raw MNIST IDX files:
    - `train-images-idx3-ubyte`
    - `train-labels-idx1-ubyte`
    - `t10k-images-idx3-ubyte`
    - `t10k-labels-idx1-ubyte`
- `docs/`
  - Architecture and contract docs: `ARCHITECTURE.md`, `ANALYTICS.md`, `BENCHMARK_PLUGIN.md`, `HPO_RANDOM_SEARCH.md`, `BIO_SWEEP_QUEUE.md`, `HYBRID_ADAPTER_CAPABILITIES.md`.
  - Code review and pipeline analysis notes: `CODE_REVIEW_2026-03-03.md`, `CODE_REVIEW_FIX_TODOS_2026-03-03.md`, `FULL_REPOSITORY_PIPELINES_AND_CODE_REVIEW_2026-03-04.md`, `FULL_REPOSITORY_PIPELINES_AND_CODE_REVIEW_DETAILED_2026-03-04.md`.
- `legacy/standalone_mlp_app/`
  - Older standalone MLP + interpretability codepath (`include/` and `src/`), kept for reference.

### 2.3 Generated / Runtime Artifacts

- `build/`, `bin/`, `obj/`: compiled artifacts.
- `runs/`: run logs and manifests (can be large; 130+ run directories).
- `reports/`: generated plots and experiment summaries (70+ report directories).
- `output/`: simulation outputs (cells diagnostics, cellengine evolution outputs, leaf tissue outputs).
- `logs/`: additional runtime logs.
- `test-results/`: test execution metadata.
- `ui/node_modules/`, `ui/dist/`: frontend dependencies/build output.

`.gitignore` excludes build and runtime artifact directories (`build/`, `bin/`, `obj/`, `output/`, `runs/`, `reports/`) plus Python/editor cache files.

## 3. Build and Execution

### 3.1 Toolchain

- Compiler: `g++` by default (`CXX ?= g++`).
- Standard: C++17 (`-std=c++17`).
- Build modes:
  - `BUILD_TYPE=release` (default, `-O3 -DNDEBUG`)
  - `BUILD_TYPE=debug` (`-O0 -g3`)
- Optional flags:
  - `OPENMP=1` (default) for `-fopenmp`
  - `NATIVE=1` (default) for `-march=native -mtune=native`

### 3.2 CUDA Behavior

`Makefile` supports `USE_CUDA`:

- `USE_CUDA=auto` (default): enables CUDA if `CUDA_HOME` libs/headers are found.
- `USE_CUDA=1`: requires CUDA; build fails if unavailable.
- `USE_CUDA=0`: CPU-only build.

Runtime tensor offload controls:

- `TENSOR_USE_CUDA`
- `TENSOR_FORCE_CUDA`
- `TENSOR_CUDA_MIN_FLOPS`

Implemented in `core/tensor/cuda_backend.cpp`.

CellEngine also has its own CUDA compilation unit (`cellengine_cuda.cpp`).

### 3.3 Main Build Targets

- `make benchmarks`: builds all 26 benchmark binaries.
- `make run_all`: executes all benchmarks sequentially (fail-fast).
- `make test_smoke`: builds/runs core smoke tests.
- `make test_cells_deep`: builds/runs deep cells tests.
- `make test_leaf_cells`: builds/runs leaf tissue tests.
- `make test_cellengine`: builds/runs cellengine tests.
- `make test_cells_ogre_smoke`: OGRE visualization smoke test (optional, `REQUIRE_OGRE=1` for strict mode).
- `make test_cells_suite`: composite target running all 5 test targets above.
- `make cells_prod_check`: runs `scripts/cells_production_check.sh` for production readiness.
- `make clean`: removes `build/` and `bin/`.
- `./build_all.sh`: wrapper for `make benchmarks`.

### 3.4 Benchmark Binaries (26)

MNIST family benchmarks:
- `benchmark_mlp`
- `benchmark_cnn`
- `benchmark_rnn`
- `benchmark_lstm`
- `benchmark_transformer`
- `benchmark_vit`
- `benchmark_trees`
- `benchmark_forests`
- `benchmark_clustering`
- `benchmark_hebbian`
- `benchmark_actor_critic`
- `benchmark_diffusion`
- `benchmark_gnn`
- `benchmark_forward_forward`
- `benchmark_reinforcement`
- `benchmark_markov`
- `benchmark_continuous`

Discrete/control benchmarks:
- `benchmark_tictactoe`
- `benchmark_games`

Cell/simulation benchmarks:
- `benchmark_cells`
- `benchmark_cell_circuit`
- `benchmark_cell_circuit_compare`
- `benchmark_leaf_cells`
- `benchmark_cell_tasks`
- `benchmark_cell_developmental`
- `benchmark_cellengine`

## 4. Core Architecture Contracts

### 4.1 Shared Model Contract

`core/online/trainable_model.h` defines `TrainableModel`:

- `forward`, `backward`
- `parameters`, `gradients`, `zero_grad`
- `train`, `eval`
- `id`

Optional specialized routing contract:

- `ActionOwnerAwareModel` for owner-group action mapping (used by per-piece control modes).

### 4.2 Discrete Benchmark Contract

`core/benchmark/discrete_env.h` defines `benchmark::DiscreteEnv`:

- Metadata: `benchmark_id`, `benchmark_name`, `task_type`
- Dimensions: `observation_bits`, `num_actions`
- Interaction: `reset`, `legal_actions`, `step`
- Optional teacher/tactical signals and action-owner metadata.

`core/benchmark/discrete_q_runtime.h` defines `DiscreteQRuntime` with:

- Deep-Q style train/eval loops
- Epsilon schedule
- Eval metrics (win/draw/loss, tactical capture/miss rates, latency, confusion-compatible inference samples)
- Hook/callback system for logging/instrumentation

### 4.3 Run Logging Contract

`core/io/run_logger.h` provides `RunLogger`:

- Creates `runs/<run_id>/...` structure.
- Writes:
  - `manifest.json`
  - `learning/*.csv`
  - `deployment/*.csv`
  - `model_specific/<family>/*.csv` (through `append_csv_row`)
  - runtime heartbeat updates
- Captures run metadata including benchmark identity, params JSON, git commit/dirty status, process ID, `trial_uuid`, and `job_origin`.

### 4.4 Biology Runtime

`core/bio/bio_runtime.h` provides `BioRuntime` with a `BioConfig` that includes:

- Structural plasticity
- Homeostasis
- Three-factor modulation
- Myelination / conduction delays
- Bioelectric coupling
- Excitatory/inhibitory sign constraints
- Wiring cost controls

Default safety mode is observational (`active_parameter_updates = false`) unless explicitly enabled (`BIO_ACTIVE=1`).

Implementation is split across `bio_runtime_core.cpp` (core logic) and `bio_runtime_runtime.cpp` (runtime integration), with internal helpers in `bio_runtime_internal.h` and affine utilities in `bio_affine.h`.

## 5. Model Families

Each family has code in `models/<family>/src/`.

- `mlp` logic is assembled mainly in benchmark/registry paths via shared `core/nn` blocks.
- `cnn`: convolutional classifiers and variants.
- `rnn`/`lstm`/`gru`: sequence models for image-as-sequence settings and adapter use.
- `transformer`: token-sequence transformer classifier.
- `vit`: patch/embedding vision transformer classifier.
- `hebbian`: Hebbian feature-learning + probe classification workflows.
- `actor_critic`: policy/value style networks and RL loops.
- `reinforcement`: tabular + DQN/DDQN + MuZero-lite style algorithms.
- `markov`: markovian/statistical sequence baselines over MNIST structure.
- `trees`/`forests`: decision-tree and ensemble methods.
- `clustering`: kmeans/minibatch/GMM/agglomerative clustering workflows.
- `diffusion`: denoising process benchmarks on MNIST-like data.
- `gnn`: graph-style message-passing models and diagnostics.
- `forward_forward`: forward-forward training variant.
- `hybrid`: composed multi-expert systems + fusion network.
- `cells`: biologically inspired simulation and cell-circuit modules, including:
  - **Cell circuit** (`cell_circuit.h|.cpp`): fixed-graph MNIST cell circuit where cells process signals through a wired graph topology.
  - **Cell module** (`cell_module.h|.cpp`): individual cell dynamics unit with internal state.
  - **Cell tasks** (`cell_tasks.h|.cpp`): task-driven cell controllers for chemotaxis, distribution-shift, and damage-recovery benchmarks. Supports a baseline comparison matrix against homeostatic, RNN, GRU, LSTM, and transformer controllers.
  - **Cell tissue 2D** (`cell_tissue2d.h|.cpp`): 2D tissue morphogenesis simulation.
  - **Cell developmental** (`cell_developmental.h|.cpp`): migrating self-wiring developmental tissue that grows its own connectivity, benchmarked against a frozen fixed-graph baseline. Supports internal-teacher curriculum blending.
  - **CellEngine** (`cellengine/`): a comprehensive physics-based evolutionary cell simulator. Cells have internal state (gene expression, metabolism, signaling), exist in a physical environment with forces and boundaries, and evolve via a population-level evolutionary algorithm (ODD protocol). Features include CUDA acceleration, RL replay integration, trace I/O for visualization, and environment body physics. Detailed specification in `models/cellspec.md`.
  - **Standalone OGRE viewer** (`standalone_ogre/`): CMake-based 3D visualization sidecar project for rendering cell simulations using OGRE.

Notes:

- `models/*/configs`, `models/*/interpret`, and `models/*/tests` directories exist broadly but are currently mostly placeholders.

## 6. Hybrid Adapter and Registry

`models/hybrid/src/model_registry.*` provides adapter registration for the shared `TrainableModel` path.

Adapter-supported families:

- `mlp`, `cnn`, `rnn`, `gru`, `lstm`, `transformer`, `vit`, `hebbian`, `actor_critic`

Intentionally non-adapter families (separate runtime/training contracts):

- `markov`, `reinforcement`, `gnn`, `forward_forward`, `diffusion`, `trees`, `forests`, `clustering`, `continuous`

`models/hybrid/src/hybrid_models.*` builds `HybridSystem` that fuses logits from expert models through learned fusion layers.

## 7. Benchmark Coverage and Behavior

### 7.1 MNIST Family Benchmarks

These benchmarks generally emit common run artifacts and family-specific CSV telemetry:

- `benchmark_mlp`
- `benchmark_cnn`
- `benchmark_rnn`
- `benchmark_lstm`
- `benchmark_transformer`
- `benchmark_vit`
- `benchmark_trees`
- `benchmark_forests`
- `benchmark_clustering`
- `benchmark_hebbian`
- `benchmark_actor_critic`
- `benchmark_diffusion`
- `benchmark_gnn`
- `benchmark_forward_forward`
- `benchmark_reinforcement`
- `benchmark_markov`
- `benchmark_continuous`
- `benchmark_cell_circuit`
- `benchmark_cell_circuit_compare` (writes two run logs: no-shortcut cell circuit + frozen linear baseline)

### 7.2 Discrete/Control Benchmarks

- `benchmark_tictactoe`
  - Uses bit-bridge adapters.
  - Supports single models and `hybrid:<...>` model specs.
  - Emits tactical and deployment summaries under `model_specific/<family>/`.
- `benchmark_games`
  - Unified entrypoint for `connect_four`, `battleship`, `go`, `chess`, `cartpole`.
  - Supports control modes (including per-piece/per-owner setups in chess).

### 7.3 Cell Simulation Benchmarks

- `benchmark_cells`
  - Physics/simulation benchmark writing CSV diagnostics to `output/cells/cells_diagnostics.csv`.
  - Not primarily a `runs/<run_id>` benchmark plugin flow.
- `benchmark_leaf_cells`
  - Morphogenesis/evolution benchmark writing `output/cells/*` CSV artifacts.
  - Also not primarily a run-logger plugin flow.
- `benchmark_cell_tasks`
  - Task-driven cell controller benchmark: chemotaxis, distribution-shift, damage-recovery.
  - Supports a baseline comparison matrix (`CELL_TASK_CONTROLLERS` env var) against homeostatic, RNN, GRU, LSTM, and transformer controllers.
- `benchmark_cell_developmental`
  - Developmental tissue benchmark: migrating self-wiring cells vs frozen fixed-graph baseline.
  - Supports internal-teacher curriculum blending (`CELL_DEV_INTERNAL_TEACHER`, `CELL_DEV_INTERNAL_BLEND`, `CELL_DEV_INTERNAL_AFTER_EPOCH`).
- `benchmark_cellengine`
  - Full physics-based evolutionary cell engine benchmark.
  - Evolves populations of cells with genomes parameterizing behavior in physical environments.
  - CUDA-accelerated; writes trace/evolution outputs to `output/cellengine_*/`.

## 8. Runtime Configuration Surface

Environment variable surface is extensive (280+ unique vars found across benchmarks), organized mostly by prefix:

- `MLP_*`, `CNN_*`, `RNN_*`, `LSTM_*`, `TRANSFORMER_*`, `VIT_*`
- `HEBBIAN_*`, `FF_*`, `GNN_*`, `DIFFUSION_*`
- `ACTOR_CRITIC_*`, `A2C_*`, `PPO_*`
- `RL_*`, `MARKOV_*`
- `TREE_*`, `FORESTS_*`, `CLUSTERING_*`
- `TICTACTOE_*`, `GAME_*`
- `ONLINE_*` (continuous/hybrid runtime)
- `CELL_*`, `CELLS_*`, `LEAF_*`, `CELL_DEV_*`, `CELL_TASK_*` (cell benchmarks)
- Shared logging/runtime knobs such as `BATCH_LOG_EVERY`, `GAME_TRACE_EVERY`, `TICTACTOE_TRACE_EVERY`

Global execution knobs:

- OpenMP: `OMP_NUM_THREADS`, `OMP_PROC_BIND`, `OMP_PLACES`
- Tensor backend: `TENSOR_USE_CUDA`, `TENSOR_FORCE_CUDA`, `TENSOR_CUDA_MIN_FLOPS`
- Biology toggles: `BIO_ACTIVE` and `BIO_ENABLE_*` family plus advanced `BIO_*` controls.

## 9. Run Artifact Schema

Canonical run structure:

- `runs/<run_id>/manifest.json`
- `runs/<run_id>/learning/epoch_metrics.csv`
- Optional:
  - `learning/batch_metrics.csv`
  - `learning/class_metrics.csv`
  - `learning/confusion_matrix.csv`
- Optional deployment:
  - `deployment/inference_metrics.csv`
  - `deployment/calibration_bins.csv`
  - `deployment/system_metrics.csv`
- Optional model-specific:
  - `model_specific/<family>/*.csv`
- Runtime metadata:
  - `runtime/heartbeat.json` (for active/stale detection)

This schema is consumed by analytics scripts and the UI API discovery endpoints.

CellEngine benchmarks write separate trace/evolution artifacts under `output/cellengine_*/` rather than the `runs/` schema.

## 10. Automation Scripts

- `scripts/random_search.py`
  - Random hyperparameter search across model families.
  - Launches benchmark binaries, tags runs via `TRIAL_UUID`, scores via `learning/epoch_metrics.csv`.
  - Outputs:
    - `reports/hpo_<timestamp>/summary.csv`
    - `reports/hpo_<timestamp>/best_configs.json`
- `scripts/bio_sweep_queue.py`
  - Sequential bio-ablation sweeps (`full`, `pairwise`, `core`) across selected models.
  - Supports profiles (`quick`, `balanced`, `full`), resume mode, deterministic seed modes.
  - Outputs:
    - `queue_plan.csv`
    - `summary.csv`
    - `meta.json`
- `scripts/prune_runs_keep_best.py`
  - Keeps best run per model family based on scored metrics and tie-breakers.
  - Dry-run by default; apply deletion with `--apply`.
  - Writes decision report CSV (default `reports/prune_runs_keep_best_report.csv`).
- `scripts/smoke_test.sh`
  - Executes `make test_smoke`.
- `scripts/cells_ogre_smoke.sh`
  - OGRE visualization smoke test for the standalone viewer.
- `scripts/cells_production_check.sh`
  - Production readiness checks for the cells subsystem.

## 11. Analytics Stack

### 11.1 Python Plotting

- Main entrypoint: `analytics/plots/common/plot_bundle.py`
  - Generates common bundle + model-specific bundle.
- Common plotting modules:
  - learning curves (`plot_learning_curves.py`)
  - confusion matrix (`plot_confusion_matrix.py`)
  - calibration (`plot_calibration.py`)
  - latency/QPS (`plot_latency_qps.py`)
  - error gallery (`plot_error_gallery.py`)
- Model-specific plotting spec:
  - `analytics/plots/common/model_specific_plots.py`
  - Uses `MODEL_PLOT_SPECS` per family and skips missing/invalid CSVs safely.
- Per-family plot modules in `analytics/plots/<family>/plot_<family>.py`:
  - `mlp`, `cnn`, `rnn`, `lstm`, `transformer`, `vit`, `trees`, `forests`, `clustering`, `hebbian`, `actor_critic`, `diffusion`, `gnn`, `forward_forward`, `reinforcement`.
- Batch generation: `analytics/plots/generate_all.py`.

### 11.2 Streamlit Dashboard

- File: `analytics/dashboard/app.py`
- Pages:
  - Overview
  - Training
  - Deployment
  - Model-Specific
  - Compare Runs
  - Data/Drift
- Reads directly from `runs/` and infers effective model family for mixed benchmark/model identities (notably TicTacToe adapter runs).

## 12. UI + API Stack

### 12.1 Frontend

- React + Vite app in `ui/src/`.
- Main app component: `ui/src/App.jsx`.
- Dev proxy (`ui/vite.config.js`):
  - `/api` -> `http://localhost:8787`
  - `/reports` -> `http://localhost:8787`

Component library (`ui/src/components/`):
- Top-level panels: `TrainingPanel.jsx`, `EvaluationPanel.jsx`, `ComparisonPanel.jsx`, `TasksPanel.jsx`, `RunDetailPanel.jsx`, `GameReplayPanel.jsx`, `CellEngineReplayPanel.jsx`, `ViewAllChart.jsx`.
- Dashboard sections: `DashboardOverviewSections.jsx`, `DashboardAssetSections.jsx`, `ModelArtifactPanels.jsx`.
- 3D visualization: `Plot3D.jsx`, `Visualization3D.jsx`.
- Run detail sub-panels (`components/run-detail/`): `FamilyVisualizations.jsx`, `FamilyVisualizationsCore.jsx`, `FamilyVisualizationsAdvanced.jsx`, `FamilyVisualizationsGames.jsx`.
- Comparison sub-panels (`components/comparison/`): `BioAblationPanel.jsx`, `ComparisonTables.jsx`.
- CellEngine visualization subsystem (`components/cellengine/`):
  - Core: `core.js`, `body.js`, `uiPrimitives.jsx`, `legends.jsx`, `surfaceUtils.jsx`.
  - Views: `cellViews.jsx`, `discoveryViews.jsx`, `genomeViews.jsx`, `libraryViews.jsx`, `surfacePreviews.jsx`, `oddPanels.jsx`, `replayScenes.jsx`, `replayExplainers.jsx`.
  - State management hooks: `useCellEngineState.js`, `useCellEngineDerived.js`, `useCellEngineActions.js`.
  - Sections (`components/cellengine/sections/`): `ReplaySection.jsx`, `ReplayTopGrid.jsx`, `DiscoverSection.jsx`, `CompareSection.jsx`, `ReportsSection.jsx`.

Hooks (`ui/src/hooks/`):
- `usePolling.js` — real-time data polling.
- `useAppQueries.js` — API data fetching.
- `useAppDerivedMetrics.js` — computed dashboard metrics.
- `useAppWorkspace.js` — workspace state management.

Lib utilities (`ui/src/lib/`):
- `fetchJson.js`, `requestJson.js` — API communication.
- `appLandscapes.js`, `appViewDataset.js`, `appViewMeta.js` — view/data management.
- `dashboardShared.js` — shared styling/utilities.

### 12.2 API Server

- Express server: `ui/api/server.mjs`
- Core responsibilities:
  - Run discovery and aggregation.
  - CSV reading with pagination/downsample/tail filters.
  - Task lifecycle management (train/evaluate/start/kill/delete).
  - Bio ablation summary analysis.
  - Static report image serving (`/reports`).
  - Cache layers for run summaries, CSV parsing, family/benchmark coverage, reports, bio studies.
  - Task persistence under `runs/_tasks`.

Key endpoints:

- `GET /api/health`
- `GET /api/tasks`
- `POST /api/tasks/train`
- `POST /api/tasks/evaluate`
- `POST /api/tasks/:taskId/start`
- `POST /api/tasks/:taskId/kill`
- `DELETE /api/tasks/:taskId`
- `GET /api/bio/ablation/studies`
- `GET /api/bio/ablation`
- `GET /api/runs`
- `GET /api/families`
- `GET /api/benchmarks`
- `GET /api/runs/:runId/manifest`
- `GET /api/runs/:runId/model-specific-files`
- `GET /api/runs/:runId/csv`
- `GET /api/runs/:runId/detail`
- `GET /api/reports`

### 12.3 CellEngine Support Scripts

- `ui/scripts/cellengine_burnin.mjs` — burn-in harness for cellengine evolution runs.
- `ui/scripts/cellengine_runtime_harness.mjs` — runtime harness for cellengine execution.
- `ui/scripts/stop_dev_stack.sh` — stops the development server stack.

## 13. Testing and Quality Gates

- `tests/test_core_smoke.cpp`
  - Broad numerical and integration smoke checks (tensor ops, conv/batchnorm stability, loss gradients, optimizer behavior, replay/runtime sanity, RL/Markov/cell pieces).
- `tests/test_cells_deep.cpp`
  - Multi-phase deep checks for cell simulation behavior (determinism, membrane/neighbor grid/fluid/coupling constraints).
- `tests/test_leaf_cells.cpp`
  - 2D leaf/tissue determinism and growth sanity checks.
- `tests/test_cellengine.cpp`
  - CellEngine-specific tests: physics primitives, population evolution, trace I/O, determinism.

Test targets:

- `make test_smoke` — fast core gate.
- `make test_cells_deep` — deep cell simulation checks.
- `make test_leaf_cells` — leaf tissue checks.
- `make test_cellengine` — cellengine checks.
- `make test_cells_ogre_smoke` — OGRE visualization smoke (optional, `REQUIRE_OGRE=1` for strict).
- `make test_cells_suite` — composite target running all 5 above.
- `make cells_prod_check` — production readiness script.

## 14. Documentation Inventory

Primary operational docs:

- `docs/ARCHITECTURE.md`
- `docs/ANALYTICS.md`
- `docs/BENCHMARK_PLUGIN.md`
- `docs/HPO_RANDOM_SEARCH.md`
- `docs/BIO_SWEEP_QUEUE.md`
- `docs/HYBRID_ADAPTER_CAPABILITIES.md`

Code review and pipeline analysis docs:

- `docs/CODE_REVIEW_2026-03-03.md`
- `docs/CODE_REVIEW_FIX_TODOS_2026-03-03.md`
- `docs/FULL_REPOSITORY_PIPELINES_AND_CODE_REVIEW_2026-03-04.md`
- `docs/FULL_REPOSITORY_PIPELINES_AND_CODE_REVIEW_DETAILED_2026-03-04.md`

Cells subsystem docs (in `models/cells/`):

- `CELL_FIRST_AI_COMPLETE_GUIDE.md` — canonical long-form guide.
- `IMPLEMENTATION_STATUS.md` — implementation ledger.
- `TODO_CELL_FIRST_AI.md` — outstanding work items.
- Report MDs: `CELL_CIRCUIT_3SEED_REPORT_2026-03-04.md`, `CELL_CIRCUIT_LONG_RUN_REPORT_2026-03-04.md`.

CellEngine docs (in `models/cells/cellengine/`):

- `README.md`, `DEEP_DIVE.md`, `ODD_GOLD_STANDARD.md`, `GAP_TODO.md`, `RESULTS_2026-03-09.md`.

CellEngine specification:

- `models/cellspec.md` — full scientific and engineering specification for the evolutionary cell benchmark.

## 15. Legacy Subproject

`legacy/standalone_mlp_app/` contains:

- Independent matrix/network/layer/optimizer/loss stack
- Legacy MNIST loader and interpretability components
- Standalone `main.cpp`

This path is retained for historical reference and is not the canonical benchmark stack.

## 16. Operational Notes and Caveats

- Repository is large primarily due to `runs/` (130+ run directories), `ui/node_modules`, and generated reports/output.
- Not every benchmark uses identical artifact richness:
  - Most modern benchmark families follow `RunLogger` schema.
  - `benchmark_cells` and `benchmark_leaf_cells` are primarily output-file simulation workflows.
  - `benchmark_cellengine` writes to `output/cellengine_*/` with its own trace format.
- UI/API and plotting code include model-family inference logic to disambiguate benchmark identity vs model identity (especially for TicTacToe bit-bridge runs).
- Hybrid adapter path intentionally excludes some families with non-shared training contracts.
- CellEngine has CUDA compilation but can also run CPU-only.
- Three core subdirectories (`interpret/`, `training/`, `utils/`) exist as empty placeholders.

## 17. Quick Start Commands

Build benchmarks:

```bash
make -j"$(nproc)" benchmarks
```

Run one benchmark:

```bash
./bin/benchmark_mlp
```

Run smoke tests:

```bash
make test_smoke
```

Run full cells test suite:

```bash
make test_cells_suite
```

Cells production readiness:

```bash
make cells_prod_check
```

Launch Streamlit analytics:

```bash
streamlit run analytics/dashboard/app.py
```

Launch UI + API:

```bash
cd ui
npm install
npm run dev
```
