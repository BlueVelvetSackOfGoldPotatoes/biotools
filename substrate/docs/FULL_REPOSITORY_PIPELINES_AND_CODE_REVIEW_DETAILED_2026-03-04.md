# Full Repository Pipeline and Code Review (Detailed)
Date: 2026-03-04

## Scope
This document reviews the maintained repository code and docs in depth, with two goals:
1. Explain all runtime pipelines end-to-end (build, training, evaluation, bio runtime, continuous runtime, optimization, analytics, dashboard).
2. Review each source/config/documentation file with practical engineering critique (what it does, how it fits, risks, and improvements).

Reviewed inventory: 168 maintainable files (excluding generated runs/reports/build artifacts and vendored `ui/node_modules`).

## System Pipeline Overview

### 1) Build and Binary Pipeline
- The build is controlled by `Makefile` and compiles a shared core + model families + benchmark executables.
- CUDA is optional (`USE_CUDA=auto|0|1`), OpenMP is enabled for CPU parallel loops.
- Outputs are benchmark binaries under `bin/` (e.g. `benchmark_mlp`, `benchmark_reinforcement`, `benchmark_games`).
- Key risk: benchmark targets are repetitive in the make graph; drift risk grows as new benchmarks are added.

### 2) Training and Evaluation Pipeline (Supervised Families)
1. Benchmark executable parses env vars/flags and chooses architecture variant.
2. Data is loaded via `MNISTLoader` from IDX files.
3. Model executes mini-batch train loop (`forward -> loss -> backward -> optimizer step`).
4. Metrics are aggregated (loss, accuracy, grad norms, throughput).
5. Artifacts are written through `RunLogger` into `runs/<run_id>/`:
   - `epoch_metrics.csv`
   - `batch_metrics.csv`
   - `model_specific.csv`
   - optional confusion/calibration/error outputs
   - `manifest.json`
6. Evaluation path reads held-out split and writes deployment-oriented metrics.

### 3) Control/Game Benchmark Pipeline
- `DiscreteEnv` is the benchmark abstraction for game/control tasks.
- `BitCodec` and bridge models convert arbitrary environment state/action spaces to model-compatible bit vectors.
- `DiscreteQRuntime` owns interaction loop (episode rollout, policy selection, replay/update, logging).
- `benchmark_tictactoe.cpp` runs one environment; `benchmark_games.cpp` generalizes to multiple environments (connect four, battleship, go-lite, chess-lite, cartpole).
- This is the main mechanism that decouples benchmark identity from model implementation.

### 4) Bio Addon Runtime Pipeline
- The bio layer is integrated at runtime through hooks, not by rewriting every model architecture.
- Hook order:
  1. `before_forward`
  2. `record_linear_activity` / transport preparation
  3. `after_backward`
  4. `after_optimizer_step`
  5. epoch snapshots and bio metrics logging
- Implemented effects include:
  - structural masking/rewiring
  - homeostasis scaling
  - three-factor plasticity update channel
  - delay/myelin influence
  - bioelectric state trackers
  - E/I sign constraints
- Important reality check: behavior is meaningful only when bio runtime is active (`BIO_ACTIVE=1`); default mode is observational/no-op for safety.

### 5) Continuous Learning Pipeline (Always-On)
- `ContinuousRuntime` supports active and sleep phases:
  - active: task-driven updates
  - sleep: replay/noise rehearsal at lower budget
- Tracks cycle-level metrics like active efficiency, sleep consolidation, and target-accuracy stop logic.
- This provides practical "never fully shut down" behavior in current framework constraints.

### 6) Hyperparameter Optimization and Sweep Pipeline
- `scripts/random_search.py` samples architecture-specific hyperparameters, executes trials, scores runs, and saves best configs.
- `scripts/bio_sweep_queue.py` schedules bio-feature combinatorics (full, pairwise, core, custom), supports resume and queue ownership.
- `scripts/prune_runs_keep_best.py` curates run storage by keeping best runs per family.

### 7) Analytics and Visualization Pipeline
- `analytics/plots/common/*` provides reusable plotting primitives against standard run CSVs.
- Family scripts (`analytics/plots/<family>/plot_*.py`) are thin wrappers around shared plotting bundle.
- Streamlit dashboard (`analytics/dashboard/app.py`) offers quick analysis mode independent from React UI.

### 8) Live React Dashboard Pipeline
- `ui/api/server.mjs` scans run folders, parses metrics, manages tasks (train/eval/start/kill/delete), and exposes benchmark/model coverage endpoints.
- `ui/src/App.jsx` renders control-room UX with tabs for live runs, training, evaluation, model-specific charts, comparison views, bio visuals, and game replay.
- Polling hooks (`usePolling.js`) refresh live charts and tables.
- Main technical debt hotspot: frontend and API monolith size.

## Cross-Cutting Critical Findings

### Architecture and Maintainability
- `ui/src/App.jsx` and `ui/api/server.mjs` centralize too much logic; this is the biggest regression and onboarding risk.
- Benchmark executables contain repeated training/evaluation boilerplate; shared harness abstraction would reduce defects.
- Some capability boundaries are implicit (e.g., which models can act through bit-bridge/hybrid adapters). These should always be explicit in API metadata and UI.

### Correctness and Reliability
- Core guardrails improved significantly (shape checks, dataloader bounds checks, run-id collision hardening).
- Bio runtime now performs nontrivial updates, but interpretability of its effects still depends on disciplined ablation logging.
- Metrics semantics differ by benchmark type (classification accuracy vs tactical game metrics); UI must avoid collapsing them into a single generic meaning.

### Performance
- CUDA acceleration currently focuses on GEMM and still pays host-device transfer overhead per call.
- Backend API reads/parses CSVs synchronously under polling pressure; this can stall at scale.
- Many loops are parallelized with OpenMP, but system-wide throughput is still gated by I/O and orchestration overhead in long experiments.

### Recommended Refactor Priorities
1. Break UI and API monoliths into modules with typed contracts and shared selectors/caches.
2. Extract shared C++ benchmark harnesses (supervised loop, control loop, logging adapters).
3. Add async/memoized run indexing in API (mtime + incremental parsing).
4. Expand capability manifests so unsupported model-benchmark pairs fail early and clearly.
5. Continue GPU backend expansion beyond GEMM to reduce transfer and improve end-to-end speed.

---

## File-by-File Detailed Review

Notes for this section:
- "Responsibility" explains why the file exists.
- "Internals" highlights key APIs/classes/logic.
- "Critical review" focuses on risks, quality, and practical improvements.

### Root and Build

#### `.gitignore`
- Responsibility: Keeps generated artifacts, build outputs, run folders, and cache files out of versioned source scope.
- Internals: Includes ignores for C++ build directories, Python cache, and runtime outputs.
- Critical review: Mostly correct. Ensure new generated directories are added immediately to avoid accidental large commits.

#### `Makefile`
- Responsibility: Canonical build and orchestration entrypoint for all C++ binaries and test targets.
- Internals: Defines compile/link flags, object targets, benchmark binaries, CUDA/OpenMP configuration, and smoke test targets.
- Critical review: Functional and explicit. Repetitive benchmark target definitions should be data-driven (template macro/list) to prevent divergence.

#### `README.md`
- Responsibility: Primary operator documentation (build, run, benchmarking, UI, optimization, bio toggles).
- Internals: Explains commands and environment variables used across scripts and executables.
- Critical review: Rich but dense. Split into quickstart + advanced architecture references to reduce cognitive load.

#### `build_all.sh`
- Responsibility: Convenience wrapper to build benchmark set.
- Internals: Runs `make benchmarks`.
- Critical review: Simple and valid; could forward `-j` and include more explicit error context for missing toolchains.

### Core Runtime: Tensor, NN, Optimization, Data, IO

#### `core/tensor/tensor.h`
- Responsibility: Defines the central dense `Tensor` abstraction used by all models.
- Internals: Constructors, shape/storage metadata, arithmetic ops, reductions, reshaping, and factory utilities.
- Critical review: Good single-source tensor API. Add more compile-time and runtime invariants for risky reshapes and broadcasting paths.

#### `core/tensor/tensor.cpp`
- Responsibility: Implements tensor operations and math kernels.
- Internals: CPU kernels, OpenMP loops, matrix multiplication dispatch to CUDA backend when eligible.
- Critical review: Correct and stable. Performance remains limited by frequent allocations and host/device copies; fusion/memory-pooling are next-level optimizations.

#### `core/tensor/cuda_backend.h`
- Responsibility: Declares CUDA GEMM helper interface.
- Internals: Runtime selection helpers and host-facing matmul entrypoint.
- Critical review: Clean minimal API, but narrow scope prevents broader GPU acceleration without API extension.

#### `core/tensor/cuda_backend.cpp`
- Responsibility: Provides cuBLAS-backed matrix multiplication path.
- Internals: Device buffer reuse, size-threshold gating, fallback behavior when CUDA unavailable.
- Critical review: Good first stage GPU support. End-to-end throughput still constrained because only GEMM is offloaded.

#### `core/nn/module.h`
- Responsibility: Defines module/layer interfaces and sequential composition primitives.
- Internals: Layer base class, common modules, integration hooks for bio runtime in linear pathways.
- Critical review: Strong core abstraction. Growing feature set suggests splitting modules into smaller headers by concern.

#### `core/nn/module.cpp`
- Responsibility: Implements forward/backward for modules and training plumbing.
- Internals: Layer logic, gradient propagation, and bio-affine callback points.
- Critical review: Core correctness is strong, but complexity is increasing; targeted unit tests per layer would reduce regression risk.

#### `core/optim/optimizer.h`
- Responsibility: Optimizer and scheduler interfaces for both sequential models and adapter/vector models.
- Internals: SGD, Adam, RMSProp, LR scheduler interfaces, vector-step API.
- Critical review: Useful dual API design. Scheduler usage in benchmarks is still inconsistent.

#### `core/optim/optimizer.cpp`
- Responsibility: Optimizer update implementations.
- Internals: Adam/RMSProp/SGD steps, grad clipping path, optimizer state allocation, parameter update math.
- Critical review: Solid implementation and improved safety. Ensure clipping semantics remain consistent across all model update paths.

#### `core/losses/losses.h`
- Responsibility: Declares loss functions used across supervised and some RL settings.
- Internals: Cross entropy, MSE, BCE, NLL, KL, Huber interfaces.
- Critical review: API breadth is good. Keep argument conventions (logits vs probs) explicitly documented to prevent misuse.

#### `core/losses/losses.cpp`
- Responsibility: Loss computation and gradients.
- Internals: Adds shape guards and numerical protection in loss/grad calculations.
- Critical review: Strong safety improvements. Continue expanding tests for edge cases (empty tensors, extreme logits).

#### `core/metrics/metrics.h`
- Responsibility: Declares metric APIs shared by benchmarks and analytics exports.
- Internals: Accuracy, confusion matrix, precision/recall/F1 helpers.
- Critical review: Interface is straightforward; metric semantics for non-classification tasks should remain separate.

#### `core/metrics/metrics.cpp`
- Responsibility: Implements metric calculations.
- Internals: Classification metrics, size checks, and aggregation helpers.
- Critical review: Better guarded now. Some functions still assume specific target formatting; add stricter validation and error messages.

#### `core/data/dataloader.h`
- Responsibility: Data loading and mini-batch iteration contract (MNIST-focused).
- Internals: `MNISTLoader`, batch sampling/iteration config.
- Critical review: Practical and simple. A benchmark-agnostic dataset API would help long-term multi-benchmark growth.

#### `core/data/dataloader.cpp`
- Responsibility: IDX parsing and batch materialization.
- Internals: Header parsing, payload checks, normalized tensor creation, bounded batch iteration.
- Critical review: Safety checks are materially better. For scale, add memory-mapped mode and background prefetch.

#### `core/io/run_logger.h`
- Responsibility: Unified artifact writing API for runs.
- Internals: Run id management, heartbeat, metrics CSV appenders, manifest writing.
- Critical review: Good central contract; file naming and schema discipline are major strengths.

#### `core/io/run_logger.cpp`
- Responsibility: Implements durable run artifact writing.
- Internals: Run-id collision hardening, JSON escaping, trial/job metadata support, heartbeat updates.
- Critical review: Reliability improved significantly. Could benefit from batched writes and buffered flush strategy for high-frequency logging.

### Core Runtime: Benchmark Abstractions

#### `core/benchmark/discrete_env.h`
- Responsibility: Environment interface for control/game benchmarks.
- Internals: Step/reset/action/state/reward contracts.
- Critical review: Clean abstraction and key enabler for benchmark-model separation.

#### `core/benchmark/bit_codec.h`
- Responsibility: Encodes numeric/categorical environment state into bit vectors.
- Internals: Bit packing and decode helpers used by bridge adapters.
- Critical review: Good for model-agnostic interoperability. Must keep precision/quantization assumptions explicit per environment.

#### `core/benchmark/bit_codec.cpp`
- Responsibility: Implements bit-level feature transform utilities.
- Internals: Packing/unpacking routines and shape conversion helpers.
- Critical review: Utility is practical. Needs explicit benchmarks for codec overhead in high-throughput control runs.

#### `core/benchmark/bit_bridge_model.h`
- Responsibility: Wraps generic models for control tasks via bit vectors.
- Internals: Adapter interface for predict/update on encoded states and actions.
- Critical review: Strong decoupling mechanism; capability mismatches should always be surfaced as explicit errors.

#### `core/benchmark/bit_bridge_model.cpp`
- Responsibility: Bridge implementation from environment state to model input/output.
- Internals: Encoding glue, action scoring, update path integration.
- Critical review: Useful and pragmatic. Keep conversion costs and dimensional assumptions visible in logs.

#### `core/benchmark/piecewise_bit_bridge_model.h`
- Responsibility: Extends bridge model for action-owner/piece-specialist routing.
- Internals: Per-owner submodel selection contract.
- Critical review: Important for chess-like benchmarks and modular control.

#### `core/benchmark/piecewise_bit_bridge_model.cpp`
- Responsibility: Implements per-piece/per-action-owner dispatch.
- Internals: Routes state/action to correct submodel and merges outputs.
- Critical review: High-value flexibility; routing correctness must stay heavily tested.

#### `core/benchmark/discrete_q_runtime.h`
- Responsibility: Generic discrete control runtime interface.
- Internals: Episode scheduler, policy/evaluation hooks, metrics collection.
- Critical review: Good reusable runtime layer.

#### `core/benchmark/discrete_q_runtime.cpp`
- Responsibility: Executes training/eval loops for discrete environments.
- Internals: Rollout logic, epsilon/exploration policy integration, per-episode logging.
- Critical review: Central for control benchmarks. Consider extracting reusable logger adapters to reduce per-benchmark duplication.

#### `core/benchmark/tictactoe_env.h`
- Responsibility: TicTacToe environment contract and tactical metric exposure.
- Internals: Board state encoding, legal move generation, terminal checks.
- Critical review: Clear and focused; strong benchmark for tactical metric instrumentation.

#### `core/benchmark/tictactoe_env.cpp`
- Responsibility: TicTacToe game mechanics.
- Internals: Move application, winner detection, tactical opportunities (missed wins/blocks) tracking.
- Critical review: Good metric-rich environment. Continue validating edge cases around illegal action handling.

#### `core/benchmark/game_envs.h`
- Responsibility: Declares additional environments (connect four, battleship, go-lite, chess-lite, cartpole).
- Internals: Shared environment factory and metadata.
- Critical review: Broad benchmark surface. Complexity warrants stronger per-env unit tests.

#### `core/benchmark/game_envs.cpp`
- Responsibility: Implements game/control environments used in generalized benchmark.
- Internals: State transitions, reward shaping, action ownership metadata for piecewise routing.
- Critical review: Valuable but complex. Needs disciplined test expansion to keep rules/reward semantics stable.

### Core Runtime: Bio and Continuous

#### `core/bio/bio_affine.h`
- Responsibility: Lightweight integration helpers between linear layers and bio runtime.
- Internals: Hooks to apply transport modifiers and record activity around affine operations.
- Critical review: Good low-friction integration point.

#### `core/bio/bio_runtime.h`
- Responsibility: Defines bio runtime configuration, state, and hook API.
- Internals: Config toggles for homeostasis, myelin/delay, rewiring, three-factor updates, bioelectric tracking.
- Critical review: Coherent API. Config explosion risk is increasing; introduce grouped profiles and validation constraints.

#### `core/bio/bio_runtime.cpp`
- Responsibility: Implements bio dynamics, mask updates, activity stats, and bio telemetry output.
- Internals: Epoch/batch hook behavior, feature-specific update channels, CSV exports for bio diagnostics.
- Critical review: Meaningful implementation exists now. Some dynamics remain simplified proxies; prioritize calibration tests and ablation sanity checks.

#### `core/online/trainable_model.h`
- Responsibility: Unifying model interface for continuous runtime and control adapters.
- Internals: Predict/update API, optional action-owner grouping support.
- Critical review: Critical abstraction for interoperability; keep interface minimal and stable.

#### `core/online/continuous_runtime.h`
- Responsibility: Defines always-on runtime scheduler and metric structures.
- Internals: Config and contract for active/sleep cycles.
- Critical review: Good formalization for continuous learning loop.

#### `core/online/continuous_runtime.cpp`
- Responsibility: Executes continuous training cycles.
- Internals: Active data pass, sleep replay/noise pass, per-cycle logging and stop criteria.
- Critical review: Useful realization of always-on concept; stronger replay policy controls could improve realism.

### Benchmarks

#### `benchmarks/logging_utils.h`
- Responsibility: Shared logging helpers for benchmark executables.
- Internals: Common CSV row helpers, run metadata utility glue.
- Critical review: Useful utility, but further centralization (single benchmark harness) would reduce duplicate metric logic.

#### `benchmarks/hpo_utils.h`
- Responsibility: Shared hyperparameter parsing/sampling helpers used by benchmark executables.
- Internals: Env parsing, optimization option selection, score formatting support.
- Critical review: Good utility layer. Should evolve toward one canonical schema for all benchmark knobs.

#### `benchmarks/benchmark_mlp.cpp`
- Responsibility: MLP MNIST benchmark with train/eval and rich model-specific telemetry.
- Internals: Builds MLP variants, runs mini-batch loop, writes neuron/layer sparsity/pixel importance diagnostics.
- Critical review: One of the best-instrumented benchmarks. Boilerplate could be extracted to shared supervised runner.

#### `benchmarks/benchmark_cnn.cpp`
- Responsibility: CNN benchmark (simple, LeNet, ResNet-like variants).
- Internals: Conv architecture selection, optimizer/loss options, filter/feature-map telemetry.
- Critical review: Strong practical implementation. Similar training loop duplicated across supervised benchmarks.

#### `benchmarks/benchmark_rnn.cpp`
- Responsibility: RNN/GRU/BiRNN MNIST sequence benchmark.
- Internals: Sequence formatting from images, BPTT training loop, timestep gradient diagnostics.
- Critical review: Useful recurrent diagnostics; shares repeated loop code with LSTM benchmark.

#### `benchmarks/benchmark_lstm.cpp`
- Responsibility: LSTM/BiLSTM benchmark for sequentialized MNIST.
- Internals: Gate statistics, cell-state diagnostics, configurable optimization/loss.
- Critical review: Good architecture-specific observability. Candidate for consolidation with `benchmark_rnn.cpp` under common sequence harness.

#### `benchmarks/benchmark_transformer.cpp`
- Responsibility: Transformer MNIST benchmark.
- Internals: Configurable depth/heads/width/dropout, attention entropy/head specialization logs.
- Critical review: High-value instrumentation. Efficiency guardrails needed for large attention configs.

#### `benchmarks/benchmark_vit.cpp`
- Responsibility: Vision Transformer MNIST benchmark.
- Internals: Patch settings, transformer encoder configuration, ViT-specific diagnostics.
- Critical review: Good architecture support; memory/runtime costs should be surfaced clearly in UI comparisons.

#### `benchmarks/benchmark_hebbian.cpp`
- Responsibility: Hebbian/Hopfield benchmark.
- Internals: Local rule updates, probe training, associative memory metrics and energy diagnostics.
- Critical review: Important non-backprop path. Keep unsupervised and probe-supervised phases clearly separated in reporting.

#### `benchmarks/benchmark_forward_forward.cpp`
- Responsibility: Forward-Forward benchmark.
- Internals: Goodness-based local objective, positive/negative passes, margin diagnostics.
- Critical review: Good experimental coverage. Susceptible to hyperparameter instability; keep robust defaults.

#### `benchmarks/benchmark_trees.cpp`
- Responsibility: Decision tree benchmark on MNIST.
- Internals: Depth/split controls and tree diagnostics.
- Critical review: Useful classical baseline; include calibration notes for probability interpretation.

#### `benchmarks/benchmark_forests.cpp`
- Responsibility: Forest ensemble benchmark (RandomForest/ExtraTrees).
- Internals: OOB metrics and ensemble agreement diagnostics.
- Critical review: Good baseline breadth; could parallelize tree growth more aggressively.

#### `benchmarks/benchmark_clustering.cpp`
- Responsibility: Unsupervised clustering benchmark suite.
- Internals: KMeans/GMM/Agglomerative/DBSCAN-style runs with cluster quality metrics.
- Critical review: Strong non-supervised coverage. Ensure dashboards label these metrics separately from classifier accuracy.

#### `benchmarks/benchmark_gnn.cpp`
- Responsibility: Graph benchmark mapping images to graph structures.
- Internals: Grid graph creation, GCN/GAT training, graph-level readout metrics.
- Critical review: Valuable graph pathway. Dense graph ops may become expensive at higher resolutions.

#### `benchmarks/benchmark_diffusion.cpp`
- Responsibility: Diffusion model benchmark.
- Internals: Noise schedule training loop, reverse-process/sample telemetry.
- Critical review: Useful but intentionally compact implementation; should be framed as research baseline, not production image diffusion stack.

#### `benchmarks/benchmark_actor_critic.cpp`
- Responsibility: Actor-critic benchmark in contextual MNIST framing.
- Internals: A2C/PPO-style training, policy/value metrics.
- Critical review: Good RL comparison harness; environment is synthetic contextual task, so generalization claims should remain bounded.

#### `benchmarks/benchmark_reinforcement.cpp`
- Responsibility: Multi-algorithm RL benchmark suite including MuZero-lite.
- Internals: Tabular Q/SARSA and value-based deep RL flows with shared telemetry outputs.
- Critical review: Strong experimental breadth. Must keep explicit note that MuZero variant is simplified.

#### `benchmarks/benchmark_markov.cpp`
- Responsibility: Markov-chain classifier benchmark.
- Internals: Symbolization of inputs, partial-fit schedule, context/emission diagnostics.
- Critical review: Excellent probabilistic baseline with interpretable outputs. State space growth can be expensive.

#### `benchmarks/benchmark_tictactoe.cpp`
- Responsibility: Benchmark-agnostic TicTacToe evaluation through bit-bridge and discrete runtime.
- Internals: Tactical metrics (wins/losses/draws/missed wins/missed blocks), replay traces, deployment stats.
- Critical review: Strong benchmark-model separation example. Shares logic with `benchmark_games.cpp`; should refactor common control harness.

#### `benchmarks/benchmark_games.cpp`
- Responsibility: Generalized multi-game benchmark runner.
- Internals: Environment factory selection, full-side vs per-piece specialist control, logging across environments.
- Critical review: High-value extensibility file. Large and complex; break into per-env modules + shared runtime wrappers.

#### `benchmarks/benchmark_continuous.cpp`
- Responsibility: Always-on active/sleep benchmark runner for single/hybrid/suite models.
- Internals: Cycle scheduling, target stop logic, replay/noise sleep configuration and metrics.
- Critical review: Good realization of continuous-learning objective within current architecture.

### Models

#### `models/cnn/src/cnn_models.h`
- Responsibility: CNN family interfaces and builders.
- Internals: Declares simple/LeNet/ResNet-like classifiers and related helper blocks.
- Critical review: Clear public contract; keep architecture options synchronized with benchmark/UI enumerations.

#### `models/cnn/src/cnn_models.cpp`
- Responsibility: CNN family implementation.
- Internals: Conv blocks, pooling/readout, residual-style composition, forward/backward wiring.
- Critical review: Functional and readable. Further modularization of architecture assembly would improve extensibility.

#### `models/rnn/src/rnn_models.h`
- Responsibility: RNN/GRU interface and state definitions.
- Internals: Declares recurrent cells, sequence classifier APIs, and training-facing methods.
- Critical review: Good interface but exposes complexity; maintain strict tests around state shape/time semantics.

#### `models/rnn/src/rnn_models.cpp`
- Responsibility: RNN/GRU implementations with explicit unroll.
- Internals: Forward through timesteps, BPTT-style backward logic, recurrent parameter updates.
- Critical review: Manual recurrent backprop is intricate and regression-prone; numerical gradient checks should remain part of smoke coverage.

#### `models/lstm/src/lstm_models.h`
- Responsibility: LSTM/BiLSTM model interfaces.
- Internals: Cell/gate state APIs and classifier wrappers.
- Critical review: Interface quality is good; keep gate/cell telemetry contracts stable for dashboard consumers.

#### `models/lstm/src/lstm_models.cpp`
- Responsibility: LSTM implementations.
- Internals: Gate computations, cell-state transitions, bidirectional pass, gradients.
- Critical review: Good transparency and instrumentation. Complexity suggests extracting reusable gate kernels/helpers.

#### `models/transformer/src/transformer_models.h`
- Responsibility: Transformer model interfaces.
- Internals: Declares attention blocks, encoder stack, and classifier head contracts.
- Critical review: Strong API; future sparse-attention variants may need separate interfaces to avoid option overload.

#### `models/transformer/src/transformer_models.cpp`
- Responsibility: Transformer implementation.
- Internals: Multi-head attention, FFN, norm/dropout, positional encoding, training plumbing.
- Critical review: Feature-rich and valuable; computationally heavy, so profiling and caching optimizations are important.

#### `models/vit/src/vit_models.h`
- Responsibility: Vision Transformer interfaces.
- Internals: Patch embedding settings, encoder config, classifier contracts.
- Critical review: Good separation between config and implementation.

#### `models/vit/src/vit_models.cpp`
- Responsibility: ViT implementation.
- Internals: Patchify/project, transformer encoder, CLS-token readout and training support.
- Critical review: Solid implementation for MNIST-scale vision; memory scaling should be explicitly surfaced in benchmark stats.

#### `models/hebbian/src/hebbian_models.h`
- Responsibility: Hebbian/Hopfield interfaces.
- Internals: Declares local rule updates, associative memory APIs, diagnostics structures.
- Critical review: Clear non-backprop model boundary.

#### `models/hebbian/src/hebbian_models.cpp`
- Responsibility: Hebbian and Hopfield implementations.
- Internals: Weight updates by local rules, memory energy behaviors, readout/probe logic.
- Critical review: Good differentiator. Maintain careful metric labeling to distinguish representation quality vs classifier outcome.

#### `models/forward_forward/src/ff_models.h`
- Responsibility: Forward-Forward model interfaces.
- Internals: Layer goodness APIs, training/prediction contracts.
- Critical review: Focused interface and useful alternative learning path.

#### `models/forward_forward/src/ff_models.cpp`
- Responsibility: Forward-Forward implementation.
- Internals: Positive/negative pass goodness computation and local updates.
- Critical review: Good implementation; very sensitive to thresholds, so defaults and config docs are critical.

#### `models/trees/src/tree_models.h`
- Responsibility: Decision tree interfaces.
- Internals: Node structure, split criteria configuration, classifier/regressor APIs.
- Critical review: Interface is coherent; consider separating classifier and regressor APIs to reduce branching complexity.

#### `models/trees/src/tree_models.cpp`
- Responsibility: CART-style tree implementation.
- Internals: Split search, impurity metrics, recursive growth/pruning, feature importance.
- Critical review: Mature baseline implementation; add edge-case tests for small/degenerate datasets.

#### `models/forests/src/forest_models.h`
- Responsibility: Ensemble-over-trees interfaces.
- Internals: RandomForest/ExtraTrees constructors and prediction/diagnostic contracts.
- Critical review: Practical API; parallel execution strategy should be more explicit in config.

#### `models/forests/src/forest_models.cpp`
- Responsibility: Forest implementation.
- Internals: Bootstrap sampling, per-tree training, aggregation, OOB/permutation stats.
- Critical review: Good classic baseline. Compute can be heavy; dynamic thread partitioning can improve utilization.

#### `models/clustering/src/clustering_models.h`
- Responsibility: Clustering algorithm interfaces.
- Internals: KMeans, GMM, Agglomerative, DBSCAN, Spectral contracts and result structs.
- Critical review: Broad and useful interface; keep algorithm-specific options from leaking into generic contracts.

#### `models/clustering/src/clustering_models.cpp`
- Responsibility: Clustering algorithms implementation.
- Internals: Multiple clustering methods with shared utilities and diagnostics.
- Critical review: Valuable breadth, but large file complexity suggests splitting by algorithm family.

#### `models/gnn/src/gnn_models.h`
- Responsibility: GNN interfaces.
- Internals: GCN/GAT layers, graph classifier contracts, graph conversion helpers.
- Critical review: Good graph-focused abstraction for benchmark integration.

#### `models/gnn/src/gnn_models.cpp`
- Responsibility: GNN implementations.
- Internals: Message passing layers, attention, graph-level pooling/classification.
- Critical review: Strong graph experimentation path; sparse kernel support would improve scaling.

#### `models/diffusion/src/diffusion_models.h`
- Responsibility: Diffusion model interfaces.
- Internals: Schedules, denoiser interfaces, train/sample APIs.
- Critical review: Useful compact contract, should remain explicit that it targets benchmark-scale diffusion.

#### `models/diffusion/src/diffusion_models.cpp`
- Responsibility: Diffusion implementation.
- Internals: Forward noising, denoising objective, reverse sampling trajectory logic.
- Critical review: Good minimal DDPM-style path; add profiling hooks for sampling-step cost.

#### `models/actor_critic/src/ac_models.h`
- Responsibility: Actor-critic model interfaces.
- Internals: Policy/value network contracts and update APIs.
- Critical review: Clean RL API and good fit for benchmark runner integration.

#### `models/actor_critic/src/ac_models.cpp`
- Responsibility: Actor-critic implementations.
- Internals: Advantage/value updates, policy updates, entropy/value telemetry.
- Critical review: Good baseline RL implementations. Stability options (entropy schedules, normalization) can be expanded.

#### `models/reinforcement/src/rl_models.h`
- Responsibility: Reinforcement family interfaces (tabular + deep + planner-lite).
- Internals: Replay buffer, Q-learning/SARSA contracts, DQN/DDQN, MuZero-lite API.
- Critical review: Rich interface set; keep naming explicit about simplified planner scope.

#### `models/reinforcement/src/rl_models.cpp`
- Responsibility: RL family implementations.
- Internals: Tabular updates, replay training, target-network logic, planning-lite path.
- Critical review: High experimental value. Complexity warrants more targeted algorithm-specific tests.

#### `models/markov/src/markov_models.h`
- Responsibility: Markov classifier interface.
- Internals: Context order/binning/smoothing config and prediction/training contracts.
- Critical review: Compact and interpretable interface.

#### `models/markov/src/markov_models.cpp`
- Responsibility: Markov chain classifier implementation.
- Internals: Tokenization/bucketing, transition/emission counting, posterior scoring and diagnostics.
- Critical review: Strong probabilistic baseline; context explosion should be actively guarded.

#### `models/hybrid/src/hybrid_models.h`
- Responsibility: Hybrid composition interfaces.
- Internals: Multi-expert composition contracts and fusion controls.
- Critical review: Core interoperability mechanism; boundary conditions for unsupported experts must remain explicit.

#### `models/hybrid/src/hybrid_models.cpp`
- Responsibility: Hybrid composition implementation.
- Internals: Expert routing/combination and training-time update delegation.
- Critical review: Useful and pragmatic. As complexity grows, explicit routing tests are needed.

#### `models/hybrid/src/model_registry.h`
- Responsibility: Registry interfaces for model adapter creation.
- Internals: Lookup/creation APIs by family string.
- Critical review: Keeps benchmark runtime decoupled from concrete model classes.

#### `models/hybrid/src/model_registry.cpp`
- Responsibility: Registry implementation and adapter mapping.
- Internals: Family-to-adapter creation, capability filtering.
- Critical review: Good central capability map; update discipline is required whenever adding families.

### Analytics and Plotting

#### `analytics/__init__.py`
- Responsibility: Package marker.
- Internals: No runtime logic.
- Critical review: No issues.

#### `analytics/dashboard/__init__.py`
- Responsibility: Package marker for dashboard module.
- Internals: No runtime logic.
- Critical review: No issues.

#### `analytics/dashboard/app.py`
- Responsibility: Streamlit dashboard for run exploration and plot inspection.
- Internals: Loads run directories, parses CSV metrics, renders summary, comparison, and model-specific views.
- Critical review: Useful secondary analysis UI; overlapping responsibilities with React dashboard create duplicate maintenance burden.

#### `analytics/requirements.txt`
- Responsibility: Python analytics dependency lock-lite.
- Internals: Minimal package list (`pandas`, `numpy`, `matplotlib`, `streamlit`).
- Critical review: Good minimal footprint, but consider pinning tighter versions for reproducibility.

#### `analytics/plots/__init__.py`
- Responsibility: Plot package marker.
- Internals: None.
- Critical review: No issues.

#### `analytics/plots/generate_all.py`
- Responsibility: Batch plot generator across known families.
- Internals: Iterates family scripts and dispatches plot generation.
- Critical review: Practical script; static family list can drift from actual supported families.

#### `analytics/plots/common/__init__.py`
- Responsibility: Common plotting package marker.
- Internals: None.
- Critical review: No issues.

#### `analytics/plots/common/io_utils.py`
- Responsibility: Shared file I/O helper for plotting.
- Internals: Safe CSV loading, directory creation, figure persistence utilities.
- Critical review: Good utility cohesion and defensive checks.

#### `analytics/plots/common/plot_learning_curves.py`
- Responsibility: CLI wrapper for learning-curve chart generation.
- Internals: Reads run path and calls common renderer.
- Critical review: Thin and stable.

#### `analytics/plots/common/plot_confusion_matrix.py`
- Responsibility: CLI wrapper for confusion matrix plotting.
- Internals: Delegates to shared common plot code.
- Critical review: Thin and stable.

#### `analytics/plots/common/plot_calibration.py`
- Responsibility: CLI wrapper for calibration plotting.
- Internals: Delegates to shared renderer.
- Critical review: Thin and stable.

#### `analytics/plots/common/plot_latency_qps.py`
- Responsibility: CLI wrapper for latency/throughput charts.
- Internals: Delegates to shared renderer.
- Critical review: Thin and stable.

#### `analytics/plots/common/plot_error_gallery.py`
- Responsibility: CLI wrapper for misclassification gallery.
- Internals: Delegates to shared renderer.
- Critical review: Thin and stable.

#### `analytics/plots/common/common_plots.py`
- Responsibility: Implements common chart generation across families.
- Internals: Parses standard run CSVs and emits learning/quality/deployment plots.
- Critical review: Important shared module; add robust downsampling for very large runs.

#### `analytics/plots/common/model_specific_plots.py`
- Responsibility: Defines model-specific plot spec registry and generic renderers.
- Internals: Schema-driven chart generation per family and bio-specific overlays.
- Critical review: Strong abstraction. Better warnings for missing expected columns would improve diagnosability.

#### `analytics/plots/common/plot_bundle.py`
- Responsibility: CLI wrapper for generating full plot bundle.
- Internals: Dispatches common + model-specific rendering.
- Critical review: Thin and practical.

#### `analytics/plots/common/bundle.py`
- Responsibility: High-level plot bundling orchestration.
- Internals: Family inference from manifest and dispatch pipeline.
- Critical review: Useful orchestration; family detection must stay aligned with benchmark naming.

#### `analytics/plots/actor_critic/plot_actor_critic.py`
- Responsibility: Actor-critic family plot entrypoint.
- Internals: Calls shared bundle for actor_critic family.
- Critical review: Simple wrapper; low risk.

#### `analytics/plots/clustering/plot_clustering.py`
- Responsibility: Clustering family plot entrypoint.
- Internals: Calls shared bundle for clustering family.
- Critical review: Simple wrapper; low risk.

#### `analytics/plots/cnn/plot_cnn.py`
- Responsibility: CNN family plot entrypoint.
- Internals: Calls shared bundle for cnn family.
- Critical review: Simple wrapper; low risk.

#### `analytics/plots/diffusion/plot_diffusion.py`
- Responsibility: Diffusion family plot entrypoint.
- Internals: Calls shared bundle for diffusion family.
- Critical review: Simple wrapper; low risk.

#### `analytics/plots/forests/plot_forests.py`
- Responsibility: Forests family plot entrypoint.
- Internals: Calls shared bundle for forests family.
- Critical review: Simple wrapper; low risk.

#### `analytics/plots/forward_forward/plot_forward_forward.py`
- Responsibility: Forward-Forward family plot entrypoint.
- Internals: Calls shared bundle for forward_forward family.
- Critical review: Simple wrapper; low risk.

#### `analytics/plots/gnn/plot_gnn.py`
- Responsibility: GNN family plot entrypoint.
- Internals: Calls shared bundle for gnn family.
- Critical review: Simple wrapper; low risk.

#### `analytics/plots/hebbian/plot_hebbian.py`
- Responsibility: Hebbian family plot entrypoint.
- Internals: Calls shared bundle for hebbian family.
- Critical review: Simple wrapper; low risk.

#### `analytics/plots/lstm/plot_lstm.py`
- Responsibility: LSTM family plot entrypoint.
- Internals: Calls shared bundle for lstm family.
- Critical review: Simple wrapper; low risk.

#### `analytics/plots/mlp/plot_mlp.py`
- Responsibility: MLP family plot entrypoint.
- Internals: Calls shared bundle for mlp family.
- Critical review: Simple wrapper; low risk.

#### `analytics/plots/reinforcement/__init__.py`
- Responsibility: Reinforcement plots package marker.
- Internals: None.
- Critical review: No issues.

#### `analytics/plots/reinforcement/plot_reinforcement.py`
- Responsibility: Reinforcement family plot entrypoint.
- Internals: Calls shared bundle for reinforcement family.
- Critical review: Simple wrapper; low risk.

#### `analytics/plots/rnn/plot_rnn.py`
- Responsibility: RNN family plot entrypoint.
- Internals: Calls shared bundle for rnn family.
- Critical review: Simple wrapper; low risk.

#### `analytics/plots/transformer/plot_transformer.py`
- Responsibility: Transformer family plot entrypoint.
- Internals: Calls shared bundle for transformer family.
- Critical review: Simple wrapper; low risk.

#### `analytics/plots/trees/plot_trees.py`
- Responsibility: Trees family plot entrypoint.
- Internals: Calls shared bundle for trees family.
- Critical review: Simple wrapper; low risk.

#### `analytics/plots/vit/plot_vit.py`
- Responsibility: ViT family plot entrypoint.
- Internals: Calls shared bundle for vit family.
- Critical review: Simple wrapper; low risk.

### Scripts and Testing

#### `scripts/random_search.py`
- Responsibility: Hyperparameter random search orchestrator across model families.
- Internals: Family-specific search spaces, trial launching, score extraction, best-config summary export.
- Critical review: Useful and production-practical. Random search can waste compute; add optional successive halving or Bayesian optimization later.

#### `scripts/bio_sweep_queue.py`
- Responsibility: Queue executor for bio feature ablations/combinatorics.
- Internals: Profile generation (full/pairwise/core), resume support, run ownership metadata.
- Critical review: Strong experimentation automation. Serial execution is safe but slow; optional bounded parallel mode could help.

#### `scripts/prune_runs_keep_best.py`
- Responsibility: Storage curation utility to retain best runs per family.
- Internals: Score-based run ranking, dry-run mode, deletion execution.
- Critical review: Useful lifecycle tool. Add benchmark-aware retain policies to avoid deleting valuable non-family-best runs.

#### `scripts/smoke_test.sh`
- Responsibility: Minimal shell wrapper to run smoke tests.
- Internals: Delegates to make test command.
- Critical review: Thin and fine.

#### `tests/test_core_smoke.cpp`
- Responsibility: Main automated C++ smoke coverage.
- Internals: Validates tensor ops, loss/metric guards, optimizer behavior, model basics, replay/markov, and invariants.
- Critical review: Strong baseline but still mostly unit-level; add long-running integration tests for API/UI and control pipelines.

### UI and API

#### `ui/package.json`
- Responsibility: Frontend package manifest and script definitions.
- Internals: `dev` script launches API and Vite client; lists React, Recharts, Plotly, Express dependencies.
- Critical review: Practical script setup. Keep dependency growth controlled to prevent bundle bloat.

#### `ui/package-lock.json`
- Responsibility: NPM lock file for reproducible installs.
- Internals: Full dependency tree snapshot.
- Critical review: Generated artifact; do not manually edit. Ensure lockfile updates are intentional.

#### `ui/vite.config.js`
- Responsibility: Vite build/dev configuration.
- Internals: React plugin and proxy settings for API.
- Critical review: Clean config; production base path and deployment modes can be added as needed.

#### `ui/index.html`
- Responsibility: Root HTML shell for React app.
- Internals: Mount node and minimal metadata.
- Critical review: Correct and minimal.

#### `ui/src/main.jsx`
- Responsibility: React entrypoint.
- Internals: Root render and StrictMode wrapper.
- Critical review: Standard. Ensure side effects in hooks remain idempotent under StrictMode.

#### `ui/src/lib/fetchJson.js`
- Responsibility: Shared fetch wrapper for API calls.
- Internals: Request dedupe map, no-cache behavior, normalized error propagation.
- Critical review: Good utility. Add explicit timeout support if API delays grow.

#### `ui/src/hooks/usePolling.js`
- Responsibility: Polling lifecycle hook for live updates.
- Internals: Interval polling, visibility handling, abort signal use, manual refresh nonce support.
- Critical review: Good defensive implementation. SSE/WebSocket should be considered for lower polling overhead.

#### `ui/src/styles.css`
- Responsibility: Main UI style sheet.
- Internals: Theme tokens, layout, panel, chart, control, responsive styles.
- Critical review: Visual quality is good; file is large and could be split by feature to improve maintainability.

#### `ui/api/server.mjs`
- Responsibility: Backend API for run metadata, tasks, model/benchmark coverage, CSV access, report discovery.
- Internals: Filesystem run scanning, endpoint routing, training/evaluation task control, cleanup actions.
- Critical review: Functionally rich but monolithic and sync-IO heavy. Highest backend priority is modularization + async indexed caching.

#### `ui/src/App.jsx`
- Responsibility: Main control-room React application.
- Internals: Global state, polling orchestration, tabs/panels, graph rendering, comparison views, view-all workflows, bio visuals, game replay controls, task actions.
- Critical review: Powerful UX but far too large for maintainability (~9k LOC). Split into domain modules/components and shared selectors immediately.

### Documentation Files

#### `docs/ARCHITECTURE.md`
- Responsibility: High-level architecture summary.
- Internals: Core/model/benchmark layout and guiding structure.
- Critical review: Useful entry doc but now under-represents expanded control/UI/bio systems.

#### `docs/ANALYTICS.md`
- Responsibility: Analytics pipeline and output schema documentation.
- Internals: Plot generation usage and expected run artifacts.
- Critical review: Good operational reference; keep synchronized with evolving CSV schemas.

#### `docs/BENCHMARK_PLUGIN.md`
- Responsibility: Plugin contract for adding new benchmarks.
- Internals: Required interfaces, artifact expectations, integration checklist.
- Critical review: Strong extensibility doc; should be enforced with template scaffolding.

#### `docs/BIO_SWEEP_QUEUE.md`
- Responsibility: How to run bio ablation queues.
- Internals: Profiles, examples, and expected outputs.
- Critical review: Useful for experimentation reproducibility.

#### `docs/HPO_RANDOM_SEARCH.md`
- Responsibility: Hyperparameter search usage documentation.
- Internals: CLI/env controls and output interpretation.
- Critical review: Good practical guide; include explicit score semantics per benchmark family.

#### `docs/HYBRID_ADAPTER_CAPABILITIES.md`
- Responsibility: States which models are adapter-capable for hybrid/bridge runtime.
- Internals: Supported/unsupported family map.
- Critical review: Important capability contract; must be updated whenever model support changes.

#### `docs/CODE_REVIEW_2026-03-03.md`
- Responsibility: Prior review snapshot.
- Internals: Historical findings and recommendations.
- Critical review: Good audit history; append status updates for closed issues.

#### `docs/CODE_REVIEW_FIX_TODOS_2026-03-03.md`
- Responsibility: Detailed backlog derived from prior review.
- Internals: Task-level remediation actions.
- Critical review: Useful execution artifact; should include completion tracking to avoid stale TODO drift.

#### `docs/FULL_REPOSITORY_PIPELINES_AND_CODE_REVIEW_2026-03-04.md`
- Responsibility: Earlier full-repo review draft.
- Internals: Pipeline summary and concise file-level notes.
- Critical review: Good baseline, but too shallow for deep implementation understanding; superseded by this detailed report.

### Legacy Standalone MLP App (Reference/Archive)

#### `legacy/standalone_mlp_app/include/matrix.h`
- Responsibility: Legacy matrix class interface.
- Internals: Basic dense ops used by old standalone pipeline.
- Critical review: Archive-only code; should remain isolated from active stack.

#### `legacy/standalone_mlp_app/include/activation.h`
- Responsibility: Legacy activation API declarations.
- Internals: Sigmoid/ReLU-related helpers for old MLP path.
- Critical review: Reference only; avoid accidental coupling with modern modules.

#### `legacy/standalone_mlp_app/include/loss.h`
- Responsibility: Legacy loss interface.
- Internals: Basic loss definitions used by standalone app.
- Critical review: Archive-only.

#### `legacy/standalone_mlp_app/include/layer.h`
- Responsibility: Legacy dense layer interface.
- Internals: Weights/biases and forward/backward declarations.
- Critical review: Archive-only.

#### `legacy/standalone_mlp_app/include/network.h`
- Responsibility: Legacy MLP network interface.
- Internals: Layer composition and train/infer declarations.
- Critical review: Archive-only.

#### `legacy/standalone_mlp_app/include/optimizer.h`
- Responsibility: Legacy optimizer interface.
- Internals: Basic update rule declarations.
- Critical review: Archive-only.

#### `legacy/standalone_mlp_app/include/mnist_loader.h`
- Responsibility: Legacy MNIST loader interface.
- Internals: IDX read declarations used by old app.
- Critical review: Archive-only.

#### `legacy/standalone_mlp_app/include/interpretability.h`
- Responsibility: Legacy interpretability output interface.
- Internals: Diagnostic dump declarations for old MLP analysis.
- Critical review: Archive-only; useful historical reference for present telemetry design.

#### `legacy/standalone_mlp_app/src/matrix.cpp`
- Responsibility: Legacy matrix implementation.
- Internals: Dense math ops for old standalone app.
- Critical review: Archive-only and superseded by `core/tensor`.

#### `legacy/standalone_mlp_app/src/activation.cpp`
- Responsibility: Legacy activation implementations.
- Internals: Nonlinear function math.
- Critical review: Archive-only.

#### `legacy/standalone_mlp_app/src/loss.cpp`
- Responsibility: Legacy loss implementations.
- Internals: Basic objective calculations.
- Critical review: Archive-only.

#### `legacy/standalone_mlp_app/src/layer.cpp`
- Responsibility: Legacy layer forward/backward logic.
- Internals: Matrix-based dense transformations.
- Critical review: Archive-only.

#### `legacy/standalone_mlp_app/src/network.cpp`
- Responsibility: Legacy network orchestration.
- Internals: Layer chaining and update loop.
- Critical review: Archive-only.

#### `legacy/standalone_mlp_app/src/optimizer.cpp`
- Responsibility: Legacy optimizer implementation.
- Internals: Basic parameter update logic.
- Critical review: Archive-only.

#### `legacy/standalone_mlp_app/src/mnist_loader.cpp`
- Responsibility: Legacy MNIST parsing implementation.
- Internals: IDX reads and sample formatting.
- Critical review: Archive-only.

#### `legacy/standalone_mlp_app/src/interpretability.cpp`
- Responsibility: Legacy interpretability export implementation.
- Internals: Feature and neuron-level analysis dumps.
- Critical review: Archive-only but conceptually useful.

#### `legacy/standalone_mlp_app/src/main.cpp`
- Responsibility: Legacy app executable entrypoint.
- Internals: End-to-end old standalone training flow.
- Critical review: Archive-only; explicitly mark as unsupported to avoid confusion.


## Priority Fix Plan (Post-Review)

### P0 (Highest Impact)
1. Split `ui/src/App.jsx` into feature modules (`runs`, `training`, `evaluation`, `comparison`, `bio`, `games`, `tasks`) and centralize state selectors.
2. Split `ui/api/server.mjs` into route modules + run index service + task scheduler service; migrate sync FS reads to async with mtime cache.
3. Extract common supervised benchmark loop into shared runtime helper to remove duplicated train/eval/log code.

### P1
1. Add explicit capability manifest endpoint (model x benchmark x mode) and consume it in UI to prevent invalid combinations.
2. Extend GPU backend beyond GEMM and reduce host-device transfer overhead.
3. Add benchmark-type-specific metric ontology in UI (`accuracy` only for classification; tactical/game metrics for control).

### P2
1. Split large model implementation files (`clustering`, `transformer`, `rl_models`) into smaller units by algorithm/block.
2. Add long-running integration tests (task start/stop, live polling correctness, game replay consistency).
3. Add automated documentation checks to keep family lists/capabilities synchronized across code/docs/UI.

## Final Assessment
- The repository is no longer a simple MNIST project; it is now a multi-runtime experimentation platform with supervised, control, RL, hybrid, bio-addon, and continuous-learning capabilities.
- Core numerical foundations are in solid shape, and adapter abstractions are directionally correct.
- The main bottleneck to production-grade robustness is orchestration-layer complexity (frontend/API/benchmark duplication), not model math.
