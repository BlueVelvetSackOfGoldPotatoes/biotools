# Substrate — C++ Benchmark Suite

Pure C++17 neural-network, ML, and biological cell simulation benchmark suite with a shared run contract, 26 benchmarks across 17 model families, and a React dashboard for live monitoring. Covers MNIST classification, discrete control tasks (chess, go, pong, cartpole), and evolutionary cell-based intelligence.

## Canonical Code Path

Use the `core/ + models/ + benchmarks/` stack as the maintained path.

- `core/`: tensor ops, modules/layers, losses, metrics, optimizers, data loading, biology runtime, discrete environment framework, game engines
- `models/`: 17 model families — CNN, RNN/LSTM, Transformer, ViT, trees/forests, clustering, diffusion, GNN, RL/MuZero-lite, Markov chains, Hebbian, actor-critic, forward-forward, hybrid, plus the cells/cellengine biological simulation subsystem
- `benchmarks/`: 26 executable benchmarks for each family and task type

## Cells Subsystem

The cells subsystem spans multiple simulation scales with dedicated benchmarks, a standalone physics engine, and optional 3D visualization:

Canonical documentation:

- `models/cells/CELL_FIRST_AI_COMPLETE_GUIDE.md` — long-form guide
- `models/cells/IMPLEMENTATION_STATUS.md` — implementation ledger
- `models/cellspec.md` — full CellEngine specification (scientific basis, cell model, evolutionary algorithm, evaluation)

Primary cells entry points:

- `./bin/benchmark_cells`: raw deterministic 3D cell simulator diagnostics
- `./bin/benchmark_cell_tasks`: chemotaxis, distribution-shift, and damage-recovery task harness with baseline controller comparison matrix
- `./bin/benchmark_cell_circuit`: fixed-graph MNIST cell circuit
- `./bin/benchmark_cell_circuit_compare`: cell circuit vs frozen linear baseline (dual run logs)
- `./bin/benchmark_cell_developmental`: migrating self-wiring developmental tissue vs fixed-graph baseline
- `./bin/benchmark_leaf_cells`: 2D tissue / morphogenesis benchmark
- `./bin/benchmark_cellengine`: full physics-based evolutionary cell engine (CUDA-accelerated, population evolution, RL replay)

Optional viewer sidecar:

- `models/cells/standalone_ogre/`: standalone CMake + OGRE 3D visualization project

## Legacy Path

`legacy/standalone_mlp_app/src/ + legacy/standalone_mlp_app/include/` contains an older standalone MLP + interpretability app.
It is kept for reference, but benchmark development should happen in the canonical stack.

## Build

```bash
make benchmarks
```

Performance-oriented build (OpenMP + native CPU + CUDA auto-detection):

```bash
make -j$(nproc) benchmarks
```

Force specific backend choices at build time:

```bash
# CPU-only build
make clean && make USE_CUDA=0 OPENMP=1 benchmarks

# Require CUDA/cuBLAS (fails fast if unavailable)
make clean && make USE_CUDA=1 benchmarks
```

Or:

```bash
./build_all.sh
```

## Run

Run one benchmark:

```bash
./bin/benchmark_mlp
```

Run Markov-chain benchmark:

```bash
./bin/benchmark_markov
```

Run TicTacToe benchmark:

```bash
./bin/benchmark_tictactoe
```

Run generalized game/physics benchmarks (`connect_four`, `battleship`, `go`, `chess`, `cartpole`):

```bash
GAME_BENCHMARK=chess \
GAME_MODEL=mlp \
GAME_CONTROL_MODE=full_side \
./bin/benchmark_games
```

Chess per-piece specialist control (one model per piece-group, or one replicated model):

```bash
GAME_BENCHMARK=chess \
GAME_CONTROL_MODE=per_piece \
GAME_PIECE_MODELS=mlp,mlp,mlp,mlp,mlp,mlp \
./bin/benchmark_games
```

Run TicTacToe with any supported architecture via the shared bit-bridge:

```bash
TICTACTOE_MODEL=transformer \
TICTACTOE_EPISODES=12000 \
TICTACTOE_LR=0.0015 \
./bin/benchmark_tictactoe
```

Supported `TICTACTOE_MODEL` values:
`mlp`, `cnn`, `rnn`, `gru`, `lstm`, `transformer`, `vit`, `hebbian`, `actor_critic`, or `hybrid:<comma-separated-models>`.

For adapter scope (supported vs intentionally non-adapter families), see `docs/HYBRID_ADAPTER_CAPABILITIES.md`.

Hybrid composition works too:

```bash
TICTACTOE_MODEL=hybrid:cnn,transformer,mlp ./bin/benchmark_tictactoe
```

Run continuous-learning / hybrid-composition benchmark:

```bash
./bin/benchmark_continuous
```

Useful runtime knobs:

```bash
# modes: single | hybrid | suite
ONLINE_MODE=hybrid \
ONLINE_HYBRID_SPEC=cnn,transformer,mlp \
ONLINE_CYCLES=16 \
ONLINE_ACTIVE_STEPS=24 \
ONLINE_SLEEP_STEPS=10 \
ONLINE_TARGET_ON_FULL_TEST=1 \
ONLINE_STOP_ON_TARGET=1 \
./bin/benchmark_continuous
```

Run all benchmarks (fails fast on first failing benchmark):

```bash
make run_all
```

### Cells-Specific Runs

Raw cell simulation:

```bash
./bin/benchmark_cells
```

Cell tasks with baseline comparison matrix:

```bash
# choose any comma-separated subset:
# homeostatic,rnn,gru,lstm,transformer
CELL_TASK_CONTROLLERS=homeostatic,gru,lstm,transformer \
./bin/benchmark_cell_tasks
```

Developmental MNIST with internal-teacher curriculum:

```bash
CELL_DEV_INTERNAL_TEACHER=1 \
CELL_DEV_INTERNAL_BLEND=0.35 \
CELL_DEV_INTERNAL_AFTER_EPOCH=2 \
./bin/benchmark_cell_developmental
```

CellEngine evolutionary simulation:

```bash
./bin/benchmark_cellengine
```

## Testing

Fast deterministic core sanity check:

```bash
make test_smoke
```

or

```bash
./scripts/smoke_test.sh
```

Full cells test suite (core smoke + cells deep + leaf cells + cellengine + OGRE smoke):

```bash
make test_cells_suite
```

Individual test targets:

```bash
make test_cells_deep       # deep cell simulation checks
make test_leaf_cells       # 2D leaf/tissue determinism
make test_cellengine       # cellengine physics/evolution tests
```

Cells production readiness checks:

```bash
make cells_prod_check
./scripts/cells_production_check.sh
```

Optional OGRE strict mode:

```bash
REQUIRE_OGRE=1 make test_cells_ogre_smoke
```

## Analytics

All benchmark binaries emit run logs under `runs/<run_id>/` using the common schema:

- `manifest.json`
- `learning/*.csv`
- `deployment/*.csv`
- `model_specific/<model_family>/*.csv`

CellEngine writes separate trace artifacts under `output/cellengine_*/`.

Generate plots:

```bash
python analytics/plots/common/plot_bundle.py \
  --run_dir runs/<run_id> \
  --out_dir reports/<run_id> \
  --format png
```

Launch Streamlit dashboard:

```bash
streamlit run analytics/dashboard/app.py
```

React dashboard (live training monitor + eval/model visuals + cellengine replay):

```bash
cd ui
npm install
npm run dev
```

Then open `http://localhost:5173`.

Use the `Benchmark` selector in the top toolbar to switch dashboard scope across discovered benchmarks.
Use the `Game Live` tab on control-task runs to replay move-by-move traces with:
- `real-time` playback (uses logged training-time deltas)
- `human` playback (fixed slower delay)
- `free hand` stepping (manual back/forward arrows)

The CellEngine panel provides evolution replay, genome inspection, discovery views, surface previews, and cross-run comparison.

Random hyperparameter search across architectures and benchmarks:

```bash
./scripts/random_search.py --models all --trials 8
```

Details and model-specific search knobs:
[docs/HPO_RANDOM_SEARCH.md](docs/HPO_RANDOM_SEARCH.md)

## Runtime Acceleration Knobs

```bash
# CPU parallelism
export OMP_NUM_THREADS=$(nproc)
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

# Tensor backend selection
export TENSOR_USE_CUDA=1          # 1 on, 0 off
export TENSOR_FORCE_CUDA=0        # 1 forces CUDA for eligible matmuls
export TENSOR_CUDA_MIN_FLOPS=2e7  # offload threshold

# Live training telemetry
export BATCH_LOG_EVERY=10         # write learning/batch_metrics.csv every N train batches
export GAME_TRACE_EVERY=1         # benchmark_games trace sampling stride (1 = every move)
export GAME_TRACE_MAX_ROWS=250000 # cap trace rows per run
export TICTACTOE_TRACE_EVERY=1
export TICTACTOE_TRACE_MAX_ROWS=250000
```

Example (GPU-enabled run):

```bash
OMP_NUM_THREADS=$(nproc) TENSOR_USE_CUDA=1 ./bin/benchmark_mlp
```

## Biology Runtime Mode

Benchmarks include a cross-family biology instrumentation layer.

- Default (safe): observational only (no parameter mutation).
- Active biology updates: set `BIO_ACTIVE=1` before running benchmarks.
- Fine-grained biology feature toggles:
  - `BIO_ENABLE_STRUCTURAL_PLASTICITY`
  - `BIO_ENABLE_HOMEOSTASIS`
  - `BIO_ENABLE_THREE_FACTOR`
  - `BIO_ENABLE_MYELINATION`
  - `BIO_ENABLE_BIOELECTRIC`
  - `BIO_ENABLE_EI_SIGN_CONSTRAINTS`

Example:

```bash
BIO_ACTIVE=1 ./bin/benchmark_mlp
```

More details: [docs/ANALYTICS.md](docs/ANALYTICS.md).
Bio sweep queue: [docs/BIO_SWEEP_QUEUE.md](docs/BIO_SWEEP_QUEUE.md).

Benchmark plug-in contract: [docs/BENCHMARK_PLUGIN.md](docs/BENCHMARK_PLUGIN.md).

## Data

MNIST IDX files are expected in `data/`:

- `train-images-idx3-ubyte`
- `train-labels-idx1-ubyte`
- `t10k-images-idx3-ubyte`
- `t10k-labels-idx1-ubyte`
