# Benchmark Plug-in Contract

This repository supports multiple benchmarks in one shared dashboard/runtime. A benchmark is considered "plugged in" when its executable writes run artifacts under `runs/<run_id>/` using the common schema.

## Required Artifacts

Each benchmark run must emit:

- `manifest.json`
- `learning/epoch_metrics.csv`

The manifest must include:

- `benchmark_id` (stable machine id, e.g. `mnist`, `tictactoe`)
- `benchmark_name` (human label)
- `task_type` (e.g. `classification`, `control`)
- standard run/model fields already written by `RunLogger`

## Strongly Recommended Artifacts

For full dashboard functionality (live + evaluation + model-specific tabs), emit:

- `learning/batch_metrics.csv` (for real-time batch line charts)
- `learning/confusion_matrix.csv` (when applicable)
- `deployment/inference_metrics.csv`
- `deployment/calibration_bins.csv`
- `deployment/system_metrics.csv`
- `model_specific/<model_family>/*.csv` (custom architecture/benchmark visuals)

## Discovery Rules

The API discovers runs from `runs/*/manifest.json` and derives:

- benchmark coverage: `/api/benchmarks`
- family coverage (optionally benchmark-scoped): `/api/families?benchmark=<benchmark_id>`
- run list (optionally benchmark-scoped): `/api/runs?benchmark=<benchmark_id>`

No hardcoded benchmark registry is required in the UI.

## Integration Checklist

1. Add benchmark executable in `benchmarks/`.
2. Add build target to `Makefile` and include it in `BENCHMARKS`.
3. Use `RunLogger` with benchmark metadata (`benchmark_id`, `benchmark_name`, `task_type`).
4. Emit `epoch_metrics.csv` and `batch_metrics.csv` during training.
5. Emit evaluation/deployment CSVs for dashboard tabs.
6. Optional: add model-specific CSV visualizations.
7. Optional: add random-search sampler and model spec in `scripts/random_search.py`.

## Model/Benchmark Decoupling Pattern

For cross-architecture compatibility, prefer a two-layer adapter:

- `DiscreteEnv` implementation for benchmark dynamics + legal actions.
- Bit observation bridge that maps benchmark bits to model input width and projects model logits to benchmark action space.

This lets a single benchmark run with multiple architectures without rewriting model internals.

### Discrete Plug-in Interfaces

Core interfaces live in `core/benchmark/`:

- `discrete_env.h`: benchmark contract (`reset`, `step`, `legal_actions`, `observation_bits`, metadata).
- `bit_codec.h`: shared bit encoders/decoders.
- `bit_bridge_model.h`: model adapter (`bits -> bridge -> backbone -> action logits`).
- `discrete_q_runtime.h`: generic train/eval runtime with callback hooks for logging.

For a new control benchmark:

1. Implement `DiscreteEnv` (environment state machine + reward/outcome contract).
2. Build any `TrainableModel` backbone (single model or `HybridSystem` composition).
3. Wrap it in `BitBridgeModel` with benchmark-specific bit/action dimensions.
4. Run with `DiscreteQRuntime` and emit standard `RunLogger` artifacts.

### Model Adapter Registry

TicTacToe benchmark model binding is now registry-driven (no hardcoded `if/else` factory chain):

- Registry API: `models/hybrid/src/model_registry.h`
- Built-in adapters + aliases: `models/hybrid/src/model_registry.cpp`
- Hybrid entrypoint consumption: `models/hybrid/src/hybrid_models.cpp`

To add a new architecture to benchmark plug-ins:

1. Implement a `TrainableModel` adapter for the architecture.
2. Register it with `register_model_adapter("<canonical>", factory, {"alias1", ...})`.
3. It becomes available in `TICTACTOE_MODEL` and hybrid specs automatically.
