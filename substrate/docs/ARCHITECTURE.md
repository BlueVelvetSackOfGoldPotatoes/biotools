# Architecture Notes

## Primary Stack

The active implementation path is:

- `core/`
- `models/`
- `benchmarks/`

Recent additions for continual/bio experiments:

- `core/online/`: trainable abstraction, generic parameter optimizer, continuous active/sleep runtime
- `core/benchmark/`: benchmark environment interfaces, bit-bridge adapters, discrete-control runtime
- `models/hybrid/`: end-to-end composable multi-architecture hybrid systems
- `benchmarks/benchmark_continuous.cpp`: continuous-learning + hybrid benchmark entry point

This is the stack used by `make benchmarks`, `make run_all`, and `build_all.sh`.

## Legacy Stack

- `src/`
- `include/`

This stack is legacy and retained for historical/reference purposes.
Avoid adding new benchmark features there.

## Build Reliability

- Make now tracks headers via generated `.d` files.
- `make benchmarks` builds actual benchmark targets directly.
- `benchmark_forests` links both forest and tree model objects.
- `run_all` is fail-fast by design.
