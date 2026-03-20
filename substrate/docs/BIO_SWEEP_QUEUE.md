# Bio Combinatorics Queue

Use `scripts/bio_sweep_queue.py` to run sequential sweeps across model families and biology-feature combinations.

## What It Does

- Enumerates model families (`--models all` or a subset).
- Enumerates bio feature combos (`--combo-mode full|pairwise|core`).
- Runs one task at a time (strict queue).
- Writes:
  - `queue_plan.csv` (planned tasks),
  - `summary.csv` (results per executed task),
  - `meta.json` (combo + config metadata).

## Combo Modes

- `full`: baseline observational run + full factorial of active bio toggles.
- `pairwise`: pairwise covering set for bio toggles.
- `core`: baseline + all-off + single-feature ablations + all-on.

## Typical Runs

Full factorial sweep (all models, quick profile):

```bash
python3 -u ./scripts/bio_sweep_queue.py \
  --models all \
  --combo-mode full \
  --include-baseline \
  --profile quick
```

Resume an interrupted run:

```bash
python3 -u ./scripts/bio_sweep_queue.py \
  --models all \
  --combo-mode full \
  --include-baseline \
  --profile quick \
  --out-dir reports/bio_sweep_all_full_YYYYMMDD_HHMMSS \
  --resume
```

Pairwise sweep for selected models:

```bash
python3 -u ./scripts/bio_sweep_queue.py \
  --models mlp,cnn,transformer,vit \
  --combo-mode pairwise \
  --profile balanced
```

## Profiles

- `quick`: reduced training budget per benchmark (fast comparisons).
- `balanced`: medium budget.
- `full`: benchmark defaults (slowest, highest cost).

## Notes

- GPU and CPU parallel runtime hints are enabled by default for queued runs:
  - `TENSOR_USE_CUDA=1`
  - `OMP_NUM_THREADS=<nproc>`
  - `OMP_PROC_BIND=spread`
  - `OMP_PLACES=cores`
- Add global env overrides with `--extra-env KEY=VALUE,...`.
