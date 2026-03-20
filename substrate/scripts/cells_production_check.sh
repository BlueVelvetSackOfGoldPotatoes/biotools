#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[cells-prod] running test suite"
make test_cells_suite

echo "[cells-prod] building cells benchmarks"
make -j"$(nproc)" benchmark_cells benchmark_cell_tasks benchmark_cell_circuit benchmark_cell_developmental benchmark_leaf_cells

echo "[cells-prod] running short benchmark smokes"
CELLS_STEPS=80 ./bin/benchmark_cells >/tmp/cells_prod_benchmark_cells.log
CELL_TASK_STEPS=100 ./bin/benchmark_cell_tasks >/tmp/cells_prod_benchmark_tasks.log
CELL_DEV_EPOCHS=1 CELL_DEV_TRAIN_SAMPLES=64 CELL_DEV_TEST_SAMPLES=32 CELL_DEV_FREE_SETTLE=1 CELL_DEV_TAUGHT_SETTLE=1 ./bin/benchmark_cell_developmental >/tmp/cells_prod_benchmark_dev.log

echo "[cells-prod] PASS"
