# Code Review Report (2026-03-03)

## Scope and method

Reviewed the maintained stack (`core/`, `models/`, `benchmarks/`, `scripts/`, `ui/`) with focus on correctness, reliability, performance, duplication, and maintainability.

Validation run:

- `make -j$(nproc) test_smoke`
- Result: `34 passed, 0 failed`

Generated/vendor directories were not treated as source of truth (`ui/node_modules`, `ui/dist`, `build`, `runs`, `reports`, `logs`).

---

## Critical findings

### 1) Run ID collision under parallel launch can corrupt/merge runs
- Evidence: `core/io/run_logger.cpp:290-307`
- Problem: run id format uses `family + UTC second + millisecond`. Two same-family runs started within the same millisecond can collide and write into the same run directory.
- Impact: silent data corruption, mixed metrics, impossible provenance.
- Fix:
  - Add entropy (PID + atomic counter + random suffix or UUIDv4).
  - Create run dir with exclusive semantics and retry on collision.

### 2) Manifest JSON is not safely encoded
- Evidence: `core/io/run_logger.cpp:72-87`, `90-105`
- Problem: string fields (`model_variant`, `data_version`, etc.) are injected into JSON without escaping; `params_json_` is trusted raw.
- Impact: malformed `manifest.json` if fields contain quotes/newlines; downstream API parsing breaks.
- Fix:
  - Use a JSON writer/serializer (or at minimum a robust JSON-escape helper for all string fields).
  - Validate `params_json_` on logger creation.

### 3) Multiple loss functions can access out-of-bounds memory
- Evidence: `core/losses/losses.cpp:43-55`, `60-77`, `83-101`, `106-122`, `127-148`
- Problem:
  - `MSE`, `BCE`, `KLDiv`, `Huber` iterate `pred.data.size()` and index `targets.data[i]` without shape check.
  - `NLLLoss` casts target to class index and indexes `pred(i, cls)` without validating class bounds.
- Impact: UB/crashes or silent wrong gradients with malformed inputs.
- Fix:
  - Add strict `TENSOR_CHECK` shape checks in `forward` and `backward` for all losses.
  - In `NLLLoss`, validate `targets.cols>=1`, class index range, and integer semantics.

### 4) Metric helpers can read past vector bounds
- Evidence: `core/metrics/metrics.cpp:24-31`, `33-40`, `58-66`
- Problem: `accuracy(preds, labels)` and `confusion_matrix(...)` iterate `preds.size()` and index `labels[i]` without size equality checks.
- Impact: UB if caller supplies mismatched lengths.
- Fix:
  - Add `TENSOR_CHECK(preds.size()==labels.size(), ...)` in both overloads.

### 5) MNIST format validation disappears in release builds
- Evidence: `core/data/dataloader.cpp:31-33`, `58-60`
- Problem: IDX magic checks are `assert(...)`; in release (`-DNDEBUG`) these are removed.
- Impact: corrupted/wrong files can flow through silently and poison training.
- Fix:
  - Replace `assert` with runtime checks + hard failure.
  - Verify full read lengths for headers/payload.

---

## High findings

### 6) Continuous benchmark logs wrong semantic mode
- Evidence: `benchmarks/benchmark_continuous.cpp:113-139`, call at `155-161`
- Problem: `json_config(... mode ...)` receives `family` (`continuous`/`hybrid`) instead of runtime mode (`single|hybrid|suite`).
- Impact: inaccurate experiment metadata and wrong grouping/analysis.
- Fix:
  - Pass actual execution mode variable to `json_config`.

### 7) Hard-coded grad clip in continuous runtime
- Evidence: `core/online/continuous_runtime.cpp:189`
- Problem: `optimizer.step(..., lr_scale, 5.0)` ignores config-driven clipping.
- Impact: hidden behavior, harder tuning/repro, inconsistent training dynamics.
- Fix:
  - Add `grad_clip_norm` to `ContinuousConfig`; thread through env/config and logging.

### 8) BatchNorm unbiased variance can divide by zero
- Evidence: `core/nn/module.cpp:200-201`, `329-330`
- Problem: Bessel correction uses `N/(N-1)` and `M/(M-1)` with no guard for `N==1` or `M==1`.
- Impact: NaNs/infs on tiny batches.
- Fix:
  - Guard denominator: if sample count <= 1, skip unbiased correction (use biased var) or keep previous running var.

### 9) Hybrid adapter coverage is incomplete vs “any model” usage
- Evidence: `models/hybrid/src/model_registry.cpp:278-313`
- Problem: registry includes only `mlp, cnn, rnn, gru, lstm, transformer, vit, hebbian, actor_critic`.
- Impact: architecture-agnostic pathways fail for other implemented families (`markov`, `reinforcement`, `gnn`, `forward_forward`, `diffusion`, `trees`, `forests`, `clustering`, `continuous`).
- Fix:
  - Add adapters for all supported model families or enforce capability manifest and clear UI constraints.

### 10) Bio E/I sign constraints applied on likely wrong axis
- Evidence: `core/bio/bio_runtime.cpp:78-87`, validation path `445-460`
- Problem: sign enforcement is per destination column; in common linear layout `[in_features, out_features]`, biological sign is typically constrained per presynaptic output.
- Impact: semantics mismatch, harder interpretation, potential training distortion.
- Fix:
  - Make sign axis explicit via metadata and enforce consistently (source- or destination-type by design).

### 11) Bio epoch passes scale poorly
- Evidence: `core/bio/bio_runtime.cpp:263-341`, `343-389`, `476-481`
- Problem: full-parameter scans + sorts + per-edge distance loops every epoch.
- Impact: severe runtime cost on larger models; defeats “fast experimentation” goal.
- Fix:
  - Use sampled rewiring, top-k heap selection, cached distance blocks, and sparse active-edge structures.
  - Move expensive passes to lower frequency schedules.

---

## Medium findings

### 12) Duplicated env parsing logic (inconsistent boolean semantics)
- Evidence: `benchmarks/benchmark_continuous.cpp:25-66` vs `benchmarks/hpo_utils.h:27-68`
- Problem: duplicate helpers (`env_int/env_double/env_bool/...`) with different accepted values.
- Impact: drift and config surprises.
- Fix:
  - Reuse `hpo_utils.h` only; remove local duplicates.

### 13) DataLoader drops remainder samples and lacks batch index guard
- Evidence: `core/data/dataloader.cpp:113`, `115-120`
- Problem:
  - `num_batches()` is floor division; tail samples are always discarded.
  - `get_batch` does not validate `batch_idx` bounds.
- Impact: systematic data underuse; possible OOB if caller overshoots.
- Fix:
  - Add `drop_last` config and optional final partial batch.
  - Add bounds checks in `get_batch`.

### 14) API backend is synchronous and repeatedly reparses entire CSVs
- Evidence: `ui/api/server.mjs:218-229`, `265-285`, `625-646`, `648-729`
- Problem: sync `fs` + full CSV parse on each poll across many endpoints.
- Impact: event-loop stalls, high CPU/IO, poor scalability with many runs.
- Fix:
  - Switch to async I/O + incremental caches keyed by mtime.
  - Add pagination/downsampling for `/csv` and slim responses for `/detail`.

### 15) “Active run” detection is heuristic and can misclassify
- Evidence: `ui/api/server.mjs:174-201`, `296-302`
- Problem: active state relies on mtimes of a fixed file list and stale timeout.
- Impact: false “active/stale/finished” states for atypical logging patterns.
- Fix:
  - Add explicit heartbeat file (`runtime/heartbeat.json`) written by benchmarks.
  - Use PID liveness when available.

### 16) Frontend is a monolith (high regression risk)
- Evidence: `ui/src/App.jsx:1` and file length `8017`
- Problem: one giant file mixes transport, derivation logic, and rendering.
- Impact: hard to reason about, test, and safely evolve.
- Fix:
  - Split into feature modules/hooks/components (`LiveRuns`, `Comparison`, `Training`, `Evaluation`, `ModelSpecific`).
  - Add unit tests for data transforms.

### 17) Polling architecture can overload backend
- Evidence: global poller `ui/src/App.jsx:64-106`, per-detail poller `4151-4173`
- Problem: concurrent timers + full payload fetches, multiplied by expanded panels.
- Impact: high request volume; perceived UI flakiness under load.
- Fix:
  - Centralize polling with shared cache/store.
  - Use conditional requests or WebSocket/SSE for incremental updates.

### 18) React StrictMode doubles effects in dev, amplifying polling side effects
- Evidence: `ui/src/main.jsx:6-10`
- Problem: effect double-invocation in dev can look like duplicate refresh loops.
- Impact: confusing behavior during development diagnostics.
- Fix:
  - Keep StrictMode, but make polling hooks idempotent/cancel-safe and clearly document dev behavior.

### 19) HPO/sweep scripts can misattribute runs under concurrency
- Evidence:
  - `scripts/random_search.py:386-419`
  - `scripts/bio_sweep_queue.py:294-341`
- Problem: identifies new runs via before/after directory diff only.
- Impact: if another process writes to `runs/`, scores/new_runs can be assigned to the wrong trial.
- Fix:
  - Tag runs with trial UUID in env and require matching manifest field before scoring.

---

## Low findings / hygiene / duplication

### 20) Legacy code path duplicates maintained stack
- Evidence: `README.md:13-17`, plus `src/` and `include/` trees.
- Problem: two divergent implementations in same repo.
- Impact: confusion, duplicated maintenance surface.
- Fix:
  - Archive legacy path under `legacy/` with explicit non-build status, or remove if not needed.

### 21) Python cache artifacts are present and not ignored
- Evidence: `find analytics scripts -type d -name '__pycache__'` returns multiple dirs; `.gitignore:1-14` lacks `__pycache__/` and `*.pyc`.
- Impact: noisy diffs and accidental artifact commits.
- Fix:
  - Add Python ignores; remove tracked cache dirs from git.

### 22) Git commit provenance is always null
- Evidence: `core/io/run_logger.cpp:78`, `96`
- Problem: `git_commit` not populated.
- Impact: weaker reproducibility and auditability.
- Fix:
  - Resolve commit hash at run start (`git rev-parse --short HEAD`) with fallback when unavailable.

---

## Positive notes

- Core smoke suite passes (`34/34`), including gradient checks and RL/Markov sanity tests.
- CUDA matmul path has clean fallback behavior and runtime gating (`TENSOR_USE_CUDA`, threshold-based offload).
- Run schema and model-specific logging are broad and useful for dashboard diagnostics.

---

## Priority fix order

1. Run identity + manifest serialization hardening (Findings 1, 2).
2. Loss/metrics/dataloader correctness guards (Findings 3, 4, 5, 13).
3. Continuous/bio training correctness and scaling issues (Findings 6, 7, 8, 10, 11).
4. API/UI scalability and architecture cleanup (Findings 14, 15, 16, 17, 18).
5. Script reproducibility + repository hygiene (Findings 19, 20, 21, 22).

