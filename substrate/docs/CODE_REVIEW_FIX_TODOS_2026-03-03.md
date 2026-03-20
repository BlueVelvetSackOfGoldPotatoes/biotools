# Code Review Fix TODOs (2026-03-03)

This document expands each finding from `CODE_REVIEW_2026-03-03.md` into implementation-level TODOs.

## 1) Run ID collision under parallel launch
### Files
- `core/io/run_logger.h`
- `core/io/run_logger.cpp`
- `ui/api/server.mjs` (only if run id format assumptions exist)

### Implementation TODO
1. Add a dedicated run-id generator helper that includes:
1. UTC timestamp (`YYYYMMDDTHHMMSSmmm`).
2. Process id.
3. Thread-safe monotonic counter (atomic).
4. Short random suffix (8+ hex chars).
2. Keep `model_family` prefix but sanitize to `[a-z0-9_]+` to avoid filesystem issues.
3. Replace direct `make_run_id(model_family)` use with a retry loop that checks if `runs/<run_id>` already exists.
4. Retry up to a bounded limit (for example 32 attempts) with regenerated suffixes.
5. If all attempts fail, throw explicit runtime error with collision diagnostics.
6. Add a unit/integration test that creates many loggers in parallel and verifies unique run directories.

### Validation TODO
1. Launch 1k+ logger creations concurrently in a stress test.
2. Verify no duplicate run ids and no merged files.
3. Verify run id still parseable by existing UI listing logic.

### Acceptance criteria
1. No collisions in parallel stress test.
2. Existing dashboard still shows runs normally.
3. Failures are explicit, never silent merges.

## 2) Manifest JSON not safely encoded
### Files
- `core/io/run_logger.cpp`
- optionally `core/io/json_utils.h/.cpp` (new)

### Implementation TODO
1. Stop hand-writing JSON strings via stream concatenation.
2. Implement a minimal JSON writer utility for string escaping and object serialization.
3. Serialize all string fields with proper escaping (`\`, `"`, control chars, newlines).
4. Parse and validate `params_json_` once at logger construction:
1. If valid object/array JSON, embed as JSON value.
2. If invalid, write it as escaped string in `params_raw` and set `params_parse_error`.
5. Write `manifest.json.tmp` then atomically rename to `manifest.json` to avoid partial writes.
6. Ensure `write_manifest_start` and `write_manifest_end` share one serializer path to avoid drift.
7. Add tests for quotes/newlines/unicode/backslashes in model fields.

### Validation TODO
1. Create runs with intentionally tricky strings.
2. Parse all produced manifests with strict JSON parser.
3. Ensure UI `/api/runs` and `/api/runs/:id/manifest` still parse cleanly.

### Acceptance criteria
1. 100% valid manifests under fuzzed input strings.
2. No malformed JSON in logs.

## 3) Loss functions OOB risk
### Files
- `core/losses/losses.cpp`
- `core/losses/losses.h`
- `tests/test_core_smoke.cpp`

### Implementation TODO
1. Add shape checks in `MSELoss`, `BCELoss`, `KLDivLoss`, and `HuberLoss` forward/backward.
2. In `NLLLoss`:
1. Require `targets.rows == pred.rows` and `targets.cols >= 1`.
2. Validate each class index is finite integer-like and in `[0, pred.cols)`.
3. Throw clear error including offending row/value.
3. Add fast path for valid shapes with no extra allocations.
4. Ensure OpenMP loops remain correct after checks.
5. Add tests for mismatch shapes and invalid class indices.

### Validation TODO
1. Run smoke tests.
2. Add negative tests expecting throws.
3. Run benchmarks quickly to ensure no perf regression beyond noise.

### Acceptance criteria
1. No undefined behavior on malformed inputs.
2. Errors are explicit and actionable.

## 4) Metrics vector bounds safety
### Files
- `core/metrics/metrics.cpp`
- `core/metrics/metrics.h`
- `tests/test_core_smoke.cpp`

### Implementation TODO
1. Add size equality checks to both `accuracy` overloads and `confusion_matrix`.
2. Add optional empty-input policy:
1. If both empty, return `0.0` (or throw consistently, choose and document).
3. Keep current range checks in confusion matrix and improve error messages.
4. Add tests for mismatched lengths and empty vectors.

### Validation TODO
1. Verify expected throws for mismatch.
2. Confirm existing callers pass checks.

### Acceptance criteria
1. No out-of-bounds reads from labels.
2. Deterministic behavior for empty input policy.

## 5) MNIST validation removed in release
### Files
- `core/data/dataloader.cpp`
- `core/data/dataloader.h`
- benchmark entry points that load MNIST

### Implementation TODO
1. Replace `assert(magic == ...)` with runtime checks and exceptions/errors.
2. Validate header read success at each step (`read_u32` should fail explicitly if read < 4 bytes).
3. Validate `rows`, `cols`, `num` are sane before allocating memory.
4. Validate payload length against expected bytes.
5. If short read, hard fail instead of warning + partial tensor use.
6. Return `StatusOr`-style result or throw and handle at caller level consistently.
7. Add corrupt-file tests (wrong magic, truncated file).

### Validation TODO
1. Run valid MNIST load path.
2. Run with synthetic corrupt IDX files.
3. Verify clear errors and no partial tensors used.

### Acceptance criteria
1. Release builds reject invalid IDX deterministically.
2. No silent partial data ingestion.

## 6) Continuous benchmark logs wrong mode
### Files
- `benchmarks/benchmark_continuous.cpp`

### Implementation TODO
1. Introduce explicit runtime mode enum/string (`single|hybrid|suite`).
2. Pass actual mode to `json_config` instead of family.
3. Keep family and mode both in params JSON.
4. Add a small parser test/print assertion in benchmark startup logs.

### Validation TODO
1. Run each mode once.
2. Inspect generated `manifest.json` params for correct `mode`.

### Acceptance criteria
1. Metadata reflects true execution mode in all variants.

## 7) Hard-coded grad clip in continuous runtime
### Files
- `core/online/continuous_runtime.h`
- `core/online/continuous_runtime.cpp`
- `benchmarks/benchmark_continuous.cpp`

### Implementation TODO
1. Add `grad_clip_norm` to `ContinuousConfig` with sensible default.
2. Parse env/config value in benchmark.
3. Pass config value to `optimizer.step` instead of literal `5.0`.
4. Log clip norm to params JSON and, if possible, epoch metrics.
5. Add guard: if <=0 then disable clipping.

### Validation TODO
1. Run with clip off and on.
2. Confirm behavior changes and values appear in manifests.

### Acceptance criteria
1. No hard-coded clip values remain in continuous runtime.

## 8) BatchNorm divide-by-zero on tiny batches
### Files
- `core/nn/module.cpp`
- `tests/test_core_smoke.cpp`

### Implementation TODO
1. In `BatchNorm1d`, guard unbiased variance correction when `N <= 1`.
2. In `BatchNorm2d`, guard correction when `M <= 1`.
3. Decide fallback behavior and document:
1. Use biased variance directly.
2. Or skip running-var update for that batch.
4. Add finite checks in debug assertions for running stats.
5. Add tests with batch size 1 for BN1d/BN2d train/eval paths.

### Validation TODO
1. Verify no NaN/inf in outputs and running vars.
2. Confirm normal behavior unchanged for larger batches.

### Acceptance criteria
1. Tiny-batch BN is numerically stable.

## 9) Hybrid adapter coverage incomplete
### Files
- `models/hybrid/src/model_registry.cpp`
- `models/hybrid/src/model_registry.h`
- `models/hybrid/src/hybrid_models.cpp`
- `README.md` (supported model list)

### Implementation TODO
1. Define capability interface for adapter eligibility (input/output assumptions).
2. Add adapters for missing families where meaningful (`markov`, `reinforcement`, `gnn`, `forward_forward`, `diffusion`, `trees`, `forests`, `clustering`, `continuous`).
3. If some families are not feasible as direct classifiers, mark explicitly unsupported with reason.
4. Make `supported_model_names()` derived from actual registry, no hardcoded docs drift.
5. Add tests that each advertised model name builds successfully.

### Validation TODO
1. Loop over supported names and instantiate adapters.
2. Run one forward/backward smoke per adapter.

### Acceptance criteria
1. “Supported” list and actual registry are aligned.
2. No runtime surprise for documented names.

## 10) Bio E/I sign constraint axis ambiguity
### Files
- `core/bio/bio_runtime.h`
- `core/bio/bio_runtime.cpp`
- benchmark bio config wiring

### Implementation TODO
1. Introduce explicit enum config: `EiConstraintAxis::{Source,Destination,Disabled}`.
2. Annotate each `ParamState` with matrix semantic metadata (`rows=source`, `cols=dest`) when attaching.
3. Apply sign enforcement based on chosen axis and metadata.
4. Update sign-violation metric computation to same axis semantics.
5. Add unit tests with known matrices for both axis modes.
6. Add manifest logging for selected axis.

### Validation TODO
1. Run bio benchmarks with each axis mode.
2. Check sign-violation fractions behave as expected.

### Acceptance criteria
1. Constraint semantics are explicit and consistent.

## 11) Bio epoch pass scalability
### Files
- `core/bio/bio_runtime.cpp`
- `core/bio/bio_runtime.h`

### Implementation TODO
1. Add scheduling knobs:
1. `rewire_every_epochs` already exists; enforce low-frequency defaults.
2. Separate myelin update frequency.
2. Replace full sort with partial selection (`nth_element`/heap) everywhere possible.
3. Cache edge distances per param tensor and invalidate only when geometry changes.
4. Track active-edge index lists to avoid scanning masked-off edges each epoch.
5. Add sampling mode for rewiring on large tensors (percent or capped edge budget).
6. Add benchmark timing instrumentation for each bio sub-pass.
7. Add adaptive budget cap tied to wall-clock target per epoch.

### Validation TODO
1. Compare epoch wall time before/after on same seed/model.
2. Verify metric outputs are still produced and numerically sane.

### Acceptance criteria
1. Significant reduction in bio overhead on large models.
2. No functional regressions in bio metrics logging.

## 12) Duplicated env parsing logic
### Files
- `benchmarks/benchmark_continuous.cpp`
- `benchmarks/hpo_utils.h`

### Implementation TODO
1. Remove local `env_*` helpers from `benchmark_continuous.cpp`.
2. Replace calls with `hpo::env_int/env_double/env_bool/env_string/env_size`.
3. Normalize boolean semantics once in `hpo_utils.h`.
4. Add a tiny unit-like benchmark self-check for accepted bool strings.

### Validation TODO
1. Run benchmark with mixed-case boolean env values.
2. Confirm behavior matches shared parser.

### Acceptance criteria
1. Single source of truth for env parsing.

## 13) DataLoader remainder and bounds
### Files
- `core/data/dataloader.h`
- `core/data/dataloader.cpp`
- callers in benchmarks

### Implementation TODO
1. Add `drop_last` option to `DataLoader` ctor/config.
2. Make `num_batches()` return ceil when `drop_last=false`.
3. Update `get_batch` to:
1. Check `batch_idx < num_batches()`.
2. Compute actual batch size for final partial batch.
4. Adjust `extract_batch` to support variable batch size per call safely.
5. Update callers that assume fixed batch size.
6. Add tests for exact division and remainder cases.

### Validation TODO
1. Verify final partial batch content matches expected indices.
2. Verify out-of-range `batch_idx` throws.

### Acceptance criteria
1. No silent sample drops unless explicitly requested.

## 14) API sync I/O and full CSV reparsing
### Files
- `ui/api/server.mjs`

### Implementation TODO
1. Move CSV reads to async fs APIs (`fs/promises`).
2. Introduce per-file cache keyed by `(path, mtimeMs, size)`.
3. Add parsed-row cache with TTL + max memory bound.
4. Add query options for `/api/runs/:runId/csv`:
1. `limit`, `offset`.
2. `from_step` / `from_epoch`.
3. `downsample`.
5. Add slim mode for `/detail` so heavy arrays are optional (`include=` query params).
6. Add request-level timing logs to identify hotspots.

### Validation TODO
1. Load-test API under multiple UI clients.
2. Verify event-loop lag and CPU usage drop.
3. Confirm output parity for existing consumers.

### Acceptance criteria
1. Backend stays responsive under polling load.
2. Large-run detail requests no longer block API.

## 15) Active run detection heuristic
### Files
- `ui/api/server.mjs`
- benchmark binaries (heartbeat writer integration)

### Implementation TODO
1. Define `runtime/heartbeat.json` contract:
1. `pid`, `started_utc`, `last_update_utc`, `phase`, `epoch`, `step`.
2. Add helper in C++ runtime/logging layer to update heartbeat periodically.
3. Update all long-running benchmarks to write heartbeat every N seconds/batches.
4. In API, prefer heartbeat-based activity state.
5. Fallback to mtime heuristic only when heartbeat absent.
6. Add stale-state transitions and reason field (`no_heartbeat`, `ended`, `stale`).

### Validation TODO
1. Start and kill a benchmark abruptly; confirm state transitions.
2. Confirm no false inactive during long epochs.

### Acceptance criteria
1. Active/stale/finished labels reflect process reality.

## 16) Frontend monolith (`App.jsx`)
### Files
- `ui/src/App.jsx`
- new files under `ui/src/features/*`, `ui/src/components/*`, `ui/src/hooks/*`, `ui/src/lib/*`

### Implementation TODO
1. Split by domain:
1. `features/live-runs`.
2. `features/comparison`.
3. `features/training`.
4. `features/evaluation`.
5. `features/model-specific`.
2. Move fetch and polling code into hooks (`useRuns`, `useRunDetail`, `useReports`).
3. Move transformation utilities into pure modules with tests.
4. Keep top-level `App` as route/shell + shared state composition only.
5. Introduce lightweight state container for cache and selection state.
6. Add lint rules for max file length/complexity to prevent relapse.

### Validation TODO
1. Snapshot or integration tests for each tab.
2. Manual regression pass over existing flows.

### Acceptance criteria
1. `App.jsx` reduced to orchestrator-level size.
2. Behavior unchanged while maintainability improves.

## 17) Polling overload architecture
### Files
- `ui/src/App.jsx` and extracted hooks
- `ui/api/server.mjs`

### Implementation TODO
1. Replace per-component timers with one shared polling scheduler.
2. De-duplicate requests by key and share in-flight promises.
3. Use adaptive poll intervals:
1. active runs: fast.
2. inactive views/background tabs: slow.
4. Add conditional fetch support (`If-Modified-Since`/etag-equivalent token).
5. Optionally add SSE/WebSocket stream for live training updates.
6. Cap concurrent heavy-detail fetches.

### Validation TODO
1. Measure requests/min before/after.
2. Verify live charts remain timely and stable.

### Acceptance criteria
1. Significant drop in request volume without losing freshness.

## 18) StrictMode dev effect amplification
### Files
- `ui/src/main.jsx`
- polling hooks (`usePolling` and successors)

### Implementation TODO
1. Keep StrictMode enabled.
2. Make polling hooks fully idempotent:
1. AbortController on every fetch.
2. Sequence token to ignore stale responses.
3. Guaranteed timer cleanup.
3. Add debug logging guard to detect duplicate fetch loops in development.
4. Document expected StrictMode double-invoke behavior in `ui/README`.

### Validation TODO
1. Run dev mode and ensure no duplicated visible state churn.
2. Verify production mode remains unchanged.

### Acceptance criteria
1. Dev experience is stable despite StrictMode.

## 19) HPO/sweep run attribution race
### Files
- `scripts/random_search.py`
- `scripts/bio_sweep_queue.py`
- `core/io/run_logger.cpp` (manifest extra fields)

### Implementation TODO
1. Generate per-trial `trial_uuid` in scripts.
2. Pass `TRIAL_UUID` env var into benchmark process.
3. Extend run manifest params to include `trial_uuid` and `job_origin`.
4. In scripts, score only run dirs whose manifest matches current `trial_uuid`.
5. Keep directory diff as fallback only with explicit warning.
6. Add timeout/exit handling that still records unmatched trials.

### Validation TODO
1. Run two sweep scripts concurrently.
2. Verify each trial maps only to its own run ids.

### Acceptance criteria
1. No cross-process run misattribution.

## 20) Legacy path duplication (`src/` + `include/`)
### Files
- `README.md`
- `Makefile`
- potentially move files under `legacy/`

### Implementation TODO
1. Decide policy: archive or delete legacy path.
2. If archive:
1. Move `src/` and `include/` to `legacy/standalone_mlp_app/`.
2. Exclude from default build and tests.
3. Add clear “unsupported/archived” notice.
3. If keep:
1. Add explicit ownership and sync boundaries.
2. Add CI guard to prevent accidental cross-dependency.

### Validation TODO
1. Ensure main build targets unaffected.
2. Ensure docs reflect exact supported paths.

### Acceptance criteria
1. No ambiguity about canonical code path.

## 21) Python cache artifacts and ignore policy
### Files
- `.gitignore`
- repository tracked files

### Implementation TODO
1. Update `.gitignore` with:
1. `__pycache__/`
2. `*.pyc`
3. `.pytest_cache/`
4. virtualenv patterns if used
2. Remove tracked cache directories from repository history tip (`git rm -r --cached ...`).
3. Add cleanup script target (`make clean_pycache` optional).

### Validation TODO
1. Confirm `git status` stays clean after Python script runs.
2. Confirm no cache artifacts show in diffs.

### Acceptance criteria
1. Python artifacts are never tracked again.

## 22) Git commit provenance missing
### Files
- `core/io/run_logger.cpp`
- build/runtime env handling

### Implementation TODO
1. Add helper to resolve commit hash in this order:
1. `GIT_COMMIT` env override.
2. `git rev-parse --short HEAD` command.
3. fallback `null`.
2. Cache commit hash once per process to avoid repeated subprocess calls.
3. Add optional dirty flag (`git diff --quiet` -> clean/dirty) if affordable.
4. Write `git_commit` and `git_dirty` into manifest start/end.
5. Fail gracefully when git is unavailable (no hard crash).

### Validation TODO
1. Run inside git repo and outside git repo.
2. Verify manifests include expected commit metadata.

### Acceptance criteria
1. Reproducibility metadata is present for normal runs.

---

## Execution recommendation

1. Execute fixes in the same priority order from the review report.
2. For each finding, create one PR branch and include tests before merge.
3. Add a tracking board with 22 cards mapped to this document IDs.
4. Block benchmark mega-runs until all Critical + High items are closed.
