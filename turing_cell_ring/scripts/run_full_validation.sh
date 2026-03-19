#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/results/final_validation}"
REPORT_TEX="$ROOT_DIR/report/turing_ring_validation_report.tex"
REPORT_DIR="$ROOT_DIR/report"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18091}"

PIPELINE_LOG="$OUTPUT_DIR/pipeline.log"
BUILD_LOG="$OUTPUT_DIR/backend_build.log"
CTEST_LOG="$OUTPUT_DIR/ctest.log"
FRONTEND_INSTALL_LOG="$OUTPUT_DIR/frontend_npm_ci.log"
FRONTEND_BUILD_LOG="$OUTPUT_DIR/frontend_build.log"
SERVER_LOG="$OUTPUT_DIR/http_server.log"
SMOKE_LOG="$OUTPUT_DIR/http_smoke.log"
VALIDATE_LOG="$OUTPUT_DIR/validate.log"
FIGURE_LOG="$OUTPUT_DIR/hypothesis_figures.log"
PDFLATEX_LOG="$OUTPUT_DIR/pdflatex.log"

mkdir -p "$OUTPUT_DIR" "$REPORT_DIR"
: >"$PIPELINE_LOG"

log() {
  printf '[%s] %s\n' "$(date -Iseconds)" "$*" | tee -a "$PIPELINE_LOG"
}

run_logged() {
  local logfile="$1"
  shift
  log "Running: $*"
  if {
    printf '== %s ==\n' "$*"
    "$@"
    printf '\n'
  } >>"$logfile" 2>&1; then
    log "Completed: $*"
  else
    log "Failed: $*"
    tail -n 60 "$logfile" >&2 || true
    exit 1
  fi
}

curl_json() {
  local outfile="$1"
  shift
  if curl -fsS "$@" >"$outfile" 2>/dev/null; then
    return 0
  fi
  return 1
}

SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

cd "$ROOT_DIR"

run_logged "$BUILD_LOG" cmake -S . -B build
run_logged "$BUILD_LOG" cmake --build build -j
run_logged "$CTEST_LOG" ctest --test-dir build --output-on-failure

run_logged "$FRONTEND_INSTALL_LOG" npm --prefix frontend ci
run_logged "$FRONTEND_BUILD_LOG" npm --prefix frontend run build

log "Starting backend server on http://$HOST:$PORT"
if command -v stdbuf >/dev/null 2>&1; then
  stdbuf -oL -eL ./build/turing_ring_backend --host "$HOST" --port "$PORT" >"$SERVER_LOG" 2>&1 &
else
  ./build/turing_ring_backend --host "$HOST" --port "$PORT" >"$SERVER_LOG" 2>&1 &
fi
SERVER_PID=$!

ready=0
for _ in $(seq 1 60); do
  if curl_json "$OUTPUT_DIR/health.json" "http://$HOST:$PORT/api/health"; then
    ready=1
    break
  fi
  sleep 1
done

if [[ "$ready" -ne 1 ]]; then
  log "Backend failed to become ready"
  tail -n 60 "$SERVER_LOG" >&2 || true
  exit 1
fi

: >"$SMOKE_LOG"
log "Running HTTP smoke checks" | tee -a "$SMOKE_LOG"

ROOT_STATUS="$(curl -sS -o "$OUTPUT_DIR/root.html" -w '%{http_code}' "http://$HOST:$PORT/")"
printf 'GET / -> %s\n' "$ROOT_STATUS" | tee -a "$SMOKE_LOG"
if [[ "$ROOT_STATUS" != "200" ]]; then
  log "Static root smoke check failed"
  exit 1
fi

curl_json "$OUTPUT_DIR/presets.json" "http://$HOST:$PORT/api/presets"
printf 'GET /api/presets -> ok\n' | tee -a "$SMOKE_LOG"

curl_json "$OUTPUT_DIR/analyze.json" \
  -H 'Content-Type: application/json' \
  -d '{"preset":"paper-quick","engine":"reduced_xy"}' \
  "http://$HOST:$PORT/api/analyze"
printf 'POST /api/analyze -> ok\n' | tee -a "$SMOKE_LOG"

curl_json "$OUTPUT_DIR/simulate.json" \
  -H 'Content-Type: application/json' \
  -d '{"preset":"paper-quick","engine":"full_chemistry","seed":3,"dt":0.01,"totalTime":80.0,"enableNoise":true,"noiseScale":1.0,"captureStride":20,"incipientCaptureGamma":0.0625}' \
  "http://$HOST:$PORT/api/simulate"
printf 'POST /api/simulate -> ok\n' | tee -a "$SMOKE_LOG"

curl_json "$OUTPUT_DIR/batch.json" \
  -H 'Content-Type: application/json' \
  -d '{"preset":"paper-quick","engine":"reduced_xy","replicates":16,"seed":1}' \
  "http://$HOST:$PORT/api/batch"
printf 'POST /api/batch -> ok\n' | tee -a "$SMOKE_LOG"

curl_json "$OUTPUT_DIR/analyze_table2.json" \
  -H 'Content-Type: application/json' \
  -d '{"preset":"paper-table2-stable-ring","familyId":"example2_table2","engine":"reduced_xy"}' \
  "http://$HOST:$PORT/api/analyze"
printf 'POST /api/analyze (Table 2) -> ok\n' | tee -a "$SMOKE_LOG"

curl_json "$OUTPUT_DIR/simulate_wave_e.json" \
  -H 'Content-Type: application/json' \
  -d '{"preset":"paper-wave-e-travelling","familyId":"oscillatory_case_e","engine":"reduced_xy","enableNoise":false}' \
  "http://$HOST:$PORT/api/simulate"
printf 'POST /api/simulate (wave e) -> ok\n' | tee -a "$SMOKE_LOG"

curl_json "$OUTPUT_DIR/simulate_historical.json" \
  -H 'Content-Type: application/json' \
  -d '{"preset":"paper-quick","familyId":"section10_example1","engine":"reduced_xy","executionMode":"historical_paper_constrained","executionProfileId":"historic_1952_baseline","seed":3}' \
  "http://$HOST:$PORT/api/simulate"
printf 'POST /api/simulate (historical) -> ok\n' | tee -a "$SMOKE_LOG"

run_logged \
  "$VALIDATE_LOG" \
  ./build/turing_ring_validate --output-dir "$OUTPUT_DIR" --replicates 512 --seed-search 64

run_logged \
  "$FIGURE_LOG" \
  env VALIDATION_DIR="$OUTPUT_DIR" python3 report/generate_hypothesis_figures.py

run_logged \
  "$PDFLATEX_LOG" \
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory "$REPORT_DIR" "$REPORT_TEX"
run_logged \
  "$PDFLATEX_LOG" \
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory "$REPORT_DIR" "$REPORT_TEX"

log "Validation pipeline completed successfully"
