# Turing Paper-Example Platform

This repository is now a paper-example toolkit for A. M. Turing's *The Chemical Basis of Morphogenesis*, not only a Section 10 ring reconstruction. It combines a native C++ backend, a React frontend, reproducible validation scripts, and a LaTeX report.

## Scope

Implemented families:

- `section10_example1`
  - the twenty-cell stationary ring from Section 10 / Table 1
  - engines: `reduced_xy`, `full_chemistry`
  - execution modes: `modern`, `historical_paper_constrained`
- `example2_table2`
  - the second two-species chemical example and the Table 2 six-cell stable ring workflow
  - engine: generic reduced family engine
  - execution modes: `modern`, `historical_paper_constrained`
- `oscillatory_case_e`
  - three-morphogen travelling-wave toolkit preset for Turing's Section 8 case `(e)`
- `oscillatory_case_f`
  - three-morphogen extreme-short-wave oscillatory toolkit preset for Section 8 case `(f)`

The historical branch is explicitly approximate. It is a paper-constrained reconstruction with fixed-step Euler updates, decimal rounding, and a seeded LCG; it is not a claim of exact Manchester-machine emulation.

## What the platform provides

Backend:

- model-family registry with per-family species, diffusion, equilibrium, Jacobian, and nonlinear drift
- generalized `N`-species ring analysis using complex eigenspectra
- stationary vs oscillatory classification
- travelling-wave diagnostics: dominant frequency, phase velocity, neighbour phase offset, travelling consistency
- Section 10 dual engines:
  - `reduced_xy`
  - `full_chemistry`
- paper-constrained historical execution for the worked numerical examples
  - profiles: `historic_1952_baseline`, `historic_1952_coarse_rounding`, `historic_1952_fine_rounding`, `historic_1952_synchronous`
- JSON API for analysis, single runs, and Monte Carlo batches
- validation runner that writes grouped artifacts under `results/final_validation/`

Frontend:

- family, preset, engine, and execution-mode selection
- dynamic controls for family parameters such as `example2F`
- generic multi-species profile charts
- mode-spectrum and mode-histogram views
- time-by-cell heatmaps for oscillatory examples
- modern-vs-historical comparison view for supported families
- Table 1 / Table 2 reference-comparison tables

Validation/reporting:

- C++ regression tests via `ctest`
- production frontend build
- HTTP smoke checks
- saved JSON artifacts by family / engine / execution mode
- hypothesis and expansion figures under `report/figures/`
- compiled LaTeX report in `report/turing_ring_validation_report.pdf`

## Main presets

- `paper-quick`
- `paper-slow`
- `paper-table2-stable-ring`
- `paper-example2-scan`
- `paper-wave-e-travelling`
- `paper-wave-f-short-oscillation`

## Build

Backend:

```bash
cmake -S . -B build
cmake --build build -j
```

Frontend:

```bash
npm --prefix frontend ci
npm --prefix frontend run build
```

Run the backend:

```bash
./build/turing_ring_backend --host 127.0.0.1 --port 8080
```

When launched from the project root, the backend serves `frontend/dist` automatically if it exists.

## Validation pipeline

Run the full end-to-end pipeline:

```bash
./scripts/run_full_validation.sh
```

That script performs:

- backend configure/build
- `ctest`
- frontend `npm ci` and production build
- HTTP smoke checks for `/`, `/api/health`, `/api/presets`, `/api/analyze`, `/api/simulate`, `/api/batch`
- family-aware validation artifact generation
- report figure generation
- LaTeX compilation

The default output directory is:

```text
results/final_validation/
```

Main artifact groups:

- `results/final_validation/section10/`
- `results/final_validation/example2_table2/`
- `results/final_validation/oscillatory_toolkit/`
- `results/final_validation/historical_comparisons/`

## API

Endpoints:

- `GET /api/health`
- `GET /api/presets`
- `POST /api/analyze`
- `POST /api/simulate`
- `POST /api/batch`

`GET /api/presets` returns both:

- `presets`
- `families`

Important additive request fields:

- `preset`
- `familyId`
- `engine`
- `analysisMode`
- `executionMode`
- `executionProfileId`
- `speciesOrder`
- `familyParameters`

The legacy Section 10 path remains backward-compatible: existing `preset + engine` requests still resolve to the current Section 10 family by default.

## Project layout

```text
backend/
  include/turing/
  src/
  tests/
frontend/
  src/
    components/
report/
  figures/
results/
scripts/
```

## Important limitations

- The Section 10 reconstruction is much stronger than the broader paper extensions. It remains the most validated part of the repository.
- The Table 2 stable ring workflow is now a dynamic relaxation toward the published six-cell profile using an inferred forcing and a weak stabilizing term. That is still an explicit reconstruction choice, not a fully derived law from the reduced family alone.
- The oscillatory presets reproduce the intended qualitative regimes from the paper, but they should be read as toolkit exemplars, not final historical reconstructions.
- The historical branch is sensitivity analysis under paper-constrained assumptions, not exact historical emulation.

## Documents

- high-level technical contract: `SPEC.md`
- final report: `report/turing_ring_validation_report.tex`
- compiled report: `report/turing_ring_validation_report.pdf`
