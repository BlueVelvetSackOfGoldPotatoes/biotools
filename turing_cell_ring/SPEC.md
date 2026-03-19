# Turing Paper-Example Platform Spec

This file is the implementation contract for the repository as it exists now.

## 1. Scope

The package covers four paper-facing families:

1. `section10_example1`
2. `example2_table2`
3. `oscillatory_case_e`
4. `oscillatory_case_f`

Section 10 remains the default path and the most thoroughly validated path. The rest of the repository expands the platform toward broader paper coverage without claiming that every branch has Section 10-level historical fidelity.

## 2. Architectural Contract

### 2.1 Shared scientific core

The backend must be family-driven rather than hard-coded around one two-species example.

Each family defines:

- `id`
- `name`
- `description`
- `analysis_mode`
- species order
- diffusion coefficients
- equilibrium solver
- Jacobian / linear matrix
- nonlinear cell drift
- supported execution modes
- preset set
- embedded reference data where available

Current implementation:

- family registry: `backend/src/model.cpp`
- generic config and results types: `backend/include/turing/types.hpp`
- generic JSON mapping: `backend/src/json.cpp`

### 2.2 Generalized linear analysis

Linear analysis is no longer fixed to `X/Y`.

For every admissible ring mode, the analyzer constructs the family mode matrix and computes its complex eigenvalues. The public output must include:

- dominant mode
- dominant real growth
- dominant imaginary frequency
- stationary vs oscillatory classification
- wavelength in cells
- phase velocity when oscillatory

Current implementation:

- `backend/src/analysis.cpp`
- Eigen-based complex eigensolve

### 2.3 Generic simulation state

Snapshots and mode traces are generic over species name.

Required output shape:

- `speciesOrder`
- `species`
- `speciesModeAmplitudes`
- `primaryModeAmplitudes`

Backward-compatibility requirement for two-species families:

- legacy `x`
- legacy `y`
- legacy `yModeAmplitudes`

## 3. Family-Level Scientific Scope

### 3.1 Section 10 family

This family implements the twenty-cell stationary ring example from Section 10.

Supported engines:

- `reduced_xy`
- `full_chemistry`

Supported execution modes:

- `modern`
- `historical_paper_constrained`

Required diagnostics:

- mode 3 / mode 4 linear gap
- e-fold lead
- arrest timing for first `Y = 0`
- Table 1 reference comparisons

Required presets:

- `paper-quick`
- `paper-slow`

### 3.2 Example 2 / Table 2 family

This family covers the second worked two-species chemical example and the Table 2 workflow.

Required presets:

- `paper-table2-stable-ring`
- `paper-example2-scan`

Required outputs:

- homogeneous equilibrium for chosen `f` / `k`
- stability scan summaries
- six-cell stable-ring reference comparison

Current implementation detail:

- the six-cell Table 2 profile is reconstructed as a dynamically stable target using an inferred forcing field plus a weak restoring term around the published profile

This must be stated explicitly in docs and report. It is a pragmatic reconstruction choice, not a hidden claim of complete derivation from the reduced law alone.

### 3.3 Oscillatory toolkit

This family layer is the first-class home for the paper’s oscillatory three-morphogen cases.

Required shipped presets:

- `paper-wave-e-travelling`
- `paper-wave-f-short-oscillation`

Required diagnostics:

- oscillatory/stationary classification
- dominant oscillatory mode
- frequency
- phase velocity
- neighbour phase offset
- travelling consistency

Required frontend views:

- real/imaginary spectrum
- multi-species profiles
- time-by-cell heatmaps

## 4. Execution Modes

### 4.1 `modern`

Modern integration branch used by default.

Current implementation:

- predictor-corrector style update in the generic family simulator
- native stochastic increments

### 4.2 `historical_paper_constrained`

Approximate historical sensitivity branch for worked numerical examples only.

This mode is explicitly not an exact machine emulator.

Current profiles:

- `historic_1952_baseline`
- `historic_1952_coarse_rounding`
- `historic_1952_fine_rounding`
- `historic_1952_synchronous`

Assumed behaviors:

- fixed-step explicit Euler
- cell-major sequential update
- decimal rounding after updates
- seeded LCG plus Gaussian conversion
- sparse record cadence inherited from the simulation capture stride

Supported families:

- `section10_example1`
- `example2_table2`

Not supported in first implementation:

- oscillatory toolkit families
- Section 10 `full_chemistry`

## 5. API Contract

### 5.1 Endpoints

- `GET /api/health`
- `GET /api/presets`
- `POST /api/analyze`
- `POST /api/simulate`
- `POST /api/batch`

### 5.2 Preset catalog response

`GET /api/presets` must return:

- `presets`
- `families`

### 5.3 Additive request fields

The public JSON request surface supports:

- `preset`
- `familyId`
- `engine`
- `analysisMode`
- `executionMode`
- `executionProfileId`
- `speciesOrder`
- `familyParameters`

Backward-compatibility:

- existing Section 10 requests that only provide `preset` and `engine` must keep working

## 6. Frontend Contract

The frontend must provide parity with the scientific scope above.

Required controls:

- family selector
- preset selector
- engine selector
- execution-mode selector
- family-parameter inputs where relevant

Required displays:

- generic multi-species line plots
- linear mode spectrum
- final dominant-mode histogram for batches
- oscillatory heatmaps
- reference-comparison table
- modern-vs-historical comparison panel for supported families

The Section 10 view remains the default landing experience, but the UI must not be structurally locked to it.

## 7. Validation Contract

Validation artifacts are grouped under `results/final_validation/`.

Required subtrees:

- `section10/`
- `example2_table2/`
- `oscillatory_toolkit/`
- `historical_comparisons/`

### 7.1 Section 10 checks

Required:

- linear regression around `gamma = 0`
- quick and slow Monte Carlo batches
- Table 1 seed-search artifacts

### 7.2 Table 2 checks

Required:

- correct homogeneous equilibrium for selected `f` / `k`
- scan summary across multiple `f` values
- stable-ring persistence artifact

### 7.3 Oscillatory checks

Required:

- case `(e)` oscillatory finite-wave diagnostics
- case `(f)` short-wave oscillatory diagnostics with near-antiphase neighbours

### 7.4 Historical checks

Required:

- reproducibility for fixed seed
- side-by-side saved modern vs historical artifacts
- explicit statement in the report that the historical branch is approximate

## 8. Report Contract

The LaTeX report must cover:

- source fidelity and modeling choices
- implementation changes
- Section 10 validation
- second chemical example / Table 2
- oscillatory / travelling-wave toolkit
- historical sensitivity versus modern execution
- explicit hypothesis verdict
- limitations

Required figures now include:

- H1–H6 hypothesis figures
- Table 2 stable-ring overlay
- oscillatory toolkit figure
- modern-vs-historical comparison figure

## 9. Known limitations that must stay explicit

- Section 10 remains the strongest reconstruction; the broader family additions are less historically constrained.
- The Table 2 stable-ring workflow currently uses an inferred forcing/stabilization term to preserve the published six-cell pattern.
- The oscillatory cases are qualitative paper-matched presets rather than exact historical reconstructions.
- The historical execution branch is approximate and paper-constrained, not exact emulation.

Any document or UI copy that suggests otherwise is out of spec.
