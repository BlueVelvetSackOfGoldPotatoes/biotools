# CellEngine Benchmark

Dedicated implementation of the cell-based benchmark family described in `models/cellspec.md` and `/home/pessegueirolunar/Downloads/cellengine_spec.md`.

## Implemented Vs Planned
| Area | Current state |
| --- | --- |
| Shared genome + developmental phenotype map | implemented |
| `fixed2d`, `grown2d`, `grown3d` bodies | implemented |
| Task set: cartpole / mass-spring / worm / pong | implemented |
| Replay + RL comparator reuse in UI path | implemented |
| ODD telemetry: current / stress / mechanics / gap decomposition | implemented |
| ODD interventions and counterfactual viewers | partial |
| Neurite / axon-like long-range wiring | planned |
| Rich GRN, migration, division during runtime | planned |
| Fully GPU-resident batched rollouts | planned |

## Files
- `cellengine.h`
- `cellengine.cpp`
- `DEEP_DIVE.md`
- `ODD_GOLD_STANDARD.md`

## Entry Points
- Benchmark: `./bin/benchmark_cellengine`
- Regression test: `make test_cellengine`

## Modes
- `CELLENGINE_MODE=full`
- `CELLENGINE_MODE=cell_only`
- `CELLENGINE_MODE=rl_only`
- `CELLENGINE_MODE=replay`

## Tasks
- `CELLENGINE_TASK=cartpole_balance`
- `CELLENGINE_TASK=mass_spring_balance`
- `CELLENGINE_TASK=worm_drag_race`
- `CELLENGINE_TASK=pong_return`

Task-specific launch knobs:
- replay initial conditions:
  - balance tasks use `CELLENGINE_REPLAY_THETA_DEG`
  - `worm_drag_race` uses `CELLENGINE_REPLAY_TASK_A` for initial bend
  - `pong_return` uses `CELLENGINE_REPLAY_TASK_A` for serve height and `CELLENGINE_REPLAY_TASK_B` for vertical serve drift
- benchmark/discovery difficulty:
  - `worm_drag_race` uses `CELLENGINE_WORM_GOAL_DISTANCE` and `CELLENGINE_WORM_MAX_BACKWARD`
  - `pong_return` uses `CELLENGINE_PONG_TARGET_HITS`, `CELLENGINE_PONG_BALL_SPEED`, and `CELLENGINE_PONG_PADDLE_HALF`

The current cell controller and RL comparator both expose a single scalar actuation stream. That makes continuous-control tasks a good fit. Discrete board tasks such as tic-tac-toe need a different multi-action controller interface and are not wired into this benchmark yet.

## Body Modes
- `CELLENGINE_BODY_MODE=fixed2d`
- `CELLENGINE_BODY_MODE=grown2d`
- `CELLENGINE_BODY_MODE=grown3d`

## CUDA
- `CELLENGINE_USE_CUDA=1` forces the GPU fast-cell-tick path
- `CELLENGINE_USE_CUDA=0` forces CPU
- default `auto` keeps GPU use conservative

The cell engine CUDA path uses NVRTC at runtime, so it does not require `nvcc` to be present in the build shell.

## RL Selectors
- `CELLENGINE_RL_ALGO=dqn`
- `CELLENGINE_RL_ALGO=a2c`
- `CELLENGINE_RL_ALGO=all`

## Reload A Saved Champion
```bash
CELLENGINE_MODE=full \
CELLENGINE_LOAD_GENOME=reports/cellengine_latest/champion_genome.csv \
CELLENGINE_RL_ALGO=all \
./bin/benchmark_cellengine
```

## Watch Tasks In React
```bash
make benchmark_cellengine

cd ui
npm run dev
```

Then open `http://localhost:5173/?tab=cellengine`.

The React viewer calls the C++ binary through the UI API and stores replay artifacts under `reports/cellengine_ui/...`.
Use the task strip at the top of `CellEngine Lab` to switch between tasks without leaving the page. The launch form, replay scenes, charts, gallery filters, and replay library all follow the selected task.

## Headless Replay From The CLI
```bash
CELLENGINE_MODE=replay \
CELLENGINE_LOAD_GENOME=reports/cellengine_latest/champion_genome.csv \
CELLENGINE_FINAL_TICKS=500 \
CELLENGINE_REPLAY_STDOUT=0 \
CELLENGINE_OUT=reports/cellengine_replay \
./bin/benchmark_cellengine
```

Replay outputs:
- `reports/cellengine_replay/replay_trace.csv`
- `reports/cellengine_replay/summary.json`

## Default Outputs
- `reports/cellengine_latest/report.md`
- `reports/cellengine_latest/summary.json`
- `reports/cellengine_latest/champion_genome.csv`

Full benchmark runs now also emit integrated ODD results inside `report.md` and `summary.json`:
- ranked mechanism ablations
- ranked gene-channel ablations
- ranked per-cell force ablations
- ranked edge ablations
- relative importance shares
- temporal correlations from a traced autonomous episode
- structural confounds between body geometry and expressed gene channels
- morphology-vs-legacy-body attribution
- diffusion-vs-local-chemistry attribution

Trial-context artifacts are now persisted both for replay and for benchmark reports:
- replay mode:
  - `trial_context.json`
  - `trial_pre_reset_cells.csv`
  - `trial_post_reset_cells.csv`
- benchmark/report outputs:
  - `clean_teacher_trial_context.json`
  - `clean_autonomous_trial_context.json`
  - `damaged_autonomous_trial_context.json`
  - matching `*_trial_pre_reset_cells.csv` and `*_trial_post_reset_cells.csv`

## Useful Env Vars
- `CELLENGINE_TASK`
- `CELLENGINE_BODY_MODE`
- `CELLENGINE_POP`
- `CELLENGINE_GENS`
- `CELLENGINE_SEARCH_TRIALS`
- `CELLENGINE_SEARCH_TICKS`
- `CELLENGINE_FINAL_TRIALS`
- `CELLENGINE_FINAL_TICKS`
- `CELLENGINE_AUTO_ATTEMPTS`
- `CELLENGINE_DEVELOP_STEPS`
- `CELLENGINE_SEED_HALF_WIDTH`
- `CELLENGINE_MAX_CELLS`
- `CELLENGINE_BODY_EXTENT_X`
- `CELLENGINE_BODY_EXTENT_Y`
- `CELLENGINE_BODY_EXTENT_Z`
- `CELLENGINE_GROWTH_THRESHOLD`
- `CELLENGINE_CHEM_DIFF_STEPS`
- `CELLENGINE_CHEM_DIFF_RATE`
- `CELLENGINE_CHEM_DECAY`
- `CELLENGINE_USE_CUDA`
- `CELLENGINE_LOAD_GENOME`
- `CELLENGINE_REPLAY_SLEEP_MS`
- `CELLENGINE_REPLAY_FRAME_STRIDE`
- `CELLENGINE_REPLAY_CLEAR`
- `CELLENGINE_REPLAY_STDOUT`
- `CELLENGINE_REPLAY_DAMAGE_TICK`
- `CELLENGINE_REPLAY_THETA_DEG`
- `CELLENGINE_REPLAY_TASK_A`
- `CELLENGINE_REPLAY_TASK_B`
- `CELLENGINE_WORM_GOAL_DISTANCE`
- `CELLENGINE_WORM_MAX_BACKWARD`
- `CELLENGINE_PONG_TARGET_HITS`
- `CELLENGINE_PONG_BALL_SPEED`
- `CELLENGINE_PONG_PADDLE_HALF`
- `CELLENGINE_RL_ALGO`
- `CELLENGINE_RL_EPISODES`
- `CELLENGINE_RL_FINETUNE`
- `CELLENGINE_OUT`
- `CELLENGINE_ODD`
- `CELLENGINE_ODD_TRIALS`
- `CELLENGINE_ODD_TICKS`
