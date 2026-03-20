# Cell Circuit Long Run Report (MNIST)

- Date: 2026-03-04
- Run ID: `cells_20260304T182826785_p60457_c000000_888e5d44`
- Run dir: `runs/cells_20260304T182826785_p60457_c000000_888e5d44`
- Command:
  - `CELL_CIRCUIT_EPOCHS=12 CELL_CIRCUIT_TRAIN_SAMPLES=20000 CELL_CIRCUIT_TEST_SAMPLES=5000 CELL_CIRCUIT_SEED=42 ./bin/benchmark_cell_circuit`

## Aggregate Metrics

- Test epochs: `12`
- Start test accuracy: `0.7228`
- Final test accuracy: `0.7644`
- Absolute gain: `+0.0416`
- Best test accuracy: `0.7644` (epoch `12`)
- Best test F1 macro: `0.7620` (epoch `12`)
- Start test loss: `2.0341`
- Final test loss: `2.0359`
- Mean test accuracy: `0.7173`
- Std dev test accuracy: `0.0294`
- Epoch-to-epoch acc moves: `7 up`, `4 down`
- Total train wall time (sum of per-epoch): `155.16s`
- Mean throughput: `1937.56 samples/s`

## Top 3 Test Epochs

1. Epoch `12`: acc `0.7644`, loss `2.0359`, f1 `0.7620`
2. Epoch `7`: acc `0.7486`, loss `2.0245`, f1 `0.7196`
3. Epoch `6`: acc `0.7376`, loss `2.0122`, f1 `0.7270`

## Per-Epoch Test Curve

| Epoch | Test Acc | Test Loss | Test F1 |
|---:|---:|---:|---:|
| 1 | 0.7228 | 2.0341 | 0.7214 |
| 2 | 0.7362 | 2.0437 | 0.7331 |
| 3 | 0.6806 | 2.0040 | 0.6485 |
| 4 | 0.7086 | 2.0356 | 0.7035 |
| 5 | 0.6636 | 2.0126 | 0.6427 |
| 6 | 0.7376 | 2.0122 | 0.7270 |
| 7 | 0.7486 | 2.0245 | 0.7196 |
| 8 | 0.7130 | 2.0049 | 0.6747 |
| 9 | 0.7308 | 2.0360 | 0.7124 |
| 10 | 0.6750 | 2.0117 | 0.6555 |
| 11 | 0.7266 | 2.0358 | 0.7198 |
| 12 | 0.7644 | 2.0359 | 0.7620 |

## Notes

- Accuracy improves overall and peaks at the final epoch, but with visible oscillation.
- Loss is relatively flat while accuracy/F1 rise, indicating confidence calibration/noise dynamics dominate more than CE minimization in this local three-factor setup.
- Logs for analysis:
  - `runs/cells_20260304T182826785_p60457_c000000_888e5d44/learning/epoch_metrics.csv`
  - `runs/cells_20260304T182826785_p60457_c000000_888e5d44/model_specific/cells/circuit_dynamics.csv`
