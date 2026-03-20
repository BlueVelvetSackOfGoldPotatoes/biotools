# Cell Circuit 3-Seed Long Run Report (MNIST)

- Date: 2026-03-04
- Schedule per seed:
  - `epochs=12`
  - `train_samples=20000`
  - `test_samples=5000`
- Seeds: `42, 43, 44`
- Run mapping file: `output/cells/cell_circuit_3seed_runs.txt`

## Seed-Level Summary

| Seed | Run ID | Best Epoch | Best Test Acc | Final Test Acc | Mean Test Acc | Test Acc Std (within seed) |
|---:|---|---:|---:|---:|---:|---:|
| 42 | `cells_20260304T183538518_p63341_c000000_32e97993` | 12 | 0.7644 | 0.7644 | 0.7173 | 0.0294 |
| 43 | `cells_20260304T183812593_p63751_c000000_467cae0c` | 2  | 0.7456 | 0.7424 | 0.7018 | 0.0292 |
| 44 | `cells_20260304T184043985_p64338_c000000_59e1dac3` | 10 | 0.7440 | 0.7110 | 0.7010 | 0.0266 |

## Cross-Seed Convergence (Mean/Std/CI)

Confidence band is computed as `mean ± 1.96 * (std/sqrt(3))` per epoch.

| Epoch | Mean Test Acc | Std | 95% CI Low | 95% CI High | Mean Test F1 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.7217 | 0.0047 | 0.7163 | 0.7270 | 0.7147 |
| 2 | 0.7223 | 0.0266 | 0.6921 | 0.7524 | 0.7170 |
| 3 | 0.6979 | 0.0147 | 0.6812 | 0.7145 | 0.6751 |
| 4 | 0.7055 | 0.0071 | 0.6974 | 0.7135 | 0.6847 |
| 5 | 0.6695 | 0.0062 | 0.6625 | 0.6765 | 0.6506 |
| 6 | 0.6865 | 0.0382 | 0.6433 | 0.7298 | 0.6728 |
| 7 | 0.7079 | 0.0336 | 0.6699 | 0.7460 | 0.6862 |
| 8 | 0.7001 | 0.0195 | 0.6781 | 0.7222 | 0.6745 |
| 9 | 0.6997 | 0.0325 | 0.6629 | 0.7366 | 0.6752 |
| 10 | 0.7037 | 0.0294 | 0.6705 | 0.7369 | 0.6949 |
| 11 | 0.7262 | 0.0096 | 0.7153 | 0.7371 | 0.7165 |
| 12 | 0.7393 | 0.0219 | 0.7145 | 0.7641 | 0.7357 |

## Key Outcomes

- Best mean test accuracy occurred at epoch `12`: `0.7393`.
- Final epoch mean test accuracy: `0.7393` with std `0.0219`.
- Final 95% confidence band (normal approx, n=3): `[0.7145, 0.7641]`.
- Cross-seed variability is moderate and largest around epochs `6-10`.

## Output Artifacts

- Per-seed mapping: `output/cells/cell_circuit_3seed_runs.txt`
- Per-row raw extraction (test split): `output/cells/cell_circuit_3seed_test_raw.csv`
- Per-seed summary: `output/cells/cell_circuit_3seed_seed_summary.csv`
- Epoch confidence bands: `output/cells/cell_circuit_3seed_epoch_bands.csv`

