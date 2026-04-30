# Migration Summary: Normalization v1 → v2

## Formula change

| | v1 | v2 |
|---|---|---|
| Baseline | `progress(0)` of each individual trajectory | `progress(input_grid, target_grid)` — shared task property |
| Human zero point | After participant's 1st edit | Same as CodeIt: input_grid level |
| CodeIt zero point | Input grid | Input grid (unchanged) |
| Aggregation | Median of per-trajectory normalised curves | Same |
| IQR bands | Not stored | p25 / p75 stored per group |

## Starting-point comparison (v1 vs v2 median_curve[0])

| task_id | baseline | human_success v1→v2 | human_failed v1→v2 | codeit_success v1→v2 |
|---|---|---|---|---|
| 0c9aba6e | 0.000 | 0.000 → 0.458 | 0.000 → 0.750 | 0.000 → 0.000 |
| 195ba7dc | 0.000 | 0.000 → 0.467 | 0.000 → 0.133 | 0.000 → 0.000 |
| 009d5c81 | 0.837 | 0.000 → -5.125 | 0.000 → -2.469 | 0.000 → 0.000 |
| 00576224 | 0.000 | 0.000 → 0.000 | 0.000 → 0.694 | 0.000 → 0.000 |
| 3194b014 | 0.000 | 0.000 → 0.000 | 0.000 → 0.000 | 0.000 → 0.000 |

## Interpretation of v2 starting points

- **CodeIt always starts at 0**: first trace grid is the input_grid, so norm=0 by construction.
- **Human start > 0**: participant's blank output is partially correct (target has background-colored cells).
- **Human start < 0**: participant's 1st edit was worse than the input grid (rare).
- **AUC may exceed v1 values** for human groups on tasks with high blank-output overlap.

## Downstream impact

Scripts 07 and 08 updated to read `progress_curves_v2.json` with new key structure.