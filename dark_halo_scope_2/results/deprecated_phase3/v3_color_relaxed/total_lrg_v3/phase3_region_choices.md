# Phase 3 Region Choices – total_lrg_v3


This file documents the automatic selection of regions for Phase 3.

- Variant: `v3_color_relaxed`
- Ranking mode: `total_lrg_v3`
- Number of regions requested: 5

## Selected regions

| phase3_region_rank | region_id | total_area_deg2 | mean_lrg_density_v3_color_relaxed | total_n_lrg_v3_color_relaxed | phase3_score |
| --- | --- | --- | --- | --- | --- |
| 1.000 | 413.000 | 0.998 | 2425.591 | 2421.000 | 2421.000 |
| 2.000 | 2303.000 | 0.562 | 2329.477 | 1309.000 | 1309.000 |
| 3.000 | 396.000 | 0.437 | 2423.533 | 1059.000 | 1059.000 |
| 4.000 | 884.000 | 0.374 | 2430.216 | 910.000 | 910.000 |
| 5.000 | 1856.000 | 0.375 | 2325.325 | 871.000 | 871.000 |

## Notes

- `phase3_region_rank = 1` is the highest-scoring region for this ranking mode.
- In later phases, you can compare performance as a function of rank (e.g., does including rank 5 actually help the model?).
