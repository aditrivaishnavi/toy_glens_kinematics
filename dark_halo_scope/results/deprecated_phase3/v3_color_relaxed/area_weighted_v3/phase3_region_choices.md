# Phase 3 Region Choices – area_weighted_v3


This file documents the automatic selection of regions for Phase 3.

- Variant: `v3_color_relaxed`
- Ranking mode: `area_weighted_v3`
- Number of regions requested: 5

## Selected regions

| phase3_region_rank | region_id | total_area_deg2 | mean_lrg_density_v3_color_relaxed | total_n_lrg_v3_color_relaxed | phase3_score |
| --- | --- | --- | --- | --- | --- |
| 1.000 | 413.000 | 0.998 | 2425.591 | 2421.000 | 2423.294 |
| 2.000 | 2303.000 | 0.562 | 2329.477 | 1309.000 | 1746.220 |
| 3.000 | 396.000 | 0.437 | 2423.533 | 1059.000 | 1602.037 |
| 4.000 | 1034.000 | 0.312 | 2677.551 | 836.000 | 1496.139 |
| 5.000 | 884.000 | 0.374 | 2430.216 | 910.000 | 1487.110 |

## Notes

- `phase3_region_rank = 1` is the highest-scoring region for this ranking mode.
- In later phases, you can compare performance as a function of rank (e.g., does including rank 5 actually help the model?).
