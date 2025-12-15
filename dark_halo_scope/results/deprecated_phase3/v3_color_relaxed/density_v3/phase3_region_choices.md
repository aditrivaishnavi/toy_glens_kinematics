# Phase 3 Region Choices – density_v3


This file documents the automatic selection of regions for Phase 3.

- Variant: `v3_color_relaxed`
- Ranking mode: `density_v3`
- Number of regions requested: 5

## Selected regions

| phase3_region_rank | region_id | total_area_deg2 | mean_lrg_density_v3_color_relaxed | total_n_lrg_v3_color_relaxed | phase3_score |
| --- | --- | --- | --- | --- | --- |
| 1.000 | 1635.000 | 0.062 | 4388.646 | 274.000 | 4388.646 |
| 2.000 | 25.000 | 0.125 | 4041.283 | 504.000 | 4041.283 |
| 3.000 | 1283.000 | 0.059 | 3889.405 | 229.000 | 3889.405 |
| 4.000 | 2179.000 | 0.062 | 3885.215 | 240.000 | 3885.215 |
| 5.000 | 1638.000 | 0.060 | 3844.972 | 232.000 | 3844.972 |

## Notes

- `phase3_region_rank = 1` is the highest-scoring region for this ranking mode.
- In later phases, you can compare performance as a function of rank (e.g., does including rank 5 actually help the model?).
