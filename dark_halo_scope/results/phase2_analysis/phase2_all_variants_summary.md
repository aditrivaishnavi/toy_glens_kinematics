# Phase 2 Hypergrid Analysis - All Variants Summary

## Analysis Parameters

- **Hypergrid CSV**: `results/phase2_results.csv`
- **Output directory**: `results/phase2_analysis`
- **Bricks source**: TAP (https://datalab.noirlab.edu/tap)
- **Percentiles**: (90.0, 95.0, 99.0)
- **Min region area**: 2.0 deg²
- **Max region area**: 400.0 deg²
- **RA/Dec filter**: None (full sky)

## LRG Selection Variants

| Variant | z_mag_max | r-z min | z-W1 min | Description |
|---------|-----------|---------|----------|-------------|
| v1_pure_massive | 20.0 | 0.5 | 1.6 | Strictest cut: pure massive LRGs only |
| v2_baseline_dr10 | 20.4 | 0.4 | 1.6 | Baseline DR10 LRG selection |
| v3_color_relaxed | 20.4 | 0.4 | 0.8 | Relaxed z-W1 color cut |
| v4_mag_relaxed | 21.0 | 0.4 | 0.8 | Deeper magnitude limit (z < 21) |
| v5_very_relaxed | 21.5 | 0.3 | 0.8 | Most inclusive: faint + relaxed colors |

## Individual Variant Reports

- [v1_pure_massive](./v1_pure_massive/phase2_hypergrid_analysis.md)
- [v2_baseline_dr10](./v2_baseline_dr10/phase2_hypergrid_analysis.md)
- [v3_color_relaxed](./v3_color_relaxed/phase2_hypergrid_analysis.md)
- [v4_mag_relaxed](./v4_mag_relaxed/phase2_hypergrid_analysis.md)
- [v5_very_relaxed](./v5_very_relaxed/phase2_hypergrid_analysis.md)
