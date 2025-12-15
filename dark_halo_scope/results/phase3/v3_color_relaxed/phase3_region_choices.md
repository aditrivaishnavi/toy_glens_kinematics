# Phase 3 - Field definition


This document records the exact fields used to build the Phase 3 LRG parent catalog.
The selection is based on the Phase 2 v3_color_relaxed hypergrid analysis.

## Inputs
- Regions summary CSV: `results/phase2_analysis/v3_color_relaxed/phase2_regions_summary.csv`
- Regions bricks CSV: `results/phase2_analysis/v3_color_relaxed/phase2_regions_bricks.csv`

## Selected regions

| region_id | n_bricks | area_deg2 | center_ra_deg | center_dec_deg | lrg_density_v3 | notes |
|-----------|----------|-----------|----------------|-----------------|----------------|-------|
| 413 | 16 | 0.9981 | 35.805 | 19.281 | 2425.6 | Primary Phase 3 field |
| 2303 | 9 | 0.5619 | 322.161 | 12.445 | 2329.5 | Secondary or comparison field |

## Aggregate brick coverage for Phase 3

- Total bricks across all selected regions: 25
- Total geometric area (sum of brick areas): 1.560 deg^2
- RA range: [35.177, 322.436] deg
- Dec range: [11.750, 20.000] deg

These RA and Dec ranges define the approximate Phase 3 footprint.
In the parent catalog builder, I restrict DR10 sweeps to this footprint to avoid unnecessary input and output outside the Phase 3 fields.

## Notes on scope and assumptions

- Phase 3 uses only the DR10 South footprint, as in Phase 2.
- Parent LRGs are defined using the v3_color_relaxed cuts as implemented in Phase 2.
- I retain region IDs so that later phases can compare results across fields.
- This document is meant to be pasted directly into the research log as a record of exact field choices before simulation and training.