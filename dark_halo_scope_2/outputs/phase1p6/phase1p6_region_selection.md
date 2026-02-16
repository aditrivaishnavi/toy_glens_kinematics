# Phase 1.6 Region Selection Summary

This step joins EMR-derived LRG counts per brick with the DR10
bricks table, applies the Phase1p5 brick-level quality cuts, and
selects primary and backup contiguous regions using the existing
region_scout logic.

## Configuration

- TAP URL: `https://datalab.noirlab.edu/tap`
- Bricks table: `ls_dr10.bricks_s`
- Tractor table (for context): `ls_dr10.tractor_s`
- Footprint: RA [150.0, 250.0] deg, Dec [0.0, 30.0] deg
- Region area target: [100.0, 400.0] deg^2

## Brick statistics

- Total bricks in footprint: 46280
- Bricks passing quality cuts: 38138
- Bricks with nonzero LRG density: 37851

## Primary region

- Total area: 2.75 deg^2
- Total LRG count: 391
- Mean LRG surface density: 142.2 per deg^2
- Number of bricks: 44
- RA range: [232.625, 236.375] deg
- Dec range: [0.000, 0.750] deg

## Backup region

- Total area: 3.81 deg^2
- Total LRG count: 452
- Mean LRG surface density: 118.6 per deg^2
- Number of bricks: 61
- RA range: [237.375, 240.625] deg
- Dec range: [0.500, 2.500] deg
