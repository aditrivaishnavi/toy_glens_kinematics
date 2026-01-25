# S3 Locations Reference

**Project**: Dark Halo Scope  
**Last Updated**: 2026-01-25  
**Author**: Aditrivaishnavi Balaji

This document provides a complete reference of all S3 paths used in the Dark Halo Scope pipeline, with clear guidance on which paths to use for downstream stages.

---

## Quick Reference: Paths for Phase 4c and Beyond

### USE THESE PATHS

| Purpose | S3 Path |
|---------|---------|
| Task Manifests | `s3://darkhaloscope/phase4_pipeline/phase4a/v3_color_relaxed/manifests_filtered/` |
| Bricks Manifest | `s3://darkhaloscope/phase4_pipeline/phase4a/v3_color_relaxed/bricks_manifest_filtered/` |
| Coadd Cache | `s3://darkhaloscope/dr10/coadd_cache/` |
| Stage 4a Config | `s3://darkhaloscope/phase4_pipeline/phase4a/v3_color_relaxed/_stage_config.json` |

### DO NOT USE THESE PATHS (contain 221 bad bricks)

| Purpose | S3 Path | Reason |
|---------|---------|--------|
| Original Manifests | `s3://darkhaloscope/phase4_pipeline/phase4a/v3_color_relaxed/manifests/` | Contains tasks for 221 non-existent bricks |
| Original Bricks Manifest | `s3://darkhaloscope/phase4_pipeline/phase4a/v3_color_relaxed/bricks_manifest/` | Contains 221 non-existent bricks |

---

## Phase 3 Outputs

### Phase 3a: Region Metrics

| Path | Purpose | Status |
|------|---------|--------|
| `s3://darkhaloscope/phase3_pipeline/phase3a/v3_color_relaxed/bricks_with_region/` | Brick-level metadata (PSF, depth, EBV, region assignment) | Complete |
| `s3://darkhaloscope/phase3_pipeline/phase3a/v3_color_relaxed/region_metrics/` | Region-level aggregated metrics | Complete |
| `s3://darkhaloscope/phase3_pipeline/phase3a/v3_color_relaxed/region_metrics_csv/` | CSV export of region metrics | Complete |

### Phase 3b: Region Selections

| Path | Purpose | Status |
|------|---------|--------|
| `s3://darkhaloscope/phase3_pipeline/phase3b/v3_color_relaxed/region_selections/` | Region selection rankings by different strategies | Complete |
| `s3://darkhaloscope/phase3_pipeline/phase3b/v3_color_relaxed/_stage_config.json` | Stage 3b configuration (1523 bytes) | Complete |

### Phase 3c: Parent Catalog

| Path | Purpose | Status |
|------|---------|--------|
| `s3://darkhaloscope/phase3_pipeline/phase3c/v3_color_relaxed/parent_union_parquet/` | Raw parent catalog (many small files) | Complete |
| `s3://darkhaloscope/phase3_pipeline/phase3c/v3_color_relaxed/_stage_config.json` | Stage 3c configuration (1154 bytes) | Complete |

### Phase 3.5: Compacted Parent Catalog

| Path | Purpose | Status |
|------|---------|--------|
| `s3://darkhaloscope/phase3_pipeline/phase3p5/v3_color_relaxed/parent_compact/` | Compacted parent catalog (291 files, 1.34 GB) | Complete |

---

## Phase 4a Outputs: Task Manifests

### Configuration and Metadata

| Path | Size | Purpose |
|------|------|---------|
| `s3://.../phase4a/v3_color_relaxed/_stage_config.json` | 1414 bytes | Stage 4a parameters (seed, replicates, control fractions) |
| `s3://.../phase4a/v3_color_relaxed/_stage_config.json.bak.2026-01-24` | 1367 bytes | Backup of previous config |
| `s3://.../phase4a/v3_color_relaxed/brick_blacklist.json` | 3608 bytes | 221 bricks that don't exist on NERSC |

### Original Manifests (DO NOT USE FOR 4c)

These contain tasks for 221 bricks that don't exist on NERSC.

| Path | Experiment | Status |
|------|------------|--------|
| `s3://.../phase4a/v3_color_relaxed/bricks_manifest/` | 180,373 unique bricks | Has 221 bad bricks |
| `s3://.../phase4a/v3_color_relaxed/manifests/debug_stamp64_bandsgrz_gridgrid_small/` | Debug tier, 64px stamps | Has bad bricks |
| `s3://.../phase4a/v3_color_relaxed/manifests/debug_stamp96_bandsgrz_gridgrid_small/` | Debug tier, 96px stamps | Has bad bricks |
| `s3://.../phase4a/v3_color_relaxed/manifests/grid_stamp64_bandsgrz_gridgrid_medium/` | Grid tier, 64px stamps | Has bad bricks |
| `s3://.../phase4a/v3_color_relaxed/manifests/grid_stamp96_bandsgrz_gridgrid_medium/` | Grid tier, 96px stamps | Has bad bricks |
| `s3://.../phase4a/v3_color_relaxed/manifests/train_stamp64_bandsgrz_gridgrid_small/` | Train tier, 64px stamps | Has bad bricks |
| `s3://.../phase4a/v3_color_relaxed/manifests/train_stamp96_bandsgrz_gridgrid_small/` | Train tier, 96px stamps | Has bad bricks |

### Filtered Manifests (USE THESE FOR 4c)

These have the 221 bad bricks removed.

| Path | Experiment | Status |
|------|------------|--------|
| `s3://.../phase4a/v3_color_relaxed/bricks_manifest_filtered/` | 180,152 valid bricks | USE THIS |
| `s3://.../phase4a/v3_color_relaxed/manifests_filtered/debug_stamp64_bandsgrz_gridgrid_small/` | Debug tier, 64px stamps | USE THIS |
| `s3://.../phase4a/v3_color_relaxed/manifests_filtered/debug_stamp96_bandsgrz_gridgrid_small/` | Debug tier, 96px stamps | USE THIS |
| `s3://.../phase4a/v3_color_relaxed/manifests_filtered/grid_stamp64_bandsgrz_gridgrid_medium/` | Grid tier, 64px stamps | USE THIS |
| `s3://.../phase4a/v3_color_relaxed/manifests_filtered/grid_stamp96_bandsgrz_gridgrid_medium/` | Grid tier, 96px stamps | USE THIS |
| `s3://.../phase4a/v3_color_relaxed/manifests_filtered/train_stamp64_bandsgrz_gridgrid_small/` | Train tier, 64px stamps | USE THIS |
| `s3://.../phase4a/v3_color_relaxed/manifests_filtered/train_stamp96_bandsgrz_gridgrid_small/` | Train tier, 96px stamps | USE THIS |

---

## Phase 4b Outputs: Coadd Cache

### Coadd Cache Location

```
s3://darkhaloscope/dr10/coadd_cache/
```

**Total Brick Directories**: 180,152

### Files Per Brick (11 total)

| File | Purpose | Added By |
|------|---------|----------|
| `_SUCCESS` | Brick cache completion marker | Phase 4b |
| `legacysurvey-{brick}-image-g.fits.fz` | g-band science image (~11-15 MB) | Phase 4b |
| `legacysurvey-{brick}-image-r.fits.fz` | r-band science image (~11-15 MB) | Phase 4b |
| `legacysurvey-{brick}-image-z.fits.fz` | z-band science image (~11-15 MB) | Phase 4b |
| `legacysurvey-{brick}-invvar-g.fits.fz` | g-band inverse variance (~11 MB) | Phase 4b |
| `legacysurvey-{brick}-invvar-r.fits.fz` | r-band inverse variance (~11 MB) | Phase 4b |
| `legacysurvey-{brick}-invvar-z.fits.fz` | z-band inverse variance (~11 MB) | Phase 4b |
| `legacysurvey-{brick}-maskbits.fits.fz` | Mask bits (~450 KB) | Phase 4b |
| `legacysurvey-{brick}-psfsize-g.fits.fz` | g-band PSF FWHM map (~400 KB) | Phase 4b2 |
| `legacysurvey-{brick}-psfsize-r.fits.fz` | r-band PSF FWHM map (~400 KB) | Phase 4b2 |
| `legacysurvey-{brick}-psfsize-z.fits.fz` | z-band PSF FWHM map (~400 KB) | Phase 4b2 |

### Phase 4b Metadata

| Path | Purpose |
|------|---------|
| `s3://.../phase4b/v3_color_relaxed/_stage_config.json` | Original Phase 4b configuration (468 bytes) |
| `s3://.../phase4b/v3_color_relaxed/assets_manifest/` | Original 4b caching manifest (7 files per brick) |

### Phase 4b2 PSFsize Repair Metadata

| Path | Purpose |
|------|---------|
| `s3://.../phase4b_psfsize/v3_color_relaxed/_psfsize_repair_config.json` | PSFsize repair configuration (717 bytes) |
| `s3://.../phase4b_psfsize/v3_color_relaxed/psfsize_repair_manifest/` | PSFsize repair manifest (400 parquet files) |

---

## Summary Statistics

| Dataset | Count |
|---------|-------|
| Phase 3 LRG Objects | 19,687,747 |
| Phase 3 Unique Regions | 811 |
| Phase 3 Unique Bricks | 256,208 |
| Phase 4a Bricks (original) | 180,373 |
| Phase 4a Bricks (filtered) | 180,152 |
| Phase 4a Blacklisted Bricks | 221 |
| Phase 4b Cached Bricks | 180,152 |
| Phase 4b Files Per Brick | 11 |
| Phase 4b Total Cached Files | ~1,981,672 |

---

## Example: Constructing Paths for a Brick

For brick `0001m002`:

```
# Coadd cache
s3://darkhaloscope/dr10/coadd_cache/0001m002/legacysurvey-0001m002-image-r.fits.fz
s3://darkhaloscope/dr10/coadd_cache/0001m002/legacysurvey-0001m002-invvar-r.fits.fz
s3://darkhaloscope/dr10/coadd_cache/0001m002/legacysurvey-0001m002-maskbits.fits.fz
s3://darkhaloscope/dr10/coadd_cache/0001m002/legacysurvey-0001m002-psfsize-r.fits.fz
```

---

## Version History

| Date | Change |
|------|--------|
| 2026-01-25 | Initial creation with complete Phase 3, 4a, 4b, 4b2 paths |

