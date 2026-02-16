# Reply: Training LRG Sampling Details

Thank you for the thorough analysis. Here are the full details of how training LRGs are sampled, with code and hyperparameters.

---

## 1. Training LRG Selection Pipeline

### Phase 2: LRG Candidate Selection

LRGs are selected from DR10 South sweeps using color-magnitude cuts. Multiple variants are defined:

```python
# From spark_phase3_pipeline.py
LRG_VARIANTS = {
    "v1_pure_massive":    {"z_max": 20.0, "rz_min": 0.5, "zw1_min": 1.6},
    "v2_baseline_dr10":   {"z_max": 20.4, "rz_min": 0.4, "zw1_min": 1.6},
    "v3_color_relaxed":   {"z_max": 20.4, "rz_min": 0.4, "zw1_min": 0.8},  # USED
    "v4_mag_relaxed":     {"z_max": 21.0, "rz_min": 0.4, "zw1_min": 0.8},
    "v5_very_relaxed":    {"z_max": 21.5, "rz_min": 0.3, "zw1_min": 0.8},
}
```

**Gen5 uses `v3_color_relaxed`** (via `--require-lrg-flag is_v3_color_relaxed`).

### Phase 3: Region Selection and Parent Sample

1. **3a**: Compute region metrics for all connected components in DR10 South
2. **3b**: Select top-K regions by various ranking modes (density, n_lrg, area_weighted, psf_weighted)
3. **3c**: Build parent catalog by scanning DR10 South sweeps for selected regions

**Result**: ~145,000 unique LRG targets

---

## 2. Manifest Generation (Phase 4a)

The manifest is built from the existing Phase 4a SOTA manifest:

```
s3://darkhaloscope/phase4_pipeline/phase4a/v4_sota_moffat/manifests/train_stamp64_bandsgrz_gridgrid_sota
```

### Manifest Schema (relevant columns)

| Column | Description |
|--------|-------------|
| `ra`, `dec` | Sky coordinates (J2000) |
| `brickname` | DR10 brick identifier |
| `psfsize_r` | PSF FWHM in r-band (arcsec) |
| `psfdepth_r` | 5σ point source depth in r-band (nMgy^-2) |
| `region_split` | train/val/test assignment (hash-based) |
| `selection_set_id` | Which region ranking mode produced this sample |
| `is_control` | 0 = injection, 1 = control (no lens) |
| `theta_e_arcsec` | Einstein radius (injection parameter) |
| `src_dmag` | Source magnitude relative to lens (injection) |
| `src_reff_arcsec` | Source effective radius (injection) |
| `src_e` | Source ellipticity (injection) |
| `shear` | External shear magnitude (injection) |

### Manifest Statistics

| Metric | Value |
|--------|-------|
| Total rows | ~12 million |
| Unique bricks | ~133,000 |
| Unique (ra, dec) pairs | ~145,000 LRGs |
| Selection sets | 12 (4 ranking modes × 3 variants) |
| Train/Val/Test split | ~80% / ~10% / ~10% (hash-based) |

---

## 3. Injection Grid Configuration

For Gen5 training, we use `grid_small` (default for `--grid-train`):

```python
# From spark_phase4_pipeline_gen5.py lines 387-392
def _get_config_grid(name: str) -> List[InjectionConfig]:
    if name == "grid_small":
        theta = [0.3, 0.6, 1.0]          # Einstein radius (arcsec)
        dmag = [1.0, 2.0]                 # Source mag offset from lens
        reff = [0.08, 0.15]               # Source effective radius (arcsec)
        e = [0.0, 0.3]                    # Source ellipticity
        shear = [0.0, 0.03]               # External shear
    # ... cross product generates 3×2×2×2×2 = 48 configs
```

### Full Grid Used (grid_small)

| Parameter | Values | Count |
|-----------|--------|-------|
| `theta_e_arcsec` | [0.3, 0.6, 1.0] | 3 |
| `src_dmag` | [1.0, 2.0] | 2 |
| `src_reff_arcsec` | [0.08, 0.15] | 2 |
| `src_e` | [0.0, 0.3] | 2 |
| `shear` | [0.0, 0.03] | 2 |
| **Total configs** | | **48** |

---

## 4. Sampling Strategy (Phase 4a Manifest)

### Train Tier Sampling Logic

```python
# From spark_phase4_pipeline_gen5.py lines 1624-1677

# Key parameters
n_total_train_per_split = 200000  # Samples per (selection_set, split)
control_frac_train = 0.50         # 50% controls (no lens)
unpaired_controls = 1             # Controls from DIFFERENT galaxy positions
replicates = 2                    # 2 replicates per config
split_seed = 1337                 # Deterministic hash seed

# Stratified sampling across PSF/depth bins (4×4 = 16 bins)
# This ensures uniform coverage across observing conditions
wbin = Window.partitionBy("selection_set_id", "region_split", "psf_bin", "depth_bin") \
             .orderBy(F.xxhash64(F.col("row_id"), F.lit(split_seed)))
per_bin = ceil(n_total_train_per_split / 16)

# Each LRG gets ONE random config (not all configs like grid tier)
cfg_rand = F.xxhash64(F.col("row_id"), F.lit(split_seed + 5001))
cfg_idx = F.pmod(cfg_rand, F.lit(n_cfg))  # Random config assignment

# Control assignment: 50% become controls, 50% get lens injections
ctrl_hash = F.xxhash64(F.col("row_id"), F.col("brickname"), 
                       F.col("region_split"), F.lit("train"), 
                       F.lit(split_seed + 7003))
is_control = (ctrl_hash % 1000000) / 1000000.0 < control_frac
```

### Train/Val/Test Split (Hash-Based)

```python
# Deterministic split using brickname hash
split_hash = F.xxhash64(F.col("brickname"), F.lit(split_seed))
split_u = (F.pmod(split_hash, F.lit(1000000)) / 1000000.0)

# Distribution: 80% train, 10% val, 10% test
# train: split_u < 0.80
# val:   0.80 <= split_u < 0.90  
# test:  split_u >= 0.90
```

---

## 5. Phase 4c Processing (Injection & Image Generation)

### Full Command Used

```bash
spark-submit spark_phase4_pipeline_gen5.py \
  --stage 4c \
  --output-s3 s3://darkhaloscope/phase4_pipeline \
  --variant cosmos_corrected \
  --experiment-id train_stamp64_bandsgrz_cosmos_corrected \
  --parent-s3 s3://darkhaloscope/phase4_pipeline/phase4a/v4_sota_moffat/manifests/train_stamp64_bandsgrz_gridgrid_sota \
  --coadd-s3-cache-prefix s3://darkhaloscope/dr10/coadd_cache \
  --bands g,r,z \
  --source-mode cosmos \
  --cosmos-bank-h5 s3://darkhaloscope/data/cosmos_banks/cosmos_bank_20k_gen5.h5 \
  --cosmos-salt production_v1 \
  --psf-model moffat \
  --moffat-beta 3.5 \
  --sweep-partitions 600 \
  --skip-if-exists 1
```

### Key Phase 4c Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--source-mode` | `cosmos` | Use COSMOS real galaxy templates |
| `--cosmos-bank-h5` | `cosmos_bank_20k_gen5.h5` | 20k galaxy HDF5 bank |
| `--psf-model` | `moffat` | Realistic PSF model |
| `--moffat-beta` | `3.5` | Moffat shape parameter |
| `--bands` | `g,r,z` | 3-band imaging |
| `stamp_size` | `64×64` | Stamp dimensions (pixels) |
| `pixscale` | `0.262 arcsec/pix` | DR10 pixel scale |

---

## 6. Source Brightness Calibration (Current)

### How Source Magnitude is Computed

```python
# From spark_phase4_pipeline_gen5.py
# Source magnitude = lens_r_mag + src_dmag

# Example: If lens has r=19.0 and src_dmag=1.5
# Source r_mag = 19.0 + 1.5 = 20.5 (source is 1.5 mag fainter than lens)

# Convert to flux (nanomaggies)
src_flux_nmgy = 10.0 ** ((22.5 - src_r_mag) / 2.5)
```

### Current src_dmag Distribution

| src_dmag | Meaning | Approximate arc brightness |
|----------|---------|---------------------------|
| 1.0 | Source 1 mag fainter than lens | Bright arc |
| 2.0 | Source 2 mag fainter than lens | Moderate arc |

**Note**: This is the current calibration. The LLM suggested recalibrating based on Tier-A anchor arc_snr distribution.

---

## 7. COSMOS Source Template Selection

### COSMOS Bank Structure

| Property | Value |
|----------|-------|
| Source | GalSim COSMOS RealGalaxy catalog |
| N galaxies | ~20,000 (curated subset) |
| Resolution | 0.03 arcsec/pixel (HST/ACS) |
| Band | F814W (I-band) |
| Storage | HDF5 with per-galaxy arrays |

### Template Selection Logic

```python
# Deterministic selection based on task_id
cosmos_idx = hash(task_id + cosmos_salt) % n_cosmos_templates
template = cosmos_bank[cosmos_idx]

# Template is unit-flux normalized
# Scaled to target flux during injection:
src_surface_brightness = template * src_flux_nmgy / (cosmos_pixscale**2)
```

---

## 8. Response to LLM's Specific Questions

### "How training LRG is sampled"

**Answer**: 
1. LRGs are selected from DR10 South sweeps using `v3_color_relaxed` cuts (z_max=20.4, r-z>0.4)
2. ~145,000 unique LRG positions are identified in Phase 3c parent sample
3. For training, 200,000 samples per (selection_set, split) are drawn with stratification across 16 PSF/depth bins
4. 50% become controls (no injection), 50% get random injection configs from `grid_small`
5. Split is deterministic (hash-based on brickname)

### "What are the hyperparameters and configs passed"

See sections 3, 4, and 5 above for complete parameter listing.

---

## 9. Questions for LLM

### Q1: Is the current src_dmag range appropriate?

Current: [1.0, 2.0] mag fainter than lens
- This produces relatively bright arcs
- Should we extend to [0.5, 1.0, 1.5, 2.0, 2.5, 3.0] to match fainter Tier-A anchors?

### Q2: Is the theta_e range correct for ground-based detection?

Current grid_small: [0.3, 0.6, 1.0] arcsec
- Median PSF FWHM in DR10 South: ~1.2 arcsec
- theta_e/PSF for smallest: 0.3/1.2 = 0.25 (marginally detectable)
- Should we shift to [0.5, 0.75, 1.0, 1.25, 1.5] for better detectability coverage?

### Q3: Source effective radius range?

Current: [0.08, 0.15] arcsec
- COSMOS galaxies have intrinsic HLR of 0.1-0.5 arcsec typically
- Should we expand to [0.05, 0.10, 0.15, 0.25] to cover more morphologies?

### Q4: Control fraction 50% - is this optimal?

- Higher control fraction (60-70%) could improve specificity
- Lower (40%) gives more positive examples
- What does literature suggest for lens finder training?

### Q5: Is hash-based brickname split sufficient for region disjointness?

Current implementation uses `xxhash64(brickname)` for splits.
- This ensures different bricks go to different splits
- But adjacent bricks may share sky systematics
- Should we use contiguous sky-region blocks instead?

---

## 10. Proposed Arc Brightness Recalibration

Based on LLM's recommendation for target arc_snr distribution:

| arc_snr Range | Target Fraction | Current | Notes |
|---------------|-----------------|---------|-------|
| 0.8–2 | 40% | ~5% | Near-threshold (hard) |
| 2–8 | 40% | ~35% | Moderate |
| 8–20 | 15% | ~40% | Easy |
| 20+ | 5% | ~20% | Extreme |

**Implementation Plan**:
1. Compute arc_snr distribution of Tier-A anchors
2. Invert to find target src_dmag distribution
3. Sample src_dmag from this distribution during injection
4. Add rejection sampling if arc_snr falls outside target bins

---

## 11. Updated Priority Plan (Based on LLM Feedback)

| Priority | Task | Est. Time | Status |
|----------|------|-----------|--------|
| **1** | Center-masked ablation (quick diagnostic) | 1-2 days | PENDING |
| **2** | Build Tier-A anchors (Legacy Surveys ML candidates) | 2-3 days | PENDING |
| **3** | Hard-negative dataset (Galaxy Zoo DECaLS) | 1-2 days | PENDING |
| **4** | Injection brightness recalibration (arc_snr targeting) | 2-3 days | PENDING |
| **5** | Region-disjoint split framework (brick blocks) | 1 day | PENDING |
| **6** | Retrain Gen5' with all fixes | 2-3 days | PENDING |
| **7** | Evaluate on Tier-A + SLACS/BELLS stress test | 1 day | PENDING |

**Key Changes from Original Plan**:
- Moved center-masked ablation to #1 (fast falsification test)
- Deferred photometric jitter until after core fixes validated
- Added explicit "Tier-A first" before any retraining

---

## 12. Additional Diagnostics to Run Before Proceeding

Per LLM recommendation, before investing in new anchors:

| Check | Status | Result |
|-------|--------|--------|
| Column audit (leakage check) | ✅ DONE | PASS - no leakage |
| Stratified AUC analysis | ✅ DONE | PASS - gap explained by difficulty |
| Pipeline parity check | ✅ DONE | PASS - pipelines agree |
| bad_pixel_frac distribution | PENDING | Need to run |
| maskbit_frac distribution | PENDING | Need to run |
| invvar summaries by class | PENDING | Need to run |
| bandset consistency check | PENDING | Need to run |
| Null-injection test | PENDING | Inject zero-flux, verify classifier can't separate |

**Will run these before proceeding to Tier-A anchor building.**
