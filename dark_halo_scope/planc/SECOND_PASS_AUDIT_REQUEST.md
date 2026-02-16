# Second-Pass Audit Request: Stratified Sampling and Cutout Validation

Following your excellent first-pass review, please audit these two additional components for correctness and Paper IV parity.

---

## 1. Stratified Sampling (`spark_stratified_sample.py`)

### Goal
Reproduce Paper IV's 100:1 negative:positive ratio per stratum, while maintaining 85:15 N1:N2 ratio within negatives.

### Current Implementation

```python
# Constants
NEG_POS_RATIO = 100  # 100 negatives per positive
N1_RATIO = 0.85  # 85% from N1 pool
N2_RATIO = 0.15  # 15% from N2 pool

# Strata definitions
NOBS_Z_BINS = ["1-2", "3-5", "6-10", "11+"]
TYPE_BINS = ["SER", "DEV", "REX", "EXP"]  # NOTE: Should this now be ["SER", "DEV", "REX"]?
```

### Sampling Logic (lines 141-175)

```python
# Compute target counts per stratum
target_counts = {}
for (nobs_bin, type_bin), pos_count in pos_counts.items():
    neg_target = pos_count * args.neg_pos_ratio
    n1_target = int(neg_target * args.n1_ratio)
    n2_target = neg_target - n1_target
    target_counts[(nobs_bin, type_bin)] = {
        "total": neg_target,
        "n1": n1_target,
        "n2": n2_target,
    }

# Count available negatives per stratum
neg_stratum_counts = negatives_df.groupBy("nobs_z_bin", "type_bin", "pool").count()

# Sample from each pool proportionally
sampled_dfs = []
for (nobs_bin, type_bin), targets in target_counts.items():
    # Sample N1
    n1_pool = negatives_df.filter(
        (F.col("nobs_z_bin") == nobs_bin) &
        (F.col("type_bin") == type_bin) &
        (F.col("pool") == "N1")
    )
    n1_available = n1_pool.count()
    n1_sample_size = min(targets["n1"], n1_available)
    
    if n1_sample_size > 0:
        fraction = n1_sample_size / n1_available
        sampled_n1 = n1_pool.sample(fraction=min(fraction * 1.1, 1.0), seed=42)
        sampled_n1 = sampled_n1.limit(n1_sample_size)
        sampled_dfs.append(sampled_n1)
    
    # Sample N2 (similar logic)
    # If N2 insufficient, backfill from N1
```

### Questions for Stratified Sampling

1. **Does Paper IV use 100:1 per stratum?** Or is it 100:1 overall with some other stratification scheme?

2. **TYPE_BINS mismatch**: The code still references `["SER", "DEV", "REX", "EXP"]` but we've now excluded EXP from N1. Should TYPE_BINS be updated to `["SER", "DEV", "REX"]`?

3. **Stratum granularity**: We stratify by `(nobs_z_bin, type_bin)` giving 4×4=16 strata (or 4×3=12 without EXP). Is this the right granularity? Should we add seeing or depth bins?

4. **N2 backfill logic**: When N2 is insufficient for a stratum, we backfill from N1. Is this the right approach, or should we oversample N2 globally regardless of stratum?

5. **Sampling seed determinism**: We use `seed=42` for reproducibility. Is this sufficient, or should we use the HEALPix-based split to ensure spatial independence?

---

## 2. Cutout Validation (`spark_validate_cutouts.py`)

### Goal
Detect data quality issues and shortcut features that could bias training.

### Current Implementation

#### Quality Validation (lines 83-130)

```python
def validate_cutout_quality(cutout: np.ndarray) -> Dict:
    h, w, c = cutout.shape
    
    # Size check
    size_ok = (h == CUTOUT_SIZE and w == CUTOUT_SIZE and c == 3)
    
    # NaN analysis
    nan_mask = np.isnan(cutout)
    total_nan_frac = np.sum(nan_mask) / cutout.size
    
    # Central region (50x50)
    center = h // 2
    margin = 25
    central = cutout[center-margin:center+margin, center-margin:center+margin, :]
    central_nan_frac = np.sum(np.isnan(central)) / central.size
    
    # Core region (16x16 around center)
    core = cutout[center-CORE_RADIUS:center+CORE_RADIUS, 
                  center-CORE_RADIUS:center+CORE_RADIUS, :]
    core_nan_frac = np.sum(np.isnan(core)) / core.size
    
    # Bands present
    bands_present = [not np.all(np.isnan(cutout[:, :, i])) for i in range(c)]
    all_bands_present = all(bands_present)
    
    # Quality gate
    quality_ok = (
        size_ok and
        central_nan_frac < MAX_NAN_FRAC_CENTER and  # 2%
        all_bands_present
    )
    
    return {
        "size_ok": size_ok,
        "total_nan_frac": float(total_nan_frac),
        "central_nan_frac": float(central_nan_frac),
        "core_nan_frac": float(core_nan_frac),
        "has_g": bands_present[0],
        "has_r": bands_present[1],
        "has_z": bands_present[2],
        "all_bands_present": all_bands_present,
        "quality_ok": quality_ok,
    }
```

#### Shortcut Feature Extraction (lines 133-200)

```python
def extract_shortcut_features(cutout: np.ndarray) -> Dict:
    h, w, c = cutout.shape
    center = h // 2
    
    # Use r-band (index 1) for brightness features
    r_band = cutout[:, :, 1]
    
    # Create distance map from center
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((y - center)**2 + (x - center)**2)
    
    # Regions
    core_mask = dist <= CORE_RADIUS  # 8 pixels
    annulus_mask = (dist >= 20) & (dist <= 40)  # where arcs appear
    outer_mask = dist >= OUTER_MARGIN  # edge pixels
    
    # Core brightness (shortcut: lenses have bright cores)
    core_pixels = r_band[core_mask]
    valid_core = core_pixels[~np.isnan(core_pixels)]
    core_brightness = float(np.median(valid_core)) if len(valid_core) > 0 else None
    core_max = float(np.max(valid_core)) if len(valid_core) > 0 else None
    
    # Annulus brightness (where lensing arcs should appear)
    annulus_pixels = r_band[annulus_mask]
    valid_annulus = annulus_pixels[~np.isnan(annulus_pixels)]
    annulus_brightness = float(np.median(valid_annulus)) if len(valid_annulus) > 0 else None
    annulus_std = float(np.std(valid_annulus)) if len(valid_annulus) > 0 else None
    
    # Outer background
    outer_pixels = r_band[outer_mask]
    valid_outer = outer_pixels[~np.isnan(outer_pixels)]
    outer_brightness = float(np.median(valid_outer)) if len(valid_outer) > 0 else None
    
    # Radial gradient (shortcut: lenses have steep gradients)
    radial_gradient = core_brightness - outer_brightness if (core_brightness and outer_brightness) else None
    
    # MAD for noise estimation
    mad_r = float(np.median(np.abs(valid_core - np.median(valid_core)))) if len(valid_core) > 0 else None
    
    # Color features (g-r color in core vs annulus)
    # ... similar logic for color gradient
```

#### AUC Computation for Shortcut Detection (lines 250-280)

```python
def compute_auc(pos_values: List[float], neg_values: List[float]) -> float:
    """Compute AUC for a single feature to detect shortcuts."""
    from sklearn.metrics import roc_auc_score
    
    labels = [1] * len(pos_values) + [0] * len(neg_values)
    values = pos_values + neg_values
    
    # Filter NaN
    valid = [(v, l) for v, l in zip(values, labels) if v is not None and not np.isnan(v)]
    if len(valid) < 10:
        return 0.5  # Not enough data
    
    values, labels = zip(*valid)
    return roc_auc_score(labels, values)

# Shortcut detection: flag features with AUC > 0.70
shortcut_features = {}
for feature in ["core_brightness", "radial_gradient", "core_max", "annulus_std"]:
    pos_vals = [r[feature] for r in pos_results if r.get(feature) is not None]
    neg_vals = [r[feature] for r in neg_results if r.get(feature) is not None]
    auc = compute_auc(pos_vals, neg_vals)
    shortcut_features[feature] = {
        "auc": auc,
        "is_shortcut": auc > AUC_THRESHOLD,  # 0.70
    }
```

### Questions for Cutout Validation

6. **NaN threshold**: We reject cutouts with >2% NaN in central 50×50. Is this too strict or too lenient?

7. **Band ordering verification**: We assume channel order is [g, r, z]. Should we add an explicit check (e.g., verify relative brightness or WCS metadata)?

8. **Core radius of 8 pixels**: This equals ~2.1 arcsec at 0.262"/pix. Is this appropriate for strong lens cores, or should it be larger (e.g., 4 arcsec)?

9. **Annulus range 20-40 pixels**: This equals 5.2-10.5 arcsec. Is this the right range for detecting lensing arcs? Paper IV may use different radii.

10. **AUC threshold of 0.70**: We flag features as shortcuts if AUC > 0.70. Is this threshold appropriate? Should it be 0.60 or 0.75?

11. **Missing shortcut features**: Should we also check:
    - Azimuthal symmetry (rings vs arcs)
    - Color gradients (arcs are often blue)
    - Edge sharpness / contrast
    - PSF concentration

12. **Normalization validation**: We don't currently validate that cutouts are consistently normalized. Should we add checks for:
    - Consistent flux scale across bands
    - No extreme outliers (e.g., saturated pixels)
    - Proper background level

---

## Summary of Questions (Q1-12)

| # | Topic | Question |
|---|-------|----------|
| 1 | Stratified | Does Paper IV use 100:1 per stratum? |
| 2 | Stratified | Should TYPE_BINS exclude EXP now? |
| 3 | Stratified | Is (nobs_z, type) the right stratification? |
| 4 | Stratified | Is N2 backfill from N1 correct? |
| 5 | Stratified | Is seed=42 sufficient for reproducibility? |
| 6 | Validation | Is 2% NaN threshold appropriate? |
| 7 | Validation | Should we verify band ordering explicitly? |
| 8 | Validation | Is 8-pixel core radius appropriate? |
| 9 | Validation | Is 20-40 pixel annulus range correct for arcs? |
| 10 | Validation | Is AUC > 0.70 the right shortcut threshold? |
| 11 | Validation | What shortcut features are we missing? |
| 12 | Validation | Should we validate normalization consistency? |

---

## Requested Output

1. **Stratified sampling audit**: Confirm the 100:1 logic matches Paper IV, or specify corrections.

2. **TYPE_BINS fix**: Confirm whether to remove EXP from the strata list.

3. **Validation thresholds**: For each threshold (NaN, core radius, annulus, AUC), confirm or provide Paper IV values.

4. **Missing QC checks**: List any critical quality checks we're missing.

5. **Shortcut detection improvements**: Recommend additional features to monitor.
