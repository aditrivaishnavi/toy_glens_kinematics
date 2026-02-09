# Shortcut Detection in Cutout Validation

## Overview

The validation pipeline computes AUC (Area Under ROC Curve) for various image features
to detect potential "shortcuts" - features that trivially distinguish positives from
negatives and might lead to a classifier that doesn't learn true lensing physics.

## Expected Physics vs Shortcuts

Not all high-AUC features are shortcuts. Some features **should** separate positives
from negatives because they reflect real physical differences:

### Expected Physics Features

These features are **expected** to have high AUC and are not considered shortcuts:

| Feature | Why It Should Separate |
|---------|----------------------|
| `azimuthal_asymmetry` | Lensed arcs are inherently azimuthally asymmetric - that's the lensing signature |
| `annulus_brightness_r` | Lens systems have arc flux in the annulus region (4-16 pixels from center) |
| `annulus_max_r` | Peak arc brightness - lenses have bright arc segments |
| `annulus_std_r` | Arc structure variability - arcs have non-uniform brightness |

### True Shortcuts (Problems)

These features would indicate data quality or selection bias issues:

| Feature | Why It's a Problem |
|---------|-------------------|
| `core_brightness_r` | Core brightness shouldn't differ (both have galaxies) |
| `core_r4/r8/r12_brightness` | Multi-radius core checks - same reasoning |
| `mad_r` | Noise level shouldn't differ between classes |
| `edge_artifact_score` | Edge artifacts indicate image quality issues |
| `radial_gradient_r` | Radial profile shouldn't systematically differ |

## AUC Thresholds

- **Green (OK)**: AUC lower CI < 0.60
- **Yellow (Warning)**: AUC lower CI > 0.60 but ≤ 0.70
- **Red (Shortcut)**: AUC lower CI > 0.70

## Gate Logic

The validation passes ("GO") if:

1. `all_positives_ok`: All positive cutouts pass quality checks
2. `negatives_98pct_ok`: ≥98% of negative cutouts pass quality checks  
3. `no_core_shortcuts`: No red flags in core brightness features
4. `no_trivial_shortcuts`: No red flags in non-physics features

Features marked as "expected physics" are excluded from shortcut counting.

## Interpreting Results

When reviewing validation results:

1. Check the `is_expected_physics` flag for each feature
2. Features with `is_expected_physics=True` and high AUC are **good** - they confirm the signal exists
3. Features with `is_expected_physics=False` and high AUC are **problems** to investigate

## Example Interpretation

```json
{
  "azimuthal_asymmetry": {
    "auc": 0.721,
    "is_red": true,
    "is_expected_physics": true,
    "is_shortcut": false  // NOT a shortcut because it's expected physics
  },
  "core_brightness_r": {
    "auc": 0.55,
    "is_red": false,
    "is_expected_physics": false,
    "is_shortcut": false  // OK - core brightness doesn't separate (as expected)
  }
}
```

## References

- Paper IV methodology: 100:1 negative:positive ratio per stratum
- LLM recommendations: Multi-radius core analysis, bootstrap CIs
