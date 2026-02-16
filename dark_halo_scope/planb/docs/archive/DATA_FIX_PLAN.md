# Data Fix Plan: Core-Neutral Arc Injection

**Date:** 2026-02-06
**Status:** In Progress
**Goal:** Eliminate core brightness shortcut (Core LR AUC: 0.9497 → < 0.65)

---

## Problem Statement

The current arc injection process adds lensed arc flux that overlaps with the core region of the lens galaxy. This creates a systematic brightness difference:

- **Lens image** = galaxy + injected arc → brighter core
- **Non-lens image** = galaxy only → normal core

A simple logistic regression on the central 10×10 pixels achieves **AUC = 0.9497**, indicating 95% of the classification signal is in the core brightness shortcut.

---

## Root Cause Analysis

### Why arc flux ends up in the core

1. **Small Einstein radius (θ_E)**: For θ_E < 1", the arc images are close to the lens center
2. **Extended source profiles**: Sersic profiles with larger reff create broader lensed images
3. **PSF convolution**: Even arcs that avoid the core get spread into it by PSF
4. **Source position**: Sources near the lens center produce more centrally concentrated images

### Evidence from prior analysis

| θ_E Range | % Arc Flux in Core |
|-----------|-------------------|
| < 0.75" | 41% |
| 0.75-1" | 30% |
| 1-1.5" | 21% |
| 1.5-2" | 15% |
| >= 2" | 7.7% |

The shortcut is strongest for small θ_E lenses.

---

## Proposed Fix: Core-Neutral Injection

### Strategy

After rendering the lensed arc (including PSF convolution), explicitly zero out the central pixels before adding to the background galaxy.

```python
# After rendering arc and PSF convolution:
add_b = render_lensed_source(...)

# Apply core mask
half = stamp_size // 2
y, x = np.ogrid[:stamp_size, :stamp_size]
r2 = (x - half)**2 + (y - half)**2
core_radius_pix = args.core_mask_radius  # e.g., 5 pixels = 1.3 arcsec
core_mask = r2 < core_radius_pix**2
add_b[core_mask] = 0.0

# Then add to background
imgs[b] = (imgs[b] + add_b).astype(np.float32)
```

### Design Choices

| Choice | Decision | Rationale |
|--------|----------|-----------|
| **Where to mask** | After PSF convolution | Ensures no PSF-spread light enters core |
| **Mask shape** | Circular | Rotationally invariant |
| **Radius** | Configurable (default 5 pix = 1.3") | Can tune based on gate results |
| **Flux redistribution** | None (just zero out) | Simple, explicit, reviewers understand it |
| **Apply to both lens/ctrl** | Lens only | Ctrl has no arc, no need to mask |

### Trade-offs

**Pros:**
- Simple and explicit
- Directly addresses the shortcut
- Reviewers will understand what we did
- Preserves arc morphology in the annulus

**Cons:**
- May look artificial (sharp cutoff)
- Reduces signal for small θ_E lenses
- May affect detectability of compact arcs

### Validation Gate

After implementing the fix:
- Regenerate pilot dataset (20k pairs)
- Compute Core LR AUC
- **Target: Core LR AUC < 0.65**

---

## Implementation Plan

### Phase 1: Modify Pipeline (1-2 hours)

1. Add `--core-mask-radius` argument to pipeline
2. Add `mask_core_flux()` helper function
3. Apply mask after `render_lensed_source()` call
4. Update documentation

### Phase 2: Pilot Validation (4-6 hours)

1. Generate 20k pairs with core masking (r=5 pixels)
2. Compute Core LR AUC on the new data
3. If gate fails, try larger radius (r=7 or r=10)
4. Iterate until gate passes

### Phase 3: Full Regeneration (8-12 hours)

Once pilot passes:
1. Regenerate full training set (~250k pairs)
2. Regenerate validation and test sets
3. Verify split integrity
4. Upload to S3

### Phase 4: Retrain (12-24 hours)

1. Train new baseline on fixed data
2. Verify all gates pass
3. Run ablation comparisons

---

## Code Changes

### File: `spark_phase4_pipeline_gen5.py`

#### 1. Add argument parser option

```python
# Around line 150 (in argument parsing section)
parser.add_argument(
    "--core-mask-radius",
    type=int,
    default=0,
    help="Radius in pixels to zero out from arc center (0=disabled). "
         "Use 5-10 to mitigate core brightness shortcut."
)
```

#### 2. Add helper function

```python
# Around line 650 (before render_lensed_source)
def mask_core_flux(img: np.ndarray, core_radius_pix: int) -> np.ndarray:
    """
    Zero out flux in the central circular region.
    
    This prevents arc flux from contributing to core brightness,
    mitigating the core brightness shortcut in lens classification.
    
    Args:
        img: 2D image array
        core_radius_pix: Radius in pixels to mask
        
    Returns:
        Image with central pixels zeroed
    """
    if core_radius_pix <= 0:
        return img
    
    ny, nx = img.shape
    center_y, center_x = ny // 2, nx // 2
    y, x = np.ogrid[:ny, :nx]
    r2 = (x - center_x)**2 + (y - center_y)**2
    core_mask = r2 < core_radius_pix**2
    
    img_masked = img.copy()
    img_masked[core_mask] = 0.0
    return img_masked
```

#### 3. Apply mask at injection site

```python
# Around line 2680 (after render_lensed_source call)
add_b = render_lensed_source(...)

# Apply core masking if enabled
if args.core_mask_radius > 0:
    add_b = mask_core_flux(add_b, args.core_mask_radius)

imgs[b] = (imgs[b] + add_b).astype(np.float32)
```

---

## Alternative Fixes (If Core Mask Insufficient)

### Alt 1: Raise minimum θ_E

Only train on θ_E > 1.0" (larger Einstein radii where core flux is lower).

**Pros:** Naturally avoids problematic regime
**Cons:** Limits what lenses we can detect

### Alt 2: Smooth transition (apodization)

Instead of hard cutoff, use a smooth transition:

```python
# Apodization window
r = np.sqrt(r2)
window = 0.5 * (1 + np.tanh((r - core_radius_pix) / taper_width))
add_b = add_b * window
```

**Pros:** Less artificial looking
**Cons:** Still some core flux leakage

### Alt 3: Core-matched negatives

Add negatives that have similar core brightness (but no arcs).

**Pros:** Forces model to distinguish arc structure from brightness
**Cons:** Doesn't fix the underlying data bias

---

## Success Criteria

| Metric | Current | Target | Method |
|--------|---------|--------|--------|
| Core LR AUC | 0.9497 | < 0.65 | LR on central 10×10 pixels |
| Core masked drop | 0.08% | < 10% | AUROC with masked vs unmasked |
| AUROC (synthetic) | 0.99 | > 0.85 | Standard test set evaluation |

---

## Timeline

| Task | ETA | Status |
|------|-----|--------|
| Implement code changes | Today | 🔄 In Progress |
| Generate pilot (20k) | Today + 4h | ⏳ Pending |
| Validate Core LR AUC | Today + 6h | ⏳ Pending |
| Full regeneration | Tomorrow | ⏳ Pending |
| Retrain baseline | Tomorrow + 12h | ⏳ Pending |

---

## Notes

- The current ablation runs will complete on the OLD data (with shortcut)
- These provide a "before" comparison for the paper
- After data fix, we retrain to show "after" improvement
- This is valuable documentation of the shortcut problem and solution
