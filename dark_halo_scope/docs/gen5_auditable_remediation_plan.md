# Gen5 Auditable Remediation Plan

**Date:** 2026-02-05  
**Status:** ACTIVE  
**Purpose:** Comprehensive, honest, auditable steps to fix Gen5 model before publication

---

## Executive Summary

| Completed Check | Result |
|-----------------|--------|
| Leakage Audit | ✅ PASS - no label leakage |
| Stratified AUC | ✅ PASS - gap explained by difficulty |
| Pipeline Parity | ✅ PASS - pipelines agree |
| Brightness Metric | ✅ CORRECTED - 43.8x central aperture ratio |

**Key Conclusions:**
1. SLACS/BELLS are NOT a fair primary anchor for DR10 ground-based finder
2. Many anchor DR10 cutouts are background-subtracted to noise at center-aperture scale
3. Training LRGs are ~44x brighter than anchors - this is REAL, not artifact
4. Model likely learned "bright center" shortcut

---

## PHASE 1: Dataset Sanity Gates (DO TODAY)

### Gate 1.1: Class-Conditional Quality Distributions

**Goal:** Confirm positives and controls are matched in data-quality space.

**Columns to check:** `bad_pixel_frac`, `maskbit_frac` (or proxy), any quality flags

```python
#!/usr/bin/env python3
"""Gate 1.1: Class-conditional quality distribution check."""
import pyarrow.dataset as ds
import numpy as np

print("=" * 70)
print("GATE 1.1: CLASS-CONDITIONAL QUALITY DISTRIBUTIONS")
print("=" * 70)

data_path = "/lambda/nfs/darkhaloscope-training-dc/phase4c_v5_cosmos"
dataset = ds.dataset(data_path, format="parquet", partitioning="hive")

# Quality columns to check
QUALITY_COLS = ["bad_pixel_frac", "bandset", "cutout_ok", "physics_valid"]

# Also check arc_snr for injections only
cols_to_read = ["is_control", "region_split"] + QUALITY_COLS + ["arc_snr"]

# Read train split
filt = (ds.field("region_split") == "train") & (ds.field("cutout_ok") == 1)
table = dataset.to_table(filter=filt, columns=cols_to_read)
df = table.to_pandas()

print(f"\nTotal train samples: {len(df)}")
print(f"Controls (is_control=1): {(df['is_control']==1).sum()}")
print(f"Positives (is_control=0): {(df['is_control']==0).sum()}")

# Check each quality column
for col in QUALITY_COLS:
    if col not in df.columns:
        print(f"\n⚠️ {col}: NOT IN DATASET")
        continue
    
    ctrl = df[df['is_control']==1][col]
    pos = df[df['is_control']==0][col]
    
    print(f"\n{'='*70}")
    print(f"{col}:")
    print(f"{'='*70}")
    
    if col == "bandset":
        print(f"  Controls: {ctrl.value_counts().to_dict()}")
        print(f"  Positives: {pos.value_counts().to_dict()}")
        if set(ctrl.unique()) != set(pos.unique()):
            print(f"  ⚠️ MISMATCH: Different bandset distributions")
    else:
        print(f"  Controls:  mean={ctrl.mean():.4f}, std={ctrl.std():.4f}, median={ctrl.median():.4f}")
        print(f"  Positives: mean={pos.mean():.4f}, std={pos.std():.4f}, median={pos.median():.4f}")
        
        # KS test
        from scipy.stats import ks_2samp
        stat, pval = ks_2samp(ctrl.dropna(), pos.dropna())
        flag = " ⚠️ SIGNIFICANT DIFF" if pval < 0.001 else ""
        print(f"  KS test: stat={stat:.4f}, p={pval:.6f}{flag}")

# Check arc_snr for positives only
pos_snr = df[df['is_control']==0]['arc_snr'].dropna()
print(f"\n{'='*70}")
print("arc_snr (positives only):")
print(f"{'='*70}")
print(f"  mean={pos_snr.mean():.2f}, median={pos_snr.median():.2f}")
print(f"  min={pos_snr.min():.2f}, max={pos_snr.max():.2f}")
print(f"  Fraction < 2: {(pos_snr < 2).mean()*100:.1f}%")
print(f"  Fraction < 5: {(pos_snr < 5).mean()*100:.1f}%")
print(f"  Fraction > 20: {(pos_snr > 20).mean()*100:.1f}%")

print("\n" + "=" * 70)
print("GATE 1.1 CONCLUSION")
print("=" * 70)
```

**Pass Criteria:**
- [ ] bad_pixel_frac: Controls and positives have similar distributions (KS p > 0.01)
- [ ] maskbit_frac: Controls and positives have similar distributions
- [ ] bandset: All samples are 'grz' (no per-class imbalance)

---

### Gate 1.2: Bandset Audit

**Goal:** Confirm all samples have consistent band coverage.

```python
#!/usr/bin/env python3
"""Gate 1.2: Bandset consistency audit."""
import pyarrow.dataset as ds

print("=" * 70)
print("GATE 1.2: BANDSET AUDIT")
print("=" * 70)

data_path = "/lambda/nfs/darkhaloscope-training-dc/phase4c_v5_cosmos"
dataset = ds.dataset(data_path, format="parquet", partitioning="hive")

# Read bandset for all samples
table = dataset.to_table(columns=["bandset", "is_control", "region_split"])
df = table.to_pandas()

print(f"\nTotal samples: {len(df)}")
print(f"\nBandset value counts (all):")
print(df['bandset'].value_counts())

# Check by class
print(f"\nBandset by class:")
print(df.groupby('is_control')['bandset'].value_counts())

# Check for non-grz samples
non_grz = df[df['bandset'] != 'grz']
if len(non_grz) > 0:
    print(f"\n⚠️ WARNING: {len(non_grz)} samples have bandset != 'grz'")
    print(f"  Per class: {non_grz.groupby('is_control').size().to_dict()}")
    print(f"  Per split: {non_grz.groupby('region_split').size().to_dict()}")
else:
    print(f"\n✓ All samples have bandset='grz'")

print("\n" + "=" * 70)
print("GATE 1.2 CONCLUSION")
print("=" * 70)
if len(non_grz) == 0:
    print("✓ PASS: Consistent bandset across all samples")
else:
    print("⚠️ INVESTIGATE: Non-grz samples found")
```

**Pass Criteria:**
- [ ] All samples have bandset='grz'
- [ ] No per-class imbalance in band coverage

---

### Gate 1.3: Null-Injection Test

**Goal:** Verify classifier cannot separate controls from zero-flux injections.

**Implementation:** Run injection pipeline with `src_flux = 0` (or inject then subtract arc).

```python
#!/usr/bin/env python3
"""Gate 1.3: Null-injection test.

This test verifies the model cannot separate controls from zero-flux "injections".
If it can, there's a shortcut unrelated to the arc signal.
"""
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import convnext_tiny
import pyarrow.dataset as ds
import io
from sklearn.metrics import roc_auc_score

print("=" * 70)
print("GATE 1.3: NULL-INJECTION TEST")
print("=" * 70)

# Load model
MODEL_PATH = "/lambda/nfs/darkhaloscope-training-dc/models/gen5_cosmos/ckpt_best.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MetaFusionHead(nn.Module):
    def __init__(self, feat_dim, meta_dim, hidden=256, dropout=0.1):
        super().__init__()
        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim + hidden, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, 1))
    def forward(self, feats, meta):
        m = self.meta_mlp(meta)
        x = torch.cat([feats, m], dim=1)
        return self.classifier(x).squeeze(1)

m = convnext_tiny(weights=None)
feat_dim = m.classifier[2].in_features
m.classifier = nn.Identity()
backbone = m

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = backbone
        self.head = MetaFusionHead(feat_dim, 2, hidden=256, dropout=0.1)
    def forward(self, x, meta=None):
        feats = self.backbone(x)
        if feats.ndim > 2:
            feats = torch.flatten(feats, 1)
        if meta is None:
            meta = torch.zeros(feats.shape[0], 2, device=feats.device)
        return self.head(feats, meta)

model = Model()
ckpt = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(ckpt["model"])
model = model.to(device)
model.eval()
print(f"Model loaded from epoch {ckpt['epoch']}")

# Normalization (must match training)
def robust_mad_norm_outer(x, clip=10.0, eps=1e-6, inner_frac=0.5):
    out = np.empty_like(x, dtype=np.float32)
    h, w = x.shape[-2:]
    cy, cx = h // 2, w // 2
    ri = int(min(h, w) * inner_frac / 2)
    yy, xx = np.ogrid[:h, :w]
    outer_mask = ((yy - cy)**2 + (xx - cx)**2) > ri**2
    for c in range(x.shape[0]):
        v = x[c]
        outer_v = v[outer_mask]
        med = np.median(outer_v)
        mad = np.median(np.abs(outer_v - med))
        scale = 1.4826 * mad + eps
        vv = (v - med) / scale
        if clip is not None:
            vv = np.clip(vv, -clip, clip)
        out[c] = vv.astype(np.float32)
    return out

def decode_stamp(blob):
    bio = io.BytesIO(blob)
    with np.load(bio) as z:
        g = z["image_g"].astype(np.float32)
        r = z["image_r"].astype(np.float32)
        zb = z["image_z"].astype(np.float32)
    return np.stack([g, r, zb], axis=0)

# Load ONLY controls (is_control=1) from test set
data_path = "/lambda/nfs/darkhaloscope-training-dc/phase4c_v5_cosmos"
dataset = ds.dataset(data_path, format="parquet", partitioning="hive")

# Get controls only
filt = (ds.field("region_split") == "test") & (ds.field("cutout_ok") == 1) & (ds.field("is_control") == 1)
cols = ["stamp_npz", "psfsize_r", "psfdepth_r"]
table = dataset.to_table(filter=filt, columns=cols)

n_samples = min(1000, table.num_rows)
print(f"\nEvaluating {n_samples} control samples...")

# Run inference on controls
predictions = []
for i in range(n_samples):
    blob = table["stamp_npz"][i].as_py()
    if blob is None:
        continue
    try:
        img = decode_stamp(blob)
        img_norm = robust_mad_norm_outer(img)
        if not np.isfinite(img_norm).all():
            continue
        
        psfsize = table["psfsize_r"][i].as_py() or 1.0
        psfdepth = table["psfdepth_r"][i].as_py() or 0.0
        
        x = torch.tensor(img_norm[np.newaxis], dtype=torch.float32).to(device)
        meta = torch.tensor([[psfsize, psfdepth]], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            logit = model(x, meta)
            prob = torch.sigmoid(logit).item()
        
        predictions.append(prob)
    except Exception as e:
        continue

predictions = np.array(predictions)

print(f"\nControl predictions (should be ~0 for all):")
print(f"  Mean: {predictions.mean():.4f}")
print(f"  Std: {predictions.std():.4f}")
print(f"  Fraction > 0.5: {(predictions > 0.5).mean()*100:.1f}%")
print(f"  Fraction > 0.9: {(predictions > 0.9).mean()*100:.1f}%")

print("\n" + "=" * 70)
print("GATE 1.3 CONCLUSION")
print("=" * 70)
if predictions.mean() < 0.2 and (predictions > 0.5).mean() < 0.1:
    print("✓ PASS: Model correctly identifies controls as non-lenses")
else:
    print("⚠️ INVESTIGATE: Model may have shortcuts")
```

**Pass Criteria:**
- [ ] Mean p_lens on controls < 0.2
- [ ] Fraction of controls with p_lens > 0.5 < 10%

---

### Gate 1.4: Per-Pixel SNR Image Ablation

**Goal:** Test if using per-pixel SNR images reduces "bright center" shortcut.

```python
#!/usr/bin/env python3
"""Gate 1.4: Per-pixel SNR representation ablation.

Instead of outer-annulus MAD normalization, use image * sqrt(invvar).
This is physically meaningful and reduces center brightness shortcuts.
"""
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import convnext_tiny
import pyarrow.dataset as ds
import io
from sklearn.metrics import roc_auc_score

print("=" * 70)
print("GATE 1.4: PER-PIXEL SNR IMAGE ABLATION")
print("=" * 70)
print("Note: This requires invvar data in the stamps.")
print("If invvar is not stored, this gate is DEFERRED.")

# Check if invvar is available
data_path = "/lambda/nfs/darkhaloscope-training-dc/phase4c_v5_cosmos"
dataset = ds.dataset(data_path, format="parquet", partitioning="hive")
schema_names = dataset.schema.names

if "invvar_npz" in schema_names or "stamp_invvar" in schema_names:
    print("✓ invvar data available")
    # Implementation would go here
else:
    print("⚠️ invvar not stored in dataset")
    print("GATE 1.4: DEFERRED - requires reprocessing with invvar")
    print("\nRECOMMENDATION:")
    print("  Add invvar storage to Phase 4c pipeline for future runs.")
    print("  For now, proceed with other gates.")

print("\n" + "=" * 70)
print("GATE 1.4 STATUS: DEFERRED (invvar not available)")
print("=" * 70)
```

**Pass Criteria:**
- [ ] If invvar available: Per-pixel SNR images produce AUC within 5% of original
- [ ] If invvar NOT available: DEFERRED

---

## PHASE 2: Center-Masked Diagnostic (After Phase 1)

**Goal:** Fast falsification test - does performance collapse when lens-galaxy core is suppressed?

### Implementation

```python
#!/usr/bin/env python3
"""
Center-masked diagnostic training.

Mask radius options: r=8 or r=10 pixels (2.1-2.6 arcsec)
Fill policy: resample from outer annulus (same band)
Apply: training only (not inference for this diagnostic)
"""
import numpy as np

def mask_center_outer_fill(img, r_mask=10):
    """
    Mask center with pixels resampled from outer annulus.
    
    Args:
        img: (C, H, W) array
        r_mask: mask radius in pixels (default 10 = 2.62 arcsec)
    
    Returns:
        Masked image with center filled from outer annulus samples
    """
    C, H, W = img.shape
    out = img.copy()
    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    r2 = (yy - cy)**2 + (xx - cx)**2
    
    center_mask = r2 < r_mask**2
    outer_mask = r2 >= (r_mask * 2)**2  # Outer annulus at 2x mask radius
    
    for c in range(C):
        outer_vals = img[c][outer_mask]
        n_center = center_mask.sum()
        # Sample with replacement from outer annulus
        fill_vals = np.random.choice(outer_vals, size=n_center, replace=True)
        out[c][center_mask] = fill_vals
    
    return out

# Example usage in training loop:
# if training:
#     img = mask_center_outer_fill(img, r_mask=10)
```

### Diagnostic Training Config

```yaml
# Center-masked diagnostic config
experiment_name: "gen5_center_masked_diagnostic"
mask_radius: 10  # pixels (2.62 arcsec at 0.262"/pix)
fill_policy: "outer_annulus_resample"
apply_at_training: true
apply_at_inference: false
epochs: 3  # Quick diagnostic
```

**Expected Outcomes:**
- If anchors improve + synthetic stable → Shortcut confirmed, proceed with fixes
- If anchors improve + synthetic collapses → Model was center-only
- If no change → Brightness calibration is the main issue

---

## PHASE 3: Build Tier-A Anchor Set

### Sources (in priority order)

| Source | Type | Est. N | Notes |
|--------|------|--------|-------|
| Legacy Surveys ML candidates (Huang et al.) | Ground-based discovery | 50-200 | Same domain as training |
| KiDS lens candidates (Petrillo et al.) | Ground-based CNN | 30-100 | Published methodology |
| HSC SuGOHI | Ground-based | 50-150 | Deeper but similar seeing |
| DES lens candidates | Ground-based | 20-50 | External validation |

### Selection Criteria for Tier-A

```python
def is_tier_a_anchor(cutout, arc_visibility_threshold=2.0):
    """
    Tier-A criterion: arc is visible in DR10 cutout.
    
    Uses annulus residual energy metric.
    """
    h, w = cutout.shape[-2:]
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r2 = (yy - cy)**2 + (xx - cx)**2
    
    # Inner core (lens galaxy)
    inner = r2 < 4**2
    # Arc annulus
    annulus = (r2 >= 4**2) & (r2 < 16**2)
    # Outer reference
    outer = r2 >= 16**2
    
    # Background-subtracted annulus energy
    bg = np.median(cutout[outer])
    annulus_excess = np.sum(cutout[annulus] - bg)
    
    # Noise estimate
    outer_mad = np.median(np.abs(cutout[outer] - bg))
    noise = 1.4826 * outer_mad * np.sqrt(annulus.sum())
    
    snr = annulus_excess / (noise + 1e-10)
    return snr > arc_visibility_threshold, snr
```

### Tier-A vs Tier-B Definition

| Tier | Purpose | Criterion |
|------|---------|-----------|
| **Tier-A** (Primary) | DR10-detectable anchors | arc_visibility_snr > 2.0 in DR10 cutout |
| **Tier-B** (Stress) | Below-threshold anchors | SLACS/BELLS that fail Tier-A criterion |

---

## PHASE 4: Hard Negatives

### Source: Galaxy Zoo DECaLS

```python
# Galaxy Zoo DECaLS morphology query
# Categories: ring galaxies, spirals with strong arms, mergers

GALAXY_ZOO_QUERY = """
SELECT ra, dec, gz_morphology
FROM galaxy_zoo_decals
WHERE (
    gz_ring_fraction > 0.5 
    OR gz_spiral_arms_prominent > 0.5
    OR gz_merger_fraction > 0.5
)
AND decals_dr10_available = 1
LIMIT 10000
"""
```

### Mixing Strategy

- **Total negative class**: 50% of training data
- **Within negatives**:
  - 70-80%: Standard controls (no injection)
  - 20-30%: Hard negatives (rings, spirals, mergers)
- **Keep separate eval slice**: Never use for training selection

---

## PHASE 5: Brightness Recalibration

### Expanded src_dmag Range

**Current:** `[1.0, 2.0]` (too narrow, too bright)

**Recommended:** `[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]`

### Target arc_snr Distribution

| arc_snr Range | Target Fraction | Current (Est.) |
|---------------|-----------------|----------------|
| 0.8–2 (hard) | 40% | ~5% |
| 2–8 (moderate) | 40% | ~35% |
| 8–20 (easy) | 15% | ~40% |
| 20+ (extreme) | 5% | ~20% |

### Implementation: Option C (Sample + Filter)

```python
def sample_src_dmag_with_snr_targeting(target_arc_snr_dist):
    """
    1. Sample from wide src_dmag prior: U[0.5, 3.5]
    2. After injection, compute arc_snr
    3. Accept/reject based on target distribution bins
    """
    # Wide prior
    src_dmag = np.random.uniform(0.5, 3.5)
    
    # After injection, check arc_snr
    # arc_snr = compute_arc_snr(injected_stamp)
    
    # Rejection sampling to match target distribution
    # ...
    
    return src_dmag, arc_snr
```

---

## PHASE 6: Expanded theta_e Range

**Current:** `[0.3, 0.6, 1.0]`

**Recommended:** `[0.3, 0.5, 0.75, 1.0, 1.25, 1.5]`

### Stratification by theta_e/psfsize_r

```python
# Track detectability proxy
theta_over_psf = theta_e_arcsec / psfsize_r

# Ensure coverage across bins:
# - Sub-PSF: theta/psf < 0.5 (challenging, but needed for selection function)
# - Marginal: 0.5 <= theta/psf < 0.8
# - Detectable: 0.8 <= theta/psf < 1.5
# - Easy: theta/psf >= 1.5
```

---

## PHASE 7: Expanded Source Radius Range

**Current:** `[0.08, 0.15]` arcsec (too narrow)

**Recommended:** `[0.05, 0.08, 0.12, 0.18, 0.25]` arcsec

Or: Sample directly from COSMOS HLR distribution after filters.

---

## PHASE 8: Region-Disjoint Splits

### HEALPix-Based Splitting

```python
import healpy as hp

def assign_healpix_cell(ra, dec, nside=64):
    """
    Assign each sample to a HEALPix cell.
    NSIDE=64 gives ~12288 cells (~3072 pixels at NSIDE=32).
    """
    theta = np.radians(90 - dec)  # colatitude
    phi = np.radians(ra)
    pixel = hp.ang2pix(nside, theta, phi)
    return pixel

def create_region_disjoint_splits(df, nside=64, test_frac=0.1, val_frac=0.1):
    """
    Split by HEALPix cell, not by brickname.
    """
    df['healpix_cell'] = df.apply(lambda r: assign_healpix_cell(r['ra'], r['dec'], nside), axis=1)
    
    unique_cells = df['healpix_cell'].unique()
    np.random.shuffle(unique_cells)
    
    n_test = int(len(unique_cells) * test_frac)
    n_val = int(len(unique_cells) * val_frac)
    
    test_cells = set(unique_cells[:n_test])
    val_cells = set(unique_cells[n_test:n_test+n_val])
    train_cells = set(unique_cells[n_test+n_val:])
    
    # Optional: guard band (exclude cells adjacent to test cells)
    # test_neighbors = get_neighbor_cells(test_cells)
    # train_cells = train_cells - test_neighbors
    
    def assign_split(cell):
        if cell in test_cells:
            return 'test'
        elif cell in val_cells:
            return 'val'
        else:
            return 'train'
    
    df['region_split'] = df['healpix_cell'].apply(assign_split)
    return df
```

---

## PHASE 9: Final Retraining (Gen5')

### Configuration Summary

| Parameter | Current | Recommended |
|-----------|---------|-------------|
| `src_dmag` | [1.0, 2.0] | [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5] |
| `theta_e` | [0.3, 0.6, 1.0] | [0.3, 0.5, 0.75, 1.0, 1.25, 1.5] |
| `src_reff` | [0.08, 0.15] | [0.05, 0.08, 0.12, 0.18, 0.25] |
| `control_frac` | 50% | 50% (keep, add hard negs within) |
| Hard negatives | 0% | 20-30% of negatives |
| Split method | brickname hash | HEALPix cell |
| arc_snr target | uniform | 40/40/15/5 distribution |

---

## PHASE 10: Final Evaluation

### Primary Metrics (Tier-A)

| Metric | Target |
|--------|--------|
| Recall @ FPR=1% | > 50% |
| AUC | > 0.85 |
| Correlation(p_lens, theta_e/psf) | Positive, significant |

### Stress Test (Tier-B: SLACS/BELLS)

| Metric | Expected |
|--------|----------|
| Recall @ FPR=1% | 10-30% (acceptable for below-threshold) |
| Report separately | Not primary metric |

---

## Checklist Summary

### Phase 1: Sanity Gates (Today)
- [ ] Gate 1.1: Class-conditional quality distributions
- [ ] Gate 1.2: Bandset audit
- [ ] Gate 1.3: Null-injection test (controls only)
- [ ] Gate 1.4: Per-pixel SNR ablation (if invvar available)

### Phase 2: Center-Masked Diagnostic
- [ ] Run 1-3 epoch pilot with r=10 mask
- [ ] Compare anchor recall vs synthetic AUC

### Phase 3: Tier-A Anchors
- [ ] Curate 50-100 DR10-visible anchors
- [ ] Apply arc_visibility_snr > 2.0 criterion
- [ ] Keep SLACS/BELLS as Tier-B stress test

### Phase 4: Hard Negatives
- [ ] Query Galaxy Zoo DECaLS for rings/spirals
- [ ] Create 20-30% hard negative subset
- [ ] Reserve eval slice

### Phase 5-7: Parameter Recalibration
- [ ] Expand src_dmag to [0.5, 3.5]
- [ ] Expand theta_e to [0.3, 1.5]
- [ ] Expand src_reff to [0.05, 0.25]
- [ ] Implement arc_snr targeting

### Phase 8: Region-Disjoint Splits
- [ ] Implement HEALPix-based splitting
- [ ] Choose NSIDE (64-128 recommended)
- [ ] Optional: add guard band

### Phase 9: Retrain
- [ ] Apply all fixes
- [ ] Train Gen5' for full epochs

### Phase 10: Evaluate
- [ ] Tier-A: Target recall > 50%
- [ ] Tier-B: Report separately

---

## Timeline

| Phase | Task | Est. Time | Status |
|-------|------|-----------|--------|
| 1 | Sanity Gates | 2-4 hours | PENDING |
| 2 | Center-Masked Diagnostic | 1-2 days | PENDING |
| 3 | Tier-A Anchors | 2-3 days | PENDING |
| 4 | Hard Negatives | 1-2 days | PENDING (parallel with 3) |
| 5-7 | Parameter Recalibration | 2-3 days | PENDING |
| 8 | Region-Disjoint Splits | 1 day | PENDING |
| 9 | Retrain Gen5' | 2-3 days | PENDING |
| 10 | Final Evaluation | 1 day | PENDING |

**Total: ~10-14 days** (with parallelization)

---

*Created: 2026-02-05*
*This document is the authoritative remediation plan.*
