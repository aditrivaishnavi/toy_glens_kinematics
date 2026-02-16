# Gen5 Remediation - Independent LLM Review Prompt

**Date:** 2026-02-05  
**Purpose:** Independent review of Gen5 model remediation analysis and implementation

---

## Context

We are building a CNN-based strong gravitational lens finder for DESI Legacy Survey DR10 imaging. The model classifies 64×64 pixel image stamps (g, r, z bands) as containing a gravitational lens or not.

**Problem:** Our Gen5 model achieved excellent synthetic test metrics (AUC=0.9945) but only 4.4% recall on real SLACS/BELLS anchor lenses. Investigation revealed:
1. SLACS/BELLS lenses are ~40x fainter than our training injections in DR10 imaging
2. Model may have learned shortcuts based on galaxy brightness rather than arc morphology
3. Training data distribution may not match real detection targets

**Remediation approach:** Run sanity gates and diagnostics before retraining.

---

## Phase 1: Sanity Gates

### Gate 1.1: Class-Conditional Quality Distributions

**Purpose:** Verify positives (injected lenses) and controls (no injection) have identical quality characteristics to rule out shortcuts.

**Script:**
```python
#!/usr/bin/env python3
"""
Gate 1.1: Class-conditional quality distribution check.
Verifies positives and controls are matched in data-quality space.
"""
import pyarrow.dataset as ds
import numpy as np
import json
from scipy.stats import ks_2samp
from datetime import datetime, timezone

RESULTS = {"gate": "1.1", "timestamp": datetime.now(timezone.utc).isoformat(), "checks": []}

data_path = "/lambda/nfs/darkhaloscope-training-dc/phase4c_v5_cosmos"
dataset = ds.dataset(data_path, format="parquet", partitioning="hive")

QUALITY_COLS = ["bad_pixel_frac", "cutout_ok", "physics_valid"]
cols_to_read = ["is_control", "region_split"] + QUALITY_COLS + ["arc_snr", "bandset"]

filt = (ds.field("region_split") == "train") & (ds.field("cutout_ok") == 1)
table = dataset.to_table(filter=filt, columns=cols_to_read)
df = table.to_pandas()

RESULTS["total_samples"] = int(len(df))
RESULTS["n_controls"] = int((df['is_control']==1).sum())
RESULTS["n_positives"] = int((df['is_control']==0).sum())

for col in QUALITY_COLS:
    if col not in df.columns:
        RESULTS["checks"].append({"column": col, "status": "MISSING"})
        continue
    
    ctrl = df[df['is_control']==1][col].dropna()
    pos = df[df['is_control']==0][col].dropna()
    
    if len(ctrl) < 2 or len(pos) < 2:
        RESULTS["checks"].append({
            "column": col,
            "status": "INSUFFICIENT_DATA",
            "ctrl_count": int(len(ctrl)),
            "pos_count": int(len(pos))
        })
        continue
    
    stat, pval = ks_2samp(ctrl, pos)
    passed = bool(pval > 0.01)
    
    RESULTS["checks"].append({
        "column": col,
        "ctrl_mean": float(ctrl.mean()),
        "ctrl_std": float(ctrl.std()),
        "pos_mean": float(pos.mean()),
        "pos_std": float(pos.std()),
        "ks_stat": float(stat),
        "ks_pval": float(pval),
        "passed": passed
    })

# arc_snr distribution for positives
pos_snr = df[df['is_control']==0]['arc_snr'].dropna()
if len(pos_snr) > 0:
    RESULTS["arc_snr_distribution"] = {
        "count": int(len(pos_snr)),
        "mean": float(pos_snr.mean()),
        "median": float(pos_snr.median()),
        "frac_lt_2": float((pos_snr < 2).mean()),
        "frac_lt_5": float((pos_snr < 5).mean()),
        "frac_gt_20": float((pos_snr > 20).mean())
    }

# Overall pass/fail
valid_checks = [c for c in RESULTS["checks"] if c.get("passed") is not None]
all_passed = bool(all(c.get("passed", False) for c in valid_checks)) if valid_checks else False
RESULTS["overall_passed"] = all_passed

with open("gate_1_1_results.json", "w") as f:
    json.dump(RESULTS, f, indent=2)

print(json.dumps(RESULTS, indent=2))
print(f"\nGATE 1.1: {'PASS' if all_passed else 'FAIL'}")
```

**Results:**
```json
{
  "gate": "1.1",
  "timestamp": "2026-02-05T04:39:27.960721+00:00",
  "checks": [
    {
      "column": "bad_pixel_frac",
      "ctrl_mean": 0.06833405926967702,
      "ctrl_std": 0.21094190379721678,
      "pos_mean": 0.06815684317846295,
      "pos_std": 0.21031819064892157,
      "ks_stat": 0.0009006987430056901,
      "ks_pval": 0.6308449100957321,
      "passed": true
    },
    {
      "column": "cutout_ok",
      "ctrl_mean": 1.0,
      "ctrl_std": 0.0,
      "pos_mean": 1.0,
      "pos_std": 0.0,
      "ks_stat": 0.0,
      "ks_pval": 1.0,
      "passed": true
    },
    {
      "column": "physics_valid",
      "status": "INSUFFICIENT_DATA",
      "ctrl_count": 0,
      "pos_count": 1380816
    }
  ],
  "total_samples": 2755872,
  "n_controls": 1375056,
  "n_positives": 1380816,
  "arc_snr_distribution": {
    "count": 1380816,
    "mean": 8.51224682690963,
    "median": 4.670472621917725,
    "frac_lt_2": 0.2161330691417249,
    "frac_lt_5": 0.5254516170148665,
    "frac_gt_20": 0.08994826247668046
  },
  "overall_passed": true
}
```

**Status: PASS**

---

### Gate 1.2: Bandset Audit

**Purpose:** Verify all samples have consistent band coverage (g, r, z).

**Script:**
```python
#!/usr/bin/env python3
"""
Gate 1.2: Bandset consistency audit.
Verifies all samples have consistent band coverage.
"""
import pyarrow.dataset as ds
import json
from datetime import datetime, timezone

RESULTS = {"gate": "1.2", "timestamp": datetime.now(timezone.utc).isoformat()}

data_path = "/lambda/nfs/darkhaloscope-training-dc/phase4c_v5_cosmos"
dataset = ds.dataset(data_path, format="parquet", partitioning="hive")

table = dataset.to_table(columns=["bandset", "is_control", "region_split"])
df = table.to_pandas()

RESULTS["total_samples"] = int(len(df))
bandset_counts = df['bandset'].value_counts().to_dict()
RESULTS["bandset_counts"] = {str(k): int(v) for k, v in bandset_counts.items()}

ctrl_bandset = df[df['is_control']==1]['bandset'].value_counts().to_dict()
pos_bandset = df[df['is_control']==0]['bandset'].value_counts().to_dict()
RESULTS["bandset_by_class"] = {
    "controls": {str(k): int(v) for k, v in ctrl_bandset.items()},
    "positives": {str(k): int(v) for k, v in pos_bandset.items()}
}

non_grz = df[df['bandset'] != 'grz']
RESULTS["non_grz_count"] = int(len(non_grz))
RESULTS["non_grz_by_class"] = {str(k): int(v) for k, v in non_grz.groupby('is_control').size().to_dict().items()} if len(non_grz) > 0 else {}

RESULTS["overall_passed"] = len(non_grz) == 0

with open("gate_1_2_results.json", "w") as f:
    json.dump(RESULTS, f, indent=2)

print(json.dumps(RESULTS, indent=2))
print(f"\nGATE 1.2: {'PASS' if RESULTS['overall_passed'] else 'FAIL'}")
```

**Results:**
```json
{
  "gate": "1.2",
  "timestamp": "2026-02-05T04:40:50.301747+00:00",
  "total_samples": 10648570,
  "bandset_counts": {
    "grz": 10648570
  },
  "bandset_by_class": {
    "controls": {
      "grz": 5309930
    },
    "positives": {
      "grz": 5338640
    }
  },
  "non_grz_count": 0,
  "non_grz_by_class": {},
  "overall_passed": true
}
```

**Status: PASS**

---

### Gate 1.3: Null-Injection Test

**Purpose:** Verify the trained model correctly classifies controls (no lens injection) as non-lenses.

**Script:**
```python
#!/usr/bin/env python3
"""
Gate 1.3: Null-injection test.
Verifies model correctly identifies controls as non-lenses.
"""
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import convnext_tiny
import pyarrow.dataset as ds
import io
import json
from datetime import datetime, timezone

RESULTS = {"gate": "1.3", "timestamp": datetime.now(timezone.utc).isoformat()}

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

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = m
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

RESULTS["model_epoch"] = int(ckpt["epoch"])

def robust_mad_norm_outer(x, clip=10.0, eps=1e-6, inner_frac=0.5):
    """Normalize using outer annulus statistics to avoid leaking center brightness."""
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

data_path = "/lambda/nfs/darkhaloscope-training-dc/phase4c_v5_cosmos"
dataset = ds.dataset(data_path, format="parquet", partitioning="hive")
filt = (ds.field("region_split") == "test") & (ds.field("cutout_ok") == 1) & (ds.field("is_control") == 1)
table = dataset.to_table(filter=filt, columns=["stamp_npz", "psfsize_r", "psfdepth_r"])

n_samples = min(1000, table.num_rows)
predictions = []
errors = 0
for i in range(n_samples):
    blob = table["stamp_npz"][i].as_py()
    if blob is None:
        errors += 1
        continue
    try:
        img = decode_stamp(blob)
        img_norm = robust_mad_norm_outer(img)
        if not np.isfinite(img_norm).all():
            errors += 1
            continue
        psfsize = table["psfsize_r"][i].as_py() or 1.0
        psfdepth = table["psfdepth_r"][i].as_py() or 0.0
        x = torch.tensor(img_norm[np.newaxis], dtype=torch.float32).to(device)
        meta = torch.tensor([[psfsize, psfdepth]], dtype=torch.float32).to(device)
        with torch.no_grad():
            logit = model(x, meta)
            prob = torch.sigmoid(logit).item()
        predictions.append(prob)
    except:
        errors += 1
        continue

predictions = np.array(predictions)
RESULTS["n_evaluated"] = int(len(predictions))
RESULTS["n_errors"] = int(errors)
RESULTS["mean_p_lens"] = float(predictions.mean())
RESULTS["std_p_lens"] = float(predictions.std())
RESULTS["frac_gt_0.5"] = float((predictions > 0.5).mean())
RESULTS["frac_gt_0.9"] = float((predictions > 0.9).mean())
RESULTS["overall_passed"] = bool(predictions.mean() < 0.2 and (predictions > 0.5).mean() < 0.1)

with open("gate_1_3_results.json", "w") as f:
    json.dump(RESULTS, f, indent=2)

print(json.dumps(RESULTS, indent=2))
print(f"\nGATE 1.3: {'PASS' if RESULTS['overall_passed'] else 'FAIL'}")
```

**Results:**
```json
{
  "gate": "1.3",
  "timestamp": "2026-02-05T04:42:41.139317+00:00",
  "model_epoch": 6,
  "n_evaluated": 1000,
  "n_errors": 0,
  "mean_p_lens": 0.014233729973710979,
  "std_p_lens": 0.07646836062119619,
  "frac_gt_0.5": 0.004,
  "frac_gt_0.9": 0.0,
  "overall_passed": true
}
```

**Status: PASS** - Controls correctly classified with mean p_lens = 0.014

---

### Gate 1.4: SNR Ablation Check

**Purpose:** Check if inverse variance is available for per-pixel SNR experiments.

**Script:**
```python
#!/usr/bin/env python3
"""
Gate 1.4: Check if invvar is available for per-pixel SNR representation.
"""
import pyarrow.dataset as ds
import json
from datetime import datetime, timezone

RESULTS = {"gate": "1.4", "timestamp": datetime.now(timezone.utc).isoformat()}

data_path = "/lambda/nfs/darkhaloscope-training-dc/phase4c_v5_cosmos"
dataset = ds.dataset(data_path, format="parquet", partitioning="hive")
schema_names = dataset.schema.names

RESULTS["schema_columns"] = schema_names
RESULTS["has_invvar_npz"] = "invvar_npz" in schema_names
RESULTS["has_stamp_invvar"] = "stamp_invvar" in schema_names

if RESULTS["has_invvar_npz"] or RESULTS["has_stamp_invvar"]:
    RESULTS["status"] = "AVAILABLE - implement SNR ablation"
    RESULTS["overall_passed"] = None
else:
    RESULTS["status"] = "DEFERRED - invvar not stored in current dataset"
    RESULTS["recommendation"] = "Add invvar to Phase 4c for future runs"
    RESULTS["overall_passed"] = "DEFERRED"

with open("gate_1_4_results.json", "w") as f:
    json.dump(RESULTS, f, indent=2)

print(json.dumps(RESULTS, indent=2))
```

**Results:**
```json
{
  "gate": "1.4",
  "has_invvar_npz": false,
  "has_stamp_invvar": false,
  "status": "DEFERRED - invvar not stored in current dataset",
  "recommendation": "Add invvar to Phase 4c for future runs",
  "overall_passed": "DEFERRED"
}
```

**Status: DEFERRED** - invvar not available in current dataset

---

## Phase 2: Center-Masked Diagnostic

**Purpose:** Test whether the model relies on the central lens galaxy (brightness shortcut) vs arc morphology.

**Script:**
```python
#!/usr/bin/env python3
"""
Phase 2: Center-masked diagnostic.
Tests if model relies on lens-galaxy core for classification.
"""
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import convnext_tiny
import pyarrow.dataset as ds
import io
import json
from datetime import datetime, timezone

def mask_center_outer_fill(img, r_mask=10, seed=None):
    """
    Mask center with pixels resampled from outer annulus.
    
    Args:
        img: (C, H, W) array
        r_mask: mask radius in pixels (default 10 = 2.62 arcsec at 0.262"/pix)
        seed: random seed for reproducibility
    
    Returns:
        Masked image with center filled from outer annulus samples
    """
    if seed is not None:
        np.random.seed(seed)
    
    C, H, W = img.shape
    out = img.copy()
    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    r2 = (yy - cy)**2 + (xx - cx)**2
    
    center_mask = r2 < r_mask**2
    outer_mask = r2 >= (r_mask * 2)**2
    
    for c in range(C):
        outer_vals = img[c][outer_mask]
        if len(outer_vals) == 0:
            continue
        n_center = center_mask.sum()
        fill_vals = np.random.choice(outer_vals, size=n_center, replace=True)
        out[c][center_mask] = fill_vals
    
    return out

# Load model (same as Gate 1.3)
# ... [model loading code omitted for brevity - same as above] ...

# For each sample, compare predictions with and without center masked
MASK_RADII = [8, 10, 12]  # pixels

for r_mask in MASK_RADII:
    original_probs = []
    masked_probs = []
    
    for i in range(500):  # 500 positive samples
        # Get original prediction
        img = decode_stamp(blob)
        img_norm = robust_mad_norm_outer(img)
        prob_orig = model_predict(img_norm)
        
        # Get masked prediction
        img_masked = mask_center_outer_fill(img, r_mask=r_mask, seed=i)
        img_masked_norm = robust_mad_norm_outer(img_masked)
        prob_masked = model_predict(img_masked_norm)
        
        original_probs.append(prob_orig)
        masked_probs.append(prob_masked)
    
    drop = np.mean(original_probs) - np.mean(masked_probs)
    drop_pct = drop / np.mean(original_probs) * 100
    print(f"r_mask={r_mask}px: drop = {drop:.3f} ({drop_pct:.1f}%)")
```

**Results:**
```json
{
  "phase": "2",
  "timestamp": "2026-02-05T05:07:15.974386+00:00",
  "results_by_radius": [
    {
      "r_mask_pixels": 8,
      "r_mask_arcsec": 2.096,
      "mean_p_original": 0.9552,
      "mean_p_masked": 0.9026,
      "drop": 0.0526,
      "drop_percent": 5.51
    },
    {
      "r_mask_pixels": 10,
      "r_mask_arcsec": 2.62,
      "mean_p_original": 0.9552,
      "mean_p_masked": 0.8480,
      "drop": 0.1072,
      "drop_percent": 11.23
    },
    {
      "r_mask_pixels": 12,
      "r_mask_arcsec": 3.144,
      "mean_p_original": 0.9552,
      "mean_p_masked": 0.7411,
      "drop": 0.2141,
      "drop_percent": 22.42
    }
  ],
  "interpretation": "MODERATE RELIANCE ON CENTER - Model uses mix of center and arc features",
  "recommendation": "Center-masked training may improve anchor recall",
  "overall_max_drop_percent": 22.42
}
```

**Interpretation:** When center is masked (r=12px = 3.14"), predictions drop by 22.4%. This indicates the model uses both center galaxy features and arc features.

---

## Phase 3: Tier-A Anchor Classification

**Purpose:** Classify known lenses into Tier-A (visible in ground-based imaging) and Tier-B (too faint).

**Method:** Compute `arc_visibility_snr` for each lens:
```python
def arc_visibility_snr(cutout, inner_r=4, outer_r=16):
    """
    Compute arc visibility SNR in annulus region.
    
    Args:
        cutout: 2D array (r-band)
        inner_r: inner radius of arc annulus (pixels)
        outer_r: outer radius of arc annulus (pixels)
    
    Returns:
        snr: visibility SNR
        is_tier_a: True if snr > 2.0
    """
    h, w = cutout.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r2 = (yy - cy)**2 + (xx - cx)**2
    
    annulus = (r2 >= inner_r**2) & (r2 < outer_r**2)
    outer = r2 >= outer_r**2
    
    bg = np.median(cutout[outer])
    annulus_excess = np.sum(cutout[annulus] - bg)
    
    outer_mad = np.median(np.abs(cutout[outer] - bg))
    noise = 1.4826 * outer_mad * np.sqrt(annulus.sum())
    
    snr = annulus_excess / (noise + 1e-10)
    return float(snr), snr > 2.0
```

**Results:**
```json
{
  "phase": "3",
  "threshold": 2.0,
  "tier_a": [
    {"name": "SDSSJ0029-0055", "ra": 7.4543, "dec": -0.9254, "source": "SLACS", "arc_visibility_snr": 3.51},
    {"name": "SDSSJ0252+0039", "ra": 43.1313, "dec": 0.6651, "source": "SLACS", "arc_visibility_snr": 3.16},
    {"name": "SDSSJ0959+0410", "ra": 149.7954, "dec": 4.1755, "source": "SLACS", "arc_visibility_snr": 3.91},
    {"name": "SDSSJ0832+0404", "ra": 128.2038, "dec": 4.0725, "source": "BELLS", "arc_visibility_snr": 7.95}
  ],
  "tier_b": [
    {"name": "SDSSJ0037-0942", "source": "SLACS", "arc_visibility_snr": 0.12},
    {"name": "SDSSJ0330-0020", "source": "SLACS", "arc_visibility_snr": 1.30},
    {"name": "SDSSJ0728+3835", "source": "SLACS", "arc_visibility_snr": -1.08},
    {"name": "SDSSJ0737+3216", "source": "SLACS", "arc_visibility_snr": -0.80},
    {"name": "SDSSJ0912+0029", "source": "SLACS", "arc_visibility_snr": -4.47},
    {"name": "SDSSJ1016+3859", "source": "SLACS", "arc_visibility_snr": -4.65},
    {"name": "SDSSJ1020+1122", "source": "SLACS", "arc_visibility_snr": -2.82},
    {"name": "SDSSJ0747+5055", "source": "BELLS", "arc_visibility_snr": -1.37},
    {"name": "SDSSJ0755+3445", "source": "BELLS", "arc_visibility_snr": 1.27},
    {"name": "SDSSJ0801+4727", "source": "BELLS", "arc_visibility_snr": -0.89},
    {"name": "SDSSJ0830+5116", "source": "BELLS", "arc_visibility_snr": -3.48}
  ],
  "summary": {
    "n_tier_a": 4,
    "n_tier_b": 11
  }
}
```

**Key Finding:** Only 4 of 15 SLACS/BELLS lenses have visible arcs in DR10 (SNR > 2.0). Most are below detection threshold.

---

## Phase 5: Arc SNR Distribution Analysis

**Purpose:** Analyze the arc_snr distribution of training data.

**Results:**
```
Current Distribution (n=1,380,816 positives):
  Mean arc_snr: 8.51
  Median arc_snr: 4.67
  Std arc_snr: 14.58

  Percentiles:
    5th: 0.51
    10th: 1.10
    25th: 2.26
    50th: 4.67
    75th: 9.65
    90th: 18.74
    95th: 28.04

  Bin distribution:
    [0, 2): 21.6%
    [2, 5): 30.9%
    [5, 10): 23.5%
    [10, 20): 15.0%
    [20, 50): 7.4%
    [50, inf): 1.6%

Current injection parameters:
  src_dmag: [0.5, 1.0, 1.5, 2.0]
  theta_e: [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5]
```

**Recommended expansion:**
```
src_dmag: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
theta_e: [0.3, 0.5, 0.7, 1.0, 1.3, 1.6]
src_reff: [0.05, 0.10, 0.15, 0.25]
```

---

## Phase 6-7: Parameter Grid Updates

**Current:** 12 configurations (3×2×2)
**Recommended:** 180 configurations (6×6×5)

**Rationale:**
- theta_e extended to 1.6" (1.5× PSF FWHM)
- src_dmag extended to 3.0 for fainter sources
- src_reff extended to include compact and extended sources

---

## Phase 8: HEALPix Region-Disjoint Splits

**Purpose:** Ensure train/val/test splits have no spatial overlap.

**Implementation:**
```python
import healpy as hp

def assign_healpix_split(ra, dec, nside=32, train_frac=0.7, val_frac=0.15, seed=42):
    """
    Assign sample to train/val/test based on HEALPix pixel.
    """
    theta = np.radians(90 - dec)
    phi = np.radians(ra)
    pix = hp.ang2pix(nside, theta, phi)
    
    np.random.seed(seed + pix)
    r = np.random.random()
    
    if r < train_frac:
        return "train"
    elif r < train_frac + val_frac:
        return "val"
    else:
        return "test"
```

**Recommendation:** NSIDE=32 (1.83° pixels), 70/15/15 split

---

## Summary Table

| Phase | Status | Key Finding |
|-------|--------|-------------|
| Gate 1.1 | PASS | Controls/positives matched (KS p=0.63) |
| Gate 1.2 | PASS | 100% grz bandset |
| Gate 1.3 | PASS | Controls: mean p=0.014 |
| Gate 1.4 | DEFERRED | invvar not stored |
| Phase 2 | COMPLETE | 22.4% drop when center masked |
| Phase 3 | COMPLETE | 4 Tier-A, 11 Tier-B anchors |
| Phase 5 | COMPLETE | arc_snr mean=8.51, 21.6% below 2 |
| Phase 6-7 | READY | Grid expansion 12→180 |
| Phase 8 | READY | HEALPix implementation |

---

## SPECIFIC QUESTIONS FOR REVIEW

### Q1: Gate Pass Criteria
Are these thresholds appropriate?
- Gate 1.1: KS test p-value > 0.01 for quality distributions
- Gate 1.3: mean p_lens < 0.2 for controls, fraction > 0.5 < 10%

### Q2: Center-Masked Diagnostic Interpretation
The model shows 22.4% prediction drop when center (r=12px) is masked. 
- Is this concerning?
- Does this indicate shortcut learning?
- Should we apply center masking during training?

### Q3: Tier-A Anchor Set Size
Only 4 Tier-A anchors found from SLACS/BELLS.
- Is this sufficient for primary evaluation?
- What is the minimum recommended anchor set size?
- Should we source additional anchors from ground-based surveys?

### Q4: Arc SNR Distribution
Current training has 21.6% of samples with arc_snr < 2.0.
- Is this appropriate?
- Should we use rejection sampling to reshape the distribution?
- What target distribution would you recommend?

### Q5: Parameter Grid Expansion
Proposed expansion from 12 to 180 configurations.
- Is this sufficient coverage?
- Are the proposed ranges appropriate for ground-based PSF ~1.2"?
- Any parameters we're missing?

### Q6: Null-Injection Test Validity
Controls have mean p_lens = 0.014 and only 0.4% above 0.5.
- Does this prove no shortcut learning on control identification?
- Are there edge cases this test misses?

### Q7: Physics Valid Column
`physics_valid` has 0 controls and 1.38M positives.
- Is this expected (controls have no physics to validate)?
- Does this create any data leakage risk?

### Q8: Code Correctness
Please verify:
- Is `robust_mad_norm_outer` implementing outer-annulus normalization correctly?
- Is the KS test application correct for detecting distribution differences?
- Is the arc_visibility_snr calculation scientifically sound?

### Q9: Missing Checks
Are there critical checks we should run that are not included?
- Spatial distribution analysis?
- PSF size distribution by class?
- Other quality metrics?

### Q10: Priority for Remediation
Given these findings, what should be the priority order?
1. Center-masked training?
2. Parameter grid expansion?
3. HEALPix splits?
4. Hard negatives?
5. Rejection sampling for arc_snr?

---

## Honest Assessment

### What we think is correct:
- Phase 1 gates all passed, indicating no obvious data quality issues
- Null-injection test shows controls are correctly classified
- Tier-A/Tier-B distinction is valid for different evaluation purposes

### What could still be wrong:
- We only sampled 1000 controls for Gate 1.3 - could miss rare edge cases
- Tier-A set is small (n=4) - may not be representative
- arc_visibility_snr threshold of 2.0 was chosen somewhat arbitrarily
- Center-masked diagnostic is on pre-normalized data - should we mask pre-normalization?

### Assumptions we made:
- Training normalization matches inference normalization exactly
- Model architecture reconstruction matches checkpoint exactly
- 0.262"/pixel is correct for DR10

Please provide your independent assessment of this analysis.
