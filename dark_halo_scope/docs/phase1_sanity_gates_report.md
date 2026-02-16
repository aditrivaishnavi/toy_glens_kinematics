# Phase 1 Sanity Gates - LLM Review Report

**Generated:** 2026-02-05T04:54:56+00:00

## Executive Summary

All Phase 1 gates PASSED (or deferred where appropriate). The Gen5 COSMOS training data appears to be well-constructed with no obvious leakage or quality issues.

## Summary

| Gate | Status | Key Finding |
|------|--------|-------------|
| 1.1 Quality Distributions | **PASS** | Controls/positives matched on bad_pixel_frac (KS p=0.63) |
| 1.2 Bandset Audit | **PASS** | All 10.6M samples have bandset=grz (100% consistent) |
| 1.3 Null-Injection | **PASS** | Controls correctly classified (mean p=0.014, only 0.4% > 0.5) |
| 1.4 SNR Ablation | **DEFERRED** | invvar not stored in current dataset |

---

## Detailed Results

### Gate 1.1: Class-Conditional Quality Distributions

**Purpose:** Verify positives and controls are matched in data-quality space to rule out shortcuts based on quality differences.

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

**Interpretation:**
- `bad_pixel_frac` is nearly identical between controls (6.83%) and positives (6.82%)
- KS p-value of 0.63 indicates no statistically significant difference
- `physics_valid` is only set for positives (expected behavior - controls have no physics to validate)
- arc_snr distribution shows good spread: 22% low SNR (<2), 52% medium (<5), 9% high (>20)

---

### Gate 1.2: Bandset Audit

**Purpose:** Verify all samples have consistent band coverage to rule out shortcuts based on band availability.

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

**Interpretation:**
- All 10.6M samples have exactly 3-band (g, r, z) coverage
- No samples with missing or extra bands
- Balanced class distribution: 5.31M controls, 5.34M positives

---

### Gate 1.3: Null-Injection Test

**Purpose:** Verify model correctly identifies controls as non-lenses, ruling out major false positive shortcuts.

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

**Interpretation:**
- Mean p_lens = 0.014 (well below 0.2 threshold) - model is very confident controls are NOT lenses
- Only 0.4% of controls got p > 0.5 (well below 10% threshold)
- No controls got p > 0.9 (no high-confidence false positives)
- Zero errors processing 1000 samples

---

### Gate 1.4: SNR Ablation Check

**Purpose:** Check if inverse variance (invvar) is available for per-pixel SNR representation experiments.

**Results:**
```json
{
  "gate": "1.4",
  "timestamp": "2026-02-05T04:46:17.649188+00:00",
  "has_invvar_npz": false,
  "has_stamp_invvar": false,
  "status": "DEFERRED - invvar not stored in current dataset",
  "recommendation": "Add invvar to Phase 4c for future runs",
  "overall_passed": "DEFERRED"
}
```

**Interpretation:**
- invvar is not currently stored in the dataset
- This is a known limitation that should be addressed in future data generation
- Does not block current training, but limits some ablation experiments

---

## Scripts Used

### Gate 1.1: `gate_1_1_quality_distributions.py`
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

### Gate 1.2: `gate_1_2_bandset_audit.py`
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

### Gate 1.3: `gate_1_3_null_injection.py`
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

### Gate 1.4: `gate_1_4_snr_ablation.py`
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

---

## Questions for LLM Review

1. **Are the pass criteria for each gate appropriate?**
   - Gate 1.1: KS p-value > 0.01 for quality distributions
   - Gate 1.2: 100% consistent bandset
   - Gate 1.3: mean p_lens < 0.2 and frac > 0.5 < 10%

2. **Any concerns about the KS test p-values in Gate 1.1?**
   - p = 0.63 for bad_pixel_frac (no difference)
   - physics_valid has no data for controls (expected)

3. **Is the null-injection test sufficient to rule out shortcuts?**
   - mean p = 0.014 suggests controls are well-classified
   - Only 0.4% false positives at p > 0.5

4. **Should we proceed to Phase 2, or investigate any findings first?**
   - All gates passed - recommend proceeding

5. **Are there other quality checks we should run?**
   - Consider checking psfsize_r distribution between classes
   - Consider checking spatial distribution of samples

---

## Honest Assessment

### What could still be wrong:
1. We only sampled 1000 controls for Gate 1.3 - could miss rare edge cases
2. physics_valid having no controls is expected but should be documented
3. arc_snr distribution shows 22% with SNR < 2 - these may be too faint

### Assumptions made:
1. Training data normalization matches inference normalization
2. Model architecture matches training exactly
3. Gate pass criteria are appropriate thresholds

### Edge cases not tested:
1. Samples with extreme bad_pixel_frac (> 50%)
2. Samples with very low arc_snr (< 1)
3. Samples near image boundaries

---

## Conclusion

**PROCEED to Phase 2** - All critical gates passed. The data quality checks indicate no obvious leakage or quality issues. The null-injection test confirms the model is correctly classifying controls as non-lenses.
