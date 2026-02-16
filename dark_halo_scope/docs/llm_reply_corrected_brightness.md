# Reply: Corrected Brightness Analysis

Thank you for the thorough review. I've addressed your feedback on the brightness metric and have additional questions.

---

## 1. Corrected Brightness Metric

### Original (Problematic)
```python
brightness = cutout_r.mean()  # Mean over ALL 4096 pixels - sky dominated
```

### Corrected (Defensible)
```python
def central_aperture_flux(img, radius=8):
    """Compute mean flux in central aperture (r < radius pixels)."""
    if img.ndim == 3:
        img = img[0]
    h, w = img.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    mask = ((yy - cy)**2 + (xx - cx)**2) < radius**2
    return float(np.mean(img[mask]))  # ~200 pixels in galaxy core
```

---

## 2. Updated Results

**Both measured via Legacy Survey cutout service with `bands=r` (fair comparison):**

| Metric | Anchor Mean (n=10) | LRG Mean (n=15) | Ratio |
|--------|-------------------|-----------------|-------|
| Full stamp mean | 0.000627 nMgy | 0.003854 nMgy | **6.2x** |
| Central aperture r<8 | 0.000546 nMgy | 0.023957 nMgy | **43.8x** |

### Individual Anchor Measurements

| Name | Full Mean | Central r<8 |
|------|-----------|-------------|
| SDSSJ0029-0055 | 0.000328 | 0.000816 |
| SDSSJ0037-0942 | 0.000328 | 0.000822 |
| SDSSJ0252+0039 | -0.000048 | 0.000112 |
| SDSSJ0330-0020 | 0.000226 | 0.000087 |
| SDSSJ0728+3835 | 0.000341 | -0.000159 |
| SDSSJ0737+3216 | 0.000360 | 0.000027 |
| SDSSJ0912+0029 | 0.002887 | 0.002241 |
| SDSSJ0959+0410 | 0.000150 | 0.000683 |
| SDSSJ1016+3859 | 0.001464 | 0.000449 |
| SDSSJ1020+1122 | 0.000231 | 0.000385 |

### Individual LRG Measurements

| LRG | Full Mean | Central r<8 |
|-----|-----------|-------------|
| LRG_0 | 0.002562 | 0.023958 |
| LRG_1 | 0.028158 | 0.120339 |
| LRG_2 | 0.004485 | 0.015968 |
| LRG_3 | 0.001476 | 0.018049 |
| LRG_4 | 0.002288 | 0.025999 |
| LRG_5 | 0.001189 | 0.014484 |
| LRG_6 | 0.002659 | 0.016638 |
| LRG_7 | 0.003513 | 0.017275 |
| LRG_8 | 0.001307 | 0.013573 |
| LRG_9 | 0.000685 | 0.007464 |
| LRG_10 | 0.000524 | 0.005115 |
| LRG_11 | 0.001297 | 0.015336 |
| LRG_12 | 0.003582 | 0.004498 |
| LRG_13 | 0.001907 | 0.027624 |
| LRG_14 | 0.002186 | 0.033035 |

---

## 3. Full Code Used

```python
#!/usr/bin/env python3
"""
Corrected Brightness Comparison: Central Aperture Metric
Both anchors and LRGs fetched via cutout service for fair comparison.
"""
import numpy as np
import requests
import tempfile
import os
import io
from astropy.io import fits
import boto3
import pyarrow.parquet as pq

s3 = boto3.client('s3')

def fetch_cutout(ra, dec):
    """Fetch single-band r cutout from Legacy Survey."""
    url = f"https://www.legacysurvey.org/viewer/cutout.fits?ra={ra}&dec={dec}&size=64&layer=ls-dr10&pixscale=0.262&bands=r"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            return None
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            f.write(resp.content)
            tmp_path = f.name
        try:
            with fits.open(tmp_path) as hdul:
                data = hdul[0].data
                if data is None:
                    return None
                return data.astype(np.float32)
        finally:
            os.unlink(tmp_path)
    except:
        return None

def central_aperture_flux(img, radius=8):
    """Compute mean flux in central aperture (r < radius pixels)."""
    if img.ndim == 3:
        img = img[0]
    h, w = img.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    mask = ((yy - cy)**2 + (xx - cx)**2) < radius**2
    return float(np.mean(img[mask]))

def full_stamp_mean(img):
    if img.ndim == 3:
        img = img[0]
    return float(np.mean(img))

# SLACS/BELLS anchors
ANCHORS = [
    {"name": "SDSSJ0029-0055", "ra": 7.4543, "dec": -0.9254},
    {"name": "SDSSJ0037-0942", "ra": 9.3004, "dec": -9.7095},
    {"name": "SDSSJ0252+0039", "ra": 43.1313, "dec": 0.6651},
    {"name": "SDSSJ0330-0020", "ra": 52.5019, "dec": -0.3419},
    {"name": "SDSSJ0728+3835", "ra": 112.1879, "dec": 38.5900},
    {"name": "SDSSJ0737+3216", "ra": 114.4121, "dec": 32.2793},
    {"name": "SDSSJ0912+0029", "ra": 138.0200, "dec": 0.4894},
    {"name": "SDSSJ0959+0410", "ra": 149.7954, "dec": 4.1755},
    {"name": "SDSSJ1016+3859", "ra": 154.1092, "dec": 38.9937},
    {"name": "SDSSJ1020+1122", "ra": 155.1029, "dec": 11.3690},
]

# Measure anchors
anchor_full = []
anchor_central = []
for coord in ANCHORS:
    img = fetch_cutout(coord['ra'], coord['dec'])
    if img is not None:
        anchor_full.append(full_stamp_mean(img))
        anchor_central.append(central_aperture_flux(img, radius=8))

# Get training LRG coordinates from manifest
resp = s3.list_objects_v2(
    Bucket='darkhaloscope',
    Prefix='phase4_pipeline/phase4a/v4_sota_moffat/manifests/train_stamp64_bandsgrz_gridgrid_sota/',
    MaxKeys=5
)
pf = [obj['Key'] for obj in resp.get('Contents', []) if obj['Key'].endswith('.parquet')][0]
obj = s3.get_object(Bucket='darkhaloscope', Key=pf)
tbl = pq.read_table(io.BytesIO(obj['Body'].read()))
df = tbl.to_pandas()
coords = df[['ra', 'dec']].drop_duplicates().head(15)

# Measure LRGs via same cutout service
lrg_full = []
lrg_central = []
for _, row in coords.iterrows():
    img = fetch_cutout(row['ra'], row['dec'])
    if img is not None:
        lrg_full.append(full_stamp_mean(img))
        lrg_central.append(central_aperture_flux(img, radius=8))

# Summary
print(f"Full stamp:  Anchor={np.mean(anchor_full):.6f}, LRG={np.mean(lrg_full):.6f}, Ratio={np.mean(lrg_full)/np.mean(anchor_full):.1f}x")
print(f"Central r<8: Anchor={np.mean(anchor_central):.6f}, LRG={np.mean(lrg_central):.6f}, Ratio={np.mean(lrg_central)/np.mean(anchor_central):.1f}x")
```

---

## 4. Updated Conclusion

- **Original claim (95.7x)**: Was inflated due to comparing different data sources
- **Corrected claim (43.8x)**: Defensible, both via same cutout service, central aperture metric
- **Core conclusion unchanged**: Training LRGs are ~44x brighter than SLACS/BELLS anchors

---

## 5. Questions for Next Steps

### Tier-A Anchor Selection

**Q1**: For building a "DR10-detectable" anchor set, which catalogs/surveys would you recommend?

Options we're considering:
- **SuGOHI** (Subaru Hyper Suprime-Cam, ground-based, similar depth)
- **Master Lens Database** (subset with visible arcs in DR10)
- **Galaxy Zoo / Space Warps** citizen science discoveries
- **DES strong lens candidates** (Dark Energy Survey)

Which would give the most statistically robust and scientifically defensible anchor set?

### Hard Negatives

**Q2**: For hard negatives (ring galaxies, spirals with strong arms), what's the best source?

Options:
- Galaxy Zoo morphological classifications (ring/spiral tags)
- Synthetic ring injection (procedural rings without lensing physics)
- SDSS spiral catalog
- SIMBAD object type queries

Should we aim for ~10% hard negatives in training, or higher?

### Injection Brightness Calibration

**Q3**: To match DR10 detectability, which approach is better?

- **Option A**: Sample source magnitudes from Tier-A anchor arc brightness distribution
- **Option B**: Define a visibility proxy (e.g., arc_snr > 2) and filter injections
- **Option C**: Both - sample magnitudes AND filter by visibility

### Center-Masked Ablation

**Q4**: For the "shortcut-killer" center-masked training:

- Is r<8 pixels the right mask radius?
- What noise model for masked region: Gaussian from outer annulus stats, or shuffle pixels from outer region?
- Should we mask during training only, or also at inference?

### Region-Disjoint Splits

**Q5**: For publication-quality splits:

- How many independent sky regions is sufficient (e.g., 10? 20?)?
- Should we split by RA/Dec blocks or by brick boundaries?
- Should stratification by difficulty (theta/psf, psfdepth) be done within each region?

### Sample Size

**Q6**: Given the corrected 43.8x central aperture ratio:

- Is n=10 anchors + n=15 LRGs sufficient evidence for the "SLACS/BELLS are too faint" claim?
- If not, should we expand to n≥50 for both?
- Would a KS test p-value on the central aperture distributions strengthen the claim?

---

## 6. Additional Question: Training Data Validation

**Q7**: Before proceeding with Tier-A anchors and hard negatives, should we first validate that our Gen5 training data doesn't have other issues?

Specifically:
- Should we check `bad_pixel_frac`, `maskbit_frac` distributions by class?
- Should we verify `bandset` is consistent across all samples?
- Any other columns that could indirectly leak class information?

---

## 7. Proposed Priority Order

Based on your recommendations, here's our proposed order. Please confirm or adjust:

1. **Build Tier-A anchor set** (SuGOHI + Master Lens DB subset)
2. **Add hard negatives** (Galaxy Zoo rings/spirals)
3. **Run center-masked ablation** (quick diagnostic)
4. **Calibrate injection brightness** (match Tier-A arc distribution)
5. **Add photometric jitter augmentation**
6. **Create region-disjoint splits**
7. **Retrain with all fixes**
8. **Evaluate on Tier-A (primary) + SLACS/BELLS (stress test)**

Is this the right order, or should we reorder based on expected impact?
