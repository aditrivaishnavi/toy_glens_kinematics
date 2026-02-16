# Anchor & Contaminant Set: Runtime Configuration

**Date**: 2026-02-07
**Status**: Ready to execute

---

## Quick Summary

| Item | Value |
|------|-------|
| **Where to run** | Local machine + NFS storage |
| **Data targets** | 100+ anchors, 200+ contaminants |
| **Cutout download** | ~300 FITS files, ~50 MB total |
| **Runtime** | 2-4 hours (mostly download time) |
| **Dependencies** | astroquery, requests, pandas, astropy |

---

## 1. Data Collection Targets

### Anchors

| Source | Target N | How to Get | Priority |
|--------|----------|------------|----------|
| **SLACS** | 68 | Already have on NFS | ✅ Done |
| **BELLS** | 25 | Already have on NFS | ✅ Done |
| **Huang+2020 LS ML** | 50+ | Download from paper | High |
| **Jacobs+2019** | 30+ | VizieR query | Medium |
| **SL2S** | 20+ | VizieR query | Medium |
| **Total** | **100-150** | | |

**Existing data location**:
```
/lambda/nfs/darkhaloscope-training-dc/anchor_cutouts/known_lenses/
  - 68 FITS files (SLACS + BELLS)
```

### Contaminants

| Category | Target N | Source |
|----------|----------|--------|
| **Ring galaxies** | 50 | Galaxy Zoo (p_ring > 0.5) |
| **Face-on spirals** | 50 | Galaxy Zoo (p_spiral > 0.8) |
| **Mergers** | 30 | Galaxy Zoo Mergers catalog |
| **Diffraction spikes** | 20 | Gaia bright stars in DR10 |
| **Edge-on disks** | 20 | Galaxy Zoo |
| **Total** | **170-200** | |

---

## 2. Runtime Environment

### Option A: Run Locally (Recommended for catalog building)

```bash
# Install dependencies
pip install astroquery astropy pandas requests tqdm

# Create working directory
mkdir -p ~/anchor_contaminant_data/{anchors,contaminants,cutouts}
```

### Option B: Run on Lambda (for GPU-based arc visibility measurement)

```bash
# SSH to lambda
ssh lambda

# Ensure packages are installed
pip install astroquery

# Working directory
cd /lambda/nfs/darkhaloscope-training-dc/
```

---

## 3. Step-by-Step Execution

### Step 1: Build Anchor Catalog (1 hour)

```bash
# Create the script
cd /path/to/dark_halo_scope/planb

# Run catalog builder
python scripts/build_anchor_catalog.py \
    --existing-csv /lambda/nfs/darkhaloscope-training-dc/anchors/slacs_bells.csv \
    --output-csv anchors/combined_anchors.csv \
    --download-huang2020 \
    --download-jacobs2019 \
    --verbose
```

**Expected output**: `anchors/combined_anchors.csv` with 100-150 rows

### Step 2: Build Contaminant Catalog (1 hour)

```bash
python scripts/build_contaminant_catalog.py \
    --output-csv contaminants/contaminant_catalog.csv \
    --n-rings 50 \
    --n-spirals 50 \
    --n-mergers 30 \
    --n-spikes 20 \
    --anchor-csv anchors/combined_anchors.csv \
    --verbose
```

**Expected output**: `contaminants/contaminant_catalog.csv` with 170+ rows

### Step 3: Download Cutouts (1-2 hours)

```bash
python scripts/download_cutouts.py \
    --anchor-csv anchors/combined_anchors.csv \
    --contaminant-csv contaminants/contaminant_catalog.csv \
    --output-dir /lambda/nfs/darkhaloscope-training-dc/evaluation_cutouts/ \
    --size 64 \
    --pixscale 0.262 \
    --bands grz \
    --parallel 4 \
    --verbose
```

**Expected output**:
```
evaluation_cutouts/
├── anchors/
│   ├── SLACS_J0029-0055.fits
│   └── ...
└── contaminants/
    ├── ring/
    ├── spiral/
    ├── merger/
    └── spike/
```

### Step 4: Measure Arc Visibility (30 min)

```bash
# On lambda (needs GPU for model inference, or can use CPU)
python scripts/measure_arc_visibility.py \
    --anchor-csv anchors/combined_anchors.csv \
    --cutout-dir /lambda/nfs/darkhaloscope-training-dc/evaluation_cutouts/anchors/ \
    --output-csv anchors/anchors_with_arc_snr.csv \
    --method "annulus_snr"
```

### Step 5: Apply Selection Function (5 min)

```python
from planb.evaluation import AnchorSet, AnchorSelectionFunction

sf = AnchorSelectionFunction(
    theta_e_min=0.5,
    theta_e_max=3.0,
    arc_snr_min=2.0,
    on_missing_arc_visibility="TIER_B",
    on_missing_dr10_flag="EXCLUDE",
)

anchors = AnchorSet.from_csv(
    "anchors/anchors_with_arc_snr.csv",
    selection_function=sf,
)

print(anchors.summary())
# Expected: 40-60 Tier-A, 60-90 Tier-B
```

### Step 6: Evaluate Model (10 min)

```python
# Load trained model and score cutouts
model = load_model("/lambda/nfs/.../ckpt_best.pt")

scores = {}
for anchor in anchors.tier_a.itertuples():
    cutout = load_cutout(f"evaluation_cutouts/anchors/{anchor.name}.fits")
    scores[anchor.name] = model.predict(cutout)

results = anchors.evaluate(scores, threshold=0.5)
print(f"Tier-A Recall: {results['tier_a_recall']:.1%}")
```

---

## 4. Data Sources & URLs

### Anchor Catalogs

| Source | URL/Method |
|--------|------------|
| SLACS | Already on NFS |
| BELLS | Already on NFS |
| Huang+2020 | https://arxiv.org/abs/2008.04767 (Table 2 in appendix) |
| Jacobs+2019 | VizieR `J/MNRAS/484/5330` |
| SL2S | VizieR `J/A+A/556/A79` |

### Contaminant Catalogs

| Source | VizieR Catalog |
|--------|----------------|
| Galaxy Zoo 2 | `J/MNRAS/435/2835/gz2` |
| Galaxy Zoo Mergers | `J/MNRAS/401/1552` |
| Gaia DR3 | `I/355/gaiadr3` (for bright stars) |

### Cutout Service

```python
# Legacy Survey cutout URL template
url = (
    f"https://www.legacysurvey.org/viewer/fits-cutout"
    f"?ra={ra}&dec={dec}"
    f"&layer=ls-dr10"
    f"&pixscale=0.262"
    f"&bands=grz"
    f"&size=64"
)
```

---

## 5. Storage Requirements

| Item | Size |
|------|------|
| Anchor cutouts (150 × 64×64×3 FITS) | ~15 MB |
| Contaminant cutouts (200 × 64×64×3 FITS) | ~20 MB |
| CSV catalogs | < 1 MB |
| **Total** | **~40 MB** |

**Storage location**: `/lambda/nfs/darkhaloscope-training-dc/evaluation_cutouts/`

---

## 6. Dependencies

```bash
# requirements.txt additions
astroquery>=0.4.6
astropy>=5.0
requests>=2.28
tqdm>=4.64
```

---

## 7. Fallback: If astroquery Fails

Some VizieR queries can be slow or fail. Fallback options:

1. **Direct VizieR TAP**: Use `pyvo` instead of `astroquery`
2. **Pre-downloaded CSVs**: Download manually from VizieR web interface
3. **Literature tables**: Copy coordinates from paper PDFs into CSV

---

## 8. Parallelization

Cutout downloads can be parallelized:

```python
from concurrent.futures import ThreadPoolExecutor

def download_cutout(args):
    name, ra, dec = args
    url = build_cutout_url(ra, dec)
    response = requests.get(url, timeout=30)
    save_fits(response.content, f"cutouts/{name}.fits")

with ThreadPoolExecutor(max_workers=4) as executor:
    executor.map(download_cutout, anchor_list)
```

**Rate limit**: Legacy Survey allows ~10 requests/sec; with 4 workers, ~400 cutouts takes 2 minutes.

---

## 9. Checklist

- [ ] Install dependencies (`astroquery`, etc.)
- [ ] Create scripts:
  - [ ] `build_anchor_catalog.py`
  - [ ] `build_contaminant_catalog.py`
  - [ ] `download_cutouts.py`
  - [ ] `measure_arc_visibility.py`
  - [ ] `evaluate_on_real_data.py`
- [ ] Download and merge anchor catalogs (target: 100+)
- [ ] Query and build contaminant catalog (target: 170+)
- [ ] Download cutouts from Legacy Survey
- [ ] Measure arc visibility on anchors
- [ ] Apply selection function
- [ ] Evaluate trained model
- [ ] Generate report

---

## 10. Timeline

| Task | Duration | Can Start |
|------|----------|-----------|
| Create scripts | 1 hr | NOW |
| Download anchor catalogs | 30 min | NOW |
| Download contaminant catalogs | 30 min | NOW |
| Download cutouts | 1-2 hrs | After catalogs |
| Measure arc visibility | 30 min | After cutouts |
| Run evaluation | 10 min | After Phase 4 training completes |

**Total**: 3-4 hours
**Can start**: NOW (parallel with training)
