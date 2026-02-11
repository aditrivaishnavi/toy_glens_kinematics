# Data and Verification

**Purpose:** Single source of truth for data locations (S3 and local), sync steps, and verification checks.  
**Last updated:** 2026-02-10

**Self-contained:** `stronglens_calibration` does not depend on `dark_halo_scope`; all code and constants (e.g. DR10 zeropoint in `constants.py`) live in this package.

---

## 1. S3 source of truth

All production and EMR jobs use these S3 paths. Local paths are for development and testing only.

| Asset | S3 location (source of truth) | Notes |
|-------|--------------------------------|--------|
| **DESI DR1 spectroscopic catalog** | `s3://darkhaloscope/stronglens_calibration/data/external/desi_dr1/desi-sl-vac-v1.fits` | 2,176 rows. Upload via `scripts/sync_data_to_s3.sh` from local or emr-launcher. |
| **Positive catalog (lenscat)** | `s3://darkhaloscope/stronglens_calibration/data/positives/desi_candidates.csv` | 5,104 rows (435 confident, 4,669 probable). In repo; sync script uploads for EMR. |
| **Positives with DR10 crossmatch** | `s3://darkhaloscope/stronglens_calibration/positives_with_dr10/20260208_180524/` | 4,788 matched; Tier-A 389, Tier-B 4,399 (matched subset). |
| **Negative pool manifest** | `s3://darkhaloscope/stronglens_calibration/manifests/20260209_223513/` | 18.6M galaxies; N1 ~75%, N2 ~25%. |
| **Positive cutouts** | `s3://darkhaloscope/stronglens_calibration/cutouts/positives/20260208_205758/` | 5,101 .npz files. |
| **Negative cutouts (current training)** | `s3://darkhaloscope/stronglens_calibration/cutouts/negatives/20260209_040454/` | ~416K .npz (pre–N2-fix). New 510K stratified run TBD. |
| **Negative prototype (tests)** | `s3://darkhaloscope/stronglens_calibration/data/negatives/negative_catalog_prototype.csv` | Optional; for EMR-launcher runs. Local tests use repo `data/negatives/` (gitignored). |
| **Training checkpoints (Lambda)** | NFS: `/lambda/nfs/darkhaloscope-training-dc/stronglens_calibration/checkpoints/` | best.pt / last.pt; sync to S3 optional. |

---

## 2. Local-only data (not in git)

These paths exist only where someone has placed or generated the files. They are in `.gitignore`. To make them available on S3 or on emr-launcher:

1. **From your machine:** rsync to emr-launcher, then upload from there:
   ```bash
   rsync -avz --progress \
     data/external/desi_dr1/ \
     data/positives/ \
     data/negatives/ \
     EMR_LAUNCHER_HOST:/data/stronglens_calibration/
   ```
2. **On emr-launcher** (with AWS configured):
   ```bash
   cd /data/stronglens_calibration
   ./scripts/sync_data_to_s3.sh --from /data/stronglens_calibration
   ```
3. Or from **local** (if AWS is configured):
   ```bash
   cd stronglens_calibration
   ./scripts/sync_data_to_s3.sh
   ```

| Asset | Local path | In repo? |
|-------|------------|----------|
| DESI DR1 FITS | `data/external/desi_dr1/desi-sl-vac-v1.fits` | No (.gitignore) |
| Negative prototype CSV | `data/negatives/negative_catalog_prototype.csv` | No (.gitignore) |

---

## 3. Phase 4 (S4) checks

Verification that training and evaluation setup match the checklist (Phase 4).

| ID | Check | Status | Evidence |
|----|--------|--------|----------|
| 4.1 | Primary architecture ResNet18 | ✅ | `dhs/model.py` build_resnet18(3) |
| 4.2 | 20–40 epochs, cosine schedule, early stopping | ✅ | `configs/resnet18_baseline_v1.yaml`: epochs 50, early_stopping_patience 10, CosineAnnealingLR |
| 4.3 | Batch size 256 (64×64), AMP | ✅ | Config: batch_size 256, mixed_precision true |
| 4.4 | Train baseline with clean splits | ✅ | resnet18_baseline_v1 trained 2026-02-10; 16 epochs (early stopped), best val AUC 0.9592. See MODELS_PERFORMANCE.md. |
| 4.5 | Calibration curves by stratum | ✅ DONE | Run on Lambda 2026-02-10. Overall ECE 0.0027, MCE 0.0156. See `docs/EVALUATION_4.5_4.6_LLM_REVIEW.md`. |
| 4.6 | Independent validation (spectroscopic) | ✅ DONE | Tier-A n=48, recall @ 0.5 = 0.6042. Same run; see EVALUATION_4.5_4.6_LLM_REVIEW.md for LLM review. |
| 4.7 | Annulus-only classifier (strong) | ☐ PENDING | |
| 4.8 | Core-only classifier (weak) | ☐ PENDING | |
| 4.9 | Freeze model before selection function | ☐ PENDING | Model frozen after run; selection function not yet run |
| 4.10 | Safe augmentations defined | ✅ | Config: hflip, vflip, rot90; see checklist 4.10 |

---

## 4. Checklist verification reference

- **P0.1 / D.3:** Spectroscopic catalog verified where file exists (local or S3). Canonical location: S3 above; run `sync_data_to_s3.sh` to upload from local.
- **P0.5:** Tier-A = 435 confident (raw `desi_candidates.csv`); post-crossmatch Tier-A = 389 (matched set).
- **2.1 / 2.3:** Tier-based sample weights implemented in `stronglens_calibration/dhs/train.py` and `dhs/data.py` (not planc). 2.3 = DONE.
- **3A.1:** DR10 zeropoint 22.5 verified in `stronglens_calibration/constants.py` as `AB_ZEROPOINT_MAG = 22.5`. No dependency on dark_halo_scope.
- **PF.1 / PF.2:** Require `data/negatives/negative_catalog_prototype.csv`. Run locally when that file exists; see test results in this doc or checklist after run.

---

## 5. Test run results

Run from repo root: `cd stronglens_calibration && python3 tests/<script>.py` (or `python3 -m pytest tests/<script>.py -v` if pytest is available).

| Test | When run | Result | Notes |
|------|----------|--------|--------|
| test_phase1_local.py | 2026-02-10 | **PASS** (12 tests) | Pool design, HEALPix/splits, schema, quality gates, provenance (1A–1E). Run: `python3 tests/test_phase1_local.py` |
| test_pipeline_local.py | 2026-02-10 | **PASS** | 50K input → 2,555 output; N1/N2 pools, splits, 5/5 quality checks. Run: `python3 tests/test_pipeline_local.py` |

Both tests run **locally** (no S3 required). For EMR or Lambda runs that need S3, use the same test scripts on the target environment after syncing data per §2.
