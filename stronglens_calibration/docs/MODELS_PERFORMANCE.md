# Model Performance & Model Cards

**Purpose:** Full model card and data provenance for every trained model in stronglens_calibration.  
**Last updated:** 2026-02-11  
**Maintained by:** Project owner — update this file whenever a new model is trained or re-evaluated.

---

## Verification checklist (before marking a model card complete)

- [ ] All paths (checkpoint, manifest, cutout sources) exist and are correct for the host (Lambda vs local).
- [ ] Config file content matches what was used (no drift); if copied to checkpoint dir, note that.
- [ ] Train/val/test counts match the manifest (or document "from manifest" and where to get counts).
- [ ] Positive/negative cutout sources (S3 path + timestamp or NFS path) are recorded.
- [ ] Command and cwd are copy-pastable for reproduce.
- [ ] If evaluation (gates, Tier-A recall, ECE) was run, all metrics are filled; otherwise "Not run" is explicit.

---

## How to use this document

- **Add a new section** (e.g. `## Model: resnet18_baseline_v2`) for each new training run.
- **Copy the template** below and fill in every field; use `TBD` or `Not run` only where unknown, and add a short note (e.g. "To verify: negative cutout S3 timestamp").
- **Verify** paths, metrics, and commands on the machine where the model was trained (e.g. Lambda NFS) before finalizing.

---

## Template (copy for new models)

```markdown
## Model: <run_id>

### 1. Identification
| Field | Value |
|-------|--------|
| Run ID |  |
| Config file |  |
| Checkpoint location (host) |  |
| Checkpoint location (S3, if synced) |  |

### 2. Model architecture & hyperparameters
| Parameter | Value |
|-----------|--------|
| Architecture |  |
| Input channels |  |
| Input size (after preprocessing) |  |
| Optimizer |  |
| Learning rate |  |
| Weight decay |  |
| Batch size |  |
| Max epochs |  |
| Early stopping patience |  |
| LR schedule |  |
| Mixed precision (AMP) |  |
| Loss |  |

### 3. Data
| Field | Value |
|-------|--------|
| Manifest path |  |
| Dataset mode |  |
| Preprocessing |  |
| Train samples |  |
| Val samples |  |
| Test samples |  |
| Positive count |  |
| Negative count |  |
| Neg:pos ratio |  |
| Tier-A (eval) count |  |
| Tier-B (training) count |  |
| N1:N2 ratio (if applicable) |  |
| Positive cutout source (S3 or path) |  |
| Negative cutout source (S3 or path) |  |
| Split assignment method |  |
| Seed |  |

### 4. Training run
| Field | Value |
|-------|--------|
| Host |  |
| Start time (UTC) |  |
| End time (UTC) |  |
| Epochs completed |  |
| Stop reason |  |
| Command used |  |
| Git commit (code) |  |

### 5. Metrics (validation)
| Metric | Value |
|--------|--------|
| Best val AUC |  |
| Best epoch |  |
| Final train loss (last epoch) |  |

### 6. Metrics (evaluation – if run)
| Metric | Value |
|--------|--------|
| Tier-A recall @ threshold 0.5 |  |
| Tier-A AUC (test) |  |
| ECE |  |
| FPR by N2 category (if run) |  |
| Shortcut gates (core-only AUC, etc.) |  |

### 7. Artifacts
| Artifact | Path |
|----------|------|
| best.pt |  |
| last.pt |  |
| train.log |  |
| config (copy) |  |

### 8. Steps to reproduce
1. Data: …
2. Manifest: …
3. Train: …
```

---

## Model: resnet18_baseline_v1

### 1. Identification
| Field | Value |
|-------|--------|
| Run ID | resnet18_baseline_v1 |
| Config file | `configs/resnet18_baseline_v1.yaml` |
| Checkpoint location (host) | `/lambda/nfs/darkhaloscope-training-dc/stronglens_calibration/checkpoints/resnet18_baseline_v1/` |
| Checkpoint location (S3, if synced) | TBD (not synced as of 2026-02-10) |

### 2. Model architecture & hyperparameters
| Parameter | Value |
|-----------|--------|
| Architecture | ResNet-18 (torchvision, `weights=None`), conv1 adapted to 3 channels, fc → 1 output |
| Input channels | 3 (g, r, z) |
| Input size (after preprocessing) | 64×64 (center crop from 101×101) |
| Optimizer | AdamW |
| Learning rate | 0.0003 (3e-4) |
| Weight decay | 0.0001 |
| Batch size | 256 |
| Max epochs | 50 |
| Early stopping patience | 10 |
| LR schedule | CosineAnnealingLR, T_max=50 |
| Mixed precision (AMP) | true |
| Loss | BCEWithLogitsLoss (reduction='none'), then per-sample weighted mean (sample_weight from manifest) |

### 3. Data
| Field | Value |
|-------|--------|
| Manifest path | `/lambda/nfs/darkhaloscope-training-dc/stronglens_calibration/manifests/training_v1.parquet` |
| Dataset mode | `file_manifest` |
| Preprocessing | `raw_robust` (outer-annulus normalize: median/MAD from r=20-32 px, then clip +/-10) |
| Train samples | 291,509 |
| Val samples | 62,180 |
| Test samples | 62,760 |
| Positive count | 4,788 |
| Negative count | 411,661 |
| Total | 416,449 |
| Neg:pos ratio | ~86:1 |
| Tier-A (eval) count | 389 (confident); sample_weight=1.0 |
| Tier-B (training) count | 4,399 (probable); sample_weight=0.5 |
| N1:N2 ratio | 85:15 (N1=349,833 N2=61,828). Confuser categories: edge_on_proxy, ring_proxy, large_galaxy, blue_clumpy. |
| Positive cutout source | `s3://darkhaloscope/stronglens_calibration/cutouts/positives/` (4,789 files) |
| Negative cutout source | `s3://darkhaloscope/stronglens_calibration/cutouts/negatives/20260210_025117/` (416,089 files) |
| Split assignment method | HEALPix nside=128, SHA-256 hash (seed=42), 70/15/15. See `docs/SPLIT_ASSIGNMENT.md`. |
| Seed | 1337 (dataset/config) |

### 4. Training run
| Field | Value |
|-------|--------|
| Host | Lambda Labs GPU, NFS mount `darkhaloscope-training-dc` |
| Start time (UTC) | 2026-02-10 ~15:40 UTC |
| End time (UTC) | 2026-02-10 ~16:49 UTC (last.pt write time) |
| Epochs completed | 16 |
| Stop reason | Early stopping: 10 consecutive epochs without improvement in val AUC (best at epoch 6). |
| Command used | `python3 code/dhs/scripts/run_experiment.py --config code/configs/resnet18_baseline_v1.yaml` from cwd `/lambda/nfs/darkhaloscope-training-dc/stronglens_calibration` |
| Git commit (code) | Not a git repo on Lambda (code deployed via scp). Config SHA256: `2445b4c4...`. See `checkpoints/resnet18_baseline_v1/run_info.json`. |

### 5. Metrics (validation)
| Metric | Value |
|--------|--------|
| Best val AUC | 0.9592 |
| Best epoch | 6 |
| Final train loss (last epoch) | 0.0013 (epoch 16) |

### 6. Metrics (evaluation on test set, n=62,760)
| Metric | Value | 95% Bootstrap CI |
|--------|--------|------------------|
| Test AUC (overall) | 0.9579 | [0.950, 0.965] |
| Recall @ 0.5 (overall) | 0.4360 | [0.397, 0.473] |
| Precision @ 0.5 (overall) | 0.7653 | [0.721, 0.805] |
| ECE (overall, 15 bins) | 0.0027 | - |
| MCE (overall) | 0.0156 | - |
| Tier-A (held-out evaluation) n | 48 (all positive, test split) | - |
| Tier-A recall @ 0.5 | 0.6042 | [0.458, 0.750] |
| Tier-A ECE/MCE | Not reported (single-class stratum; ECE not meaningful) | - |
| Tier-B recall @ 0.5 | 0.4234 | [0.386, 0.458] |
| FPR overall (negatives) @ 0.5 | 0.0015 (92/62,072) | - |
| FPR N1 @ 0.5 | 0.0008 (41/52,670) | - |
| FPR N2 @ 0.5 | 0.0054 (51/9,402) | - |
| FPR edge_on_proxy @ 0.5 | 0.0074 (44/5,929) | - |
| FPR ring_proxy @ 0.5 | 0.0023 (4/1,752) | - |
| FPR large_galaxy @ 0.5 | 0.0029 (3/1,031) | - |
| FPR blue_clumpy @ 0.5 | 0.0000 (0/690) | - |
| Shortcut gates (core-only AUC) | Not run | - |

Bootstrap: 2,000 iterations, seed=1337. Full report: `docs/bootstrap_eval_test.json`.
FPR report: `docs/fpr_by_confuser_category.json`.
Split verification: `docs/split_verification_report.json` (0 overlaps across all split pairs).

### 7. Selection Function (injection-recovery, minimal proxy injector)
| Field | Value |
|-------|--------|
| Grid | 11 thetaE x 7 PSF x 5 depth = 385 cells |
| Injections per cell | 200 |
| Sufficient cells | 242 / 385 (63%) |
| Empty cells | 143 (shallow depth bins with few hosts) |
| Mean completeness (sufficient) | 0.526 |
| Completeness range | 0.00 - 0.99 |
| Trend: thetaE | Correct (0.43 at 0.5" to 0.58 at 3.0") |
| Trend: PSF | Correct (0.73 at 0.9" FWHM) |
| Note | Minimal proxy injector; replace with Phase4c for publication. |

Results: `docs/selection_function.csv`, `docs/selection_function_meta.json`.

### 8. Artifacts
| Artifact | Path |
|----------|------|
| best.pt | `checkpoints/resnet18_baseline_v1/best.pt` (Lambda NFS) |
| last.pt | `checkpoints/resnet18_baseline_v1/last.pt` (Lambda NFS) |
| train.log | `checkpoints/resnet18_baseline_v1/train.log` (Lambda NFS) |
| run_info.json | `checkpoints/resnet18_baseline_v1/run_info.json` (Lambda NFS) |
| config | `configs/resnet18_baseline_v1.yaml` (in repo) |
| eval JSON | `docs/eval_resnet18_baseline_v1_sanitized.json` (sanitized for supplement) |

### 9. Steps to reproduce
1. **Data:** Sync cutouts from S3 to Lambda NFS.
2. **Manifest:** `python3 scripts/generate_training_manifest_parallel.py --positives .../cutouts/positives/ --negatives .../cutouts/negatives/20260210_025117/ --output manifests/training_v1.parquet`
3. **Train:** `python3 code/dhs/scripts/run_experiment.py --config code/configs/resnet18_baseline_v1.yaml` from `/lambda/nfs/darkhaloscope-training-dc/stronglens_calibration`
4. **Evaluate:** `PYTHONPATH=. python3 dhs/scripts/run_evaluation.py --config configs/resnet18_baseline_v1.yaml --export_predictions results/test_predictions_full.csv` from `code/`
5. **Bootstrap:** `PYTHONPATH=. python3 scripts/bootstrap_eval.py --manifest ../manifests/training_v1.parquet --preds ../results/test_predictions_full.csv --out_json ../docs/bootstrap_eval_test.json` from `code/`

### 10. Notes
- **Data provenance:** Fully verified 2026-02-11. All paths, counts, and split ratios confirmed.
- **Code version:** `run_info.json` created with config SHA256, seed, command, and environment versions.
- **Tier-A wording:** "Held-out Tier-A evaluation" (not "independent spectroscopic validation"). Same imaging-candidate source, held out from training.

---

## Changelog

| Date | Change |
|------|--------|
| 2026-02-10 | Initial document; added template and resnet18_baseline_v1 card. |
| 2026-02-11 | Verified all data provenance. Added split counts, N1:N2 ratio, cutout sources, split method, bootstrap CIs, FPR by confuser category, selection function results, run_info.json. Fixed Tier-A ECE (not reported for single-class). |
