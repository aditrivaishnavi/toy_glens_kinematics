# Evaluation 4.5 & 4.6 – LLM review instructions

**Purpose:** After running the evaluation script (4.5 calibration, 4.6 independent validation), paste the outputs here for a second LLM to review.

---

## Dependencies

Same as training: `torch`, `numpy`, `pandas`, `pyarrow`, `scikit-learn`, `PyYAML`. Ensure the Lambda (or eval) environment has these.

---

## How to produce the outputs

Run on **Lambda** (or wherever the checkpoint and manifest/cutouts live):

```bash
cd /path/to/stronglens_calibration   # e.g. /lambda/nfs/darkhaloscope-training-dc/stronglens_calibration
python dhs/scripts/run_evaluation.py \
  --config configs/resnet18_baseline_v1.yaml \
  --output results/eval_resnet18_baseline_v1.json \
  --summary results/eval_resnet18_baseline_v1_summary.md
```

- **JSON** (`--output`): full metrics, reliability curve bins, by-stratum (Tier-A / Tier-B), independent validation block.
- **Summary** (`--summary`): short markdown for humans/LLM (overall + stratum + 4.6 block).

If the manifest has no `tier` column, by-stratum and independent validation will be empty; then either add tier to the manifest or run on a manifest that includes it (e.g. from `generate_training_manifest_parallel.py` with tier in NPZ metadata).

---

## Checklist 4.5 (Calibration curves by stratum)

- **ECE (Expected Calibration Error):** Lower is better; often < 0.05 is considered well calibrated. Review overall and, if present, by stratum (Tier-A / Tier-B).
- **MCE (Maximum Calibration Error):** Max deviation in any bin; should not be large (e.g. < 0.1).
- **Reliability curve:** In the JSON, `reliability_curve` has `bin_acc` (accuracy per bin) and `bin_conf` (mean predicted probability per bin). For good calibration, these should be close (diagonal in a reliability diagram). Flag if any bin is far off.

---

## Checklist 4.6 (Independent validation – spectroscopic)

- **Tier-A** (confident lenses with DR10 match) is used as the *independent spectroscopic validation set*.
- Review **independent_validation_spectroscopic** (or **by_stratum.tier_A**): AUC, recall @ 0.5, ECE.
- Check that Tier-A metrics are **consistent** with overall test set (no large drop). A moderate drop is acceptable (Tier-A is harder/more independent); a large drop or very low recall should be flagged.
- Confirm that the description states these are Tier-A (confident, DR10-matched) lenses used for independent validation.

---

## What to paste below for the reviewer LLM

1. **Summary markdown** (contents of `--summary` file).
2. **Relevant JSON excerpts** (overall, by_stratum, independent_validation_spectroscopic, and optionally reliability_curve) if the reviewer needs numeric detail.

Then ask the reviewer: *“Review the evaluation results above for checklist items 4.5 (calibration) and 4.6 (independent validation). Flag any concerns: ECE/MCE too high, Tier-A metrics inconsistent with overall, or reliability curve issues.”*

---

## Run results (2026-02-10, Lambda)

Evaluation was run on Lambda with `PYTHONPATH=code python code/dhs/scripts/run_evaluation.py --config code/configs/resnet18_baseline_v1.yaml --output results/eval_resnet18_baseline_v1.json --summary results/eval_resnet18_baseline_v1_summary.md`. Test set: 62,760 samples.

### Summary (from Lambda)

# Evaluation Summary (Checklist 4.5 & 4.6)

**Config:** `code/configs/resnet18_baseline_v1.yaml`  
**Checkpoint:** `/lambda/nfs/darkhaloscope-training-dc/stronglens_calibration/checkpoints/resnet18_baseline_v1/best.pt`  
**Split:** test  
**Samples:** 62,760  
**Time (UTC):** 2026-02-10T22:15:57+00:00

## Overall (test set)

| Metric | Value |
|--------|--------|
| AUC | 0.9579 |
| Recall @ 0.5 | 0.4360 |
| Precision @ 0.5 | 0.7653 |
| ECE (n_bins=15) | 0.0027 |
| MCE | 0.0156 |

## Calibration (4.5)

Overall ECE (0.0027) and MCE (0.0156) are low → model is well calibrated on the full test set. Reliability curve in JSON.

## By stratum (tier)

### tier_A (independent validation set, 4.6)

| Metric | Value |
|--------|--------|
| n | 48 |
| n_pos | 48 |
| n_neg | 0 |
| Recall @ 0.5 | 0.6042 |
| Precision @ 0.5 | 1.0000 |
| ECE | 0.4261 |
| MCE | 0.9994 |

### tier_B

| Metric | Value |
|--------|--------|
| n | 640 |
| n_pos | 640 |
| n_neg | 0 |
| Recall @ 0.5 | 0.4234 |
| Precision @ 0.5 | 1.0000 |
| ECE | 0.5858 |
| MCE | 0.9998 |

## Held-out Tier-A evaluation (4.6)

Tier-A = confident lenses with DR10 match, **in the test split only** (held out from training). This is **not** a separate spectroscopic-search catalog; same imaging-candidate source. See **docs/EVALUATION_HONEST_AUDIT.md** for honest limitations.

| Metric | Value |
|--------|--------|
| n | 48 |
| n_pos | 48 |
| Recall @ 0.5 | 0.6042 |

(ECE/MCE not reported for this stratum: single-class, not meaningful.)

---

**Caveats for reviewer (see EVALUATION_HONEST_AUDIT.md):** (1) Overall ECE is dominated by the majority class; it does not certify calibration on positives. (2) Tier-A is held-out evaluation, not independent spectroscopic-channel validation. (3) ECE/MCE for positive-only strata are misleading and are no longer reported in new runs.

---

## Reviewer checklist (for the second LLM)

- [ ] **4.5 ECE:** Overall ECE reported; value reasonable (< 0.05–0.10)?
- [ ] **4.5 MCE:** MCE not excessively high?
- [ ] **4.5 Reliability:** Any bin with large |acc − conf|?
- [ ] **4.6 Tier-A:** Independent validation (Tier-A) block present and described?
- [ ] **4.6 Consistency:** Tier-A AUC/recall consistent with overall (or drop explained)?
- [ ] **Stratum:** If by_stratum present, Tier-A vs Tier-B calibration/performance make sense?
- [ ] **Limitations:** Any caveats (e.g. tier missing, small Tier-A n) stated?
