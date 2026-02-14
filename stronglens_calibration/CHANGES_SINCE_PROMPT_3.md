# Changes Since Prompt 3 — Complete Inventory

**Date:** 2026-02-13
**Trigger:** Both LLM1 and LLM2 reviewed the codebase (Prompt 3 of 4).
These changes implement ALL action items from the Prompt 3 retrain decision analysis,
plus previously missed items from Prompts 1-2.

---

## A. New Diagnostic Scripts (5 items)

### 1. `scripts/annulus_comparison.py` — Q2.3 Pre-Retrain Experiment (NEW)

Compares (20,32) vs (32.5,45) annulus normalization statistics:
- Loads ~1000 val cutouts, computes median/MAD with both annulus configs
- KS tests for distribution differences
- Correlation of shifts with PSF size and depth
- Positive vs negative breakdown
- Runtime: ~1-5 min CPU

### 2. `scripts/mismatched_annulus_scoring.py` — Q2.4 Sensitivity Test (NEW)

Scores injections/negatives with v4 model but mismatched (32.5,45) preprocessing:
- Compares recall and FPR at multiple thresholds (0.3, 0.5, 0.7)
- Reports delta between native and mismatched preprocessing
- Uses `dhs.scoring_utils.load_model_and_spec()` for model loading
- Runtime: ~10-30 min GPU

### 3. `scripts/split_balance_diagnostic.py` — Q3.9/Q3.10 Data Integrity (NEW)

Verifies PSF/depth balance and positive spatial distribution:
- Distribution of positives per HEALPix pixel (clustering check)
- PSF/depth KS tests across train/val/test splits
- Tier-A counts per split
- Runtime: seconds (pure pandas/numpy)

### 4. `scripts/masked_pixel_diagnostic.py` — Q3.3 Cutout Integrity (NEW)

Checks NaN/zero/non-finite pixel fractions in cutouts:
- Samples cutouts and reports masked fractions per stamp
- Flags stamps with >5% non-finite pixels
- Summary statistics for the full sample
- Runtime: ~1-5 min CPU

### 5. `tests/test_band_order.py` — Q3.1 Band Order Verification (NEW)

Three tests verifying g=0, r=1, z=2 band ordering:
- `test_injection_engine_band_order`: flux ordering g < r < z over 500 sources
- `test_injection_output_channel_order`: corner pixels preserve host band order
- `test_preprocess_stack_preserves_band_order`: normalization doesn't scramble

---

## B. Evaluation Improvements (2 items)

### 6. `scripts/evaluate_parity.py` — TPR at Fixed FPR (MODIFIED)

**Both LLMs: "report TPR at FPR=1e-3, not only AUC"**

- Added `tpr_at_fpr_0.001` and `tpr_at_fpr_0.01` to `compute_binary_metrics()`
- Uses sklearn `roc_curve` + `np.interp` for interpolation
- Added to both print output locations (core metrics + bootstrap summary)
- New keys in results dict: `tpr_at_fpr_0.001`, `tpr_at_fpr_0.01`

### 7. `sim_to_real_validations/real_lens_scoring.py` — Tier-A Evaluation (MODIFIED)

**LLM1 Prompt 3: "real_lens_scoring.py still does not tier-filter"**

- New `--tier-a-only` flag: restricts positive evaluation to Tier-A only
- Training-split leakage guard: prints Tier-A/B counts per split after
  loading manifest (before any evaluation)
- Uses existing `TIER_COL = "tier"` column

---

## C. Injection Prior Extensions (2 items)

### 8. `dhs/injection_engine.py` — Extended R_e Range (MODIFIED)

**Both LLMs: "R_e 0.05-0.25 is too narrow"**

- Default changed: `re_arcsec_range` from `(0.05, 0.25)` to `(0.05, 0.50)`
- Docstring added citing: Herle et al. (2024), Collett (2015), observed sizes
- Note: existing grid/experiment scripts pass ranges as arguments, so old
  results are unaffected. New runs use wider range.

### 9. `dhs/injection_engine.py` — Extended Sersic n Range (MODIFIED)

**Both LLMs: "n 0.7-2.5 is too narrow. Herle finds CNN selects n >= 2.55"**

- Default changed: `n_range` from `(0.7, 2.5)` to `(0.5, 4.0)`
- Docstring citing: Herle et al. (2024), Collett (2015), standard practice

---

## D. Reproducibility (2 items)

### 10. `requirements.txt` — Pinned Torch Versions (MODIFIED)

**LLM1 Prompt 3: "torchvision::nms does not exist" error from version mismatch**

- Changed `torch>=2.7.0` to `torch==2.7.0`
- Changed `torchvision>=0.22.0` to `torchvision==0.22.0`
- Added comment explaining why pinning is needed

### 11. `tests/test_band_order.py` — See item 5 above

---

## E. Documentation (6 new sections in MNRAS_SUPPLEMENTARY_TRAINING_LOG.md)

### 12. Section 13: Pre-Retrain Experiments (Q2.1-Q2.4 Run Commands)

Documented exact run commands for all four cheap experiments with
success criteria and interpretation guidance.

### 13. Section 14: Label Noise Estimation (Q3.5-Q3.7, Q4.3)

- Tier-B FP rate: ~10% (~440 mislabeled in 4,399)
- Negative contamination: ~10-50 real lenses in 446,893 negatives
- Impact on AUC, calibration, and high-purity regime

### 14. Section 15: Success Criteria + GO/NO-GO Decision Tree

- LLM1's concrete criteria: AUC >= 0.9930, Tier-A recall >= 80%,
  bright-arc ceiling >= 40%, completeness >= 2x baseline
- GO/NO-GO decision logic flowchart
- Retrain-failure fallback plan

### 15. Section 16: Hostile-Referee Defense Strategy (Q4.1-Q4.11)

- Q4.1: 3.5% completeness framing (marginal over broad prior volume)
- Q4.2: Tier-A-only eval with leakage controls
- Q4.7-Q4.8: Paper IV confounders table (too many uncontrolled variables)
- Q4.9: No independent holdout limitation
- Q4.10: Missing source redshift dimension

### 16. Section 17: Literature Comparison

Structured comparison: Herle et al. (2024), HOLISMOKES XI, Euclid (2025),
Jacobs et al. (2019, 2021), DES CNN (Gonzalez et al. 2025), our position.

### 17. Section 18: Injection Prior Justification

Prior ranges with literature sources for R_e, n, source mag, beta_frac,
colors. HOLISMOKES acceptance criteria contrast.

---

## Test Results

All files compile cleanly (python3 -c py_compile). No linter errors introduced.

---

## Items NOT Fixed (with justification)

- **psfdepth interpretation**: LLM1 flagged as potentially wrong. Our code
  assumes inverse variance (correct for Tractor catalog columns). LLM1 may
  have confused with brick-summary columns. Awaiting Prompt 4 resolution.
- **Observation-process signatures**: Sky subtraction artifacts, noise
  correlation. Not addressable without full image simulation pipeline.
- **Linear probe experiment**: "real vs injection" classifier in embedding space.
  Recommended by both LLMs but not yet implemented.
- **Weighted loss for v5 retrain**: Both LLMs noted unweighted BCE with 93:1
  is unusual. Decision deferred to retrain planning.
