# Honest evaluation audit

**Purpose:** Critical review of the evaluation code and process. No fudging; assess whether it meets rigor for original research.

**Date:** 2026-02-10

---

## 1. What the code does (verified)

- **Data path:** Loads manifest from config, builds `LensDataset` with `split=test`, no augmentation. Same preprocessing as training (`raw_robust`). So we evaluate on the same test split the pipeline defines.
- **Order alignment:** `LensDataset` filters manifest by `split == "test"` and preserves row order. DataLoader uses `shuffle=False`. So `y_true[i]`, `y_prob[i]` correspond to the i-th test row. We then filter the manifest again by split and take `df_split.iloc[:len(y_true)]`; parquet read order is deterministic, so the i-th inference result matches the i-th row of `df_split`. Tier labels are aligned.
- **Checkpoint:** Loads `best.pt` (saved when val AUC improved). We evaluate that single checkpoint on the **test** set. No test set used for model selection. Correct.
- **Metrics:** `roc_auc_score`, recall @ 0.5, precision @ 0.5 are standard and implemented correctly. ECE/MCE use equal-frequency binning; formula is the usual one: ECE = sum_b (n_b/n) * |acc(b) - conf(b)|. No arithmetic tricks.

So the **numbers are not fabricated**. The code computes what it claims.

---

## 2. Honest limitations and overclaims

### 2.1 “Independent validation (spectroscopic)” is overstated

- **Claim in doc/checklist:** Tier-A is used as “independent spectroscopic validation” (4.6).
- **Reality:** Tier-A in this code is “rows in the **test** set whose `tier` column is A.” Those are confident lenses (DR10-matched) that happened to fall in the test split. They are **held out from training** (so no train/test leakage), but they are **not** from a different selection channel. They are still from the same imaging candidate catalog (lenscat), just the confident subset.
- **What the LLM conversation asks for:** “Independent evaluation sets from spectroscopic DESI searches” (e.g. single-fiber, pairwise DR1) to break circularity. That would be a separate catalog, not “Tier-A of our same catalog.”
- **Honest statement:** We have **Tier-A held-out evaluation**: 48 confident lenses in test, recall @ 0.5 = 0.60. We do **not** have true **independent spectroscopic-channel validation** (a different discovery pipeline). For original research, the paper should say “evaluation on held-out Tier-A anchors” and only call it “independent” in the narrow sense of “not used in training,” not “from an independent spectroscopic search.”

### 2.2 Overall ECE is dominated by the majority class

- **Observed:** Overall ECE = 0.0027, MCE = 0.0156. Test set is ~62,760 samples with only ~688 positives (~1.1%).
- **How ECE behaves here:** Equal-frequency binning gives ~4,184 samples per bin. Most bins are “almost all negatives” with very small predicted probabilities. So most of the ECE sum comes from bins where both accuracy and mean confidence are near zero → small |acc − conf|. So a low overall ECE is expected and **does not** show that the model is well calibrated on **positives** (the class that matters for recall and follow-up).
- **Honest statement:** Overall ECE shows the model is not badly overconfident on the **whole** distribution (which is mostly negatives). It **does not** certify calibration on the positive class. For that you’d need calibration metrics restricted to positives, or at least a reliability diagram that highlights the high-score (positive) region and notes the small-N and imbalance there.

### 2.3 Tier-A / Tier-B ECE and MCE are misleading

- **What the code does:** It computes ECE and MCE on Tier-A (48 positives, 0 negatives) and Tier-B (640 positives, 0 negatives). So we’re binning **only** positive labels by their predicted probability.
- **Why it’s misleading:** For a single-class subset, “accuracy” in a bin is either 1 (all positives) or undefined. The model gives many positives low scores (hence recall 0.43–0.60). So we get bins with “accuracy” 1 and “confidence” low, or vice versa, and ECE/MCE become large (e.g. 0.43, 0.99) and **do not** mean the same as “poor calibration” in the usual sense. Reporting them without caveat invites misinterpretation.
- **Honest fix:** Do **not** report ECE/MCE for positive-only (or negative-only) strata, or report them only with an explicit warning that they are not comparable to overall ECE and can be misleading. The audit doc and LLM review already add a caveat; the **script’s summary output** should do the same or omit stratum ECE/MCE when n_neg or n_pos is 0.

### 2.4 Split disjointness is assumed, not verified in this script

- **What we assume:** The manifest’s `split` column was set by an upstream process (e.g. HEALPix-based or metadata in NPZ) so that train/val/test are disjoint. We never use test for model selection.
- **What we don’t do:** The eval script does **not** verify that test IDs (e.g. galaxy_id or cutout_path) do not appear in the training set. For full rigor, an independent check (e.g. set of test IDs ∩ set of train IDs = ∅) should be run and documented.
- **Recommendation:** Add a small script or assert: load manifest, compute set(train_ids), set(test_ids), assert disjoint, and note in the paper that split disjointness was verified.

### 2.5 No uncertainty on metrics

- **Current practice:** We report point estimates (AUC, recall, precision, ECE). No confidence intervals (e.g. bootstrap or analytical).
- **For publication:** At least bootstrap 95% (or 68%) intervals for AUC and recall @ 0.5 on the test set and for Tier-A would strengthen the evaluation and satisfy “rigor for original research.” Small-N (e.g. Tier-A n=48) especially needs intervals.

---

## 3. What is solid

- No augmentation at eval; same preprocessing as training.
- Test set is used only for this evaluation; model was chosen by val AUC.
- AUC, recall, precision are computed correctly.
- Overall ECE/MCE formula and implementation are correct; the limitation is **interpretation** (majority-class dominance), not arithmetic.
- Tier alignment between inference order and manifest is correct.
- We do not report AUC for positive-only strata (we set it to `null`); that’s correct.

---

## 4. Recommendations for rigor

1. **Terminology:** In the paper and checklist, use “Tier-A held-out evaluation” or “evaluation on held-out Tier-A anchors,” and avoid “independent spectroscopic validation” unless you add a truly separate spectroscopic-channel catalog.
2. **Stratum ECE/MCE:** In the eval script, either omit ECE/MCE when a stratum has only one class (n_pos==0 or n_neg==0), or always append a short warning in the summary when they are reported for single-class strata.
3. **Split verification:** Document or run a check that test and train IDs are disjoint; mention in methods.
4. **Uncertainty:** Add bootstrap (or similar) intervals for AUC and recall @ 0.5 on test and on Tier-A; report in the same JSON/summary.
5. **Calibration on positives:** Add a sentence in the paper that overall ECE summarizes calibration over the full (imbalanced) set and that calibration on the positive class would require a separate analysis (e.g. reliability diagram in the high-score region or positive-only metrics with appropriate caveats).

---

## 5. Conclusion

- The **eval code does not fudge numbers**: metrics and calibration are computed as stated.
- **Overclaims** are in **wording** (“independent spectroscopic validation”) and **interpretation** (overall ECE as proof of calibration on positives; stratum ECE on positive-only subsets without caveat).
- For **original research**, the evaluation is a reasonable start but should be tightened: correct the independence claim, add caveats or omit misleading stratum ECE/MCE, verify split disjointness, and add uncertainty quantification where possible.
