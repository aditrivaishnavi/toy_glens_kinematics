# Prompt 15: Paper v6 -- Post-Review Fixes and Re-Run Results

**Date:** 2026-02-15

## Instructions for Reviewer

**Be direct. State facts plainly. No rosy picture.** If something is still wrong after our fixes, say so clearly.

---

## Context

In Prompt 14, four independent LLM reviewers audited two experiments (Tier-A vs Tier-B linear probe, and split verification). They identified concrete bugs and overclaims. We fixed everything they flagged and re-ran the experiment. This prompt asks you to verify the fixes are correct and the paper is now ready for human review.

---

## What the Reviewers Found (Prompt 14 Consensus)

### Code bugs (all 4 agreed):

1. **Silent zero-on-exception** in `feature_space_analysis.py`: `extract_embeddings_from_paths()` caught any exception and silently replaced the failed sample with a zero tensor, corrupting embeddings without warning.

2. **No GroupKFold by galaxy_id** in `tier_ab_probe_control.py`: Used default StratifiedKFold, which could leak near-identical samples from the same galaxy across CV folds.

3. **Arbitrary AUC interpretation thresholds**: Script categorized AUC into "HIGH/MODERATE/LOW" at hardcoded thresholds (0.8, 0.65) -- not defensible given the uncertainty.

### Paper overclaims (3/4 agreed, 4th agreed with caveats):

4. **"bounds host confounding" is a logical error**: Tier-A vs Tier-B (both on lens-type hosts) does NOT bound how much of the Tier-A vs injection AUC (0.996) is due to host population differences, because injection hosts are random negatives -- a completely different population from the lens-type hosts shared by Tier-A and Tier-B. AUCs cannot be subtracted or decomposed additively.

5. **Conclusion item (ii) too strong**: "confirms the gap is morphological rather than photometric" overstates what the evidence supports.

### Arithmetic errors (2/4 caught):

6. **"1:40 class ratio" is wrong**: Table 1 shows 3,356 positives / 312,744 negatives = 1:93. No augmentation factor gives 1:40.

7. **"4x augmentation" in Table 1 caption is misleading**: Augmentation is stochastic/online (h-flip, v-flip, random 90-degree rotation), not pre-generated copies. The manifest count IS the base count.

### Documentation gap (all 4 agreed):

8. **HEALPix verification script missing from zip**: `verify_splits.py` checks galaxy_id/cutout_path overlap, NOT HEALPix pixel overlap. The paper claims "zero overlapping HEALPix pixels" but the script that proves this (`investigate_healpix_nan.py`) was not included.

---

## What We Fixed

### Code fixes (then re-ran):

**Fix 1 -- Silent zero-on-exception**: `extract_embeddings_from_paths()` now tracks all failed paths, drops them from output arrays, and raises `RuntimeError` if any fail. Console prints explicit assertion: "Cutout loads: N/N succeeded, 0 failed."

**Fix 2 -- GroupKFold**: `tier_ab_probe_control.py` now uses `sklearn.model_selection.GroupKFold` with groups set to `galaxy_id` from the manifest (612 unique galaxy_ids across 612 samples). Falls back to StratifiedKFold with a warning if galaxy_id column is absent.

**Fix 3 -- Removed arbitrary thresholds**: Interpretation logic replaced with neutral numerical reporting. No more "HIGH/MODERATE/LOW" buckets.

### Re-run result:

| Metric | v1 (StratifiedKFold, Prompt 14) | v2 (GroupKFold, this run) |
|---|---|---|
| CV AUC mean | 0.783 | **0.778** |
| CV AUC std | 0.053 | **0.062** |
| Fold AUCs | [0.697, 0.790, 0.852, 0.760, 0.818] | **[0.776, 0.883, 0.750, 0.692, 0.789]** |
| Failed cutout loads | not tracked | **0 of 612** |
| CV method | StratifiedKFold | **GroupKFold (612 groups)** |

The AUC shifted by only 0.005 (well within noise). Zero cutout loads failed. This confirms:
- The silent-zero bug was never triggered (all cutouts load successfully)
- No duplicate-galaxy leakage was inflating the previous result

Score statistics are identical (same embeddings, same Tier-B random sample):
- Tier-A median: 0.995, Tier-B median: 0.879

### Paper text fixes (v5 --> v6):

**Fix 4 -- "bounds host confounding" rewritten** (Section 4.3):

Old: "Since both populations share the real host-galaxy distribution, the Tier-A vs Tier-B AUC (0.783) bounds the host-confounding contribution."

New: "However, this does not directly decompose the Tier-A vs injection AUC (0.996) into host and morphology components, because Tier-A and Tier-B hosts (massive ellipticals selected by lensing cross-section) differ systematically from injection hosts (random negatives drawn from the full Tractor catalogue). The substantially higher Tier-A vs injection AUC is *consistent with* injection-specific features contributing additional separation beyond any host confound, but a fully host-matched injection experiment is needed for definitive decomposition."

**Fix 5 -- Conclusion item (ii) softened**:

Old: "a linear probe confirms the gap is morphological rather than photometric"

New: "a linear probe indicates that the gap extends beyond photometric differences, consistent with a morphological mismatch between parametric injections and real lensed galaxies"

**Fix 6 -- Class ratio corrected** (Section 2.4):

Old: "approximately 1:40 before augmentation"

New: "approximately 1:93"

**Fix 7 -- Table 1 caption clarified**:

Old: "Augmented counts (horizontal and vertical flips) are 4x the base counts."

New: "All counts are unique base cutouts. Geometric augmentation (horizontal flip, vertical flip, random 90-degree rotation) is applied stochastically during training and does not change the manifest size."

**Fix 8 -- Seventh limitation rewritten** to match the reframed Section 4.3. "To bound this confound" --> "As a partial diagnostic." Added explicit statement that the A-vs-B probe does not decompose the injection AUC.

**Fix 9 -- Table 5 updated** with new AUC: 0.778 +/- 0.062 (GroupKFold).

### Documentation fix:

**Fix 10 -- HEALPix script included**: `investigate_healpix_nan.py` and its output `healpix_investigation.json` are now in the zip. The JSON confirms: Tier-A train = 274 unique HEALPix pixels, Tier-A val = 112 unique pixels, **0 overlapping pixels**. `verify_splits.py` header now points to the HEALPix script for spatial verification.

---

## Attached

- `mnras_merged_draft_v6.pdf` -- Revised paper (10 pages)
- `mnras_merged_draft_v6.tex` -- LaTeX source
- `prompt_14_code_package.zip` -- Updated code package (10 files, includes HEALPix script + results)

---

## Direct Questions -- YES/NO first, then explain

**Q1.** Read the rewritten Section 4.3 text (Fix 4 above). Does it now accurately characterize what the Tier-A vs Tier-B probe shows and does NOT show? Is the "consistent with" language appropriately hedged, or is it still too strong or now too weak?

**Q2.** Read the rewritten Seventh limitation (Fix 8). Does it now correctly state that the A-vs-B probe does not decompose the injection AUC? Is anything still overclaimed?

**Q3.** The re-run AUC changed from 0.783 (StratifiedKFold) to 0.778 (GroupKFold) -- a shift of 0.005. Is this small enough that we can treat the two results as equivalent and just report the GroupKFold number? Or does the fact that it changed at all suggest a problem?

**Q4.** The class ratio was wrong (1:40 stated, 1:93 actual). We fixed it. Is the corrected "approximately 1:93" the right number given the Table 1 counts (3,356 positives / 312,744 negatives)? Does the paper need to say anything about effective ratio after augmentation, or is the manifest ratio sufficient?

**Q5.** The Table 1 caption now says augmentation is "stochastic during training" and doesn't change the manifest size. Is this clear enough? Section 2.4 says "horizontal flip, vertical flip, 90-degree rotation" -- does the combination of these two descriptions give a referee what they need?

**Q6.** Are there any remaining issues in v6 that would prevent acceptance at Minor Revision? List them if so.

**Q7.** Is this paper ready for submission to human referees? YES or NO. If NO, what specifically must be fixed first?
