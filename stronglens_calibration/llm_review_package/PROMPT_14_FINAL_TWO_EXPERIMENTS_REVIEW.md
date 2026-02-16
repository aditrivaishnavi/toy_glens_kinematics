# Prompt 14: Review of Final Two Experiments

**Date:** 2026-02-15

## Instructions for Reviewer

**Be direct. State facts plainly. Do not give a rosy picture.** If something is wrong, say so. If the conclusion does not follow from the evidence, say so. If the experiment is flawed, say how. I need clear YES/NO signals, not hedged language.

---

## Background

We have an MNRAS paper about a CNN strong-lens finder trained on DESI Legacy Survey DR10 data. Two experiments were recently completed to strengthen the paper before submission. I need you to verify whether these experiments were done correctly and whether the conclusions drawn from them are justified.

The full code for both experiments is in the attached zip.

---

## Experiment 1: Tier-A vs Tier-B Linear Probe Control

### What this tests

Our paper reports that a linear probe (logistic regression on 1280-dim EfficientNetV2-S penultimate embeddings) achieves AUC = 0.996 when separating real Tier-A lenses from synthetic injections. A referee concern is: **how much of that 0.996 is driven by host-galaxy population differences rather than injection morphology?**

Real Tier-A lenses sit on massive elliptical hosts selected by lensing cross-section. Injections are placed on random negative hosts. So the probe might be partly separating "host types" rather than "real arc vs fake arc."

### What we did

We ran a control: linear probe separating Tier-A (spectroscopically confirmed, n=112) from Tier-B (visual candidates, n=500) lenses. Both populations sit on their **real hosts** -- no injections involved. The idea: if the probe AUC is high for A-vs-B, the CNN has learned genuine morphological quality differences. If low (~0.5), host confounding is minimal and the 0.996 is mainly about arc properties.

### The result

```json
{
  "linear_probe_tier_ab": {
    "task": "tier_a vs tier_b",
    "cv_auc_mean": 0.7832,
    "cv_auc_std": 0.0530,
    "cv_fold_aucs": [0.697, 0.790, 0.852, 0.760, 0.818]
  },
  "score_stats": {
    "tier_a": {"median": 0.9946, "mean": 0.8372, "p5": 0.0145, "p95": 0.9998},
    "tier_b": {"median": 0.8792, "mean": 0.6239, "p5": 0.0019, "p95": 0.9994}
  }
}
```

- Tier-A vs Tier-B probe AUC: **0.783 +/- 0.053**
- Tier-A vs injections probe AUC: **0.996 +/- 0.004** (from D05, separate run)
- Tier-A median CNN score: 0.995
- Tier-B median CNN score: 0.879

### Our conclusion in the paper (Section 4.3 and Seventh limitation, paper v5)

We wrote:

> "To bound the contribution of host-galaxy population differences, we performed a control experiment: a linear probe separating Tier-A (spectroscopically confirmed, n=112) from Tier-B (visual candidates, n=500) lenses, both on their real hosts. This probe achieves AUC = 0.783 +/- 0.053 -- moderate separability substantially below the Tier-A vs injection AUC of 0.996 -- indicating that host-galaxy confounding contributes some feature-space separation but cannot account for the much larger injection gap."

We also state in the limitations:

> "host-galaxy confounding contributes some feature-space separation but cannot account for the much larger injection gap. A fully host-matched injection experiment (matching hosts by colour, size, and surface brightness) would provide a more definitive decomposition."

### Implementation

Script: `scripts/tier_ab_probe_control.py` (attached in zip)

- Uses `EmbeddingExtractor` and `extract_embeddings_from_paths` from `scripts/feature_space_analysis.py`
- Model loaded via `dhs/scoring_utils.py` (ensures preprocessing matches training)
- Tier-A: all 112 val-split Tier-A lenses
- Tier-B: random 500 from val-split Tier-B lenses (seed=42)
- 5-fold CV logistic regression (C=1.0, solver=lbfgs, max_iter=1000)
- Embeddings saved as `.npz` for reproducibility

---

## Experiment 2: Training Set Accounting and Split Verification

### What this tests

A referee asked: reconcile the total 451,681 samples with train/val/Tier-A/Tier-B/negative breakdowns. Also verify train/val spatial disjointness.

### What we did

1. Loaded the training manifest (`manifests/training_parity_70_30_v1.parquet`)
2. Counted rows by split, label, and tier
3. Verified galaxy_id and cutout_path disjointness between train and val
4. Recomputed HEALPix pixels for positives (they were NaN in the original manifest)

### The result

Added Table 1 to the paper:

| | Training | Validation | Total |
|---|---|---|---|
| Tier-A positives | 277 | 112 | 389 |
| Tier-B positives | 3,079 | 1,320 | 4,399 |
| Total positives | 3,356 | 1,432 | 4,788 |
| Negatives | 312,744 | 134,149 | 446,893 |
| **Total** | **316,100** | **135,581** | **451,681** |

Spatial integrity (Section 2.3 of paper):
- Tier-A training occupies 274 unique HEALPix pixels
- Tier-A validation occupies 112 unique HEALPix pixels
- **Zero overlapping pixels** between train and val Tier-A

### Our conclusion in the paper (Section 2.3)

> "Tier-A training and validation sets occupy 274 and 112 unique HEALPix pixels respectively, with zero overlapping pixels. This confirms that the model has not seen sky regions near any validation Tier-A lens during training."

### Implementation

Script: `scripts/verify_splits.py` (attached in zip)

- Uses `common/manifest_utils.py` for column definitions
- Checks overlap on both `galaxy_id` and `cutout_path` between train/val/test
- Reports duplicate rows within each split
- Outputs JSON report

---

## Attached Files in Zip

| File | Purpose |
|---|---|
| `scripts/tier_ab_probe_control.py` | Tier-A vs Tier-B linear probe control |
| `scripts/feature_space_analysis.py` | Embedding extraction (dependency of tier_ab_probe) |
| `scripts/verify_splits.py` | Split disjointness verification |
| `scripts/bootstrap_eval.py` | Bootstrap CIs for AUC/recall/precision |
| `common/manifest_utils.py` | Shared column definitions and manifest loading |
| `dhs/scoring_utils.py` | Model + preprocessing spec loading |
| `results/tier_ab_probe_control/tier_ab_probe_results.json` | Raw results from Experiment 1 |

---

## Direct Questions -- Answer YES or NO first, then explain

### On Experiment 1 (Tier-A vs Tier-B probe):

**Q1.** Is the experimental design valid as a control for host-galaxy confounding? Specifically: does comparing Tier-A vs Tier-B (both on real hosts) actually bound how much of the 0.996 Tier-A-vs-injection AUC is due to host differences?

**Q2.** The probe AUC is 0.783. We interpret this as "moderate separability that cannot account for the 0.996." Is this interpretation logically sound, or is it a non-sequitur? Can you actually subtract AUCs like that (0.996 - 0.783 = "remaining gap")?

**Q3.** The fold-to-fold standard deviation is 0.053 (fold AUCs range from 0.697 to 0.852). Given n=112 Tier-A and n=500 Tier-B, is this variance expected, or does it indicate instability that undermines the result?

**Q4.** Are there any bugs in `tier_ab_probe_control.py`? Specifically check:
- Is the Tier-A/Tier-B selection logic correct (filtering by tier column and val split)?
- Is the embedding extraction sound (using the correct preprocessing from checkpoint)?
- Is the logistic regression setup appropriate (5-fold CV, C=1.0)?
- Are there any data leakage issues?

**Q5.** The interpretation logic in the script (lines 104-114) uses AUC > 0.8 as "HIGH", > 0.65 as "MODERATE", else "LOW". Is 0.783 meaningfully different from 0.8 or 0.65 given the uncertainty? Are these thresholds defensible?

**Q6.** Is there a better control experiment we should have run instead? If yes, what and why?

### On Experiment 2 (Split verification):

**Q7.** The paper claims "zero overlapping pixels." Does `verify_splits.py` actually test what we claim? Read the code -- does it check HEALPix overlap, or does it check something else (galaxy_id/cutout_path)?

**Q8.** The paper says "we recomputed HEALPix pixel assignments for all positives (a manifest-generation issue had left the HEALPix column as NaN for positives)." But `verify_splits.py` does not do any HEALPix recomputation -- it checks galaxy_id and cutout_path overlap. Is there a disconnect between what the paper claims and what the script actually verifies?

**Q9.** The numbers add up: 316,100 + 135,581 = 451,681 and 4,788 + 446,893 = 451,681. But are these numbers correct? Check: is it possible that the 70/30 split is applied BEFORE or AFTER augmentation? If after, the true unique samples may be ~4x fewer positives. Does the paper make this clear?

**Q10.** Is `verify_splits.py` sufficient as evidence of spatial disjointness, or do we need a separate script that explicitly checks HEALPix overlap? If the latter, we don't have one in the zip -- flag this as a gap.

### Overall:

**Q11.** Given these two experiments and their results, do they meaningfully strengthen the paper? Or are they window dressing that a referee would see through?

**Q12.** Is there anything we missed -- a test we should have run, a check we should have done, a conclusion we drew that doesn't follow from the evidence?
