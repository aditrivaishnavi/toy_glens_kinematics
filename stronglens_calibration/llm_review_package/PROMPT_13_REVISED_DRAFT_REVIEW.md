# Prompt 13: Review of Revised Draft (Post-Referee Revisions)

**Date:** 2026-02-15

## Context

Two independent referees reviewed our MNRAS draft (Prompt 12) and both recommended Major Revision. We have now addressed all their concerns. This prompt asks you to verify the revisions are adequate.

## What Changed Since the Last Review

The following specific issues were raised and addressed:

### Priority 1 Fixes (Blocking)

1. **Magnitude-to-nanomaggy conversion error** (raised by Referee 1): The paper incorrectly stated m=21 = 0.58 nmgy. The correct value is m=21 = 3.98 nmgy. **Fix:** Replaced the single erroneous example with a corrected multi-magnitude table (mag 21: 6.6 e-/pix, mag 22: 2.6, mag 23: 1.05) showing the physics holds across the range. Abstract also corrected.

2. **86-pp gap misleading** (raised by both): Comparing Tier-A recall against marginal completeness was apples-to-oranges. **Fix:** Abstract and Section 4.2 now lead with brightness-matched comparisons (18-36% for mag 19-22 injections vs 89.3% Tier-A). The 3.41% marginal is properly contextualized as "dominated by faint injections."

3. **"Falsifies" language too strong** (raised by both): **Fix:** Changed throughout to "rules out arc-level shot noise" rather than "falsifies all texture." Added explicit caveat about untested texture sources (correlated noise, PSF wings, inter-band correlations).

4. **Frechet distance NaN** (raised by both): **Fix:** Text now discloses that deeper layers are numerically unstable (n=112 < dim). FD treated as "directional only." Linear probe AUC is the primary quantitative measure.

5. **FPR=1e-4 unsupported** (raised by Referee 1): **Fix:** Relabelled to FPR ≈ 3e-4 in Tier-A table. Caption notes FPR thresholds are derived from 50,000 negatives in grid but only 3,000 in scoring.

### Priority 2 Fixes (Strengthening)

6. **Host-matching confound** (raised by Referee 1): **Fix:** New limitation paragraph (seventh) explicitly discusses host-population differences as a potential confound for the linear probe.

7. **Paired statistical tests** (raised by both): **Fix:** Added Wilcoxon signed-rank test (W=28, p=0.008) alongside the sign test.

8. **Class imbalance and training choices** (raised by Referee 2): **Fix:** New paragraph discusses 1:40 class ratio, unweighted BCE rationale, and absence of noise augmentation with its relevance to the Poisson experiment.

9. **Tier-B label noise** (raised by Referee 2): **Fix:** New limitation paragraph discusses effect of ~10% label noise in Tier-B training positives.

10. **Morphological vs textural defined** (raised by Referee 1): **Fix:** Explicit operational definition added to the Introduction.

11. **Missing citations** (raised by Referee 2): **Fix:** Lanusse et al. (2018) and Metcalf et al. (2019) added.

### Figures

All 4 figures are now generated from verified D05 data and embedded in the PDF:
- Figure 1: Completeness vs theta_E (left) and lensed magnitude (right)
- Figure 2: Two-panel UMAP (category-colored and score-colored)
- Figure 3: Score distributions (log-scale histograms with threshold lines)
- Figure 4: Bright-arc detection rate vs magnitude for all 6 conditions (signature figure)

## Questions

1. **Do the revisions adequately address the magnitude error?** Is the multi-magnitude photoelectron table clear and correct?

2. **Is the reframed 86-pp gap now fair?** Does the paper properly distinguish between the parameter-space-averaged figure and the brightness-matched comparison?

3. **Is the softened language precise enough without being too weak?** "Rules out arc-level shot noise" vs the original "falsifies missing texture."

4. **Are the figures publication-quality?** Check axis labels, legends, colors, readability at single-column width.

5. **Any remaining issues that would prevent acceptance at Minor Revision?**

## Attached

- `mnras_merged_draft_v1.pdf` — Revised paper with figures (9 pages, ~880KB)
- `mnras_merged_draft_v1.tex` — LaTeX source
- `bright_arc_all_conditions.json` — All D05 bright-arc results
- `EXPERIMENT_REGISTRY.md` — Full experiment history
