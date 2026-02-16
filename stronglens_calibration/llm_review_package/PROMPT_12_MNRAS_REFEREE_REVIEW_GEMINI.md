# Prompt 12: MNRAS Referee-Style Review of Draft Paper

**Date:** 2026-02-14

## Your Role

You are acting as an anonymous referee for Monthly Notices of the Royal Astronomical Society (MNRAS). You have been assigned to review a methods paper on injection-recovery validation for CNN strong gravitational lens finders. You have expertise in strong lensing, CNN-based astronomical classifiers, and survey selection functions.

## Paper Background

The authors trained an EfficientNetV2-S CNN to find strong gravitational lenses in DESI Legacy Survey DR10 (g/r/z bands, 101x101 pixel cutouts). The model achieves 89.3% recall on 112 spectroscopically confirmed lenses (Tier-A). They then attempted to calibrate its selection function using standard injection-recovery with parametric Sersic source profiles. The injection completeness is only 3.41% — an 86-percentage-point gap compared to real-lens recall.

The paper's central claim is that this gap is *morphological*, not *textural*. They prove this through a controlled experiment: adding physically correct Poisson noise to injections DEGRADES detection (from 3.41% to 2.37%), falsifying the hypothesis that the gap is caused by missing noise texture. A gain sweep control (gain=1e12) recovers the baseline exactly, proving it's not a code bug. They propose a linear probe AUC (0.996) as a quantitative realism gate for injection pipelines.

## Important Note on Draft Status

This is a **first draft**. The following elements are **missing or incomplete** — please do not penalise these, but do note where they would strengthen the paper:

- **Figures are placeholders** (described in captions but not yet generated). The four planned figures are: (1) completeness vs theta_E and lensed magnitude, (2) two-panel UMAP of CNN embeddings, (3) CNN score distributions, (4) detection rate vs magnitude for all experimental conditions (the "signature figure").
- **Architecture diagram** is not yet included.
- **Characterisation of the 12 missed Tier-A lenses** is deferred to future work.
- **No BibTeX** — references are in a manual thebibliography block. Some may need verification.

## Review Instructions

Please conduct your review in two parts:

### Part A: Line-by-Line Technical Review

Read the paper from abstract to appendix. For each section, note:
- Factual errors or inconsistencies
- Claims not supported by the data presented
- Missing context or citations
- Unclear or ambiguous statements
- Statistical concerns (sample sizes, test choices, confidence intervals)

### Part B: Holistic Assessment

Step back from the details and consider:

1. **Scientific significance:** Is the "morphological barrier" a genuine contribution, or is it well-known that Sersic models are unrealistic? What is genuinely NEW here?

2. **Experimental rigour:** Is the Poisson falsification experiment convincing? Are the controls adequate? What alternative explanations have NOT been ruled out?

3. **Paper framing:** The authors claim this is a methods-validation paper, not a selection-function paper. Is this framing defensible? Would a referee accept "we measured something we show is wrong" as a contribution?

4. **The elephant in the room:** The authors show parametric injections are unrealistic but do not demonstrate realistic ones. Is the paper complete without a demonstration that real galaxy stamps (e.g. HUDF) close the gap?

### Part C: Specific Questions

Please answer each of these explicitly:

1. **What are the paper's three greatest strengths?**

2. **What are the paper's three greatest weaknesses?**

3. **What is the single most likely reason a second referee would recommend rejection?**

4. **What specific changes would move your recommendation from "major revision" to "minor revision"?**

5. **Is the statistical analysis sufficient?** (Sign test p=0.008, two-proportion z=14.6, Wilson CIs throughout — is anything missing?)

6. **Is the Discussion section balanced?** The authors list six limitations. Are they honest enough? Are they hiding anything?

7. **Is the title appropriate?** Current title: "The morphological barrier: quantifying the injection realism gap for CNN strong lens finders in DESI Legacy Survey DR10"

8. **Would you recommend this paper for publication in MNRAS?** Choose one: Accept, Minor Revision, Major Revision, or Reject. Justify your recommendation.

## Attached Files (7 files + this prompt)

1. `mnras_merged_draft_v1.pdf` — The compiled paper draft (9 pages, MNRAS format)
2. `mnras_merged_draft_v1.tex` — LaTeX source for the paper
3. `injection_priors.yaml` — Injection parameter ranges (single source of truth for all priors cited in the paper)
4. `bright_arc_all_conditions.json` — Combined bright-arc detection rates for all 6 experimental conditions (baseline, Poisson, clip20, Poisson+clip20, unrestricted, gain=1e12)
5. `grid_no_poisson_meta.json` — Selection function grid metadata (baseline, no Poisson)
6. `grid_poisson_fixed_meta.json` — Selection function grid metadata (Poisson, gain=150)
7. `EXPERIMENT_REGISTRY.md` — Full experiment history D01-D05 with status and key findings
