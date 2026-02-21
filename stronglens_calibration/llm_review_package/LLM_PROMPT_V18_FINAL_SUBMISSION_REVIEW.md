# LLM Review Prompt: Final Submission Readiness (v18)

## Role

You are an expert referee for Monthly Notices of the Royal Astronomical Society (MNRAS), with deep expertise in strong gravitational lensing, convolutional neural networks for astronomical classification, and statistical methodology. You are reviewing this paper for scientific correctness, statistical rigour, clarity of presentation, and readiness for submission.

## Paper under review

**Title:** A morphological barrier: quantifying the injection realism gap for CNN strong lens finders in DESI Legacy Survey DR10

**Version:** v18 (referee-hardened pre-submission draft)

**The full LaTeX source is provided below this prompt.**

## Context on changes since v17

v18 incorporates feedback from two independent LLM referee reports on v17. The changes are:

1. **p-value wording** (5 locations): Changed `p < 0.001` to `p \leq 0.001` throughout (Abstract, §4.3, Table 3, §5.2, Conclusions (iii)). This is the correct frequentist phrasing for a finite permutation test where 0/1000 permuted values exceeded the observed statistic.

2. **Prior correction motivation** (§3.2): Added a sentence explaining *why* correcting the colour prior matters: "Correcting this prior is essential for the completeness measurement to be meaningful: if injections are generated with implausible colours, low detection rates might simply reflect colour mismatch rather than a genuine morphological barrier."

3. **AUC clarification** (§4.3): Added parenthetical explaining the marginal difference between the single-split AUC (0.998) used for the permutation test and the five-fold mean (0.997): "marginally higher than the five-fold mean of 0.997 because a single 80/20 split has a larger training set."

4. **Magnitude label disambiguation**: All magnitude references in the Poisson noise discussion (§4.4.2, Conclusions (iv)) now explicitly prefix "source" to avoid ambiguity between source and lensed magnitudes, e.g., "source mag 21-22" instead of "mag 21-22".

5. **McNemar test** (§4.4.2): Added McNemar's chi-squared test on the bright-arc paired injection data (continuity-corrected). Per-bin results: chi2=12.1, p=3.2e-4 at source mag 21-22; chi2=12.0, p=2.8e-4 at source mag 22-23. Pooled over source mag 20-23 (73 gained, 28 lost): chi2=19.2, p=1.2e-5. This complements the existing sign tests.

6. **McNemar in Conclusions** (item iv): Added McNemar chi2 and p-value to reinforce the paired analysis finding.

7. **Missed Tier-A lens appendix** (Appendix B): New appendix cataloguing the 12 Tier-A lenses missed by the CNN at p > 0.3. Table includes DESI name, r-mag, CNN score, and spectroscopic z_lens (cross-matched from DESI Strong Lensing catalogue). All 12 have r <= 20. Accompanying text discusses failure modes. Section 4.1 now references this appendix instead of deferring to a "companion analysis."

8. **Data Availability** (§Data Availability): Expanded to list specific deliverables (injection pipeline code, selection function grid results, CNN model weights, linear probe scripts, training manifest, lens catalogues) and states these will be archived on Zenodo.

9. **Introduction outline**: Updated to reference Appendix B.

## Review instructions

Please evaluate the paper on the following dimensions. For each, provide a verdict (PASS / CONCERN / FAIL) and a brief explanation.

### 1. Scientific correctness
- Are all claims supported by the data and experiments described?
- Are the experimental controls adequate (gain sweep, paired analysis, permutation test, McNemar test)?
- Is the causal reasoning (morphological barrier vs. textural mismatch) appropriately hedged?
- Are the numbers internally consistent (e.g., AUC, recall, completeness cited in abstract vs. body vs. conclusions)?
- Is the missed-lens appendix (Appendix B) scientifically informative and free of unsupported speculation?

### 2. Statistical rigour
- Is the permutation test methodology sound (label shuffling, single stratified split, 1000 iterations)?
- Is the bootstrap CI methodology appropriate (resampling held-out predictions, 5000 iterations)?
- Is the McNemar test correctly applied (continuity-corrected, on discordant pairs, pooled appropriately)?
- Are confidence intervals, p-values, and effect sizes reported correctly?
- Is the class imbalance (112 vs 500) properly acknowledged and handled?
- Are the Wilson CIs, two-proportion z-tests, and sign tests correctly applied?

### 3. Clarity and presentation
- Is the paper well-structured and readable?
- Are technical terms defined when first used?
- Are the figures and tables well-captioned and informative?
- Is the diagnostic ladder (establish gap -> propose hypothesis -> test -> conclude) clearly articulated?
- Are magnitude references unambiguous (source vs. lensed)?

### 4. Completeness
- Are all relevant limitations acknowledged?
- Is the host-galaxy confound adequately discussed?
- Are comparisons with prior work fair and accurate?
- Is the Data Availability statement adequate and specific?
- Does Appendix B adequately characterise the missed lenses?

### 5. Internal consistency
- Do the abstract, body, tables, and conclusions all report the same numbers?
- Are the v18 additions (McNemar test, missed-lens table, p-value wording) consistent with existing content?
- Are there any contradictions between different parts of the paper?

### 6. Submission readiness
- Is this paper ready for submission to MNRAS?
- What (if any) changes would you require before submission?
- What (if any) changes would you recommend but not require?

## Final verdict

Please provide an overall assessment:
- **READY FOR SUBMISSION**: No blocking issues remain.
- **MINOR REVISIONS NEEDED**: Small fixes needed before submission (list them).
- **MAJOR REVISIONS NEEDED**: Significant issues that must be addressed (list them).

---

## Full LaTeX source (v18)

The full LaTeX source is in `stronglens_calibration/paper/mnras_merged_draft_v18.tex` (661 lines, 13 pages compiled).

### Key numbers to verify for internal consistency:

- Linear probe AUC: 0.997 +/- 0.003 -- Locations: Abstract, S4.3, Table 3, S5.2, Conclusions (iii)
- Permutation test: p <= 0.001 (0/1000) -- Locations: Abstract, S4.3, Table 3, S5.2, Conclusions (iii)
- Bootstrap 95% CI: [0.996, 1.000] -- Locations: Abstract, S4.3, Table 3, S5.2, Conclusions (iii)
- Max permuted AUC: 0.690 -- Location: S4.3
- Mean permuted AUC: 0.499 -- Location: S4.3
- Tier-A recall: 89.3% [82.6%, 94.0%] -- Locations: Abstract, S4.1, Table 1, Conclusions (i)
- Missed Tier-A: 12/112 (10.7%) [5.6%, 18.1%] -- Locations: S4.1, Appendix B
- Marginal completeness: 5.18% -- Locations: Abstract, S4.2, Tables 2-3, S5.1, Conclusions (ii)
- Poisson completeness: 3.79% -- Locations: Abstract, S4.4, Tables 2-3
- Bright-arc Poisson uplift: +10.5 pp at source mag 21-22 -- Locations: Abstract, S4.4, Table 5, Conclusions (iv)
- McNemar (source mag 21-22): chi2=12.1, p=3.2e-4 -- Locations: S4.4.2, Conclusions (iv)
- McNemar (pooled 20-23): chi2=19.2, p=1.2e-5 -- Location: S4.4.2
- Control probe AUC: 0.778 +/- 0.062 -- Locations: S4.3, Table 3
