# LLM Review Prompt: Final Submission Readiness (v17)

## Role

You are an expert referee for Monthly Notices of the Royal Astronomical Society (MNRAS), with deep expertise in strong gravitational lensing, convolutional neural networks for astronomical classification, and statistical methodology. You are reviewing this paper for scientific correctness, statistical rigour, clarity of presentation, and readiness for submission.

## Paper under review

**Title:** A morphological barrier: quantifying the injection realism gap for CNN strong lens finders in DESI Legacy Survey DR10

**Version:** v17 (final pre-submission draft)

**The full LaTeX source is provided below this prompt.**

## Context on changes since v16

v17 adds the results of a permutation test and bootstrap confidence interval for the linear probe AUC, which was requested by a prior reviewer as a statistical robustness check. The changes appear in five locations:

1. **Abstract**: Added "(permutation test p < 0.001, 0/1000; bootstrap 95 per cent CI: [0.996, 1.000])" after the linear probe AUC.
2. **Section 4.3** (The CNN distinguishes real lenses from injections): Added a new paragraph describing the permutation test (1000 iterations, single 80/20 stratified split per iteration, max permuted AUC = 0.690, mean = 0.499) and bootstrap CI (5000 iterations resampling held-out CV predictions, 95% CI = [0.996, 1.000]).
3. **Table 3** (tab:probe): Added two new rows — "Permutation test (0/1000): p < 0.001" and "Bootstrap 95% CI: [0.996, 1.000]".
4. **Section 5.2** (The linear probe as a realism gate): Added permutation test and bootstrap CI to reinforce the realism gate argument.
5. **Conclusions item (iii)**: Added "(permutation test p < 0.001, 0/1000; bootstrap 95 per cent CI: [0.996, 1.000])".

## Review instructions

Please evaluate the paper on the following dimensions. For each, provide a verdict (PASS / CONCERN / FAIL) and a brief explanation.

### 1. Scientific correctness
- Are all claims supported by the data and experiments described?
- Are the experimental controls adequate (gain sweep, paired analysis, permutation test)?
- Is the causal reasoning (morphological barrier vs. textural mismatch) appropriately hedged?
- Are the numbers internally consistent (e.g., AUC, recall, completeness cited in abstract vs. body vs. conclusions)?

### 2. Statistical rigour
- Is the permutation test methodology sound (label shuffling, single stratified split, 1000 iterations)?
- Is the bootstrap CI methodology appropriate (resampling held-out predictions, 5000 iterations)?
- Are confidence intervals, p-values, and effect sizes reported correctly?
- Is the class imbalance (112 vs 500) properly acknowledged and handled?
- Are the Wilson CIs, two-proportion z-tests, and sign tests correctly applied?

### 3. Clarity and presentation
- Is the paper well-structured and readable?
- Are technical terms defined when first used?
- Are the figures and tables well-captioned and informative?
- Is the diagnostic ladder (establish gap → propose hypothesis → test → conclude) clearly articulated?

### 4. Completeness
- Are all relevant limitations acknowledged?
- Is the host-galaxy confound adequately discussed?
- Are comparisons with prior work fair and accurate?
- Is the Data Availability statement adequate?

### 5. Internal consistency
- Do the abstract, body, tables, and conclusions all report the same numbers?
- Are the v17 additions (permutation test, bootstrap CI) consistent with the existing AUC values?
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

## Full LaTeX source (v17)

The full LaTeX source is in `stronglens_calibration/paper/mnras_merged_draft_v17.tex` (627 lines, 13 pages compiled).

### Key numbers to verify for internal consistency:

| Metric | Expected value | Locations |
|--------|---------------|-----------|
| Linear probe AUC | 0.997 ± 0.003 | Abstract, §4.3, Table 3, §5.2, Conclusions (iii) |
| Permutation test | p < 0.001 (0/1000) | Abstract, §4.3, Table 3, §5.2, Conclusions (iii) |
| Bootstrap 95% CI | [0.996, 1.000] | Abstract, §4.3, Table 3, §5.2, Conclusions (iii) |
| Max permuted AUC | 0.690 | §4.3 |
| Mean permuted AUC | 0.499 | §4.3 |
| Tier-A recall | 89.3% [82.6%, 94.0%] | Abstract, §4.1, Table 1, Conclusions (i) |
| Marginal completeness | 5.18% | Abstract, §4.2, Tables 2-3, §5.1, Conclusions (ii) |
| Poisson completeness | 3.79% | Abstract, §4.4, Tables 2-3 |
| Bright-arc Poisson uplift | +10.5 pp at mag 21-22 | Abstract, §4.4, Table 5, Conclusions (iv) |
| Control probe AUC | 0.778 ± 0.062 | §4.3, Table 3 |
