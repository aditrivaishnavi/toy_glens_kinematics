# PROMPT 22 — Paper v12 Final Sign-Off Review

## Background

This is a final verification pass on paper v12 (`mnras_merged_draft_v12.tex`). Paper v11 was reviewed by three independent LLMs (Prompt 21), who identified several categories of issues:

1. **CRITICAL**: All 4 figures were still generated from D05 data (old priors), contradicting the D06 text and tables.
2. **Numerical**: Stale AUC (0.996 instead of 0.997), wrong Fréchet dimensions in Table 5 and prose.
3. **Labeling**: Bright-arc magnitude bins are *source* magnitudes, not lensed magnitudes — mislabeled in abstract, intro, Table 6, Figure 4.
4. **Statistical**: Poisson deltas at N=200 needed explicit significance testing.
5. **Missing tables**: theta_E-stratified Poisson table and paired delta table were requested by all reviewers.
6. **Wording**: "physically correct texture" (too strong), "first-principles prediction" (imprecise), "overlays exactly" (contradicts own precision).
7. **Defensive text**: Missing caveats for completeness conditional on prior, host-confound AUC bracket, unrestricted beta_frac, theta_E crossover scaling, Fréchet robustness, prior improvement validation, tested only one textural hypothesis.

## What changed in v12

All issues above have been addressed:

### Figures (CRITICAL fix)
- `generate_all_figures.py` updated to point to D06 results (`D06_20260216_corrected_priors`)
- All 4 figures regenerated from D06 data with proper UMAP (umap-learn)
- Figure 1: y-axis raised to 10% (D06 peak 8.33% at theta_E=2.5)
- Figure 2: UMAP from D06 embeddings
- Figure 3: Legend now shows correct D06 median score (0.191 for low-bf injections)
- Figure 4: x-axis changed from "Lensed apparent magnitude" to "Source apparent magnitude"; gain validation tolerance relaxed to 1.5 pp; Poisson line now visibly rises above baseline at mag 20-23

### Numerical fixes
- AUC 0.996 → 0.997 (was on line 271 of v11, Section 4.3)
- Fréchet dimensions corrected: features_1 = 24-d (was 32-d), features_3 = 64-d (was 160-d)
- "(directional only)" removed from features_3 in Table 5 (n=112 > dim=64 → statistically valid)
- Conservative 78× growth bound (0.14→10.9, block 0→2) added as robustness note

### Labeling fixes
- Abstract, Introduction: "m_lensed = 19-22" → "source magnitudes m_source = 18-22 (~1-2.5 mag brighter after lensing)"
- Table 6 column: "Mag bin" → "Source mag bin"
- Figure 4 caption: "versus lensed apparent magnitude" → "versus source apparent magnitude"
- Clarifying sentence added in Section 4.4.2 about source vs lensed magnitude
- Grid results (Section 4.2, Conclusions) correctly retain "lensed magnitude" terminology

### Statistical significance (new)
- **Table (paired delta)**: New table with sign-test p-values for all 8 magnitude bins
  - mag 21-22: 27 gained vs 6 lost, p = 3.2×10⁻⁴
  - mag 22-23: 21 gained vs 3 lost, p = 2.8×10⁻⁴
  - mag 20-21: 25 vs 19, p = 0.45 (not individually significant)
- Inline text reports these p-values

### New tables
- **Table 7 (theta_E-stratified Poisson)**: 11 rows + marginal, showing No-Poisson, Poisson, and Delta (pp). Boldface for positive-delta rows. Crossover highlighted between theta_E = 1.2 and 1.5.
- **Table 8 (paired delta analysis)**: 8 rows with Mean Δp, Gained, Lost, Net, Detection rate change, sign-test p-value. Boldface for p < 0.01 rows.

### Wording fixes
- "physically correct pixel-level texture" → "consistent with shot noise" / "realistic pixel-level texture" (3 instances)
- "Contrary to the first-principles prediction" → "Contrary to the photoelectron-budget prediction"
- Figure 4 caption: "overlays the baseline exactly" → "overlays the baseline within 0.5 pp (= 1/N)"

### Defensive text additions
1. **Fréchet robustness**: Blocks 0-3 all have n=112 > dim; conservative 78× bound supports same conclusion
2. **Completeness conditional on prior**: Explicit caveat in Section 4.2 and Conclusions (v)
3. **Prior improvement**: Median injection score 0.110→0.191 (+74%) validates corrected priors
4. **Unrestricted beta_frac**: Discussion of dramatically lower detection rates confirming geometry importance
5. **Host-confound AUC bracket**: True morphology-only AUC lies between 0.778 (Tier-A vs Tier-B) and 0.997 (Tier-A vs injection)
6. **theta_E crossover caveat**: Qualitative scaling prediction, not calibrated model
7. **One textural hypothesis**: Explicitly states only shot noise tested; correlated noise and chromatic PSF untested

---

## Review questions — please answer each directly

### Q1: Number consistency
Verify that ALL numbers in the text (abstract, intro, body, conclusions) match the corresponding tables and figures. Specifically check:
- AUC values (should be 0.997 everywhere except Tier-A vs Tier-B at 0.778)
- Detection rates in abstract/intro vs Table 6
- Fréchet dimensions in prose vs Table 5
- Completeness percentages in Conclusions vs Section 4.2
- Magnitude labels: "source magnitude" for bright-arc, "lensed magnitude" for grid

### Q2: Figure-text agreement
Do the figure captions accurately describe what the figures should show given the D06 data? In particular:
- Does Figure 4 caption correctly say "source apparent magnitude"?
- Does Figure 1 y-axis accommodate the D06 peak of 8.33%?
- Does Figure 3 legend show median consistent with 0.191?

### Q3: Statistical claims
Are the statistical significance claims for Poisson deltas adequately supported? Specifically:
- Are the sign-test p-values correctly reported?
- Is the mag 20-21 result appropriately hedged as "not individually significant"?
- Is the aggregate interpretation (three contiguous bins) reasonable?

### Q4: Defensive text placement
Are the 7 defensive text additions well-placed and correctly hedged? Do any feel like they interrupt the flow?

### Q5: Remaining weaknesses
What are the top 3 remaining weaknesses for referee review? Are there any issues we missed?

### Q6: Submission readiness
Is v12 submission-ready (YES/NO)? If NO, what specific items remain?

---

## Files included in zip

- `mnras_merged_draft_v12.tex` — the paper source
- `paper/fig1_completeness.pdf` through `fig4_brightarc.pdf` — regenerated figures
- `paper/generate_all_figures.py` — figure generation script (D06 paths)
- `results/D06_20260216_corrected_priors/d06_poisson_diagnostics.json` — paired delta data
- `results/D06_20260216_corrected_priors/provenance.json` — code provenance checksums
- `scripts/analyze_poisson_diagnostics.py` — diagnostic analysis script
