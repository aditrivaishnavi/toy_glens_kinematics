# PROMPT 23 — Paper v13 Post-Review Fixes Verification

## Background

Paper v12 was reviewed by three independent LLMs (Prompt 22: LLM1, LLM3, LLM5). All three gave a **conditional YES** for submission readiness, contingent on fixing a small number of specific issues. Paper v13 incorporates all requested fixes plus a new Figure 5 (real vs injection visual comparison). This prompt asks you to verify the fixes were applied correctly and assess final submission readiness.

---

## What changed from v12 to v13

### Fix 1: Abstract magnitude label (LLM1 Q1, LLM5 Issue 1)

The abstract had a residual `m_lensed = 21-23` for the bright-arc Poisson result, which should be source magnitudes. **Fixed**: now reads `source magnitudes m_source = 21-23, corresponding to lensed arc magnitudes ~19-21`.

### Fix 2: Table 8 theta_E labels (LLM5 Issue 2)

Five theta_E values in the Poisson table used rounded values (0.8, 1.2, 1.8, 2.2, 2.8) instead of the actual grid steps (0.75, 1.25, 1.75, 2.25, 2.75). **Fixed**: all 11 rows now match the grid steps exactly. Cascading prose in Section 4.4.4 also corrected. The crossover statement now correctly says "between theta_E = 1.25 and 1.50" throughout.

### Fix 3: "source mag" qualifiers (LLM5 Issue 3)

Conclusions item (iv) and Section 4.4.6 referred to "mag 21-22" without the "source" qualifier. **Fixed**: both now say "source mag".

### Fix 4: Figure 1 right panel y-axis clipping (User feedback)

The lensed_18-20 bar had completeness of 57.8% but the y-axis was capped at 55%. **Fixed**: y-axis raised to 65%. All figures regenerated from D06 data.

### Fix 5: Figure 1 caption bin count (LLM3 Q1)

Caption said "three bins (20-22, 22-24, 24-27)" but the figure shows four bins including 18-20. **Fixed**: caption now says "four bins (18-20, 20-22, 22-24, 24-27)".

### Fix 6: Table 5 text overlap (User feedback)

Table 5 (Linear probe diagnostics) was a single-column `\begin{table}` with long entries that overlapped adjacent text in the MNRAS two-column layout. **Fixed**: changed to `\begin{table*}` (full page width).

### Fix 7: Bonferroni correction for multiple testing (LLM1 Q3, LLM3 Q3)

Both reviewers flagged that the 8-bin sign tests were uncorrected. **Fixed**: added sentence: "Both p-values survive Bonferroni correction for eight bins (threshold 0.05/8 = 0.00625)."

### Fix 8: NEW Figure 5 — Real vs Injection Visual Comparison

A new figure was added showing real Tier-A lenses alongside brightness-matched parametric injections at three Einstein radii (theta_E = 0.75, 1.50, 2.50 arcsec), with 4 pairs per regime. Key properties:

- **Brightness matching**: all pairs matched to within < 0.1 mag in total r-band flux
- **Score contrast**: real lenses score p > 0.99 while matched injections score p < 0.01 in most cases
- **Deterministic selection**: top-scored injections at each theta_E, matched to closest Tier-A by magnitude
- **Audit trail**: `comparison_audit.json` records all pair metadata
- **Source**: D06 grid cutouts (no-Poisson baseline) and Tier-A validation cutouts

This figure directly illustrates the morphological barrier for the reader: the injections and real lenses look visually similar — especially at compact theta_E — but the CNN distinguishes them effortlessly.

---

## Cumulative v12 -> v13 change summary

| # | Fix | Source |
|---|-----|--------|
| 1 | Abstract m_lensed -> m_source for Poisson result | LLM1, LLM5 |
| 2 | Table 8 theta_E labels: 0.8->0.75, 1.2->1.25, etc. + prose | LLM5 |
| 3 | "source mag" qualifiers in Conclusions (iv) and Section 4.4.6 | LLM5 |
| 4 | Figure 1 y-axis: 55% -> 65% | User |
| 5 | Figure 1 caption: "three bins" -> "four bins (18-20, ...)" | LLM3 |
| 6 | Table 5: single-column -> full-width to fix overlap | User |
| 7 | Bonferroni correction sentence for sign tests | LLM1, LLM3 |
| 8 | New Figure 5: real vs injection visual comparison | User, all LLMs recommended |

---

## Review questions — please answer each directly with YES/NO first

### Q1: Were all v12 review issues resolved?

Verify each of the 7 fixes (items 1-7) was correctly applied. In particular:
- Is there any remaining `m_lensed` in a bright-arc context?
- Do all 11 theta_E values in Table 8 match {0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00}?
- Does the Figure 1 caption correctly list four lensed-magnitude bins?
- Is the Bonferroni sentence present and correct?

### Q2: Number consistency (final sweep)

Run a final consistency check across all numbers in the paper:
- AUC values (0.997 for Tier-A vs inj, 0.778 for Tier-A vs Tier-B)
- Detection rates in abstract/intro vs tables
- Completeness figures in Conclusions vs Section 4.2
- Frechet dimensions in prose vs Table 5
- p-values in paired delta table vs inline text
- "source magnitude" for bright-arc, "lensed magnitude" for grid — no exceptions

### Q3: Figure 5 assessment

Is the new comparison figure (Figure 5) well-placed, well-captioned, and scientifically appropriate? Specifically:
- Does the caption accurately describe the matching methodology?
- Is the placement (between the linear probe and Poisson sections) logical?
- Does the figure add genuine value for a referee, or is it redundant with the existing figures?
- Any concerns about the deterministic selection methodology?

### Q4: Layout and formatting

Are there any remaining layout issues visible from the LaTeX source:
- Table/figure overlap with text?
- Caption accuracy for all figures and tables?
- Reference labels all resolved?

### Q5: Top 3 remaining weaknesses for referee

What are the three most likely referee objections that remain? Be specific and honest.

### Q6: Submission readiness

Is v13 submission-ready (YES/NO)? If NO, list the specific blocking items.

---

## Files included in zip

- `paper/mnras_merged_draft_v13.tex` — the paper source
- `paper/fig1_completeness.pdf` through `fig5_comparison.pdf` — all 5 figures
- `paper/generate_all_figures.py` — figure generation script
- `scripts/generate_comparison_figure.py` — comparison figure script
- `results/D06_20260216_corrected_priors/comparison_figure/comparison_audit.json` — audit trail for Figure 5
- `results/D06_20260216_corrected_priors/poisson_diagnostics/d06_poisson_diagnostics.json` — paired delta data
- `results/D06_20260216_corrected_priors/provenance.json` — code provenance checksums
