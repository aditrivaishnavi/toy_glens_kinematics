# Prompt 26: Final Submission Blessing — Paper v16

## Context

This is the final review before submitting this paper. The paper has been through 14 prior versions and 6 rounds of LLM review. All previously identified issues have been addressed, including the issues found in the most recent review (Prompt 26, round 1). This version (v16) contains all fixes from the previous review.

**Paper:** "A morphological barrier: quantifying the injection realism gap for CNN strong lens finders in DESI Legacy Survey DR10"

**Attached:** `mnras_merged_draft_v16.pdf`

## What changed since the last review

All five issues identified by the previous review have been addressed:

### 1. Blank panels (FIXED)
The two blank cutout panels (Real row 1 col 2, Injection row 1 col 2) were caused by a matplotlib bug: `fig.add_subplot(gs[0, 1])` for the panel title annotation **overwrote** the image subplot at the same grid position. Fixed by using `fig.text()` for panel titles instead of creating new subplots. All 16 panels now render correctly.

### 2. Caption score counts (FIXED)
- Changed "Six are high-confidence detections (p > 0.9)" → "Four are high-confidence detections (p > 0.9), two are moderate (p ≈ 0.47–0.49)"
- The actual displayed scores are: 0.49, 1.00, 1.00, 0.93, 0.15, 0.99, 0.018, 0.47
- High (>0.9): 4 (1.00, 1.00, 0.93, 0.99). Moderate: 2 (0.49, 0.47). Missed (<0.3): 2 (0.15, 0.018).

### 3. Score display precision (FIXED)
- INJ_25_046 previously displayed as "p=0.05" (2 decimal places), creating ambiguity with the caption's "p < 0.05".
- Score formatting now shows 3 decimal places for scores between 0.001 and 0.1: INJ_25_046 displays as "p=0.047", making "p < 0.05" unambiguously correct.
- REAL_040 now displays as "p=0.018" instead of "p=0.02".

### 4. Float-too-large warning (FIXED)
- Reduced figure size from (7.5, 8.5) to (7.0, 7.0) inches. The "Float too large for page by 25.0pt" warning is eliminated in the build log.

### 5. Wording and reinforcement (FIXED)
- Changed "hand-selected" → "Manually selected illustrative examples" in the caption.
- Added note that "the selected IDs and metadata are provided in the supplementary audit file" for reproducibility.
- Added a reinforcement paragraph at the end of the Limitations section: "Because injections are rendered into real survey cutouts but do not reproduce all end-to-end survey and processing artefacts (e.g. correlated noise, chromatic PSF, deblending/resampling effects), our results should be interpreted as quantifying the realism gap of standard parametric injection pipelines, not isolating arc morphology as the sole causal factor."

### 6. Figure numbering note
The cutout comparison figure is compiled as **Figure 4** (not Figure 5) in the PDF because it is placed before the bright-arc detection plot. All cross-references use `\ref{fig:comparison}` and are automatically correct. The source file is named `fig5_comparison.pdf` for historical reasons only.

## Review checklist

Please answer each question with YES or NO first, then explain.

### Q1: Are all five issues from the previous review fully resolved?

Verify specifically:
- All 16 cutout panels render (no blank/white boxes)
- Caption score counts match the displayed values (4 high, 2 moderate, 2 missed)
- "p < 0.05" is unambiguous given the displayed injection scores
- No float warnings in the figure
- Wording changes are present and appropriate

### Q2: Does the corrected caption accurately match the figure panels?

Read each displayed score from the figure and verify the caption's categorical claims.

### Q3: Is the new Limitations reinforcement paragraph appropriate?

Does it preempt the survey-pipeline realism objection without overclaiming? Is it consistent with the paper's existing framing?

### Q4: Are there any remaining factual errors, overclaims, or unsupported statements?

Look specifically for anything we might have missed while fixing the above issues.

### Q5: Is the paper now submission-ready?

If NO, list the minimum remaining changes in priority order.
If YES, state so clearly.

## Important

- Be direct and factual. Do not give a rosy picture.
- If you find an error, state it plainly with the specific location (section, table, line).
- If the paper is ready, say so clearly. If not, say what specifically blocks submission.
- This is the final gate before submission. Thoroughness matters more than speed.
