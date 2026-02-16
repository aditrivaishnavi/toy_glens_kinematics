# Prompt 8: Final Paper Review Before Submission

## Your Role

You are an independent scientific reviewer. This is the FINAL review before we prepare the MNRAS submission. We have completed all diagnostics and made our key decisions. We now need you to:

1. Confirm our paper framing is defensible
2. Review the updated completeness map (with Poisson noise)
3. Identify any remaining gaps that must be filled before submission
4. Draft figure captions for the key figures

**Rules:**
- Read the attached code and raw results. Do not trust summaries.
- If something is wrong, say so. This is our last chance to catch issues.
- Be rigorous but constructive. We want actionable feedback.
- Think like a senior MNRAS referee.

---

## What Has Changed Since Prompt 7

### Decisions Made (Based on Prompt 7 LLM Consensus)

1. **NOT retraining for annulus fix.** Both reviewers agreed the annulus effect is cosmetic (3.6pp recall drop at 1.3 sigma, not significant; MAD unchanged; no PSF/depth correlation). Documented in Appendix.

2. **Added Poisson noise to selection function grid.** Both reviewers agreed this is the highest-priority computation. The `--add-poisson-noise` flag was added to `selection_function_grid.py` and a full grid run (D03) was completed.

3. **Ran combined Poisson + clip_range=20 diagnostic.** Tests whether these two effects are additive.

4. **Reframed paper around sim-to-real gap.** The paper is now framed as a measurement of the injection realism gap, not just a selection function.

### New Code

- `selection_function_grid.py`: Added `--add-poisson-noise` and `--gain-e-per-nmgy` CLI flags, threaded through to `inject_sis_shear()`.
- `generate_paper_umap_figure.py`: Publication-quality two-panel UMAP figure (category + score).
- `run_d03_poisson_grid.sh`: D03 runner script.

### New Documents

- `docs/PAPER_FRAMING_SIM_TO_REAL_GAP.md`: Complete paper outline, abstract draft, key numbers, referee preemption strategy.
- `docs/APPENDIX_ANNULUS_CHARACTERIZATION.md`: Four-diagnostic characterization of the annulus effect.
- `docs/COMPARISON_TABLE_PUBLISHED_RESULTS.md`: Comparison with Herle et al. 2024, HOLISMOKES XI, Euclid Prep. XXXIII.

---

## D03 Results: Selection Function Grid with Poisson Noise

### Grid Configuration
- Model: gen4 (EfficientNetV2-S, AUC=0.9921)
- Grid: 11 theta_E (0.5-3.0") x 7 PSF (0.9-1.8") x 5 depth (22.5-24.5 mag) = 385 cells
- Injections per cell: 500
- Total injections: 192,500
- Poisson noise: ON (gain=150 e-/nmgy)
- FPR-derived thresholds: 0.001, 0.0001

### Raw Results

Grid completed in 1,349 seconds (~22 min), 3.50s/cell.
110,000 successful injections (0 failures). 343/385 cells empty (no hosts matching PSF/depth bin).

FPR-derived thresholds:
- FPR=0.001 -> threshold p=0.806
- FPR=0.0001 -> threshold p=0.995

**Mean completeness by threshold (over non-empty cells):**
- p>0.3: **2.6%**
- p>0.5: **2.4%**
- p>0.7: **2.2%**
- FPR=0.001 (p>0.806): **2.0%**
- FPR=0.0001 (p>0.995): **1.0%**

**Completeness by theta_E at p>0.3:**
```
theta_E=0.50": C=0.000, mean_arc_SNR=3.0
theta_E=0.75": C=0.000, mean_arc_SNR=4.3
theta_E=1.00": C=0.011, mean_arc_SNR=4.6
theta_E=1.25": C=0.021, mean_arc_SNR=5.8
theta_E=1.50": C=0.010, mean_arc_SNR=5.4
theta_E=1.75": C=0.021, mean_arc_SNR=6.5
theta_E=2.00": C=0.056, mean_arc_SNR=7.8
theta_E=2.25": C=0.034, mean_arc_SNR=5.5
theta_E=2.50": C=0.056, mean_arc_SNR=6.7
theta_E=2.75": C=0.034, mean_arc_SNR=4.6
theta_E=3.00": C=0.034, mean_arc_SNR=5.9
```

_The complete CSV (1.6MB, all cells/thresholds/mag bins) and JSON metadata
are in the attached zip under `results/D03_20260214_poisson_grid/selection_function_poisson/`._

### KEY FINDING: Poisson noise does NOT dramatically improve marginal completeness

The previous grid (without Poisson) reported ~3.5% marginal completeness. With Poisson
noise, we get ~2.6%. This is LOWER, not higher.

**Why?** The D02 bright-arc test showed Poisson noise helps at mag 20-21 (+17.5pp)
but HURTS at mag 22-24 (e.g., mag 22-23: 31% -> 23.5%). When integrated over the
full parameter space (which includes many faint-source configurations), the faint-end
degradation offsets the bright-end improvement. The net effect is roughly neutral or
slightly negative.

This is an important result for the paper: Poisson noise alone is not sufficient
to close the realism gap. It helps for the RIGHT physical reason (adding source
texture), but the injection pipeline has other deficiencies (smooth Sersic morphology,
no correlated noise, simplified PSF) that dominate at faint magnitudes.

---

## D03 Results: Combined Poisson + clip_range=20 Diagnostic

### Individual Effects (from D02)
| Condition | mag 18-19 (p>0.3) | mag 20-21 (p>0.3) | mag 21-22 (p>0.3) |
|-----------|--------------------|--------------------|---------------------|
| Baseline (no Poisson, clip=10) | 17.0% | 27.5% | 35.5% |
| Poisson alone (clip=10) | 17.5% | 45.0% | 43.0% |
| clip_range=20 alone | 30.5% | 37.0% | 40.5% |

### Combined Effect (D03)

| mag bin | Baseline (D01) | Poisson only (D02) | clip20 only (D02) | **Poisson+clip20 (D03)** |
|---------|---------------|--------------------|--------------------|-------------------------|
| 18-19   | 17.0%         | 17.5%              | 30.5%              | **42.5%**               |
| 19-20   | 24.5%         | 31.0%              | 32.0%              | **47.5%**               |
| 20-21   | 27.5%         | 45.0%              | 37.0%              | **45.0%**               |
| 21-22   | 35.5%         | 43.0%              | 40.5%              | **30.5%**               |
| 22-23   | 31.0%         | 23.5%              | 35.0%              | **17.5%**               |
| 23-24   | 24.0%         | 5.5%               | 14.5%              | **3.5%**                |

### KEY FINDING: Effects are super-additive for bright arcs, destructive for faint

**Bright arcs (mag 18-20):** The combined effect is SUPER-ADDITIVE.
- At mag 18-19: 42.5% combined vs max(17.5%, 30.5%) = 30.5%. This is a +25.5pp improvement over baseline.
- At mag 19-20: 47.5% combined -- the highest detection rate for any injection configuration tested.

This makes physical sense: clip_range=20 preserves the bright arc's morphological
structure (which was clipped at +-10), AND Poisson noise adds the expected pixel-level
texture. Together they make the injection look more like a real astronomical source.

**The peak combined detection of 47.5% (at mag 19-20) closes more than half the
gap to real-lens recall (89.3%).**

**Faint arcs (mag 21+):** The combined effect is DESTRUCTIVE.
- At mag 21-22: 30.5% combined vs 43.0% Poisson-only -- WORSE than individual effects.
- At mag 22-23: 17.5% combined vs 23.5% Poisson-only.

This also makes physical sense: clip_range=20 expands the dynamic range, pushing
faint-arc images further from the model's training distribution (which used clip=10).
At faint magnitudes, this distribution shift overwhelms any benefit from Poisson noise.

**Implication for the paper:** Two simple injection realism fixes (Poisson + wider clip)
can close ~half the gap for bright arcs, but fundamentally different approaches are
needed for the faint end (real galaxy stamps, correlated noise, band-dependent PSF).
NOTE: clip_range=20 requires retraining for production use; this result is diagnostic.

---

## All Previous Results (Unchanged from Prompt 7)

Tier-A recall: 89.3% [82.6%, 94.0%] at p>0.3 (100/112 confirmed lenses)
Tier-B recall: 72.0% [69.5%, 74.3%]
AUC: 0.9921
Linear probe AUC: 0.991 +/- 0.010
Tier-A spatial leakage: ZERO (274 train, 112 val HEALPix pixels, 0 overlap)

---

## Proposed Paper Structure

### Title
"The injection realism gap in CNN strong lens selection functions: quantifying parametric source limitations with DESI Legacy Survey DR10"

### Key Figures
1. **UMAP two-panel** (category + score): Shows distinct manifolds for real vs injected
2. **Detection rate vs source magnitude**: Multiple injection configurations
3. **Completeness heatmap** C(theta_E, depth): With and without Poisson noise
4. **Score distributions**: Real Tier-A, injections, negatives

### Key Claims
- 89.3% recall on confirmed lenses (zero spatial leakage)
- Linear probe AUC = 0.991: first quantitative measurement of injection realism gap
- Injection completeness is a conservative lower bound, not unbiased estimate
- Poisson noise increases detection by up to +17.5pp
- Beta_frac restriction doubles detection at moderate magnitudes

---

## Questions

### 1. Paper Framing Review

**1a.** Read our draft abstract in `docs/PAPER_FRAMING_SIM_TO_REAL_GAP.md`. Is it scientifically accurate? Is anything overclaimed or underclaimed? Suggest specific edits.

**1b.** The proposed title is "The injection realism gap in CNN strong lens selection functions: quantifying parametric source limitations with DESI Legacy Survey DR10". Is this too narrow, too broad, or about right for MNRAS?

**1c.** Review our referee preemption strategy (Section 8 of the paper framing doc). Are there additional criticisms we haven't anticipated?

### 2. Updated Completeness Map

**2a.** Compare the marginal completeness with and without Poisson noise. Is the improvement consistent with what we predicted from the D02 bright-arc test? If not, why?

**2b.** Does the updated completeness map change any of the claims from Prompt 7?

**2c.** What is the correct way to present both maps (with and without Poisson) in the paper? Show one as primary and one as comparison? Show both side by side?

### 3. Combined Poisson + clip_range=20 Results

**3a.** Are the effects additive? If the combined detection is roughly (Poisson alone) + (clip20 alone) - (baseline), that supports independent mechanisms. If not, what does it mean?

**3b.** Given that clip_range=20 requires retraining for production use, how should we present this result? As a diagnostic only, or as a recommendation for future work?

### 4. Comparison with Published Results

**4a.** Review our comparison table in `docs/COMPARISON_TABLE_PUBLISHED_RESULTS.md`. Is it fair and accurate? Have we missed any important comparison papers?

**4b.** Is our claim of "first quantitative measurement of the injection realism gap" defensible? Has anyone done this before with a linear probe or equivalent metric?

### 5. Final Gaps

**5a.** Is there anything MISSING that would prevent this paper from being accepted at MNRAS? Be specific.

**5b.** If you had to rank the 5 most important things to include in the paper (that we might not have thought of), what would they be?

**5c.** Draft a 2-sentence figure caption for the UMAP two-panel figure.

---

## What We Need From You

This is the final review. Give us:
1. Specific edits to the abstract
2. Any remaining experiments that MUST be run (not nice-to-have)
3. A clear "ready to submit" or "not ready, do X first" verdict
