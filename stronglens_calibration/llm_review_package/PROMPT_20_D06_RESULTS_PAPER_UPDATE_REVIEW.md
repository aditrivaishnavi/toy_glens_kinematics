# PROMPT 20: D06 Results Review & Paper v11 Draft

**Date**: 2026-02-16
**Context**: D06 (full rerun with corrected injection priors) completed. All 10 experiments ran successfully on lambda3. This prompt presents the full results, audit trail, and asks the reviewer to (a) verify scientific coherence, (b) identify which paper sections need updating, and (c) flag any concerns about the changed interpretations before we create paper v11.

---

## BACKGROUND: WHAT CHANGED IN D06 vs D05

D05 (old priors) used: beta_frac (0.05, 0.35), Re (0.1, 1.5), n (0.5, 4.0), g-r N(0.8, 0.3), r-z N(0.4, 0.2), clumps_prob=0.6, no sky noise.

D06 (corrected priors) uses: beta_frac (0.10, 0.40), Re (0.15, 0.50), n (0.5, 2.0), g-r N(1.15, 0.30), r-z N(0.85, 0.20), clumps_prob=0.0, no sky noise, per-injection seeded RNG for Poisson.

The 6 corrections were motivated by visual comparison of injected vs real Tier-A lenses. The old priors produced compact/point-like sources, multi-blob clumps, and blue colors inconsistent with K-corrected z~1-3 sources. See `dhs/injection_engine.py` docstring for detailed rationale.

D06 does NOT use `--add-sky-noise`. Two independent LLM reviewers (Prompt 18) confirmed that adding Gaussian sky noise double-counts background variance since the host cutout already contains survey noise. Only Poisson shot noise (`--add-poisson-noise`) is used where applicable.

---

## AUDIT TRAIL

### Execution provenance

All experiments ran from a single `nohup bash scripts/run_d06_corrected_priors.sh > d06_log.txt 2>&1 &` on lambda3.

```
Run ID:     D06_20260216_corrected_priors
Start:      2026-02-16 09:47:28 UTC
End:        2026-02-16 13:55:36 UTC
Duration:   4 hours 8 minutes
Python:     3.12.3
torch:      2.7.0+cu128
```

### Per-experiment audit (from run_info JSON files)

| # | Experiment | add_sky_noise | add_poisson_noise | seed | Status |
|---|-----------|--------------|-------------------|------|--------|
| 1 | ba_baseline | False | False | 42 | OK |
| 2 | ba_poisson | False | True | 42 | OK |
| 3 | ba_clip20 | False | False | 42 | OK |
| 4 | ba_poisson_clip20 | False | True | 42 | OK |
| 5 | ba_unrestricted | False | False | 42 | OK |
| 6 | ba_gain_1e12 | False | True (gain=1e12) | 42 | OK |
| 7 | grid_no_poisson | False | False | 1337 | OK |
| 8 | grid_poisson | False | True | 1337 | OK |
| 9 | linear_probe | False | False | N/A | OK |
| 10 | tier_a_scoring | N/A | N/A | 42 | OK |

**Key verification**: `add_sky_noise=False` on ALL injection experiments. Confirmed from run_info JSON audit.

### Code that was run

The analysis script `scripts/analyze_d06_results.py` reads all result JSONs and CSVs, computes marginal completeness, completeness by theta_E, bright-arc Table 4, gain=1e12 sanity check, and saves everything to `d06_analysis_summary.json`. The script is included in the package for reproducibility.

---

## D06 RESULTS

### 1. GRID COMPLETENESS (110,000 injections, 220 non-empty cells)

**Marginal completeness:**

| Condition | p>0.3 | p>0.5 |
|-----------|-------|-------|
| No Poisson | 5.18% (5697/110000) | 4.15% |
| Poisson (gain=150) | 3.79% (4174/110000) | 2.85% |
| **Deficit** | **-1.38 pp** | **-1.30 pp** |

**Comparison with D05**: D05 had 3.41% (no Poisson) and 2.37% (Poisson), deficit -1.04 pp. D06 completeness is HIGHER (5.18% vs 3.41%) because corrected priors produce more arc-like morphologies that the CNN can detect. Poisson degradation is slightly stronger (-1.38 vs -1.04 pp).

**Completeness by theta_E (no-Poisson, p>0.3):**

| theta_E | D06 | D05 (old) |
|---------|-----|-----------|
| 0.50 | 0.22% | 0.44% |
| 0.75 | 0.33% | 1.22% |
| 1.00 | 1.53% | 2.57% |
| 1.25 | 3.39% | 3.61% |
| 1.50 | 5.59% | 4.33% |
| 1.75 | 6.52% | 4.58% |
| 2.00 | 7.99% | 4.66% |
| 2.25 | 8.02% | 4.44% |
| 2.50 | 8.33% | 4.32% |
| 2.75 | 7.90% | 4.10% |
| 3.00 | 7.15% | 3.28% |

**Note**: D06 shows a SHIFT in the peak from theta_E=2.0 (D05) to theta_E=2.5 (D06), with MUCH higher completeness at large theta_E. This is expected: the corrected priors (higher Re_min=0.15", lower n_max=2.0) produce more extended arcs that are better resolved at larger Einstein radii. The steep drop at small theta_E (0.5-0.75) is stronger in D06 because the higher Re_min means sources can't be as compact.

**Completeness by lensed magnitude (no-Poisson, p>0.3):**

| Lensed mag bin | D06 |
|---------------|-----|
| 20-22 | 12.38% |
| 22-24 | 0.73% |
| 24-27 | 0.45% |

D05 had ~20.7% at 20-22 and ~1.55% at 22-24. D06 is lower at bright magnitudes — likely because the narrower beta_frac range produces more arc-like (spread-out) morphologies that may be harder to detect than the compact doubles from the old priors.

### 2. BRIGHT-ARC TABLE 4 (detection rate at p>0.3, N=200 per bin)

| Mag bin | Baseline | Poisson (g=150) | clip=20 | Poisson+clip20 | Unrestricted | Gain=1e12 |
|---------|----------|-----------------|---------|----------------|-------------|-----------|
| 18-19 | 29.0% | 29.0% | 45.0% | 42.5% | 20.5% | 29.0% |
| 19-20 | 33.5% | 33.5% | 46.5% | 42.5% | 14.5% | 34.0% |
| 20-21 | 38.5% | 41.5% | 49.5% | 46.5% | 15.0% | 38.5% |
| 21-22 | 33.0% | 43.5% | 37.5% | 32.5% | 12.0% | 33.0% |
| 22-23 | 23.5% | 32.5% | 21.0% | 17.5% | 6.5% | 23.5% |
| 23-24 | 11.0% | 12.5% | 4.0% | 3.5% | 2.5% | 11.0% |
| 24-25 | 2.0% | 1.5% | 1.0% | 1.0% | 0.5% | 2.0% |
| 25-26 | 0.5% | 0.5% | 0.0% | 0.0% | 0.0% | 0.5% |

**CRITICAL OBSERVATION**: In D05 (old priors), Poisson noise DECREASED detection at every magnitude bin. In D06 (corrected priors), **Poisson noise INCREASES detection at mag 20-23**:
- 20-21: +3.0 pp (38.5% → 41.5%)
- 21-22: +10.5 pp (33.0% → 43.5%)
- 22-23: +9.0 pp (23.5% → 32.5%)
- 23-24: +1.5 pp (11.0% → 12.5%)

This REVERSAL changes the paper's narrative. The paper (v10) Section 4.4.1 states: "Poisson noise reduces the detection rate at every magnitude bin with non-trivial sample size." This is no longer true.

**Possible explanation**: The corrected priors produce smooth, extended arcs (higher Re_min, lower n_max, no clumps). These arcs may be anomalously smooth compared to real sky signal. Adding Poisson noise adds realistic flux-proportional texture, making the injection look MORE like real signal to the CNN, and increasing its score. This is actually a STRONGER version of the morphological barrier argument: the CNN is sensitive to pixel-level texture, and the absence of realistic noise is itself a detectable artifact.

At faint magnitudes (24+), Poisson noise is negligible compared to background, so it neither helps nor hurts significantly.

**Gain=1e12 sanity check**: ALL PASS (within 1/N tolerance). Maximum deviation is +0.5 pp (1 injection) at mag 19-20. Confirms Poisson implementation is correct.

**Clip=20 results**: clip=20 helps at bright magnitudes (45% vs 29% at 18-19) but hurts at faint magnitudes (4% vs 11% at 23-24). The model was trained with clip=10, so clip=20 is out-of-distribution for the preprocessing. Bright arcs benefit because their features are preserved; faint arcs are harmed by the changed normalization.

### 3. LINEAR PROBE

| Metric | D06 | D05 (old) |
|--------|-----|-----------|
| CV AUC (real Tier-A vs low-bf inj) | 0.9970 ± 0.0033 | 0.996 ± 0.004 |
| Median score: Real Tier-A | 0.9946 | 0.995 |
| Median score: Inj low-bf | 0.1907 | 0.110 |
| Median score: Inj high-bf | 0.0190 | N/A |
| Median score: Negatives | 0.000015 | 1.5e-5 |
| Fréchet distance (real vs low-bf) | 215.0 | N/A |

**Note**: Low-bf injection median score INCREASED from 0.110 to 0.191 with corrected priors. This means the corrected injections are scored HIGHER (less detectable as fake) than the old ones. Consistent with improved realism from corrected priors. But AUC is essentially unchanged at 0.997 — the CNN can still almost perfectly separate real from injected in embedding space.

### 4. TIER-A RECALL (unchanged)

| Threshold | Recall |
|-----------|--------|
| p>0.3 | 89.3% (100/112) |
| p>0.5 | 83.9% (94/112) |

Unchanged as expected — no injection involved.

---

## SPECIFIC REVIEW QUESTIONS

### Q1: Scientific coherence

**Q1.1**: Does the Poisson detection-rate INCREASE at mag 20-23 make physical sense? Is the "Poisson adds realistic texture to smooth arcs" explanation defensible, or is there a simpler/better explanation?

**Q1.2**: The grid-level result still shows Poisson DECREASES overall completeness (-1.38 pp). But bright-arc shows Poisson INCREASES at matched magnitudes. Are these contradictory? How should the paper reconcile them?

**Q1.3**: Completeness by theta_E shows D06 values DECREASE at small theta_E (0.50: 0.22% vs 0.44%) but INCREASE at large theta_E (2.50: 8.33% vs 4.32%). Is this consistent with the corrected priors (higher Re_min=0.15", lower n_max=2.0)?

**Q1.4**: The low-bf injection median score increased from 0.110 to 0.191. Does this mean the corrected injections are genuinely more realistic, or could there be an artifact?

### Q2: Paper update guidance

**Q2.1**: The paper's Poisson narrative (Section 4.4.1) needs rewriting. What should the new narrative be? Previously: "Poisson hurts everywhere." Now: "Poisson helps at some magnitudes, hurts overall." How should we frame this?

**Q2.2**: The Poisson-clip interaction (Section 4.4.3) previously showed amplified damage. Does D06 show the same pattern? Looking at the numbers:
- clip=20 alone at 21-22: 37.5% (+4.5 pp from baseline 33.0%)
- Poisson alone at 21-22: 43.5% (+10.5 pp from baseline)
- Combined: 32.5% (-0.5 pp from baseline)
Is this still "amplified interaction" or a different effect?

**Q2.3**: Which paper claims need QUALITATIVE changes (not just number updates)?

### Q3: Completeness of audit

**Q3.1**: Is the audit trail sufficient to confirm the run was correct? The run_info JSON for each experiment records CLI args, seed, add_sky_noise=False, add_poisson_noise setting, and timestamp.

**Q3.2**: Are there any remaining concerns about the execution? Note: git_hash shows "unknown" because the lambda3 working copy is not a git repo (it's synced via scp). The exact code that ran can be verified by comparing the files on lambda3 with the local git-tracked files.

### Q4: Should we proceed with v11?

**Q4.1**: Given the Poisson reversal and other changes, should we proceed with creating paper v11? Or should we first run additional diagnostic experiments to understand the Poisson reversal?

**Q4.2**: Are there any experiments we should add to D06 that would strengthen the paper's claims given the new results?

---

## FILES IN THIS PACKAGE

| File | Description |
|------|-------------|
| `scripts/analyze_d06_results.py` | Analysis script that produced all numbers above |
| `results/d06_analysis_summary.json` | Full machine-readable results |
| `scripts/run_d06_corrected_priors.sh` | D06 driver (what actually ran) |
| `dhs/injection_engine.py` | Engine with corrected priors + RNG fix |
| `configs/injection_priors.yaml` | Prior registry |
| `paper/mnras_merged_draft_v10.tex` | Current paper (to be updated) |
| `scripts/test_injection_visual.py` | Visual test script |
| `sim_to_real_validations/bright_arc_injection_test.py` | 2-phase bright-arc |
| `scripts/selection_function_grid.py` | Grid completeness |
| `scripts/feature_space_analysis.py` | Linear probe |

---

## INSTRUCTION TO REVIEWER

This is a critical juncture. The D06 results change the paper's narrative in at least one major way (Poisson reversal). We need an honest, rigorous assessment of whether these results are scientifically correct and how to update the paper. If you see any red flags — results that look buggy, interpretations that don't hold up, or concerns about the experimental design — state them plainly. We would rather catch problems now than after submission.

Please answer each question with YES/NO first, then explain.
