# Prompt 9: We're in Trouble — Help Us Rescue This Paper

## Your Role

You are a senior astrophysicist and machine learning expert. We need your help. We've spent weeks on this project and we're stuck. Previous LLM reviews (Prompts 1-8) led us down a path of incremental diagnostics that failed to address the core problem. We need fundamentally different thinking.

**Critical context:** The assistant writing this code has introduced bugs before (Poisson clamp bug, summary reporting bug) and has been overly optimistic in interpreting results. Please review ALL attached code with skepticism. If something looks wrong, it probably is.

**Rules:**
- Read the attached code. Do not trust any summaries — verify from raw data and code.
- Be brutally honest. We don't need encouragement, we need solutions.
- Think differently. The last 8 prompts asked variations of the same question. We need a new angle.
- Provide concrete, actionable steps with realistic time estimates.

---

## The Situation

We have a CNN strong lens finder (EfficientNetV2-S, gen4) trained on DESI Legacy Survey DR10 cutouts. It works well on real lenses. But we cannot compute a meaningful selection function because our injection pipeline produces arcs that the CNN trivially distinguishes from real lenses.

### What Works

- **Tier-A recall:** 89.3% [82.6%, 94.0%] at p>0.3 on 112 spectroscopically confirmed lenses
- **AUC:** 0.9921 on validation set
- **Zero spatial leakage:** Confirmed via HEALPix analysis
- **Annulus preprocessing bug:** Fully characterized as cosmetic (0.15 normalized units shift, no PSF/depth correlation, 3.6pp recall drop not significant at 1.3 sigma)

### What Doesn't Work

**The injection-based selection function is not meaningful.** Here are the hard numbers:

#### D04 Matched Comparison (completed 2026-02-14, both grids use identical parameters)

**Baseline (no Poisson), p>0.3: marginal completeness = 3.41% (3,755/110,000)**

By lensed apparent magnitude:
- lensed_18-20: 48.8% (20/41) — tiny sample, unreliable
- lensed_20-22: 20.7% (2,559/12,361)
- lensed_22-24: 1.55% (1,082/70,016) — dominates the marginal
- lensed_24-27: 0.34% (94/27,582)

By theta_E (peaks at 2.0"):
- theta_E=0.50: 0.44%
- theta_E=1.50: 4.33%
- theta_E=2.00: 4.66% (peak)
- theta_E=3.00: 3.28%

**Fixed Poisson (torch.poisson), p>0.3: marginal completeness = 2.35% (2,584/110,000)**

Poisson noise HURT completeness by 1.06pp. By lensed magnitude:
- lensed_18-20: 51.3% (20/39) — same within noise
- lensed_20-22: 14.5% (1,809/12,456) — DOWN 6.2pp
- lensed_22-24: 0.95% (667/70,028) — DOWN 0.6pp
- lensed_24-27: 0.32% (88/27,477) — same

Both LLM reviewers in Prompts 5-7 predicted Poisson noise would IMPROVE detection. They were wrong. The added noise makes injections score lower, not higher.

**IMPORTANT CONTEXT ON PREVIOUS POISSON RESULTS:** In D02 (before D04), bright-arc tests using the BUGGY Poisson implementation (clamp(min=1.0)) showed apparent improvements (e.g., mag 20-21: 27.5% baseline -> 45.0% with Poisson). These were artifacts of the bug — the clamp inflated annulus MAD by ~2.5x, compressing normalized pixel values and effectively making the injections look more "noisy/faint" rather than adding realistic texture. The D04 fixed implementation shows the opposite: Poisson hurts.

#### Bright-Arc Detection Rate Comparison (all use beta_frac [0.1, 0.55], theta_E=1.5", n=200 hosts, seed=42)

| mag bin | D01 baseline (clip=10, no Poisson) | D02 clip=20 (no Poisson) | D02 BUGGY Poisson (clip=10) | D04 FIXED Poisson+clip=20 |
|---------|-----------------------------------|--------------------------|----------------------------|---------------------------|
| 18-19   | 17.0% | 30.5% | 17.5% | 32.5% |
| 19-20   | 24.5% | 32.0% | 31.0% | 29.0% |
| 20-21   | 27.5% | 37.0% | 45.0% | 26.5% |
| 21-22   | 35.5% | 40.5% | 43.0% | 24.0% |
| 22-23   | 31.0% | 35.0% | 23.5% | 24.0% |
| 23-24   | 24.0% | 14.5% |  5.5% |  7.5% |

**Key observations from this table (verify from raw JSONs):**
1. clip_range=20 alone helped across all bins except 23-24 (D02 clip=20 vs D01 baseline).
2. BUGGY Poisson appeared to help at mag 20-22 (+17.5pp at 20-21) but this was an artifact of the clamp(min=1.0) bug inflating annulus MAD.
3. FIXED Poisson+clip=20 is WORSE than clip=20 alone at every bin except 18-19. Adding correct Poisson noise degrades detection.
4. The D02 BUGGY Poisson column is included so you can see what the bug artifact looked like. Those numbers misled us for days.

#### Linear Probe (D01)

A logistic regression on the CNN's 1280-dimensional penultimate features achieves **AUC = 0.991 ± 0.010** separating 112 real Tier-A lenses from 200 low-beta_frac injections at mag 19. The CNN trivially distinguishes parametric Sersic injections from real lenses.

#### The Core Problem

The 86pp gap between real-lens recall (89.3%) and injection completeness (3.41%) is driven by injection unrealism, not model performance. Sersic profiles are too smooth, lack substructure (star-forming clumps, dust lanes, tidal features), have wrong color morphology, and lack correlated noise. The CNN learned what real lenses look like and rejects anything that doesn't match.

---

## What We've Tried (and what failed)

1. **Poisson noise** — FAILED. Both buggy and fixed implementations hurt detection.
2. **clip_range=20** — HELPS for bright arcs but is incompatible with the model (trained on clip=10). Cannot use for production selection function without retraining.
3. **Beta_frac restriction** — MARGINAL. Restricting to [0.1, 0.55] gave 35.5% vs 30% unrestricted at mag 21-22. Not statistically significant at n=200.
4. **Annulus fix** — IRRELEVANT. Characterized as cosmetic.
5. **8 rounds of LLM review** — Generated diagnostics that confirmed the known problem without solving it.

---

## What We Have Available

- **Compute:** Lambda3 (GH200 480GB GPU), available for ~1-2 weeks
- **Data:** DESI Legacy Survey DR10 cutouts (g/r/z, 101x101 pixels, 0.262"/pixel), 451k training set
- **Code:** Full injection engine (SIE+shear ray-tracing, Sersic sources), training pipeline, evaluation pipeline
- **Real lenses:** 389 Tier-A (277 train / 112 val), 4399 Tier-B (3079 train / 1320 val)
- **Time budget:** Realistic — this is for an MNRAS paper, not a multi-year project

---

## Questions (Answer ALL)

### Q1: Given these results, is there ANY version of this paper that is publishable in MNRAS? Be specific about what the paper would claim, what figures/tables it would contain, and what a hostile referee would say.

### Q2: The Poisson noise result contradicts both LLMs' predictions. Why did adding physically correct shot noise HURT detection? Is there a bug we're missing, or is this a real effect? Review `dhs/injection_engine.py` lines 549-567 and `dhs/preprocess.py` carefully.

### Q3: What is the single highest-leverage change we could make to the injection pipeline to close the sim-to-real gap? Be specific (code-level), estimate the time, and estimate the expected impact on probe AUC and completeness.

### Q4: Could we reframe the paper entirely — not as a selection function paper, but as something else that uses what we already have? What would that paper be?

### Q5: Are there published papers with similar problems (high real recall, low injection completeness) that got accepted? What did they do that we haven't?

### Q6: Review ALL the attached code files for bugs. The previous assistant introduced a Poisson clamp bug and a summary reporting bug. What else might be wrong? Check the injection engine, preprocessing, scoring pipeline, and selection function grid carefully.

---

## Attached Files

### Core Code (review ALL for bugs — the code author has introduced bugs before)
- `dhs/injection_engine.py` — Injection pipeline (Poisson fix at lines 549-567). REVIEW CAREFULLY.
- `dhs/preprocess.py` — Preprocessing (annulus normalization, clip_range). REVIEW CAREFULLY.
- `dhs/utils.py` — Utility functions including normalize_outer_annulus, robust_median_mad
- `dhs/scoring_utils.py` — Scoring utilities
- `dhs/model.py` — Model architecture (EfficientNetV2-S)
- `dhs/data.py` — Data loading and cutout I/O
- `dhs/transforms.py` — Data transforms / augmentations
- `dhs/constants.py` — Constants
- `dhs/preprocess_spec.py` — Preprocessing specification
- `dhs/selection_function_utils.py` — Selection function utilities
- `dhs/calibration.py` — Calibration utilities
- `dhs/gates.py` — Quality gates
- `dhs/s3io.py` — S3 I/O helpers
- `dhs/__init__.py` — Package init

### Configs
- `configs/injection_priors.yaml` — Injection parameter priors (source of truth)
- `configs/paperIV_efficientnet_v2_s_v4_finetune.yaml` — gen4 training config

### Scripts
- `scripts/selection_function_grid.py` — Selection function grid (summary fix included)
- `scripts/build_d04_comparison.py` — D04 comparison analysis
- `scripts/feature_space_analysis.py` — Linear probe and embedding analysis
- `scripts/split_balance_diagnostic.py` — Split balance checks

### Sim-to-Real Validation
- `sim_to_real_validations/bright_arc_injection_test.py` — Bright-arc detection test

### Run Scripts
- `scripts/run_d04_matched_comparison.sh` — D04 launch script

### Raw Results (verify numbers yourself — do NOT trust the summary tables above)

**D04 matched comparison (the definitive experiment):**
- `results/D04_20260214_matched_comparison/grid_no_poisson/selection_function.csv`
- `results/D04_20260214_matched_comparison/grid_no_poisson/selection_function_meta.json`
- `results/D04_20260214_matched_comparison/grid_poisson_fixed/selection_function.csv`
- `results/D04_20260214_matched_comparison/grid_poisson_fixed/selection_function_meta.json`
- `results/D04_20260214_matched_comparison/poisson_fixed_clip20_combined/bright_arc_results_bf0.10_0.55.json`

**D01 baseline (restricted beta_frac, no Poisson, clip=10):**
- `results/D01_20260214_pre_retrain_diagnostics/q21_beta_frac/bright_arc_results_bf0.10_0.55.json`
- `results/D01_20260214_pre_retrain_diagnostics/q22_embedding_umap/feature_space_results.json` (linear probe AUC)

**D02 bright-arc variants (verify the table above):**
- `results/D02_20260214_prompt5_quick_tests/unrestricted_bf/bright_arc_results_bf0.10_1.00.json` (unrestricted beta_frac baseline)
- `results/D02_20260214_prompt5_quick_tests/clip_range_20/bright_arc_results_bf0.10_0.55.json`
- `results/D02_20260214_prompt5_quick_tests/poisson_noise/bright_arc_results_bf0.10_0.55.json` (BUGGY Poisson — for reference only)
- `results/D02_20260214_prompt5_quick_tests/tier_a_scoring/real_lens_scoring_results.json` (Tier-A recall)
- `results/D02_20260214_prompt5_quick_tests/healpix_investigation/healpix_investigation.json` (spatial leakage)

### Documentation
- `EXPERIMENT_REGISTRY.md` — All experiments and their status
- `docs/MNRAS_SUPPLEMENTARY_TRAINING_LOG.md` — Full history of what we've done
- `docs/PAPER_FRAMING_SIM_TO_REAL_GAP.md` — Current paper framing and outline
