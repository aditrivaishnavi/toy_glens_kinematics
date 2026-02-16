# Prompt 7: Should We Retrain? — You Decide

## Your Role

You are an independent scientific reviewer. We give you ALL the code, ALL the raw results, and ALL the context. We want YOU to decide what we should do next. Should we retrain the model? Should we change the injection pipeline? Should we publish as-is?

**Rules:**
- Read the attached code. Do not trust summaries — form your own conclusions from the data.
- For every recommendation you make, explain your reasoning step by step. Show your work.
- If the evidence is ambiguous, say so. Do not force a conclusion.
- Be 100% honest. If we are wasting our time, tell us.
- Think like a hostile but fair MNRAS referee who wants to see solid science.

---

## What Is in the Zip

The attached `d01_d02_full_evidence_package.zip` contains:

**Diagnostic scripts we wrote and ran (11 Python scripts + 2 shell runners):**
- `scripts/split_balance_diagnostic.py` — checks train/val split balance, PSF/depth distributions, HEALPix spatial leakage
- `scripts/masked_pixel_diagnostic.py` — checks for non-finite / zero pixels in cutouts
- `scripts/annulus_comparison.py` — compares normalization stats between old (20,32) and new (32.5,45) annuli
- `scripts/mismatched_annulus_scoring.py` — scores gen4 model with native vs mismatched annulus preprocessing
- `scripts/feature_space_analysis.py` — extracts CNN embeddings, runs linear probe (logistic regression) separating real lenses from injections, computes Frechet distances
- `scripts/investigate_healpix_nan.py` — investigates why all positives have NaN healpix_128
- `scripts/generate_umap_figure.py` — generates UMAP visualization from saved embeddings
- `sim_to_real_validations/bright_arc_injection_test.py` — injects bright arcs at varying magnitudes and scores them, with flags for --clip-range, --add-poisson-noise, --beta-frac-range
- `sim_to_real_validations/real_lens_scoring.py` — scores real lenses with --tier-a-only option
- `scripts/run_diagnostics.sh` — D01 runner (6 diagnostics)
- `scripts/run_d02_quick_tests.sh` — D02 runner (7 experiments)

**Core library modules these scripts depend on:**
- `dhs/preprocess.py`, `dhs/preprocess_spec.py`, `dhs/scoring_utils.py`, `dhs/model.py`, `dhs/injection_engine.py`, `dhs/utils.py`, `dhs/constants.py`

**Training configs:**
- `configs/paperIV_efficientnet_v2_s_v4_finetune.yaml` — current gen4 model
- `configs/gen5a_efficientnet_annulus_fix.yaml` — proposed retrain: from-scratch, corrected annulus
- `configs/gen5b_efficientnet_annulus_ft.yaml` — proposed retrain: finetune gen4, corrected annulus
- `configs/gen5c_efficientnet_weighted_loss.yaml` — proposed retrain: from-scratch, corrected annulus + weighted loss

**Injection prior registry:**
- `configs/injection_priors.yaml` — exact parameter ranges used for all injections

**ALL raw results (JSON + logs + embeddings + UMAP figures):**
- `results/D01_20260214_pre_retrain_diagnostics/` — 6 diagnostic outputs
- `results/D02_20260214_prompt5_quick_tests/` — 7 experiment outputs including UMAP

**Registry:**
- `EXPERIMENT_REGISTRY.md` — maps all experiments to their configs, checkpoints, and results

---

## The Model

- **Architecture:** EfficientNetV2-S (pretrained ImageNet, finetuned)
- **Training data:** 451,681 cutouts (101x101 pixels, 3-band g/r/z, DESI Legacy Survey DR10)
  - Train: 316,100 (277 Tier-A confirmed lenses, 3,079 Tier-B visual candidates, 312,744 negatives)
  - Val: 135,581 (112 Tier-A, 1,320 Tier-B, 134,149 negatives)
- **Current checkpoint:** gen4 (finetuned from gen2, AUC=0.9921)
- **Preprocessing:** `raw_robust` — per-band normalization using median and MAD from outer annulus of stamp, then clip to [-clip_range, +clip_range]
- **Known preprocessing issue:** Annulus radii used during training were (20, 32) pixels. The geometrically correct values for 101x101 stamps are (32.5, 45.0). Training clip_range=10.

## The Injection Pipeline

- Synthetic lensed arcs injected into real host galaxy cutouts using SIS+shear ray-tracing with Sersic source profiles
- Parameter ranges: see `configs/injection_priors.yaml`
- Poisson noise on injected arcs: OFF by default (flag exists but was not used during training or previous selection function computation)
- Selection function completeness: ~3.5% marginal over full parameter space

## Three Proposed Retrain Configs (Ready to Run)

| Config | What changes | Estimated GPU time |
|--------|-------------|-------------------|
| gen5a | From-scratch, corrected annulus (32.5,45), same hyperparams as gen1 | ~15 GPU-hours |
| gen5b | Finetune from gen4, corrected annulus | ~5 GPU-hours |
| gen5c | From-scratch, corrected annulus + weighted loss (Tier-A=1.0, Tier-B=0.5) | ~15 GPU-hours |

---

## ALL Raw Results

Below is every number from every experiment. The full JSON files and console logs are also in the zip.

### D01: Pre-Retrain Diagnostics (6 experiments)

**1. Split Balance:**
```
train: 316,100 total — 277 Tier-A, 3,079 Tier-B, 312,744 neg
val:   135,581 total — 112 Tier-A, 1,320 Tier-B, 134,149 neg
PSF balance (KS test): psfsize_r p=0.174, psfdepth_r p=0.123 (not significantly different)
All 4,788 positives have NaN healpix_128
```

**2. Masked Pixels:**
```
1000 cutouts sampled: 0 non-finite pixels, max zero fraction 0.52%
```

**3. Annulus Comparison — (20,32) vs (32.5,45):**
```
old_20_32:   median_of_medians=0.000467, median_of_MADs=0.002276
new_32p5_45: median_of_medians=0.000340, median_of_MADs=0.002197

Paired differences (new - old), N=1000:
  Median diff: mean=-0.000345, std=0.000955
  MAD diff:    mean=-0.000079, std=0.000469

KS test (median): stat=0.151, p=2.32e-10  (significant)
KS test (MAD):    stat=0.033, p=0.648     (not significant)
No correlation with PSF (r=-0.025, p=0.433) or depth (r=0.026, p=0.418)
```

**4. Mismatched Annulus Scoring (gen4 scored with native vs mismatched preprocessing):**
```
                     Native(20,32)  Mismatched(32.5,45)   Delta
Recall (p>0.3)         0.740           0.704            -0.036
Recall (p>0.5)         0.698           0.660            -0.038
Recall (p>0.7)         0.642           0.616            -0.026
FPR (p>0.3)            0.002           0.002            +0.000
Median pos score       0.9434          0.9008
Median neg score       0.0000          0.0000
N=500 pos, 500 neg (mixed Tier-A + Tier-B)
```

**5. Beta-Frac Restriction Test (theta_E=1.5", beta_frac [0.1, 0.55], 200 hosts/bin):**
```
Mag bin   p>0.3    p>0.5   median_p  median_SNR
18-19     17.0%     9.0%    0.0804    1556.2
19-20     24.5%    18.0%    0.0893     671.5
20-21     27.5%    17.0%    0.1152     250.1
21-22     35.5%    27.0%    0.1008     101.4
22-23     31.0%    27.5%    0.0430      39.3
23-24     24.0%    18.5%    0.0024      15.6
24-25      8.5%     7.0%    0.0001       6.2
25-26      1.0%     0.5%    0.0000       2.3
```

**6. Embedding Analysis + Linear Probe:**
```
Real Tier-A (n=112):       median score = 0.9946
Inj low beta_frac (n=200): median score = 0.1065
Inj high beta_frac (n=200): median score = 0.0172
Negatives (n=200):          median score = 0.0000

Linear probe (real vs low-bf injections): AUC = 0.9911 +/- 0.0102

Frechet distance: real vs low-bf = 219.67, real vs high-bf = 199.81
Per-layer FD: features_0=0.22, features_1=1.40, features_2=11.06, features_3=63.07
(features_4-7: n=112 < dim, could not compute)
```

### D02: Quick Tests (7 experiments)

**7. Clip-Range Sweep (beta_frac [0.1, 0.55], detection rate at p>0.3):**
```
clip_range   18-19   19-20   20-21   21-22   22-23   23-24   24-25   25-26
10 (baseline) 17.0%  24.5%   27.5%   35.5%   31.0%   24.0%    8.5%    1.0%
20           30.5%   32.0%   37.0%   40.5%   35.0%   14.5%    4.5%    0.0%
50            0.5%    0.0%    0.0%    1.0%    6.0%    4.0%    3.0%    0.0%
100          21.0%   11.0%    3.0%    1.5%    6.5%    3.5%    2.5%    0.5%
```

**8. Poisson Noise Test (clip_range=10, beta_frac [0.1, 0.55], detection rate at p>0.3):**
```
Condition     18-19   19-20   20-21   21-22   22-23   23-24   24-25   25-26
No Poisson    17.0%  24.5%   27.5%   35.5%   31.0%   24.0%    8.5%    1.0%
With Poisson  17.5%  31.0%   45.0%   43.0%   23.5%    5.5%    1.0%    0.0%
```

**9. Poisson + clip_range=50 Combined (p>0.3):**
```
Near-total collapse: 0.5% at mag 18-19, 2.0% at 20-21, 11.0% at 21-22
```

**10. Tier-A-Only Real Lens Scoring (112 val Tier-A lenses):**
```
Threshold     Recall    n_detected/n    95% Wilson CI
p>0.3         89.3%     100/112         [82.6%, 94.0%]
p>0.5         83.9%      94/112         [76.3%, 89.8%]
p>0.806       79.5%      89/112         [71.3%, 86.1%]
p>0.995       48.2%      54/112         [39.1%, 57.4%]

Score distribution: p5=0.014, p25=0.913, median=0.995, p75=0.999, p95=1.000
```

**11. All-Tier Scoring (1432 val positives = 112 Tier-A + 1320 Tier-B):**
```
Tier-A: p>0.3 = 89.3% (100/112) CI [82.6%, 94.0%]
Tier-B: p>0.3 = 72.0% (950/1320) CI [69.5%, 74.3%]
Combined: p>0.3 = 73.3%
```

**12. Unrestricted Beta-Frac Baseline [0.1, 1.0] (matched hosts, same seed, p>0.3):**
```
Mag bin   Unrestricted [0.1,1.0]   Restricted [0.1,0.55]   Delta
18-19           17.0%                    17.0%               0.0pp
19-20           21.5%                    24.5%              +3.0pp
20-21           28.0%                    27.5%              -0.5pp
21-22           20.0%                    35.5%             +15.5pp
22-23           17.5%                    31.0%             +13.5pp
23-24            7.0%                    24.0%             +17.0pp
24-25            4.5%                     8.5%              +4.0pp
25-26            0.0%                     1.0%              +1.0pp
```

**13. HEALPix Investigation:**
```
All 4,788 positives have valid ra/dec (NaN count = 0 for both).
healpix_128 is NaN for all positives — a manifest-generation bug; the column was never computed.

Recomputed HEALPix spatial overlap:
  Tier-A: train=277 in 274 pixels, val=112 in 112 pixels — ZERO overlap
  Tier-B: train=3079 in 2940 pixels, val=1320 in 1307 pixels — 118 overlapping pixels
  Negatives: 176,391 pixels, 0 with mixed splits (fully spatial)
```

---

## Questions

Answer each question with your reasoning. Be specific. Cite numbers.

### 1. The Retrain Question

We have three retrain configs ready (gen5a, gen5b, gen5c). Based on ALL the evidence above:

**1a.** Should we retrain at all? If yes, which config(s) and why? If no, why not? Walk through your reasoning step by step.

**1b.** The annulus normalization medians differ significantly (KS p=2.3e-10) but the MADs do not (p=0.648). The mismatched scoring shows a 3.6pp recall drop at N=500. Is this evidence that the annulus bug hurts performance, or is it just expected sensitivity to preprocessing mismatch? Explain your reasoning.

**1c.** If we retrain gen5c (weighted loss), what specific improvement do you predict and why? Be quantitative if possible.

### 2. The Injection Pipeline Question

**2a.** Adding Poisson noise raised detection at mag 20-21 from 27.5% to 45.0%. What does this tell us about what the CNN learned? Why does Poisson noise help detection of *injected* arcs when the model was never trained with Poisson noise on injections?

**2b.** The clip-range sweep shows a non-monotonic pattern: 20 helps, 50 collapses, 100 partially recovers. Explain this pattern. What does it mean for the model?

**2c.** The linear probe AUC of 0.991 means the CNN almost perfectly separates real lenses from injections. Given this, is the injection-based selection function (completeness ~3.5%) a meaningful number? What does it actually measure?

### 3. The Tier-A Recall Question

**3a.** Tier-A recall is 89.3% with zero spatial leakage. Tier-B recall is 72.0% with some spatial leakage (118 shared HEALPix pixels). Which is the correct headline number for the paper? What caveats are needed?

**3b.** 12 out of 112 Tier-A lenses are missed (score < 0.3). Without seeing the images, what properties would you expect these missed lenses to have?

### 4. What Should We Do Before Submitting the Paper?

**4a.** Give us a concrete ordered list of what we MUST do before submitting to MNRAS. For each item state: what to do, how long it takes, and what it gets us. Do NOT include anything that will not materially improve the paper.

**4b.** What claims can this paper make? What claims should it explicitly NOT make?

**4c.** A hostile referee will read this paper. What are the top 3 criticisms they will raise? For each, either tell us how to preempt it in the paper, or tell us what experiment to run to address it.

### 5. The Big Picture

**5a.** Looking at the full body of evidence — the model gets 89.3% recall on real confirmed lenses, the injection completeness is 3.5%, and the linear probe AUC is 0.991. What is the most scientifically interesting story this data tells? What should the paper be *about*?

**5b.** If you had one week of GPU time and one week of engineering time, what would you do to make this the strongest possible paper?

---

## What We Need From You

Give us a clear decision with clear reasoning. Do not hedge. If the answer is "do not retrain," say so and explain why. If the answer is "retrain gen5c and add Poisson noise before publishing," say so and explain why. We will follow your recommendation if the reasoning is solid.
