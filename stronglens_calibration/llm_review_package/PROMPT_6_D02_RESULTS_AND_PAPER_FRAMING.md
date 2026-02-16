# Prompt 6: D02 Quick-Test Results and Paper Framing

## Instructions

This is a follow-up to Prompt 5. We executed the action items both reviewers recommended. The results are surprising in several ways and we need your independent assessment before finalizing the paper.

Be **100% honest, objective, and scientifically rigorous**. We are preparing an MNRAS paper. Assume a hostile but reasonable referee. Do not tell us what we want to hear.

**Attached zip** (`d02_quick_tests_and_results.zip`) contains all code, all raw JSON results, log files, and the UMAP figures.

---

## Background (from Prompt 5)

- Architecture: EfficientNetV2-S (gen4 checkpoint, AUC=0.9921)
- Training: 451,681 cutouts (101x101, g/r/z, DESI DR10). Train: 316,100 (277 Tier-A, 3,079 Tier-B, 312,744 neg). Val: 135,581 (112 Tier-A, 1,320 Tier-B, 134,149 neg).
- Preprocessing: `raw_robust` normalization with annulus (20,32), clip_range=10.0
- Injection: SIS+shear ray-tracing with Sersic source profiles
- From Prompt 5 D01 diagnostics: linear probe AUC=0.991 (real vs injection), annulus recall drop 3.6pp (not significant), bright arcs (mag 18-19) detected at only 17%

Both Prompt 5 reviewers agreed: (a) annulus retrain is NOT justified, (b) injection realism is the binding constraint, (c) the paper should reframe around the sim-to-real gap. They recommended specific follow-up experiments. We ran ALL of them.

---

## D02 Experiments and Raw Results

All run on Lambda3 (NVIDIA GH200 480GB, Python 3.12.3, torch 2.7.0+cu128). Total runtime: ~9 minutes.

### Exact Commands

```bash
cd /lambda/nfs/darkhaloscope-training-dc/stronglens_calibration
source .venv-lambda3/bin/activate
export PYTHONPATH=.
bash scripts/run_d02_quick_tests.sh
```

The runner executes 7 experiments in sequence (see `scripts/run_d02_quick_tests.sh` in the zip for exact arguments).

---

### Experiment 1: Clip-Range Sweep

Same setup as D01 beta_frac test: theta_E=1.5", beta_frac [0.1, 0.55], 200 hosts, 8 magnitude bins. Only the preprocessing clip_range varies. The gen4 model was **trained** with clip_range=10.

#### Raw Results (detection rate at p>0.3):

| clip_range | 18-19 | 19-20 | 20-21 | 21-22 | 22-23 | 23-24 | 24-25 | 25-26 |
|------------|-------|-------|-------|-------|-------|-------|-------|-------|
| **10** (D01 baseline) | 17.0% | 24.5% | 27.5% | 35.5% | 31.0% | 24.0% | 8.5% | 1.0% |
| **20** | 30.5% | 32.0% | 37.0% | 40.5% | 35.0% | 14.5% | 4.5% | 0.0% |
| **50** | 0.5% | 0.0% | 0.0% | 1.0% | 6.0% | 4.0% | 3.0% | 0.0% |
| **100** | 21.0% | 11.0% | 3.0% | 1.5% | 6.5% | 3.5% | 2.5% | 0.5% |

#### Raw Results (detection rate at p>0.5):

| clip_range | 18-19 | 19-20 | 20-21 | 21-22 | 22-23 | 23-24 | 24-25 | 25-26 |
|------------|-------|-------|-------|-------|-------|-------|-------|-------|
| **10** (D01 baseline) | 9.0% | 18.0% | 17.0% | 27.0% | 27.5% | 18.5% | 7.0% | 0.5% |
| **20** | 12.0% | 14.0% | 23.5% | 30.0% | 29.5% | 14.0% | 3.5% | 0.0% |
| **50** | 0.0% | 0.0% | 0.0% | 0.5% | 5.0% | 3.5% | 2.0% | 0.0% |
| **100** | 7.0% | 3.5% | 1.0% | 1.0% | 5.5% | 3.5% | 1.5% | 0.5% |

#### Raw Results (median CNN score):

| clip_range | 18-19 | 19-20 | 20-21 | 21-22 | 22-23 |
|------------|-------|-------|-------|-------|-------|
| **10** | 0.0804 | 0.0893 | 0.1152 | 0.1008 | 0.0430 |
| **20** | 0.1713 | 0.1695 | 0.1509 | 0.1396 | 0.0839 |
| **50** | 0.0089 | 0.0022 | 0.0010 | 0.0007 | 0.0008 |
| **100** | 0.1209 | 0.0476 | 0.0074 | 0.0016 | 0.0013 |

---

### Experiment 2: Poisson Noise Test

Same hosts as D01, beta_frac [0.1, 0.55], theta_E=1.5", clip_range=10 (default). `--add-poisson-noise` flag enables Poisson noise on injected arc flux (normal approximation, gain=150 e-/nmgy).

#### Raw Results (detection rate at p>0.3):

| Condition | 18-19 | 19-20 | 20-21 | 21-22 | 22-23 | 23-24 | 24-25 | 25-26 |
|-----------|-------|-------|-------|-------|-------|-------|-------|-------|
| No Poisson (D01) | 17.0% | 24.5% | 27.5% | 35.5% | 31.0% | 24.0% | 8.5% | 1.0% |
| **With Poisson** | 17.5% | 31.0% | **45.0%** | **43.0%** | 23.5% | 5.5% | 1.0% | 0.0% |

#### Raw Results (detection rate at p>0.5):

| Condition | 18-19 | 19-20 | 20-21 | 21-22 | 22-23 | 23-24 | 24-25 | 25-26 |
|-----------|-------|-------|-------|-------|-------|-------|-------|-------|
| No Poisson (D01) | 9.0% | 18.0% | 17.0% | 27.0% | 27.5% | 18.5% | 7.0% | 0.5% |
| **With Poisson** | 11.5% | 20.5% | **38.0%** | **37.5%** | 21.5% | 4.0% | 0.5% | 0.0% |

#### Raw Results (median CNN score):

| Condition | 18-19 | 19-20 | 20-21 | 21-22 | 22-23 |
|-----------|-------|-------|-------|-------|-------|
| No Poisson | 0.0804 | 0.0893 | 0.1152 | 0.1008 | 0.0430 |
| **With Poisson** | 0.0469 | 0.0767 | **0.2048** | **0.1388** | 0.0022 |

---

### Experiment 3: Poisson + clip_range=50 Combined

Testing whether Poisson noise helps when clip_range is also changed. Result: detection collapses (same as clip_range=50 alone).

| Condition | 18-19 | 20-21 | 21-22 |
|-----------|-------|-------|-------|
| Poisson + clip50 (p>0.3) | 0.5% | 2.0% | 11.0% |

---

### Experiment 4: Tier-A-Only Real Lens Scoring

Scored ALL 112 val Tier-A (confirmed) lenses and 3,000 random negatives with gen4.

```
TIER-A RECALL:
  p>0.3:   89.3% (100/112)  95% Wilson CI [82.6%, 94.0%]
  p>0.5:   83.9% (94/112)   95% Wilson CI [76.3%, 89.8%]
  p>0.806: 79.5% (89/112)   95% Wilson CI [71.3%, 86.1%]
  p>0.995: 48.2% (54/112)   95% Wilson CI [39.1%, 57.4%]

Score distribution:
  p5=0.014  p25=0.913  median=0.995  p75=0.999  p95=1.000
  mean=0.837  frac>0.9=76.8%  frac<0.1=8.9%
```

### Experiment 5: All-Tier Scoring (for comparison)

```
ALL-TIER RECALL (1432 positives = 112 Tier-A + 1320 Tier-B):
  p>0.3: 73.3%
  p>0.5: 68.7%

PER-TIER BREAKDOWN:
  Tier-A: p>0.3 = 89.3% (100/112) CI [82.6%, 94.0%]
  Tier-B: p>0.3 = 72.0% (950/1320) CI [69.5%, 74.3%]
```

---

### Experiment 6: Unrestricted Beta-Frac Baseline [0.1, 1.0]

Same 200 hosts (same seed=42) as D01. Proper matched comparison.

#### Raw Results (detection rate at p>0.3):

| beta_frac range | 18-19 | 19-20 | 20-21 | 21-22 | 22-23 | 23-24 | 24-25 | 25-26 |
|-----------------|-------|-------|-------|-------|-------|-------|-------|-------|
| [0.1, 1.0] (unrestricted) | 17.0% | 21.5% | 28.0% | 20.0% | 17.5% | 7.0% | 4.5% | 0.0% |
| [0.1, 0.55] (D01 restricted) | 17.0% | 24.5% | 27.5% | 35.5% | 31.0% | 24.0% | 8.5% | 1.0% |

---

### Experiment 7: HEALPix NaN Investigation

```json
{
  "n_positives": 4788,
  "positive_ra_nan": 0,
  "positive_dec_nan": 0,
  "positive_healpix_nan": 4788,
  "negative_healpix_nan": 0,
  "positives_with_valid_radec": 4788,
  "spatial_overlap": {
    "train_unique_pixels": 3192,
    "val_unique_pixels": 1417,
    "overlapping_pixels": 130,
    "train_in_overlap": 141,
    "val_in_overlap": 133
  },
  "tier_A_overlap": {
    "train_count": 277, "val_count": 112,
    "train_pixels": 274, "val_pixels": 112,
    "overlapping_pixels": 0
  },
  "tier_B_overlap": {
    "train_count": 3079, "val_count": 1320,
    "train_pixels": 2940, "val_pixels": 1307,
    "overlapping_pixels": 118
  },
  "negative_mixed_healpix_pixels": 0,
  "negative_split_mechanism": "spatial"
}
```

Key findings:
- All positives have VALID ra/dec. The healpix_128 NaN is a manifest-generation bug (the crossmatch script used did not compute it).
- **Tier-A: ZERO spatial overlap** between train and val (0 shared HEALPix pixels)
- Tier-B: 118 shared pixels (some spatial leakage possible)
- Negatives: fully spatially split (0 mixed pixels across 176,391 total)

---

### Experiment 8: UMAP Visualization

Generated from D01 embeddings (112 real Tier-A + 200 low-bf injections + 200 high-bf injections + 200 negatives). UMAP with n_neighbors=15, min_dist=0.1. See `umap_feature_space.png` and `umap_score_colored.png` in the zip.

---

## Questions for You

### A. Clip-Range Behavior

A1. The clip-range sweep shows a non-monotonic pattern: clip_range=20 helps (17%→30.5% at mag 18-19), clip_range=50 collapses everything to ~0%, and clip_range=100 partially recovers (21% at mag 18-19 only). Why does clip_range=100 partially recover at the brightest bin but not at moderate bins? Is there a coherent physical explanation for this non-monotonic pattern, or is this evidence of a deeper issue?

A2. At clip_range=20, the detection pattern STILL shows non-monotonicity: 30.5% at mag 18-19 but 40.5% at mag 21-22. Clipping partially explains the bright-arc penalty (from 17% to 30.5%), but it does NOT fully resolve it. What else is contributing to the residual gap between mag 18-19 and 21-22?

A3. Both clip_range=50 and clip_range=100 show near-total collapse of detection. This model was trained with clip_range=10. Is this collapse purely due to domain shift (model never saw these input distributions), or does it reveal something about what features the CNN relies on? What experiment could distinguish these?

### B. Poisson Noise Impact

B1. Adding Poisson noise raises mag 20-21 detection from 27.5% to 45.0% — a +17.5pp improvement — with clip_range=10 unchanged. This is the largest improvement from any single intervention. What does this tell us about the CNN's decision boundary? Why does Poisson noise HELP detection of injected arcs?

B2. Poisson noise does NOT help at mag 18-19 (17.0% → 17.5%, within noise). At this brightness level, clipping still dominates. But at mag 22-23, Poisson noise actually HURTS (31.0% → 23.5%). Why would adding noise reduce detection at fainter magnitudes?

B3. If we could retrain the model with Poisson noise enabled in the injection pipeline during training, would this improve or degrade real-lens recall? What is the mechanism — would the model learn to recognize noise-textured arcs as positive, potentially generalizing better to real arcs?

### C. Tier-A Recall — The Headline Number

C1. Tier-A recall is 89.3% (100/112) at p>0.3, with 95% CI [82.6%, 94.0%]. Tier-B recall is 72.0% (950/1320). The previously reported "73% recall" was dominated by Tier-B. Is 89.3% the correct headline number for the paper? What caveats should accompany it?

C2. Tier-A has ZERO spatial leakage (0 overlapping HEALPix pixels between train and val). Tier-B has 118 overlapping pixels. Does this make the Tier-A number more trustworthy than the Tier-B number? Could the Tier-B number be inflated by spatial leakage?

C3. Only 12 of the 112 Tier-A lenses score below p=0.3 (i.e., are "missed"). Without looking at the actual images, what properties would you expect these 12 to have? What analysis of the missed lenses would be most informative for the paper?

### D. The Sim-to-Real Gap — Quantifying It

D1. We now have three independent lines of evidence for the sim-to-real gap:
   - Linear probe AUC = 0.991 (D01)
   - Real Tier-A median score = 0.995 vs injection median = 0.107 (D01)
   - Best injection detection = 45% with Poisson noise (D02) vs Tier-A recall = 89.3%

Is this sufficient to characterize the gap for an MNRAS paper? What additional measurement, if any, is needed?

D2. The Poisson noise result (45% with Poisson vs 27.5% without) quantifies one component of injection unrealism. Can we use this to decompose the sim-to-real gap into: (a) missing Poisson noise, (b) Sersic smoothness vs real morphology, (c) other factors? What fraction would you attribute to each?

D3. LLM2 in Prompt 5 suggested this abstract sentence: "A linear probe in CNN feature space achieves AUC = 0.991 separating parametric Sersic injections from confirmed lenses, establishing that injection-based completeness is a conservative lower bound driven by source model realism rather than classifier performance." Is this scientifically accurate given the D02 results? Would you modify it?

### E. Paper Strategy — Final Decisions

E1. Given ALL results from D01 + D02, should we:
   (a) Publish with gen4 as-is (no retrain), framing the sim-to-real gap as a contribution?
   (b) Retrain gen5c (weighted loss) before publishing?
   (c) Add Poisson noise to the injection pipeline and recompute the selection function before publishing?
   (d) Some combination of the above?

For each option, estimate: time required, expected improvement, and referee risk.

E2. The clip_range=20 result suggests that if we retrain with clip_range=20, bright-arc detection could improve. But we cannot test this at inference time alone (clip_range=50 collapses). Should we add "retrain with clip_range=20" to the gen5 options? Or is this diminishing returns given that the fundamental bottleneck is injection realism?

E3. What are the 3-5 key figures the paper needs? We have:
   - UMAP of feature space (4 categories)
   - Magnitude vs detection rate (multiple conditions)
   - clip_range sweep
   - Tier-A recall with CIs
   What else?

### F. Hostile Reviewer — Updated Scenarios

F1. A referee says: "Adding Poisson noise to injections improves detection from 27.5% to 45% at mag 20-21. This proves the injection pipeline used for the published selection function is fundamentally flawed. The selection function numbers are not meaningful." How do we respond?

F2. A referee says: "The clip_range sweep shows the CNN is extremely sensitive to a preprocessing hyperparameter. At clip_range=50, detection drops to <1%. This suggests the model has memorized the training preprocessing distribution rather than learning genuine lens morphology. How robust is this model?" How do we respond?

F3. A referee says: "Tier-A recall is 89.3% but injection-based completeness is ~3.5%. This factor-of-25 discrepancy is the central result of the paper. The authors attribute it to injection realism but provide no direct evidence beyond the linear probe. A more parsimonious explanation is that the CNN simply learned to recognize the specific visual appearance of known lenses in the training set (overfitting to the positive sample). With only 277 training and 112 validation Tier-A lenses, memorization is a real concern." How do we respond? Is this critique defensible?

### G. Final Action Items

G1. Based on ALL evidence (D01 + D02 + Prompt 5 analysis), provide a definitive list of what we must do before submitting the paper. Be concrete. Estimate time for each item. Tell us what to skip.

G2. What is the minimum viable paper? What claims can we make, and what claims should we explicitly NOT make?

G3. Draft the key contribution bullet points for the abstract/introduction (3-5 bullets).

---

## Final Request

We are close to submission. We need honest, decisive guidance. Tell us what to do, what to skip, and what a referee will actually challenge. No hand-waving, no aspirational recommendations. What is the shortest path to a publishable paper that passes peer review?
