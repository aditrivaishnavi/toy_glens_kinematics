# Prompt 5: D01 Pre-Retrain Diagnostic Results — Independent Assessment

## Instructions

You are being asked to independently evaluate the results of 6 diagnostic experiments we ran before deciding whether to retrain a CNN strong-lens finder. We need you to be **100% honest, objective, and scientifically rigorous**. Do not tell us what we want to hear. If the results are ambiguous, say so. If a diagnostic is flawed or insufficient, say so. If you think we are about to waste time, say so.

We are preparing an MNRAS paper. Assume a hostile but reasonable referee will scrutinize every claim. We need actionable conclusions, not vague recommendations.

**Attached zip** (`d01_diagnostic_code_and_results.zip`) contains:
- All 6 diagnostic scripts (Python source code)
- The runner script (`scripts/run_diagnostics.sh`)
- Core library modules they depend on (`dhs/preprocess.py`, `dhs/scoring_utils.py`, `dhs/injection_engine.py`, `dhs/model.py`, etc.)
- Training configs for the current model (gen4) and proposed retrains (gen5a/b/c)
- Injection prior registry (`configs/injection_priors.yaml`)
- All raw result JSON files and log files
- Experiment registry (`EXPERIMENT_REGISTRY.md`)

Please read the code carefully. Do not trust our interpretation — form your own.

---

## Context

### The Model

- Architecture: EfficientNetV2-S (pretrained ImageNet, finetuned)
- Training data: 451,681 cutouts (101x101 pixels, 3-band g/r/z, DESI Legacy Survey DR10)
  - Train: 316,100 (277 Tier-A confirmed lenses, 3,079 Tier-B visual candidates, 312,744 negatives)
  - Val: 135,581 (112 Tier-A, 1,320 Tier-B, 134,149 negatives)
- Current best checkpoint: gen4 (finetuned from gen2, AUC=0.9921)
- Preprocessing: `raw_robust` — per-band normalization using median and MAD computed from an outer annulus of the stamp
- Known issue: The annulus radii used during training were (20, 32) pixels, but the geometrically correct values for 101x101 stamps are (32.5, 45.0). This is a bug discovered during code review.

### The Injection Pipeline

- Synthetic lensed arcs are injected into real host galaxy cutouts using SIS+shear ray-tracing with Sersic source profiles
- Injection parameters: see `configs/injection_priors.yaml` for exact ranges
- The selection function (detection completeness) is computed by injecting arcs into real cutouts and scoring with the frozen model
- Current overall completeness: ~3.5% (marginal over all parameter space)
- Real Tier-A lens recall at p>0.3: 73.3% (at p>0.5: ~70%)

### The Question

Should we retrain the model with the corrected annulus (32.5, 45.0) before publishing the MNRAS paper? We have three candidate configs ready:
- gen5a: From-scratch training with corrected annulus, same hyperparameters as gen1
- gen5b: Finetune from gen4 with corrected annulus
- gen5c: From-scratch with corrected annulus + weighted loss (Tier-A=1.0, Tier-B=0.5)

---

## Diagnostics Executed

All diagnostics were run on Lambda3 (NVIDIA GH200 480GB, Ubuntu 22.04, Python 3.12.3, torch 2.7.0+cu128). Total runtime: 170 seconds.

### Exact Commands

```bash
# Runner script invocation:
cd /lambda/nfs/darkhaloscope-training-dc/stronglens_calibration
source .venv-lambda3/bin/activate
export PYTHONPATH=.
bash scripts/run_diagnostics.sh

# Which runs these 6 commands in sequence:

# [1/6] Split Balance (CPU)
python scripts/split_balance_diagnostic.py \
    --manifest manifests/training_parity_70_30_v1.parquet \
    --out-dir results/D01_20260214_pre_retrain_diagnostics/split_balance_check

# [2/6] Masked Pixels (CPU)
python scripts/masked_pixel_diagnostic.py \
    --manifest manifests/training_parity_70_30_v1.parquet \
    --out-dir results/D01_20260214_pre_retrain_diagnostics/masked_pixel_check \
    --n-samples 1000 --threshold 0.05

# [3/6] Annulus Comparison (CPU)
python scripts/annulus_comparison.py \
    --manifest manifests/training_parity_70_30_v1.parquet \
    --out-dir results/D01_20260214_pre_retrain_diagnostics/q23_annulus_comparison \
    --n-samples 1000

# [4/6] Mismatched Annulus Scoring (GPU)
python scripts/mismatched_annulus_scoring.py \
    --checkpoint checkpoints/paperIV_efficientnet_v2_s_v4_finetune/best.pt \
    --manifest manifests/training_parity_70_30_v1.parquet \
    --out-dir results/D01_20260214_pre_retrain_diagnostics/q24_mismatched_scoring \
    --n-samples 500

# [5/6] Beta-Frac Restriction Test (GPU)
python sim_to_real_validations/bright_arc_injection_test.py \
    --checkpoint checkpoints/paperIV_efficientnet_v2_s_v4_finetune/best.pt \
    --manifest manifests/training_parity_70_30_v1.parquet \
    --out-dir results/D01_20260214_pre_retrain_diagnostics/q21_beta_frac \
    --beta-frac-range 0.1 0.55

# [6/6] Embedding Analysis + Linear Probe (GPU)
python scripts/feature_space_analysis.py \
    --checkpoint checkpoints/paperIV_efficientnet_v2_s_v4_finetune/best.pt \
    --manifest manifests/training_parity_70_30_v1.parquet \
    --out-dir results/D01_20260214_pre_retrain_diagnostics/q22_embedding_umap \
    --n-samples 200
```

---

## Raw Results

### Diagnostic 1: Split Balance

```
split_balance_results.json:
{
  "n_total": 451681,
  "split_counts": {
    "train": {"n_total": 316100, "n_pos": 3356, "n_neg": 312744, "n_tier_a": 277, "n_tier_b": 3079},
    "val": {"n_total": 135581, "n_pos": 1432, "n_neg": 134149, "n_tier_a": 112, "n_tier_b": 1320}
  },
  "positive_spatial_distribution": {
    "healpix_col": "healpix_128",
    "n_unique_pixels": 0,
    "n_nan_healpix": 4788,
    "note": "All positive healpix values are NaN"
  },
  "psf_depth_balance": {
    "psfsize_r": {"train_vs_val": {"ks_stat": 0.00360, "p_value": 0.1738, "mean_1": 1.3482, "mean_2": 1.3491}},
    "psfdepth_r": {"train_vs_val": {"ks_stat": 0.00385, "p_value": 0.1233, "mean_1": 2284.86, "mean_2": 2271.14}}
  }
}
```

Console output:
```
train: 316100 total, 3356 pos (277 A, 3079 B), 312744 neg
val: 135581 total, 1432 pos (112 A, 1320 B), 134149 neg
WARNING: 4788 positives have NaN healpix (missing ra/dec)
psfsize_r train vs val: KS=0.0036, p=1.7384e-01
psfdepth_r train vs val: KS=0.0039, p=1.2332e-01
```

### Diagnostic 2: Masked Pixel Check

```
masked_pixel_results.json:
{
  "n_sampled": 1000,
  "n_loaded": 1000,
  "n_errors": 0,
  "threshold": 0.05,
  "nonfinite_pixels": {"mean_frac": 0.0, "median_frac": 0.0, "max_frac": 0.0, "pct_above_threshold": 0.0, "n_above_threshold": 0},
  "zero_pixels": {"mean_frac": 5.20e-06, "median_frac": 0.0, "max_frac": 0.005196}
}
```

### Diagnostic 3: Annulus Comparison — (20,32) vs (32.5,45)

```
annulus_comparison_results.json:
{
  "per_config_summary": {
    "old_20_32": {"median_of_medians": 0.000467, "median_of_mads": 0.002276, "n_valid": 1000},
    "new_32p5_45": {"median_of_medians": 0.000340, "median_of_mads": 0.002197, "n_valid": 1000}
  },
  "paired_comparison": {
    "median_diff": {"mean": -0.000345, "std": 0.000955},
    "mad_diff": {"mean": -0.000079, "std": 0.000469},
    "ks_test_median": {"statistic": 0.151, "pvalue": 2.32e-10},
    "ks_test_mad": {"statistic": 0.033, "pvalue": 0.648}
  },
  "correlation_with_psf": {"median_diff_vs_psf": {"r": -0.025, "p": 0.433}},
  "correlation_with_depth": {"median_diff_vs_depth": {"r": 0.026, "p": 0.418}},
  "shift_by_label": {
    "positives": {"n": 12, "mean_median_diff": -0.000254},
    "negatives": {"n": 988, "mean_median_diff": -0.000346}
  }
}
```

Console output:
```
old_20_32:   Median of medians: 0.000467, Median of MADs: 0.002276
new_32p5_45: Median of medians: 0.000340, Median of MADs: 0.002197

Paired differences (new - old), N=1000:
  Median diff:  mean=-0.000345, std=0.000955
  MAD diff:     mean=-0.000079, std=0.000469

KS test (median): stat=0.1510, p=2.3212e-10
KS test (MAD):    stat=0.0330, p=6.4781e-01

Correlation with PSF:  median_diff: r=-0.0250, p=4.3296e-01
Shift by label:
  Positives (N=12): median_diff=-0.000254
  Negatives (N=988): median_diff=-0.000346
```

### Diagnostic 4: Mismatched Annulus Scoring

The gen4 model was trained with annulus (20,32). We score the same val set twice: once with native (20,32) preprocessing and once with mismatched (32.5,45) preprocessing.

```
mismatched_annulus_results.json:
{
  "native_annulus": [20, 32],
  "mismatched_annulus": [32.5, 45.0],
  "native_metrics": {
    "n_pos": 500, "n_neg": 500,
    "recall_p0.3": 0.740, "recall_p0.5": 0.698, "recall_p0.7": 0.642,
    "fpr_p0.3": 0.002, "fpr_p0.5": 0.002, "fpr_p0.7": 0.002,
    "median_pos_score": 0.9434, "median_neg_score": 1.49e-05
  },
  "mismatched_metrics": {
    "n_pos": 500, "n_neg": 500,
    "recall_p0.3": 0.704, "recall_p0.5": 0.660, "recall_p0.7": 0.616,
    "fpr_p0.3": 0.002, "fpr_p0.5": 0.002, "fpr_p0.7": 0.002,
    "median_pos_score": 0.9008, "median_neg_score": 1.30e-05
  },
  "deltas": {
    "delta_recall_p0.3": -0.036,
    "delta_recall_p0.5": -0.038,
    "delta_recall_p0.7": -0.026
  }
}
```

Console output:
```
Metric                     Native   Mismatched      Delta
------------------------------------------------------------
Recall (p>0.3)              0.740        0.704     -0.036
Recall (p>0.5)              0.698        0.660     -0.038
Recall (p>0.7)              0.642        0.616     -0.026
FPR (p>0.3)                0.0020       0.0020    +0.0000
FPR (p>0.5)                0.0020       0.0020    +0.0000
FPR (p>0.7)                0.0020       0.0020    +0.0000
Median pos score            0.9434       0.9008
Median neg score            0.0000       0.0000
```

### Diagnostic 5: Beta-Frac Restriction Test

Injections with theta_E=1.5", beta_frac restricted to [0.1, 0.55] (area-weighted), source magnitudes 18-26, scored with gen4 model. 200 hosts per magnitude bin.

```
bright_arc_results_bf0.10_0.55.json:
{
  "theta_e": 1.5,
  "beta_frac_range": [0.1, 0.55],
  "add_poisson_noise": false,
  "results_by_bin": {
    "18-19": {"n_scored": 200, "detection_rate_p03": 0.170, "detection_rate_p05": 0.090, "median_score": 0.0804, "median_arc_snr": 1556.2},
    "19-20": {"n_scored": 200, "detection_rate_p03": 0.245, "detection_rate_p05": 0.180, "median_score": 0.0893, "median_arc_snr": 671.5},
    "20-21": {"n_scored": 200, "detection_rate_p03": 0.275, "detection_rate_p05": 0.170, "median_score": 0.1152, "median_arc_snr": 250.1},
    "21-22": {"n_scored": 200, "detection_rate_p03": 0.355, "detection_rate_p05": 0.270, "median_score": 0.1008, "median_arc_snr": 101.4},
    "22-23": {"n_scored": 200, "detection_rate_p03": 0.310, "detection_rate_p05": 0.275, "median_score": 0.0430, "median_arc_snr": 39.3},
    "23-24": {"n_scored": 200, "detection_rate_p03": 0.240, "detection_rate_p05": 0.185, "median_score": 0.0024, "median_arc_snr": 15.6},
    "24-25": {"n_scored": 200, "detection_rate_p03": 0.085, "detection_rate_p05": 0.070, "median_score": 0.0001, "median_arc_snr": 6.2},
    "25-26": {"n_scored": 200, "detection_rate_p03": 0.010, "detection_rate_p05": 0.005, "median_score": 0.0000, "median_arc_snr": 2.3}
  }
}
```

Console output:
```
Mag bin      N scored      p>0.3      p>0.5     median_p   median_SNR
---------------------------------------------------------------------
18-19             200      17.0%       9.0%       0.0804       1556.2
19-20             200      24.5%      18.0%       0.0893        671.5
20-21             200      27.5%      17.0%       0.1152        250.1
21-22             200      35.5%      27.0%       0.1008        101.4
22-23             200      31.0%      27.5%       0.0430         39.3
23-24             200      24.0%      18.5%       0.0024         15.6
24-25             200       8.5%       7.0%       0.0001          6.2
25-26             200       1.0%       0.5%       0.0000          2.3
```

For reference: real Tier-A lens recall is ~73% at p>0.3. Standard (unrestricted beta_frac) bright-arc detection ceiling was ~30%.

### Diagnostic 6: Embedding Feature Space Analysis + Linear Probe

Extracts penultimate-layer (1280-dim) embeddings for 4 groups, then trains a logistic regression (5-fold CV) to distinguish real Tier-A lenses from low-beta_frac injections.

```
feature_space_results.json:
{
  "n_real_tier_a": 112,
  "n_inj_low_bf": 200,
  "n_inj_high_bf": 200,
  "n_negatives": 200,
  "target_mag": 19.0,
  "theta_e": 1.5,
  "linear_probe": {
    "task": "real_tier_a vs inj_low_bf",
    "cv_auc_mean": 0.9911,
    "cv_auc_std": 0.0102
  },
  "frechet_distance_per_layer": {
    "features_0": 0.218,
    "features_1": 1.397,
    "features_2": 11.059,
    "features_3": 63.072,
    "features_4": NaN,
    "features_5": NaN,
    "features_6": NaN,
    "features_7": NaN
  },
  "frechet_distance": {
    "real_vs_low_bf": 219.67,
    "real_vs_high_bf": 199.81
  },
  "median_scores": {
    "real_tier_a": 0.9946,
    "inj_low_bf": 0.1065,
    "inj_high_bf": 0.0172,
    "negatives": 0.0000
  }
}
```

Console output:
```
(a) Real Tier-A: 112 embeddings, median score=0.9946
(d) Val negatives: 200 embeddings, median score=0.0000
(b) Low beta_frac [0.1,0.3] injections: 200 embeddings, median score=0.1065
(c) High beta_frac [0.7,1.0] injections: 200 embeddings, median score=0.0172

Linear probe (real Tier-A vs low-bf injections): 5-fold CV AUC = 0.9911 +/- 0.0102

Frechet distance:
  FD(real vs low-bf injection): 219.67
  FD(real vs high-bf injection): 199.81

Per-layer FD:
  features_0 (dim=24): 0.22
  features_1 (dim=24): 1.40
  features_2 (dim=48): 11.06
  features_3 (dim=64): 63.07
  features_4-7: too few samples for covariance (n=112 < dim)
```

---

## Questions for You

Please answer each question independently. Cite specific numbers from the raw results above.

### A. Diagnostic Validity

A1. Review the code in the attached zip. Are there any bugs, methodological flaws, or statistical errors in how these 6 diagnostics were implemented? Be specific — cite file names and line numbers if you find issues.

A2. Are the sample sizes sufficient for the conclusions we need to draw? Specifically: 500 pos + 500 neg for the mismatched scoring, 200 hosts per magnitude bin for the beta_frac test, and 112 real Tier-A + 200 injections for the linear probe. What are the confidence intervals on the key numbers?

A3. The split balance diagnostic found that ALL 4,788 positive samples have NaN in the healpix_128 column (missing ra/dec). What does this mean? Is this a data quality concern that affects any of the other diagnostics or the training itself?

### B. Annulus Question (GO/NO-GO for Retrain)

B1. The annulus comparison shows a statistically significant difference in normalization medians (KS p=2.3e-10) but NOT in MADs (KS p=0.648). What does this mean physically? Is a shift in the normalization reference point without a change in the noise scale actually meaningful for CNN performance?

B2. The mismatched scoring shows recall drops of 3.6pp (p>0.3) and 3.8pp (p>0.5) when the gen4 model is scored with (32.5,45) preprocessing. The FPR does not change. Is this drop large enough to justify retraining? What is the expected statistical uncertainty on these recall numbers given N=500?

B3. CRITICAL: The mismatched scoring test gives the gen4 model inputs preprocessed with the WRONG annulus — one it was never trained on. A recall drop is expected by default for any preprocessing change. Does this test actually tell us whether retraining with the correct annulus would IMPROVE performance? Or does it only tell us the model is sensitive to its own training preprocessing (which is trivially true for any model)?

B4. Based on B1-B3, should we retrain with the corrected annulus? Is there sufficient evidence that the annulus bug is causing performance degradation, or could the annulus be cosmetic (the model adapted to it during training)?

### C. Beta_frac and Detection Ceiling

C1. With beta_frac restricted to [0.1, 0.55], the peak detection rate is 35.5% (mag 21-22). The unrestricted rate was ~30%. Is this 5.5pp improvement statistically significant given N=200 per bin?

C2. The brightest arcs (mag 18-19, SNR>1500) have the LOWEST detection rate (17%). Why? What does this tell us about what the CNN has learned? Is this an artifact of the injection pipeline, a real property of the model, or a statistical fluctuation?

C3. Detection peaks at mag 21-22 (SNR~100), not at the brightest magnitudes. What physical interpretation, if any, should we attach to this?

### D. Injection Realism Gap

D1. The linear probe achieves AUC=0.991 separating real Tier-A lenses from low-beta_frac injections. Is this a fair test? The real lenses have diverse morphologies and were observed; the injections are parametric Sersic profiles injected into random host galaxies. Would you expect AUC~1.0 even for a perfect injection pipeline, simply because the real sample is heterogeneous?

D2. Real Tier-A lenses have median score 0.995 while low-bf injections have median score 0.107. This is a 10x gap. What fraction of this gap is attributable to (a) injection unrealism vs (b) the model being correct — i.e., many injections simply don't look like real lenses because the Sersic model is too simple?

D3. The Frechet distance grows from 0.22 at features_0 to 63.07 at features_3. What does this layer-wise progression tell us about where the CNN learns to distinguish real from injected? Is this diagnostic informative or misleading given that features_4-7 could not be computed?

D4. Given the linear probe AUC of 0.991, is injection-based completeness a meaningful metric at all? Or is it fundamentally measuring "how many Sersic profiles fool the CNN" rather than "what fraction of real lenses would the CNN find"?

### E. The Retrain Decision

E1. We have three retrain options: gen5a (from-scratch, corrected annulus), gen5b (finetune gen4, corrected annulus), gen5c (from-scratch, corrected annulus + weighted loss). Given ALL the diagnostics above, which if any should we run, and in what order? What specific improvements do you predict for each?

E2. If we retrain and see AUC improve from 0.9921 to, say, 0.9935 — is this publishable as a meaningful improvement? The AUC standard error is ~0.0026 for N~14,000 val samples. The v2-to-v4 improvement was only +0.0006.

E3. What is the minimum set of experiments needed for a publishable MNRAS paper, regardless of whether we retrain? What can we publish with just the gen4 model and these diagnostic results?

### F. Hostile Reviewer Perspective

F1. A referee writes: "The authors report injection-based completeness of ~3.5% but real-lens recall of ~73%. The linear probe AUC of 0.991 confirms the CNN trivially distinguishes Sersic injections from real lenses. How is the injection-based completeness number meaningful or useful?" How should we respond?

F2. A referee writes: "The annulus normalization bug (20,32) vs (32.5,45) is concerning. Why was this not caught before training? How do we know there are no other undiscovered preprocessing bugs?" How should we respond?

F3. A referee writes: "With only 112 Tier-A validation lenses, the binomial 95% CI on 73% recall is [64%, 81%]. This is too wide to claim any precision. The authors cannot distinguish 65% from 80% recall." Is this criticism valid? How should we address it in the paper?

### G. Next Steps

G1. Provide a prioritized list of concrete action items based on these results. For each item, state: what to do, estimated time, expected outcome, and what decision it enables. Do not include items that are unlikely to change the conclusions.

G2. If you had to choose between (a) retraining with the annulus fix, or (b) spending the same compute time on improving injection realism (e.g., using real galaxy stamps instead of Sersic), which would you recommend and why? **If it is both, say so. I want the maximum chance to be accepted to MNRAS or equally prominent journal**

G3. What is the single most important thing we should do before submitting the paper?

---

## Final Request

Please be brutally honest. We have limited time and compute. We cannot afford to chase improvements that won't materialize. Tell us what is real, what is noise, and what a reasonable referee would actually care about.
