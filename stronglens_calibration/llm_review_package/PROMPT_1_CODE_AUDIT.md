# Prompt 1 of 4: Code Audit + Pipeline Integrity

**Attach:** `stronglens_calibration_for_review_20260213.zip`
**Read first:** `CHANGES_FOR_LLM_REVIEW.md` in the zip root

---

## Context (Compressed — read fully before answering)

We are writing an MNRAS paper measuring the **selection function** of CNN-based
strong lens finders in DESI Legacy Survey DR10. The CNN (EfficientNetV2-S,
AUC=0.9921) was trained on 4,788 real lens candidates + 446,893 negatives.

The selection function is measured via **injection-recovery**: inject synthetic
lensed arcs (SIE + Sersic source) into real galaxy cutouts, preprocess, score
with the frozen CNN, measure completeness.

**Key results so far:**
- Real-lens recall (val positives, p>0.3): 73.3%
- Injection completeness (p>0.3): 3.5%
- Even at blindingly bright arcs (mag 18-19, SNR ~900): only 30% detected
- Model 2 (LRG hosts + conditioned q/PA): 0.77pp WORSE, not better
- 70 percentage point gap between real recall and injection completeness

**What happened on 2026-02-13:** A Cursor LLM made code changes to address
findings from an earlier external review. The project owner does NOT trust
these changes. You are the independent reviewer.

**Your role:** Hostile-but-reasonable peer reviewer. Verify everything
independently. Read the code. Check the math. If something is wrong, say it.
If something is right, explain WHY so we can be confident.

---

## Key Files for This Prompt

| File | What to audit |
|------|---------------|
| `dhs/utils.py` | `normalize_outer_annulus()`, `default_annulus_radii()`, `radial_mask()`, `robust_median_mad()` |
| `dhs/preprocess.py` | `preprocess_stack()` — the annulus kwargs passthrough |
| `dhs/data.py` | `DatasetConfig`, `LensDataset.__getitem__()`, `load_cutout_from_file()` |
| `dhs/train.py` | Training loop, gradient accumulation, augmentation at val time |
| `dhs/transforms.py` | Augmentation functions |
| `dhs/model.py` | `build_model()` factory |
| `dhs/injection_engine.py` | SIE deflection, Sersic rendering, `inject_sis_shear()`, `sample_source_params()`, `sample_lens_params()` |
| `configs/injection_priors.yaml` | Parameter registry |
| `configs/paperIV_efficientnet_v2_s_v5_annulus_fix.yaml` | New retrain config |
| `tests/test_preprocess_regression.py` | Behavioral tests |
| `tests/test_injection_priors.py` | AST-based prior validation |
| `scripts/beta_frac_ceiling_diagnostic.py` | Beta_frac diagnostic |
| `injection_model_2/scripts/selection_function_grid_v2.py` | Scoring path |
| `sim_to_real_validations/real_lens_scoring.py` | Real lens scoring path |
| `sim_to_real_validations/bright_arc_injection_test.py` | Bright arc test path |

---

## SECTION 1: Code Change Verification

### 1.1 The Annulus Revert in `dhs/utils.py`

The Cursor LLM reverted `normalize_outer_annulus` to hardcoded defaults
`(r_in=20, r_out=32)` and added a `default_annulus_radii()` helper for future use.

**Q1.1** The normalization uses MEDIAN and MAD (robust estimators). If only ~20%
of annulus pixels contain galaxy light, the MEDIAN may still be correct (dominated
by ~80% sky pixels). **Is the annulus bug actually impactful for median/MAD
normalization?** Work through the math: for a typical bright elliptical galaxy at
r=20-32 pixels from center in a 101×101 stamp (0.262"/pix), what fraction of
annulus pixels have galaxy flux > 3σ_sky? Is the median shifted? Is the MAD
inflated? Give concrete numbers.

**Q1.2** Test output says sky median (r>40) after preprocessing = -2.6. But this
was measured on a SYNTHETIC exponential galaxy with scale length 8 px (very
extended, R_e ≈ 13 px). **For real DR10 LRGs with R_e ~ 4-12 px, would the sky
median still be -2.6 or much closer to 0?**

**Q1.3** The formula `r_in = 0.65 * R, r_out = 0.90 * R` where `R = min(H,W)//2`:
**Is this principled or arbitrary?** Why 0.65 and 0.90? Should annulus placement
be adaptive (based on measured galaxy half-light radius, or iterative sigma-clipping)?

**Q1.4** The `logger.warning()` fires when `r_out / half < 0.70` and H > 64.
With ~316K training samples × 160 epochs = ~50M calls, **will this flood logs?**
Python's logging module does NOT deduplicate by default.

**Q1.5** The signature was reverted from `r_in: float | None = None` to
`r_in: float = 20`. **Verify no code path still expects the None-accepting
signature.** Search for callers passing `r_in=None` or `r_out=None`.

### 1.2 The preprocess_stack Passthrough in `dhs/preprocess.py`

**Q1.6** If someone passes only `annulus_r_in=32.5` without `annulus_r_out`,
the kwargs dict is `{"r_in": 32.5}` and `normalize_outer_annulus` gets
`r_in=32.5, r_out=32` (default). Inner > outer! **Trace the code path:** what
happens when `r_in > r_out`? Does `radial_mask` return an empty mask? Does
`robust_median_mad` handle an empty array? NaN or crash?

**Q1.7** The `float | None` union type requires Python 3.10+. **Is the project's
Python version compatible?** Check requirements.txt or setup.py.

### 1.3 DatasetConfig Changes in `dhs/data.py`

**Q1.8** Annulus radii use `0.0` as sentinel ("use default"). But `0.0` is a
valid float. The check is `if self.dcfg.annulus_r_in > 0`. **What if the YAML
has `annulus_r_in: 0` explicitly?** Is "sentinel via 0.0" clearly documented?

**Q1.9** `run_experiment.py` does `DatasetConfig(**cfg["dataset"])`. **Verify
that `paperIV_efficientnet_v2_s_v5_annulus_fix.yaml`'s `dataset:` keys match
DatasetConfig field names EXACTLY.** Compare key-by-key. Any typo crashes at startup.

**Q1.10** **Verify old configs (v1-v4) still work** with the new DatasetConfig.
They don't have `annulus_r_in`/`annulus_r_out` keys — do the defaults (0.0) apply
correctly? Does any old config have keys NOT in DatasetConfig?

**Q1.11** Trace the full forwarding chain: YAML `annulus_r_in: 32.5` →
`DatasetConfig(annulus_r_in=32.5)` → `crop_kwargs['annulus_r_in'] = 32.5` →
`preprocess_stack(**crop_kwargs)` → `annulus_kwargs["r_in"] = 32.5` →
`normalize_outer_annulus(x, r_in=32.5)`. **Verify this chain is unbroken.**

### 1.4 Tests in `test_preprocess_regression.py`

**Q1.12** `test_preprocessing_outer_sky_near_zero` has NO assertion — only prints.
**Should it assert sky median within some tolerance of zero?** Or is -2.6
"correct" because it matches trained models?

**Q1.13** The behavioral test uses a synthetic exponential galaxy with scale
length 8 px. **How representative is this of real training data?** Does the 20%
contamination number overstate real-world impact?

**Q1.14** Checksum `7e25b9e366471bda` — **is this cross-platform stable?** Float32
arithmetic can differ between x86 (dev) and ARM (Lambda GH200). Verified on
actual training hardware?

### 1.5 Injection Priors Registry

**Q1.15** AST parsing handles simple constant defaults. **What about computed
expressions?** If a future change makes `re_arcsec_range = (0.05, 0.5 * MAX_RE)`,
the parser returns `_SENTINEL` and silently skips. Is this acceptable?

**Q1.16** `clumps_n_range` and `clumps_frac_range` are in the YAML but NOT
validated (they're inside the function body, not defaults). **Can they drift
silently?** How to test them?

**Q1.17** The YAML says `g_minus_r_mu_sigma: [0.2, 0.25]` (Gaussian). The
PREVIOUS version of our documentation said `g-r ~ U[0.0, 1.5]` (Uniform).
**The code uses Gaussian. Confirm this is correct by reading the code, and flag
if you find any discrepancy between YAML, code defaults, and documentation.**

### 1.6 Beta_frac Diagnostic Script

**Q1.18** Part 2 uses a blank host (no galaxy, background noise only). **Are
detection rates on blank hosts meaningful, or just upper bounds?**

**Q1.19** P(beta_frac < 0.55) = 29.5% ≈ 30% ceiling. But this assumes SIS
caustic structure. **For SIE with q_lens ~ 0.5-1.0, does the SIS approximation
hold?** Sources at beta_frac > 0.55 can produce bright arcs near cusp/fold
caustics.

**Q1.20** The 29.5% ≈ 30% match may be coincidental. Detection probability is
a smooth function of beta_frac, not a step function. **How to distinguish
"beta_frac geometry explains the ceiling" from "coincidence"?**

### 1.7 The v5 Config

**Q1.21** v5 trains FROM SCRATCH with ImageNet init. v4 (best model, AUC=0.9921)
finetuned from v2's best. **Why not finetune from v4 with the new annulus?**
If annulus change only affects normalization scale, finetuning might recover
faster. If it changes feature geometry, fresh training is needed. **Which is it?**

**Q1.22** v5 doesn't specify `freeze_backbone_epochs` or `warmup_epochs`.
**What are the TrainConfig defaults?** If they differ from v2's settings,
v5 training protocol silently differs from the recipe that produced our best
models.

**Q1.23** v4 finetune achieved best AUC with 60 epochs of cosine from v2's peak.
**Should v5 also do two-phase training (160 + 60 finetune)?** Or will v5 beat
v4 at the v2-equivalent stage?

---

## SECTION 2: Training Pipeline End-to-End

**Q2.1** `train.py` gradient accumulation: `loss = loss / accum_steps` before
`scaler.scale(loss).backward()`. With mixed precision, **could float16 precision
cause issues when dividing a small loss by accum_steps=8 or 16?**

**Q2.2** Val set in `train.py` uses `AugmentConfig(hflip=False, vflip=False, rot90=False)`.
Good. **But does `selection_function_grid_v2.py` also disable augmentation during
scoring?** Trace the scoring path end-to-end: cutout → preprocess_stack → model
forward → probability. Verify no augmentation.

---

## SECTION 3: Pipeline Consistency

**Q3.1** **CONFIRMED GAP:** NONE of the scoring scripts pass annulus kwargs:
- `selection_function_grid_v2.py` line 190: `preprocess_stack(img, mode=preprocessing, crop=crop, clip_range=10.0)` — NO annulus
- `selection_function_grid_v2.py` line 563: `preprocess_stack(injected_chw_np, mode=preprocessing, crop=crop, clip_range=10.0)` — NO annulus
- `bright_arc_injection_test.py` line 193: `preprocess_stack(inj_chw, mode="raw_robust", crop=False, clip_range=10.0)` — NO annulus
- `real_lens_scoring.py` line 69: `preprocess_stack(chw, mode="raw_robust", crop=False, clip_range=10.0)` — NO annulus

There are 30+ files calling `preprocess_stack`. **After retraining with (32.5, 45.0),
every scoring path must match.** Review all callers. **Propose a safer architecture**
(e.g., storing annulus params in checkpoint metadata so scoring auto-loads them).

**Q3.2** The injection engine's `inject_sis_shear` takes HWC and returns CHW.
`preprocess_stack` expects CHW. **Verify format conversions are consistent
everywhere.** A transpose error would silently permute bands.

**Q3.3** Compare `LensDataset.__getitem__()` (training path) vs
`selection_function_grid_v2.py` (scoring path) calls to `preprocess_stack`.
**Are keyword arguments identical?** Any difference (clip_range, mode) would
invalidate the selection function.

**Q3.4** Checksum `7e25b9e366471bda` computed on Python 3.11 (dev). Lambda
GH200 is ARM. **Are float32 ops bit-identical across platforms?** If checksum
fails on Lambda, someone might update it without understanding why.

**Q3.5** Mixed precision uses float16 during training, float32 at inference.
**Could this cause normalization to produce slightly different results?** Check:
does the scoring code use autocast?

**Q3.6** Manifest `cutout_path` uses absolute Lambda NFS paths. **Is there error
handling for missing files?** More dangerous: if the file exists but contains a
DIFFERENT cutout (old version), training proceeds silently with wrong data. **Any
checksum or version tag in the manifest?**

**Q3.7** In the selection function grid injection loop, if an injection fails
(NaN in deflection), **does it count as "not detected" (deflating completeness)
or error out?** Check the error handling.

---

## SECTION 4: Full Code Audit

**Q4.1** Audit `dhs/preprocess.py`: Is `preprocess_stack()` correctly implementing
outer-annulus median/MAD normalization? Any edge cases producing different results
for real vs injected cutouts?

**Q4.2** Audit `dhs/model.py`: Architecture correct? Inference path issues?

**Q4.3** Audit `dhs/data.py`: Data loading correct for scoring? Any preprocessing
differences between training and inference?

**Q4.4** Audit `dhs/transforms.py`: Augmentations applied during training only?
Could they leak into inference?

**Q4.5** Audit `injection_model_2/host_matching.py`: Moment calculations correct?
q/PA estimation robust?

**Q4.6** Audit `injection_model_2/scripts/selection_function_grid_v2.py`: Grid
logic correct? Silent failure fixes adequate?

**Q4.7** Audit `dhs/injection_engine.py`: SIE deflection mathematically correct
(Kormann et al. 1994)? Sersic normalization matches Graham & Driver 2005? Flux
conservation verified?

---

## SECTION 5: Specific Verification Requests

**Q5.1** **Preprocessing identity:** Verify `preprocess_stack()` produces
identical normalization for real cutouts vs cutouts with injected arcs. Pay
attention to whether a bright injected arc could shift annulus statistics.

**Q5.2** **SIE deflection:** Verify against Kormann et al. 1994. Check deflection
angle formulas, softened core, coordinate rotation.

**Q5.3** **Sersic normalization:** Verify total flux integral matches Graham &
Driver 2005.

**Q5.4** **Area-weighted sampling:** Verify `beta_frac = sqrt(uniform(0.01, 1.0))`
produces P(β) ∝ β correctly.

**Q5.5** **Augmentation leakage:** Verify training augmentations (hflip, vflip,
rot90) do NOT apply during inference/scoring.

**Q5.6** **Bright-arc ceiling bug check:** Could the 30% ceiling be caused by:
(a) preprocessing clipping saturating bright injections, (b) MAD shifted by
injection, (c) any numerical artifact in the scoring path?

**Q5.7** **Flux conservation:** Verify `cutout + lensed_arc` preserves total flux
through the preprocessing pipeline.

---

## DELIVERABLES FOR THIS PROMPT

1. For each Q above, a clear PASS / FAIL / CONCERN verdict with explanation.
2. If you find ANY bugs not already identified, flag them prominently with
   [NEW BUG] label.
3. For the pipeline consistency gap (Q3.1), propose a concrete architecture fix
   — code sketch, not just description.
4. A summary table: "Top 5 issues found, ranked by severity."

**Be thorough and sincere. Do not declare things are fine without checking the
code. Do not give up on a question because it's hard.**
