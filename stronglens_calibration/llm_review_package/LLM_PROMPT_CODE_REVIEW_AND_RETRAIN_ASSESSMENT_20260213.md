# Independent Review Request: Code Changes, Pipeline Integrity, and Retraining Assessment

**Date:** 2026-02-13
**Context:** This is a SUPPLEMENTARY prompt to `LLM_PROMPT_COMPREHENSIVE_REVIEW_20260213.md`.
Read that prompt FIRST for full project context (goals, data, results, history).
This prompt focuses on RECENT CODE CHANGES made by a Cursor LLM on 2026-02-13,
and on deep pipeline-integrity questions that a hostile MNRAS referee would ask.

**Attached:** `stronglens_calibration_for_review_20260213.zip` — the complete codebase
with all changes applied. Start by reading `CHANGES_FOR_LLM_REVIEW.md` at the root.

**YOUR ROLE:** You are an independent, hostile-but-reasonable peer reviewer.
The previous LLM that made these changes claims they are correct but the project
owner does NOT trust them. You must verify EVERYTHING independently. Do not take
any claim at face value. Read the code. Check the math. Run the logic in your head.
If something is wrong, say it plainly. If something is right, explain WHY it's right
so we can be confident.

---

# SECTION A: REVIEW OF THE 2026-02-13 CODE CHANGES

A Cursor LLM made the following changes in response to findings from an earlier
external LLM review. Every change needs independent verification.

## A1. The Annulus Revert in `dhs/utils.py`

The Cursor LLM claims:

1. The normalization annulus (20, 32) is wrong for 101×101 stamps because it
   sits at 40-63% of the image half-width, overlapping galaxy light.
2. The corrected annulus (32.5, 45.0) sits at 65-90%, in the sky-dominated ring.
3. Existing trained models MUST use (20, 32) because they were trained with it.
4. The defaults were reverted to (20, 32) to avoid breaking existing models.
5. A `default_annulus_radii()` function is provided for future retraining.

**Questions for you:**

**A1.1** The normalization uses MEDIAN and MAD (median absolute deviation), which
are robust estimators. If only ~20% of the annulus pixels contain galaxy light,
the MEDIAN might still be correct (dominated by the ~80% sky pixels). **Is the
annulus bug actually impactful for median/MAD normalization?** Work through the
math: for a typical bright elliptical galaxy at r=20-32 pixels from center in a
101×101 stamp with 0.262"/pix, what fraction of annulus pixels have galaxy flux
> 3σ_sky? Is the median shifted? Is the MAD inflated? Give concrete numbers, not
hand-waving. This is crucial — if the median is robust to 20% contamination, the
entire retraining exercise may be unnecessary.

**A1.2** The test output says: "Sky median (r>40) after preprocessing: -2.6
(KNOWN ISSUE)." This means after normalization, true sky pixels are at -2.6
instead of ~0. This is a strong signal that the normalization IS biased. But
this was measured on a SYNTHETIC exponential galaxy with scale length 8 px (very
extended, R_e ≈ 13 px). **What does this look like for REAL DR10 galaxies?**
Typical LRGs in the training set have half-light radii of 1-3 arcsec (4-12 px).
For a more compact galaxy (R_e ~ 6 px), would the sky median still be -2.6?
Or would it be much closer to 0?

**A1.3** The `default_annulus_radii()` formula is `r_in = 0.65 * R, r_out = 0.90 * R`
where `R = min(H,W) // 2`. **Is this principled or arbitrary?** Why 0.65 and
0.90? A referee would ask: "What is the physical basis for these coefficients?"
Should the annulus placement instead be adaptive — e.g., based on the measured
galaxy half-light radius, or on an iterative sigma-clipping procedure?

**A1.4** The runtime warning fires whenever `r_out / half < 0.70` and H > 64.
But `preprocess_stack` is called once per training sample per epoch. With ~316K
training samples × 160 epochs = ~50M calls. **Will this flood the logs?** The
warning uses `logger.warning()` which in Python's logging module does NOT
deduplicate by default. Is this a performance concern?

**A1.5** The `normalize_outer_annulus` signature changed from accepting
`r_in: float | None = None, r_out: float | None = None` (calling
`default_annulus_radii` when None) to `r_in: float = 20, r_out: float = 32`.
**Verify this revert is complete and correct.** Check that no other code path
still expects the None-accepting signature. Search for any callers that pass
`r_in=None` or `r_out=None`.

## A2. The preprocess_stack Annulus Passthrough in `dhs/preprocess.py`

**A2.1** If someone passes only `annulus_r_in=32.5` but forgets `annulus_r_out`,
the kwargs dict will be `{"r_in": 32.5}` and `normalize_outer_annulus` will use
`r_in=32.5, r_out=32` (the default). This creates an annulus where inner > outer!
**Is there validation for this?** Trace the code path: what happens when
`r_in > r_out`? Does `radial_mask` return an empty mask? Does `robust_median_mad`
handle an empty array? Does it produce NaN or crash?

**A2.2** The function signature uses `annulus_r_in: float | None = None`. The
`float | None` union type syntax requires Python 3.10+. **Is this compatible with
the project's Python 3.11 target?** (Yes, but verify the requirements.txt
specifies Python version.)

## A3. The DatasetConfig Changes in `dhs/data.py`

**A3.1** The annulus radii use `0.0` as a sentinel meaning "use default." But
`0.0` is a valid float. The check is `if self.dcfg.annulus_r_in > 0`. **What if
the YAML has `annulus_r_in: 0` explicitly — would it be treated as "use default"?**
Is this the intended behavior? Document this clearly.

**A3.2** `run_experiment.py` line 81 does `dcfg = DatasetConfig(**cfg["dataset"])`.
If the YAML has keys that don't match DatasetConfig fields (typos, extra keys),
Python raises `TypeError: __init__() got an unexpected keyword argument`. **Verify
that the new YAML configs (`paperIV_efficientnet_v2_s_v5_annulus_fix.yaml`) have
EXACTLY the right keys.** Are there any keys in the YAML that are NOT DatasetConfig
fields? Compare the YAML `dataset:` section key-by-key against the DatasetConfig
fields.

**A3.3** The variable name `crop_kwargs` in `__getitem__` now carries both crop
parameters AND annulus parameters. This is misleading. **More importantly: are the
annulus kwargs correctly forwarded?** Trace the full path:
YAML `annulus_r_in: 32.5` → `DatasetConfig(annulus_r_in=32.5)` → `crop_kwargs['annulus_r_in'] = 32.5`
→ `preprocess_stack(**crop_kwargs)` → `annulus_kwargs["r_in"] = 32.5` →
`normalize_outer_annulus(x, r_in=32.5)`. **Verify this chain is unbroken.**

## A4. The Tests in `test_preprocess_regression.py`

**A4.1** `test_preprocessing_outer_sky_near_zero` has NO assertion — it only
prints. **This test cannot catch regressions.** Should it assert that the sky
median is within some tolerance of zero? Or is the current state (sky median = -2.6)
considered "correct" because it matches what trained models expect?

**A4.2** The behavioral test uses a synthetic exponential galaxy with scale
length 8 px. **How representative is this of real training data?** If most real
galaxies have R_e < 8 px, the test galaxy is an outlier and the 20% contamination
number may overstate the real-world impact.

**A4.3** The checksum is `7e25b9e366471bda`. **Is this checksum cross-platform
stable?** The code rounds to 6 decimal places, but float32 arithmetic can differ
between x86 and ARM (Lambda GH200 is ARM). Has this checksum been verified on
the actual training hardware?

## A5. The Injection Priors Registry (`configs/injection_priors.yaml`)

**A5.1** The test uses AST parsing to extract defaults. **What about defaults
that are computed expressions?** For example, if a future change makes
`re_arcsec_range = (0.05, 0.5 * MAX_RE)`, the AST parser would return `_SENTINEL`
and silently skip the test. **Is this failure mode acceptable?**

**A5.2** The `clumps_n_range` and `clumps_frac_range` appear in the YAML but
are NOT validated by the test (they're inside the function body, not defaults).
**These can drift silently.** Is there a way to test them?

**A5.3** The YAML says `g_minus_r_mu_sigma: [0.2, 0.25]` meaning g-r is drawn
from N(0.2, 0.25). But the comprehensive review prompt (Part 5.2, line 202) says
`g-r ~ U[0.0, 1.5]`. **Which is correct?** The code uses Gaussian (N(0.2, 0.25)).
The prompt says Uniform. **This is exactly the kind of code-to-paper drift the
registry was supposed to prevent, and it's already present IN THE PROMPT ITSELF.**
Fix this.

## A6. The Beta_frac Diagnostic Script

**A6.1** Part 2 (injection experiment) uses a blank host with `background_nmgy = 0.1`
and `noise_sigma = 0.05`. Real hosts have bright central galaxies. **Detection
rates on blank hosts are upper bounds, not realistic estimates.** Does the script
document this limitation clearly enough?

**A6.2** The hypothesis is P(beta_frac < 0.55) = 29.5% ≈ 30% ceiling. But
this assumes ONLY sources with beta_frac < 0.55 produce detectable arcs. **Is
0.55 the right threshold?** For SIE (not SIS), the caustic structure is more
complex. Sources with beta_frac > 0.55 can still produce bright arcs if they
are near a cusp or fold caustic. **Does the SIS approximation hold for SIE
with q_lens ~ 0.5-1.0?**

**A6.3** Even if the math is right, the 30% ceiling prediction assumes the CNN
detects ALL bright arcs with beta_frac < 0.55 and NONE with beta_frac > 0.55.
In reality, detection probability is a smooth function of beta_frac (and SNR,
morphology, etc.). **The 29.5% ≈ 30% match may be coincidental.** How would you
distinguish "beta_frac geometry explains the ceiling" from "it's a coincidence"?

## A7. The v5 Annulus Fix Config (`paperIV_efficientnet_v2_s_v5_annulus_fix.yaml`)

**A7.1** The v5 config trains FROM SCRATCH with `pretrained: true` (ImageNet).
The v4 finetune config loaded v2's best checkpoint and finetuned at 5e-5.
**Why is v5 not finetuning from v4's best checkpoint with the new annulus?**
The rationale matters: if the annulus change only affects normalization scale
(not feature-space geometry), finetuning might recover quickly. If it changes
what the model sees fundamentally, fresh training is needed. **Which is it?
What is the expected training time for v5 vs a hypothetical v5-finetune?**

**A7.2** The v5 config doesn't specify `freeze_backbone_epochs` or `warmup_epochs`.
The v2 config (which v5 is modeled after) has `freeze_backbone_epochs: 5`.
**What is the default in the TrainConfig dataclass?** If it defaults to 0,
v5 would NOT freeze the backbone, which is different from v2. If it defaults to 5,
v5 would match v2. **This must be verified before launching the run.**

**A7.3** The comment says "Protocol: 160 epochs, StepLR halve@130." But v4
finetune (which was our BEST model at AUC=0.9921) used 60 epochs of cosine from
v2's peak. **Should v5 also do a two-phase training (160 epochs + 60 epoch
finetune)?** Or is the expectation that v5 with corrected annulus will already
beat v4 at its v2-equivalent stage?

---

# SECTION B: PRE-TRAINING DATA PIPELINE — HOSTILE REVIEWER QUESTIONS

A referee who wants to reject this paper would probe the data pipeline. These
questions have NOT been addressed anywhere in the existing documentation.

## B1. Cutout Integrity

**B1.1** The cutouts are 101×101×3 in HWC format (g, r, z bands), stored as
`data["cutout"]` in .npz files. `load_cutout_from_file` does
`cutout.transpose(2, 0, 1)` to get CHW. **Has anyone verified that the band
order is correct?** If g and z are swapped, the model would still train (it
doesn't know band semantics), but the injection engine assumes specific band
ordering when computing g-r and r-z colors. **Verify the band order is g=0,
r=1, z=2 in both the training cutouts AND the injection engine.**

**B1.2** The cutouts are in nanomaggies. **Has anyone verified that the
nanomaggy values are correct?** Specifically: DR10 Tractor catalogs provide
`flux_g`, `flux_r`, `flux_z` in nanomaggies and `psfsize_g/r/z` in arcsec. But
the cutout pixel values come from the coadd images, not the catalog. **Are the
coadd images in nanomaggies or counts or something else?** If there's a
calibration offset, the injection engine (which works in nanomaggies) would inject
arcs at the wrong brightness.

**B1.3** Are ALL cutouts exactly 101×101? **What happens if a cutout is at the
survey edge and some pixels are masked or missing?** The preprocessing replaces
NaN with 0, but does the cutout generation pipeline produce NaN for masked pixels?
Or does it produce zeros (which would be indistinguishable from real zero-flux
pixels)?

## B2. Label Quality

**B2.1** The prompt says `sample_weight_col: sample_weight` in configs and
mentions Tier-B weight = 0.5. But ALL training configs have `unweighted_loss: true`.
Looking at `train.py` line 33: `unweighted_loss: bool = False` (default) and
line 176: `loss_fn = BCEWithLogitsLoss(reduction='none' if weighted else 'mean')`,
and line 123: `weighted = is_file_manifest and not tcfg.unweighted_loss`. With
`unweighted_loss: true`, `weighted = False`, so ALL samples have equal weight.
**The Tier-B weighting is completely ignored in practice.** Is this intentional?
If so, the 4399 Tier-B candidates (noisy labels) contribute equally to the loss.
Could this explain the AUC gap to Paper IV (which uses only 1372 confirmed lenses)?

**B2.2** Paper IV uses 1,372 confirmed lenses. We use 4,788 (389 confirmed + 4,399
visual). **What is the false-positive rate in Tier-B labels?** If 10% of Tier-B
are actually not lenses, that's ~440 mislabeled positives in a dataset with 4,788
total positives — nearly 10% label noise. **Has anyone estimated the label noise
rate? How does it affect training?**

**B2.3** A hostile reviewer would ask: "You claim AUC=0.9921. But your positive
labels include unconfirmed candidates. If 10% are false positives, your true AUC
on the clean subset could be higher or lower depending on whether the 'false
positives' are easy negatives or hard confusers. **How robust is the AUC
measurement to label noise?**"

## B3. Negative Pool

**B3.1** Paper IV removes likely lenses from negatives using a prior model
(p > 0.4 → visual inspection → remove). We don't. **How many real lenses are
in our 446,893 negatives?** The DESI lens catalog has ~4800 candidates. If
they're uniformly distributed across the sky, and our negative pool covers a
similar area, simple geometry suggests ~10-50 real lenses in the negatives.
**Is this negligible or does it matter at 93:1 ratio?**

**B3.2** The hard confuser pool N2 (~15%) contains ring galaxies, edge-on disks,
and blue clumpy galaxies. **How were these selected?** The prompt mentions
"Sérsic index > 4.0" for ring proxies and "ellipticity > 0.50" for edge-on. But
these are Tractor morphological parameters, not visual classifications. **How
reliable are these proxies?** A Sérsic index > 4 galaxy is not necessarily a
ring galaxy — it could be a compact elliptical.

## B4. Spatial Splits

**B4.1** Splits are HEALPix nside=128, hash-based. **But are the positives
spatially correlated?** Many DESI lens candidates come from specific survey
regions with deeper coverage. If most positives are in a few HEALPix pixels, the
train/val split could be heavily unbalanced. **What is the distribution of
positives across HEALPix pixels? How many pixels contain >1 positive?**

**B4.2** A hostile reviewer: "Your spatial splits prevent information leakage
from repeated imaging of the same field. But do they prevent leakage from
similar PSF/depth conditions? If most high-PSF regions are in train and most
low-PSF regions are in val, the model might learn PSF-specific features that
don't generalize." **Was PSF/depth balance across splits verified and reported?**

---

# SECTION C: TRAINING PIPELINE — DOES IT ACTUALLY WORK END-TO-END?

## C1. YAML → Model Pipeline

**C1.1** The new YAML configs have `annulus_r_in: 32.5` and `annulus_r_out: 45.0`
in the `dataset:` section. `run_experiment.py` does `DatasetConfig(**cfg["dataset"])`.
**Run through this in your head:** what are ALL the keys in
`paperIV_efficientnet_v2_s_v5_annulus_fix.yaml`'s `dataset:` section?

```yaml
dataset:
  parquet_path: ""
  manifest_path: ...
  mode: file_manifest
  preprocessing: raw_robust
  label_col: label
  cutout_path_col: cutout_path
  sample_weight_col: sample_weight
  seed: 42
  crop: false
  crop_size: 0
  annulus_r_in: 32.5
  annulus_r_out: 45.0
```

DatasetConfig fields are: `parquet_path, mode, preprocessing, label_col, seed,
manifest_path, cutout_path_col, sample_weight_col, crop, crop_size, annulus_r_in,
annulus_r_out`. **Do all 12 YAML keys match DatasetConfig field names exactly?**
Any typo (e.g., `annulus_rin` vs `annulus_r_in`) would crash at startup.

**C1.2** The original configs (v1-v4) do NOT have `annulus_r_in` or `annulus_r_out`
keys. When `DatasetConfig(**cfg["dataset"])` is called, these fields use their
defaults (0.0). **Verify that old configs still work with the new DatasetConfig.**
The concern: if any old config has a key that's NOT in DatasetConfig, it would
crash. Check all YAML configs.

## C2. Gradient Accumulation

**C2.1** `train.py` line 243-244:
```python
if accum_steps > 1:
    loss = loss / accum_steps
```
This divides the loss BEFORE `scaler.scale(loss).backward()`. With mixed precision
and gradient accumulation, is this numerically equivalent to computing the true
average loss? **Could there be precision issues when dividing a small loss by
accum_steps=8 or 16 in float16?**

## C3. Augmentation Leakage

**C3.1** In `train.py` line 118:
```python
ds_va = LensDataset(dcfg, SplitConfig(split_value="val"),
                    AugmentConfig(hflip=False, vflip=False, rot90=False))
```
Val set uses no augmentation. Good. **But what about scoring during
selection function grid runs?** Does `selection_function_grid.py` also disable
augmentation? **Trace the scoring path end-to-end**: cutout → preprocess_stack →
model forward → probability. Verify no augmentation is applied.

---

# SECTION D: THE KEY QUESTION — WILL RETRAINING ACTUALLY HELP?

The Cursor LLM claims retraining with (32.5, 45.0) annulus is necessary and will
improve the model. **This is the most expensive step (10+ GPU-hours). We need
maximum confidence before committing.**

## D1. What Does the Annulus Bug Actually Do?

**D1.1** The annulus (20, 32) on a 101×101 stamp contains galaxy light. The
median of the annulus is biased UPWARD (galaxy median > sky median). The MAD is
biased UPWARD (galaxy variance > sky noise). After normalization: `x_norm =
(x - median_biased) / MAD_biased`. The center (galaxy) is divided by a larger
MAD, so its **contrast is suppressed**. The sky is at `(sky - median_biased) /
MAD_biased` ≈ negative value (consistent with the -2.6 measured).

**D1.2** But the model was TRAINED on this suppressed-contrast representation.
It learned to detect lenses in this representation. The key question:
**Does the suppressed contrast hurt DETECTION or just change the REPRESENTATION?**

Consider: if all images (positives and negatives) are normalized the same way,
the relative contrast between "lens" and "not lens" might be preserved. The
absolute pixel values are different, but the model could adapt. **In which case,
fixing the annulus would NOT improve AUC — it would just change the scale.**

**D1.3** HOWEVER, the injection scoring uses THE SAME normalization. So injected
arcs are also suppressed. **If the suppression is the same for real and injected
arcs, it cancels out and doesn't explain the 70pp gap.** The gap would persist
with the new annulus. **If the suppression is DIFFERENT for real vs injected arcs
(e.g., because real arcs contribute differently to the annulus), it could explain
part of the gap.**

**D1.4** For a 101×101 host at annulus (20, 32): when you ADD a bright injected
arc at r ~ 5-10 px from center, the arc does NOT reach the annulus (it's at
r=5-10, annulus starts at r=20). So the arc does not change the annulus
statistics. **But for a real lens in the training data, the arc was ALWAYS THERE**
— the annulus statistics already include whatever contribution the real arc makes
at r=20-32. If the arc has extended low-surface-brightness wings that reach
r=20-32, there IS a difference: real arcs subtly shift the annulus, injected arcs
don't. **Quantify: for a typical arc at theta_E = 1.5" (5.7 px), how much flux
reaches r=20 px after PSF convolution?**

**D1.5** Bottom line for retraining decision: **Estimate the expected AUC change
from retraining.** Will it be +0.001 (negligible)? +0.01 (modest)? +0.05
(significant)? Give a concrete prediction with reasoning. **What would be the
expected change in the 30% bright-arc ceiling?**

## D2. Could There Be a Cheaper Fix?

**D2.1** Instead of retraining: could we apply the new annulus normalization to
both the host AND the injected image, THEN score? The model was trained with (20,32)
normalization. But if we preprocess the host with (20,32) for training, and then
during injection scoring preprocess the injected image ALSO with (20,32), the
representation should be consistent. **The question is whether using (32.5, 45.0)
for SCORING ONLY (both host and injected, but model trained on (20,32)) would help
or hurt.** It would hurt because the model expects (20,32)-normalized inputs.

**D2.2** Another option: don't fix the annulus at all. Instead, report in the
paper: "Our normalization annulus at (20, 32) pixels overlaps galaxy light for
101×101 stamps. This suppresses arc contrast by a factor of X. We quantify this
effect as a systematic uncertainty on the selection function." **Would a referee
accept this?**

## D3. Pre-Training Data: Does Anything Need to Regenerate?

The Cursor LLM claims: "Pre-training data (cutouts, manifests, negative sampling)
do NOT need to be regenerated."

**D3.1** This is correct IF the cutouts store raw nanomaggies and normalization
is applied at training time. **Verify this claim by reading `load_cutout_from_file`
in `dhs/data.py`.** Does it just load and transpose, or does it do any
normalization? Is there ANY preprocessing baked into the cutout files?

**D3.2** The manifests contain `psfsize_r`, `psfdepth_r`, and other metadata.
These are NOT affected by the annulus change. **But are there any derived columns
in the manifest that depend on preprocessing?** Check the manifest generation
code if available.

---

# SECTION E: HOSTILE REFEREE QUESTIONS — WHAT WOULD GET THIS PAPER REJECTED?

Think like an MNRAS referee who is looking for reasons to reject. These are
questions we MUST have answers for before submitting.

## E1. Fundamental Methodology

**E1.1** "Your injection-recovery completeness is 3.5% while your real-lens
recall is 73%. You claim this is because injections don't resemble real lenses.
But this is a circular argument: you DEFINE the selection function using
injections that you KNOW don't work. How is this useful? What does a 3.5%
completeness number actually mean for the scientific community?"

**E1.2** "You train on Tier-B candidates (visual grades only, no spectroscopic
confirmation), then measure 'real-lens recall' on Tier-A (confirmed). But Tier-A
is a subset of the training data. You're measuring recall on training examples.
This is test-set contamination." **Is this true? Are Tier-A lenses in the
training split?** If yes, the 73.3% real-lens recall is meaningless. Check
whether `real_lens_scoring.py` excludes training-split lenses.

**E1.3** "Your negative pool may contain unlabeled real lenses. You report AUC
but AUC is sensitive to label noise. What is your estimated label-noise rate in
both positives and negatives, and how does it affect your metrics?"

## E2. The Selection Function

**E2.1** "Your selection function depends on injection priors (source size, color,
offset, morphology). Different priors give different completeness. How sensitive
are your results to the choice of priors? You report a sensitivity analysis with
<1% mean shift, but your priors may be systematically wrong (e.g., R_e much too
small, colors not matching real lensed sources). A 'sensitivity analysis' that
perturbs around the wrong point is meaningless."

**E2.2** "Your source R_e range is (0.05, 0.25) arcsec. Real lensed sources at
z~1-2 have R_e ~ 0.1-1.0 arcsec (e.g., Cañameras et al. 2024 Table 1). Your
upper limit is 4× too small. This means you're only injecting very compact
sources. Extended sources produce different arc morphologies. **Isn't your
selection function biased toward compact sources?**"

**E2.3** "Your Sersic index range is (0.7, 2.5). Real high-z star-forming
galaxies often have n < 1 (disk-dominated) or have complex multi-component
morphologies that Sersic cannot represent. Your n range cuts off the disk-like
sources. How does this affect the results?"

## E3. The Comparison with Paper IV

**E3.1** "Paper IV achieves AUC=0.9987 with cleaned negatives and confirmed
positives. You achieve AUC=0.9921 with dirty negatives and noisy positives. The
gap is 0.0066. You attribute this to dirty data. But you have 3.3× more
negatives and 3.5× more positives. More data usually helps. Could the gap
instead be due to your preprocessing being wrong (the annulus bug)?"

**E3.2** "You have not replicated Paper IV's negative cleaning step. You claim
this is an 'ablation.' But an ablation requires holding everything else constant.
You also changed the positive set (added Tier-B), the framework (TensorFlow →
PyTorch), the GPU, and the normalization. You have too many uncontrolled
variables to attribute the AUC gap to any single factor."

## E4. Missing Validation

**E4.1** "You have not performed any independent validation with spectroscopic
follow-up. The Tier-A anchors are the closest you have, but they were used in
training. Where are your holdout confirmed lenses?"

**E4.2** "You report injection completeness stratified by theta_E, PSF, and
depth. But you do not report completeness stratified by source redshift. Since
source redshift determines angular size, color, and surface brightness, this is
a fundamental missing dimension."

**E4.3** "You have not compared your selection function against any published
result. Herle et al. (2024) and Cañameras et al. (2024) both report injection-
recovery completeness for CNN lens finders. How does your 3.5% compare to their
numbers? If their numbers are much higher, what's different about their setup?"

---

# SECTION F: WHAT ELSE COULD TRIP US UP?

We've been burned by incorrect assumptions before. Help us find the remaining
landmines.

## F1. Pipeline Consistency Gaps

**F1.1** **CONFIRMED GAP — THIS IS A TICKING TIME BOMB.** We have already verified
that NONE of the scoring scripts pass annulus kwargs:

- `injection_model_2/scripts/selection_function_grid_v2.py` line 190:
  `preprocess_stack(img, mode=preprocessing, crop=crop, clip_range=10.0)` — NO annulus params
- `injection_model_2/scripts/selection_function_grid_v2.py` line 563:
  `preprocess_stack(injected_chw_np, mode=preprocessing, crop=crop, clip_range=10.0)` — NO annulus params
- `sim_to_real_validations/bright_arc_injection_test.py` line 193:
  `preprocess_stack(inj_chw, mode="raw_robust", crop=False, clip_range=10.0)` — NO annulus params
- `sim_to_real_validations/real_lens_scoring.py` line 69:
  `preprocess_stack(chw, mode="raw_robust", crop=False, clip_range=10.0)` — NO annulus params

There are at least **30+ files** that call `preprocess_stack`. After retraining with
(32.5, 45.0), EVERY scoring path must be updated. If they are not, the model
will be evaluated with mismatched normalization ((32.5, 45.0) during training vs
(20, 32) during scoring), producing WRONG selection function results that could
look plausible. **Review all callers and propose a safer architecture** — e.g.,
storing the annulus parameters in the checkpoint metadata so the scoring scripts
can auto-load them.

**F1.2** The injection engine adds arcs in nanomaggies to host cutouts in
nanomaggies, THEN preprocesses. But `preprocess_stack` expects CHW format (3,
101, 101). The injection engine's `inject_sis_shear` takes HWC input and returns
CHW output. **Verify the format conversions are consistent everywhere.** A
transpose error would silently permute bands.

**F1.3** During training, `LensDataset.__getitem__` calls `preprocess_stack`.
During selection function grid scoring, the grid script calls `preprocess_stack`
separately. **Are the keyword arguments identical?** Compare the two call sites
line by line. Any difference (e.g., different `clip_range`, different `mode`)
would invalidate the selection function.

## F2. Numerical Gotchas

**F2.1** The checksum `7e25b9e366471bda` was computed with NumPy on Python 3.11.
Lambda Cloud uses GH200 (ARM). Development is on macOS (x86/ARM). **Are float32
operations bit-identical across these platforms?** If not, the checksum test will
fail on Lambda, and someone might update it without understanding why.

**F2.2** Mixed precision training uses float16 for forward/backward and float32
for weight updates. **Could float16 precision cause the normalization to produce
slightly different results during training vs inference?** During training, the
input tensor is cast to float16 inside the autocast context. During inference (in
the selection function grid), is autocast used? Check the scoring code.

## F3. Silent Failures

**F3.1** The manifest has `cutout_path` pointing to absolute paths on Lambda NFS.
If a path is wrong (file moved, NFS unmounted), `load_cutout_from_file` will
crash. **Is there any error handling?** More dangerously: if the file exists but
contains a DIFFERENT cutout (e.g., from a previous version), training proceeds
silently with wrong data. **Is there a checksum or version tag in the manifest?**

**F3.2** The selection function grid loops over theta_E, PSF, depth cells and
injects 200 arcs per cell. If an injection fails (e.g., NaN in deflection),
**does it silently count as "not detected" (deflating completeness) or does it
error out?** Check the error handling in the grid script's injection loop.

---

# SECTION G: MAXIMIZING CONFIDENCE BEFORE RETRAINING

## G1. Cheap Experiments That Should Be Done FIRST

**G1.1** Run the beta_frac diagnostic on GPU with the FULL injection experiment
(not just math mode). Restrict beta_frac_max to 0.55 and see if bright-arc
detection rate jumps from 30% to 80%+. **This requires zero retraining and
directly tests the hypothesis.** If it fails, the 30% ceiling has a different
cause.

**G1.2** Extract CNN embeddings (penultimate layer) for:
- 200 real lens cutouts (Tier-A)
- 200 bright injections (source mag 18-19)
- 200 random negatives
Plot t-SNE or UMAP. **If real lenses and bright injections cluster together, the
gap is NOT morphological. If they separate, the CNN IS distinguishing them and we
need to understand why.** This requires zero retraining.

**G1.3** Compute the annulus median and MAD for 1000 real training cutouts with
both (20, 32) and (32.5, 45.0). Compare the distributions. **If the two annuli
produce nearly identical median/MAD for most galaxies, the bug is cosmetic and
retraining won't help.** This requires zero retraining.

**G1.4** Run the existing model (v4 finetune) on 200 injections preprocessed
with (32.5, 45.0) annulus (mismatch with training). Compare completeness against
the standard (20, 32). **If completeness changes dramatically, the annulus matters.
If it doesn't change, it doesn't matter.** Note: this is deliberately a mismatch
test. We're not testing "correct" behavior — we're testing sensitivity.

## G2. What Would Make You Confident the Retrained Model Is Better?

**G2.1** Before retraining, define quantitative success criteria:
- AUC should be >= X
- Real-lens recall at p>0.3 should be >= Y
- Bright-arc ceiling should rise from 30% to >= Z
- Selection function completeness (mag 23-24, theta_E=1.5") should be >= W

**What are reasonable values for X, Y, Z, W? Provide concrete predictions.**

**G2.2** If retraining produces AUC=0.9920 (essentially unchanged) and the 30%
ceiling remains at 30%, **what would you conclude?** That the annulus bug doesn't
matter? That something else is wrong? What would be the next diagnostic step?

---

# SECTION H: TRAINING EVALUATION — QUESTIONS THAT WOULD SINK THE PAPER

## H-train.1 Tier-A Evaluation Contamination

The comprehensive prompt says "Real lens recall (scoring Tier-A anchors with
v4 finetune checkpoint): 73.3% at p>0.3."

BUT: `real_lens_scoring.py` filters `df[(df["split"] == "val") & (df["label"] == 1)]`.
This selects ALL val positives — both Tier-A (confirmed) AND Tier-B (visual only).
**There is no tier filtering in the code.**

**H-train.1a** If the 73.3% recall includes Tier-B (noisy labels), the number
is meaningless as a measure of "real lens recall." Some Tier-B positives may
not be lenses at all. **Verify: does the code filter by tier? If not, what is
the recall on Tier-A ONLY?**

**H-train.1b** Even if we restrict to Tier-A, some Tier-A lenses are in the
TRAINING split. Measuring recall on training examples is circular. **How many
Tier-A lenses are in val vs train?** If only ~130 are in val (70/30 split of
389), the recall is measured on a very small sample: 73.3% of 130 = ~95 lenses.
The 95% binomial CI is [65%, 80%]. **Is this reported?**

**H-train.1c** A hostile referee: "Your 'real-lens recall' of 73% is measured
on training data contaminated with noisy labels. Your injection completeness
of 3.5% is measured on pristine synthetic data. These numbers are not
comparable. The gap could be entirely explained by label noise — your 'real
lenses' include non-lenses that the model correctly classifies as positive,
inflating recall." **How do you rebut this?**

## H-train.2 The Best Model Peaks at Epoch 1

The v4 finetune model achieves its best AUC=0.9921 at EPOCH 1 (out of 60).
By epoch 60, AUC has dropped to 0.9794.

**H-train.2a** This means 59 out of 60 epochs of training made the model WORSE.
A hostile reviewer: "Your 'training' is actually 1 epoch of warmup from a
pre-trained checkpoint. This is not training — it's initialization." **How do
you frame this in the paper?**

**H-train.2b** The v2 model peaks at epoch 19 (AUC=0.9915) and then declines.
The v4 loads v2's epoch-19 weights and immediately achieves 0.9921 — only
0.0006 better. **Is the v4 finetune actually better, or is this within noise?
What is the standard deviation of AUC across random seeds?**

**H-train.2c** For the v5 annulus-fix retrain: if the model again peaks at
epoch 1 or epoch 19 and then declines, **how do we know the peak epoch in
advance?** If we pick the best epoch by validation AUC, we're selecting on the
val set. With 1432 val positives and no separate test set, **is this
selection biased?** (Selection bias: trying many epochs and picking the best is
effectively multiple testing.)

## H-train.3 The Unweighted Loss Problem

All configs have `unweighted_loss: true`. This means:
- All 4,788 positives contribute equally (Tier-A and Tier-B)
- All 446,893 negatives contribute equally
- The BCELoss uses `reduction='mean'`, so each sample contributes 1/N to the loss

With 93:1 negative:positive ratio and mean reduction, **the positive samples
contribute only ~1% of the total loss signal.** This is EXTREME class imbalance.

**H-train.3a** Paper IV presumably handles this differently (they have only
1,372 positives and 134K negatives — 98:1 ratio, similar). **But Paper IV
achieves AUC=0.9987 vs our 0.9921.** Could the AUC gap be due to their better
handling of class imbalance (e.g., focal loss, class weighting, or oversampling)?

**H-train.3b** The `sample_weight` column exists in the manifest and is loaded
by the DatasetConfig. But with `unweighted_loss: true`, it's completely ignored.
**Was this an intentional design choice or an oversight?** If we re-enable
weighted loss with Tier-B weight=0.5, would AUC improve?

## H-train.4 Statistical Precision of the Selection Function

**H-train.4a** With 200 injections per grid cell and true completeness of ~3.5%,
the expected number of detections per cell is 200 × 0.035 = 7. The Bayesian
binomial 95% CI (Jeffreys prior) for 7/200 is approximately [1.5%, 7.0%].
**This is a factor of 4.7× range.** Is 200 injections sufficient to make
scientifically useful completeness maps?

**H-train.4b** A referee: "Your completeness map has cells with 0/200 detections,
cells with 3/200, and cells with 15/200. The error bars overlap everywhere.
**What scientific conclusion can you draw from these data?**"

**H-train.4c** How many injections per cell would be needed to distinguish 3%
from 10% completeness at 95% confidence? (Answer: ~350-500.) **Should we
increase injections per cell to 500 or 1000 before publishing?**

## H-train.5 Missing Baselines

**H-train.5a** A reviewer would ask: "What is the expected injection completeness
for a RANDOM classifier (one that assigns p=0.5 to everything)?" At threshold
p>0.3, a random classifier has 100% detection rate. At threshold p>0.5, it has
50%. **Your 3.5% at p>0.3 is actually WORSE than random in terms of injection
detection.** This means the model is actively REJECTING injections. **Why?**

**H-train.5b** What is the injection completeness using Paper IV's pre-trained
model (if available)? If Paper IV's model also achieves ~3% injection completeness,
the gap is NOT specific to our model — it's fundamental to Sersic injections vs
CNN lens finders. If Paper IV achieves much higher injection completeness, the
problem is in our training.

## H-train.6 The AUC Metric Itself

**H-train.6a** AUC measures discrimination over the full ROC curve. But for lens
finding, we only care about the HIGH-SPECIFICITY regime (FPR < 0.1%). **What is
the partial AUC (pAUC) at FPR < 0.1%?** The model could have excellent overall
AUC but poor performance in the regime that matters.

**H-train.6b** Is the model CALIBRATED? If the model says p=0.8, is the true
probability of being a lens actually ~80%? If not, the threshold-based
completeness numbers (p>0.3, p>0.5) are arbitrary. **Has calibration been
assessed (e.g., reliability diagram)?**

## H-train.7 Injection Physics — Are the Arcs at the Right Brightness?

**H-train.7a** The injection engine draws source magnitude from r_mag ~ U[23, 26]
(AB). This is the UNLENSED source magnitude. After lensing through SIE, the
total flux is MAGNIFIED. For a typical SIE with theta_E=1.5" and source near
the caustic (beta_frac ~ 0.3), the magnification factor can be 5-20×. This means
the LENSED arc has effective magnitude ~20-24.5. **Is this physically correct?**
Real lensed sources detected in DR10 have what typical magnitudes and magnification
factors? If our unlensed source mag is too faint, the arcs would be unrealistically
dim even after magnification.

**H-train.7b** The Sersic profile normalization: `flux_nmgy_r` is the total
unlensed flux. The Sersic profile is evaluated on the source plane and then mapped
to the image plane via ray-tracing. **Does the code correctly conserve flux through
the lensing transformation?** The injection engine uses `_sersic_shape()` to get
the normalized shape and then multiplies by `flux_nmgy_r / (sum of source-plane
shape)`. This should conserve flux IF the sum is computed correctly over the
oversampled grid. **Verify this, especially for high-magnification configurations
where the arc is stretched and might be partially outside the stamp.**

**H-train.7c** The PSF is Gaussian with `sigma_r = psfsize_r / 2.355`. But the
injection engine uses the SAME sigma for ALL three bands. In reality, g-band PSF
is larger than r-band PSF, which is larger than z-band PSF (atmospheric seeing
is wavelength-dependent). **Does the code account for band-dependent PSF?** If
not, g-band arcs are too sharp and z-band arcs are too blurred. Check
`inject_sis_shear` for how PSF convolution is applied per band.

**H-train.7d** For sources with large beta_frac (> caustic radius), the SIE
produces only a SINGLE weakly-magnified image, not a multiply-imaged arc. The
CNN would not detect this as a lens. **What fraction of injections have
beta_frac > tangential caustic radius?** For SIE with q=0.7 and theta_E=1.5",
the tangential caustic radius is approximately theta_E × (1-q)/(1+q) ≈ 0.26".
With beta_frac_range = (0.1, 1.0) and area-weighted sampling, what fraction of
injections produce single images vs arcs?

---

# SECTION I: HONEST OVERALL ASSESSMENT

**I1.** Given everything above: **In your honest assessment, what is the single
most likely explanation for the 70pp gap between real-lens recall (73%) and
injection completeness (3.5%)?** Not a list of possibilities — your BEST GUESS
with a probability estimate. Apportion blame: X% annulus bug, Y% source
morphology, Z% color mismatch, W% evaluation contamination (Tier-B in recall
measurement), V% something else.

**I2.** What is the probability that retraining with the corrected annulus will
materially improve the selection function completeness (>2× improvement)?

**I3.** What is the probability that the 73.3% "real-lens recall" number is
INFLATED by Tier-B contamination (non-lenses classified as positive)?

**I4.** What is the ONE thing we are most likely wrong about that would send us
back to the drawing board?

**I5.** If you were starting this project from scratch with our data and
infrastructure, what would you do differently?

**I6.** Is there a fundamental reason why Sersic injection-recovery CANNOT
achieve >30% completeness for a model trained on real lenses? If yes, should we
abandon the injection-recovery approach entirely and switch to Model B (arc
transplant) or Model C (feature-space calibration)?

**I7.** The H-train.5a observation is troubling: at p>0.3, a random classifier
detects 100% of injections, while our CNN detects 3.5%. **The model is not just
failing to detect injections — it is actively classifying them as NOT lenses
with high confidence.** Why? What feature of Sersic injections tells the CNN
"this is definitely not a lens"?

**I8.** Given the confirmed pipeline gap (30+ scoring scripts don't pass annulus
kwargs), the annulus bug, the parameter documentation drift, and the evaluation
contamination concern — **what is the minimum set of fixes needed before ANY
result from this pipeline can be trusted?** Rank them by blast radius (how many
conclusions they invalidate if wrong).

**I9.** Estimate the total GPU-hours needed to fix everything, retrain, and
produce publishable-quality results. Be honest — is this a 1-week project or
a 3-month project?

**I10.** What cheap (<1 GPU-hour) experiments should we run BEFORE committing to
retraining, ordered by information value? For each, state: what we expect to
learn, what result would change our plan, and how long it takes.

---

**INSTRUCTIONS FOR RESPONSE:**

1. Read `CHANGES_FOR_LLM_REVIEW.md` first.
2. Then read every file listed in that document.
3. Answer EVERY numbered question above. Do not skip any.
4. For code questions, cite specific line numbers and file paths.
5. For math questions, show your work.
6. For predictions, give concrete numbers.
7. Be thorough and sincere. Do not declare things are fine without checking.
   Do not give up on a question because it's hard. Try sincerely.
8. If you find a bug we haven't identified, flag it prominently.
9. If you think the entire approach is wrong, say so and explain why.
