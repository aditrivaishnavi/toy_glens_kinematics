# Prompt 3 of 4: Retrain Decision + Data Pipeline + Hostile Referee Prep

**Attach:** `stronglens_calibration_for_review_20260213_v2.zip`
**Prior sessions:** Prompts 1-2 covered code audit, injection physics, and
training evaluation. Paste key findings here: [PASTE PROMPT 1-2 KEY FINDINGS]

## IMPORTANT: Changes Since Prompt 1

The codebase has been updated since the Prompt 1 review. Key changes:

1. **NEW: `dhs/preprocess_spec.py` + `dhs/scoring_utils.py`** — All scoring
   scripts now auto-load preprocessing config from the checkpoint, eliminating
   the "scripts forgot annulus args" class of bugs (Q3.1).
2. **NEW: `configs/paperIV_efficientnet_v2_s_v5ft_annulus_fix.yaml`** — Finetune
   from v4 config (both LLMs recommended trying this first, Q1.21/Q1.23).
3. **FIX: v5 config** now includes `freeze_backbone_epochs: 5` and
   `warmup_epochs: 5` to match v2 recipe (Q1.22).
4. **FIX: `normalize_outer_annulus`** validates r_in < r_out + min pixel count (Q1.6).
5. **FIX: Logger warning** fires once, not 152M times (Q1.4).
6. **FIX: `sample_source_params`** clumps params are now explicit, validated by
   AST-based test against YAML (Q1.16).
7. All 46 tests pass. See `CHANGES_SINCE_PROMPT_1.md` for full inventory.
8. **Cheap experiment to add:** rerun bright-arc test with `clip_range=50` to
   probe clipping artifacts (Q5.6).

---

## Context (Compressed)

**Project:** MNRAS paper — CNN lens-finder selection function via injection-recovery
in DESI DR10. Best CNN: EfficientNetV2-S, AUC=0.9921. Paper IV reference: AUC=0.9987.

**The annulus bug:** Normalization uses outer annulus (r_in=20, r_out=32) on
101×101 stamps. This was tuned for 64×64. On 101×101, it overlaps galaxy light
(~20% flux contamination). The corrected annulus (32.5, 45.0) has ~6%.

**The gap:** Real-lens recall 73.3% vs injection completeness 3.5%. Even bright
arcs (SNR ~900) → 30% detection ceiling. Model 2 (LRG hosts + conditioned q/PA)
was WORSE, not better. 4-way diagnostic: host type and lens conditioning have
zero effect.

**Training data:** 4,788 positives (389 Tier-A confirmed + 4,399 Tier-B visual) +
446,893 negatives. 70/30 train/val HEALPix split. All configs use
`unweighted_loss: true` (sample weights ignored).

**Injection priors (code values):** Source R_e U[0.05, 0.25]", n U[0.7, 2.5],
colors g-r ~ N(0.2, 0.25) r-z ~ N(0.1, 0.25), mag U[23,26] unlensed, beta_frac
area-weighted, Gaussian PSF, no Poisson noise on arcs.

---

## SECTION 1: Will Retraining Actually Help?

This is the most expensive step (10+ GPU-hours). We need maximum confidence.

### 1.1 What Does the Annulus Bug Actually Do?

**Q1.1** The annulus (20, 32) contains galaxy light → median biased UP, MAD
biased UP → normalization: `x_norm = (x - median_biased) / MAD_biased`. Center
(galaxy) divided by larger MAD → **contrast suppressed**. Sky at
`(sky - median_biased) / MAD_biased` ≈ negative value (measured: -2.6).
**But the model was TRAINED on this suppressed representation.** Key question:
**does suppressed contrast hurt DETECTION or just change the REPRESENTATION?**
If all images (pos and neg) are normalized the same way, relative contrast may
be preserved.

**Q1.2** The injection scoring uses THE SAME normalization. **If suppression is
the same for real and injected arcs, it cancels out and doesn't explain the
70pp gap.** It would persist with the new annulus. **IF suppression is DIFFERENT
for real vs injected arcs (e.g., because real arcs contribute to the annulus
differently), it could explain part of the gap.** Which is it?

**Q1.3** For injected arcs: arc is at r ~ 5-10 px from center, annulus starts at
r=20. Arc doesn't reach the annulus. Annulus statistics are UNCHANGED by injection.
**For real lenses in training data:** the arc was ALWAYS THERE — annulus stats
include whatever real-arc flux reaches r=20-32. If arc has extended low-surface-
brightness wings reaching r=20-32 after PSF convolution, there IS a difference.
**Quantify: for theta_E = 1.5" (5.7 px), how much flux reaches r=20 px after
PSF convolution with sigma ≈ 2.5 px?**

**Q1.4** **Estimate the expected AUC change from retraining.** Will it be +0.001
(negligible), +0.01 (modest), or +0.05 (significant)? Give a concrete prediction
with reasoning. **What about the expected change in the 30% bright-arc ceiling?**

### 1.2 Could There Be a Cheaper Fix?

**Q1.5** Instead of retraining: apply the new annulus to BOTH host and injected
image for SCORING ONLY (model still trained on (20,32)). This creates a mismatch.
**Would it help or hurt?** The model expects (20,32)-normalized inputs.

**Q1.6** Don't fix the annulus. Report in the paper: "Our normalization annulus
overlaps galaxy light. This suppresses arc contrast by factor X. We quantify this
as a systematic uncertainty." **Would a referee accept this?**

### 1.3 Pre-Training Data

**Q1.7** Cursor LLM claims cutouts store raw nanomaggies and normalization is
applied at training time, so cutouts DON'T need regenerating. **Verify by reading
`load_cutout_from_file` in `dhs/data.py`.** Does it just load and transpose? Any
preprocessing baked into cutout files?

**Q1.8** Manifests contain `psfsize_r`, `psfdepth_r`, metadata. **Are any derived
columns dependent on preprocessing?** If not, manifests don't need regenerating.

---

## SECTION 2: Cheap Experiments Before Retraining

**Q2.1** Run beta_frac diagnostic on GPU: restrict beta_frac_max to 0.55, check
if bright-arc detection jumps from 30% to 80%+. **Zero retraining needed.** If it
fails, the ceiling has a different cause. **Estimated runtime?**

**Q2.2** Extract CNN embeddings (penultimate layer) for 200 real lenses (Tier-A),
200 bright injections (mag 18-19), 200 negatives. Plot t-SNE/UMAP. **If real and
bright injections cluster together → gap is NOT morphological. If they separate →
CNN distinguishes them.** Zero retraining. **Estimated runtime?**

**Q2.3** Compute annulus median and MAD for 1000 real training cutouts with BOTH
(20, 32) and (32.5, 45.0). Compare distributions. **If nearly identical → bug is
cosmetic, retraining won't help.** Zero retraining. **Estimated runtime?**

**Q2.4** Run existing model (v4) on 200 injections preprocessed with (32.5, 45.0)
(MISMATCH with training annulus). Compare completeness vs standard (20, 32).
**If completeness changes dramatically → annulus matters. If unchanged → it
doesn't.** This is a sensitivity test, not a correctness test. **Estimated runtime?**

### 2.5 Success Criteria

**Q2.5** Before retraining, define quantitative success criteria:
- AUC should be >= ?
- Real-lens recall (p>0.3) should be >= ?
- Bright-arc ceiling should rise from 30% to >= ?
- Selection function completeness (mag 23-24, θ_E=1.5") should be >= ?
**Give concrete predictions.**

**Q2.6** If retraining produces AUC=0.9920 (unchanged) and 30% ceiling remains:
**What would you conclude?** What is the next diagnostic step?

---

## SECTION 3: Data Pipeline — Hostile Reviewer Questions

### 3.1 Cutout Integrity

**Q3.1** Cutouts are 101×101×3 HWC (g, r, z). `load_cutout_from_file` does
`cutout.transpose(2, 0, 1)` for CHW. **Has the band order been verified?** If g
and z are swapped, the model trains fine (doesn't know band semantics) but the
injection engine assumes g=0, r=1, z=2 for color computation. **Verify band
order in both training cutouts AND injection engine.**

**Q3.2** Cutouts are in nanomaggies. DR10 coadd images: **are they in
nanomaggies or counts?** If there's a calibration offset, the injection engine
injects arcs at the wrong brightness.

**Q3.3** Are ALL cutouts exactly 101×101? **What if a cutout is at the survey
edge with masked pixels?** Preprocessing replaces NaN with 0. Does the cutout
generation produce NaN for masked pixels, or zeros?

### 3.2 Label Quality

**Q3.4** Configs: `sample_weight_col: sample_weight` + `unweighted_loss: true`.
The weight column is loaded but ignored. **Tier-B weighting is completely
disabled.** All 4,399 Tier-B contribute equally. Could this explain the Paper IV
AUC gap (they use only 1,372 confirmed)?

**Q3.5** **What is the false-positive rate in Tier-B?** If 10% are non-lenses,
~440 mislabeled positives out of 4,788 total = ~10% label noise. **Estimated
impact on training?**

**Q3.6** A reviewer: "AUC=0.9921 with noisy labels. True AUC on clean subset
could differ. **How robust is AUC to 10% label noise?**"

### 3.3 Negative Pool

**Q3.7** Paper IV removes likely lenses from negatives (prior model p>0.4 →
visual inspection). We don't. **How many real lenses in our 446,893 negatives?**
DESI catalog has ~4,800 candidates. Simple geometry → ~10-50 in negatives.
**Negligible at 93:1 ratio?**

**Q3.8** Hard confuser pool N2 (~15%): ring proxies (Sérsic n > 4) and edge-on
(ellipticity > 0.50). **These are Tractor morphological parameters, not visual
classifications. How reliable are these proxies?**

### 3.4 Spatial Splits

**Q3.9** HEALPix nside=128, hash-based. **Are positives spatially correlated?**
Many DESI candidates from deep-coverage regions. If most positives in a few
pixels, train/val split heavily unbalanced. **What is the distribution of
positives across HEALPix pixels?**

**Q3.10** A reviewer: "Your spatial splits prevent field-level leakage. But do
they prevent PSF/depth condition leakage? If high-PSF in train and low-PSF in
val, the model learns PSF-specific features." **Was PSF/depth balance verified?**

---

## SECTION 4: Hostile MNRAS Referee Questions

### 4.1 Fundamental Methodology

**Q4.1** "Your injection completeness is 3.5% while real recall is 73%. You
claim injections don't resemble real lenses. But you DEFINE the selection function
using injections you KNOW don't work. **How is a 3.5% completeness useful to the
community?**"

**Q4.2** "You train on Tier-B candidates, then measure 'real recall' on the same
population. **Tier-A is in the training data. You're measuring recall on training
examples.** This is test-set contamination." Verify: **does `real_lens_scoring.py`
exclude training-split lenses?**

**Q4.3** "Your negatives may contain unlabeled real lenses. AUC is sensitive to
label noise. **What is your estimated label-noise rate in both positives and
negatives?**"

### 4.2 The Selection Function

**Q4.4** "Your selection function depends on injection priors. Different priors →
different completeness. Your sensitivity analysis perturbs by ±30% around
possibly wrong values. **A sensitivity analysis around the wrong point is
meaningless.** How do you know your priors are correct?"

**Q4.5** "Source R_e range (0.05, 0.25)". Real lensed sources at z~1-2 have
R_e ~ 0.1-1.0" (Cañameras et al. 2024). **Your upper limit is 4× too small.**
Extended sources produce different arc morphologies. **Isn't the selection
function biased toward compact sources?**

**Q4.6** "Sersic n range (0.7, 2.5). Real high-z star-forming galaxies often
have n < 1 or complex multi-component morphologies. **Your n range cuts off
disk-like sources.** Impact?"

### 4.3 Paper IV Comparison

**Q4.7** "Paper IV: AUC=0.9987 with cleaned negatives and confirmed positives.
You: AUC=0.9921 with dirty negatives and noisy positives. Gap = 0.0066. You
attribute this to dirty data. But you have 3.3× more negatives and 3.5× more
positives. **Could the gap be due to the annulus bug?**"

**Q4.8** "You haven't replicated Paper IV's negative cleaning. You changed the
positive set, framework, GPU, and normalization. **Too many uncontrolled variables
to attribute the gap to any single factor.**"

### 4.4 Missing Validation

**Q4.9** "No independent spectroscopic validation. Tier-A anchors used in
training. **Where are your holdout confirmed lenses?**"

**Q4.10** "Injection completeness stratified by θ_E, PSF, depth — but NOT by
source redshift. Since source z determines size, color, and surface brightness,
**this is a fundamental missing dimension.**"

**Q4.11** "No comparison against published results. Herle et al. (2024) and
Cañameras et al. (2024) report injection-recovery completeness. **How does your
3.5% compare? If theirs is much higher, what's different?**"

---

## SECTION 5: Publishability

**Q5.1** Is this paper publishable in MNRAS in its current form? Be honest.
Low injection completeness + negative Model 2 + unexplained 30% ceiling + no
independent validation. If "not in current form," **what is minimally needed?**

**Q5.2** If salvageable, provide a concrete outline: Title, abstract draft,
section structure, key figures, main claims. If further work needed, estimate
timeline (single GPU).

---

## SECTION 6: Literature Verification

**Q6.1** Verify our claims against the actual literature:
- HOLISMOKES XI (Cañameras et al. 2024, A&A 692, A72): do they use 1,574 real
  HUDF galaxies as sources? How does their methodology compare to ours? What
  injection completeness do they achieve?
- Herle, O'Riordan & Vegetti 2024 (MNRAS 534, 1093): do they focus on CNN
  selection functions? Key findings?
- Euclid 2025 lens-finding pipeline: what injection methodology?
- Any recent (2024-2026) papers on CNN lens-finder selection functions we're
  missing?

**Q6.2** Based on the literature: **where does our approach sit relative to the
state of the art?** Above, at, or below the MNRAS bar? What specific improvements
would move us from "below" to "at"?

---

## DELIVERABLES FOR THIS PROMPT

1. For Q1.4: concrete AUC change prediction with reasoning
2. For Q2.1-Q2.4: estimated runtime for each cheap experiment
3. For Q2.5: concrete success criteria (numbers, not ranges)
4. For Q4.5-Q4.6: comparison of our injection priors against published values
5. For Q5.1: honest YES/NO on publishability
6. For Q6.1: specific published injection completeness numbers for comparison
7. A "GO / NO-GO" recommendation on retraining, with conditions

**Be thorough and sincere. Research the literature. Give concrete numbers.**
