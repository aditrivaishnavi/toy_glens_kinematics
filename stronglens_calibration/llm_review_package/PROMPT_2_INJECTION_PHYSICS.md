# Prompt 2 of 4: Injection Physics + 30% Ceiling + Training Evaluation

**Attach:** `stronglens_calibration_for_review_20260213.zip`
**Prior session:** Prompt 1 covered code audit and pipeline integrity. If
that session found bugs, mention them here: [PASTE ANY PROMPT 1 FINDINGS]

---

## Context (Compressed)

**Project:** MNRAS paper measuring CNN lens-finder selection function via
injection-recovery in DESI DR10.

**CNN:** EfficientNetV2-S, AUC=0.9921 (v4 finetune, peak at epoch 1/60,
loaded from v2's epoch 19 best). Trained on 4,788 positives (389 confirmed
Tier-A + 4,399 visual Tier-B) + 446,893 negatives. 101×101×3 stamps (g,r,z),
nanomaggies, 0.262"/pix.

**Preprocessing (`raw_robust`):** Per-band: compute median and MAD from outer
annulus (r_in=20, r_out=32 pixels), subtract median, divide by MAD, clip [-10, 10].
KNOWN ISSUE: (20, 32) overlaps galaxy light on 101×101 stamps; corrected annulus
is (32.5, 45.0) but requires retraining.

**Injection engine:** SIE + external shear lens model, Sersic + optional clumps
source, Gaussian PSF, additive injection into real hosts, no Poisson noise on arc.

**Injection priors (FROM CODE, not documentation):**
- Source R_e: U[0.05, 0.25] arcsec
- Sersic n: U[0.7, 2.5]
- Source q: U[0.3, 1.0]
- Colors: g-r ~ N(0.2, 0.25), r-z ~ N(0.1, 0.25) (GAUSSIAN, not uniform)
- Source mag: U[23, 26] r-band AB (UNLENSED)
- beta_frac = sqrt(U(0.01, 1.0)), area-weighted
- Lens q: U[0.5, 1.0]
- Shear: g1,g2 ~ N(0, 0.05) per component

**Key results:**
- Real-lens recall (ALL val positives, p>0.3): 73.3%
- Injection completeness (p>0.3): 3.5%
- Bright arc ceiling: even at source mag 18-19 (SNR ~900), only 30.5% detected
- Model 2 (LRG hosts + conditioned q/PA): 0.77pp WORSE than Model 1
- 4-way diagnostic: host type and lens conditioning have ZERO effect (within 0.3pp)

**Previous LLM predicted Model 2 would raise bright-arc ceiling to 50-70%. It didn't.**

---

## SECTION 1: The Previous LLM's Failed Prediction

**Q1.1** The previous LLM predicted Model 2 would "narrow the gap substantially"
and raise the bright-arc ceiling to 50-70%. Model 2 was 0.77pp WORSE. **Was this
prediction fundamentally wrong?** What does this tell us about the reasoning?
Should we trust its other recommendations (Model 2.5, Model B, Model C)?

**Q1.2** The LLM said: "If Model 2 fails, your story is 'injections don't
resemble training positives in feature space.'" We now have confirming data.
**Do you agree with this diagnosis?** If yes:
- What SPECIFIC feature statistics differ between Sersic injections and real arcs
  as seen by EfficientNetV2-S?
- Not just "morphology" — what pixel-level or statistical properties?
- How would you measure this experimentally?

**Q1.3** The 4-way diagnostic showed neither host type nor lens parameter
conditioning has any effect (all within 0.3pp). **What does this definitively
rule out?** What hypotheses survive?

---

## SECTION 2: The 30% Bright-Arc Ceiling (Central Mystery)

**Q2.1** Even at source mag 18-19 (arc SNR ~900), only 30.5% are detected.
**Attribute concrete probability fractions to each cause:**
- (a) Source morphology: Sersic too smooth, lacks clumpy/knotty structure
- (b) Preprocessing artifact: could `raw_robust` normalization interact
  differently with real arcs vs injected arcs? Could bright arcs shift annulus stats?
- (c) Color/SED mismatch: g-r ~ N(0.2, 0.25). Real lensed sources are blue
  star-forming. Is the Gaussian centered correctly? Dispersion too wide/narrow?
- (d) Arc spatial distribution: area-weighted P(β) ∝ β may not match training
  positives' arc positions
- (e) Missing Poisson noise on arcs
- (f) Gaussian PSF vs real survey PSF
- (g) Something else — **what?**

**Q2.2** Design a controlled experiment to determine which factor DOMINATES the
30% ceiling. Provide specific methodology, not vague suggestions. What data do
you need? What script would you write?

---

## SECTION 3: Injection Physics

**Q3.1** Mean arc SNR ranges from 2.3 to 5.0 across the θ_E grid. **Are these
physically reasonable for detectable lenses?** Real DR10 lens candidates must
have higher arc SNR to be visible. If typical injection has SNR ~4, is it
reasonable to expect CNN detection? What SNR do real detected lenses have?

**Q3.2** Mean host q for Model 2 is ~0.83 (nearly round), constant across ALL
θ_E bins. Is this physically reasonable for DEV/SER galaxies? Could moment
estimation (`host_matching.py`) be biased toward round shapes? If q_host ~0.83,
then q_lens ~ clip(q_host + N(0, 0.05), 0.5, 1.0) ≈ centered at 0.83. Rounder
lenses → less elongated arcs → harder to detect. **Could this explain Model 2
being WORSE?**

**Q3.3** We do NOT add Poisson noise to injected arcs. Rationale: real noise
already present in host. But real arcs contribute Poisson noise ∝ arc signal.
**Are injected arcs "too clean"?** Could CNN detect the difference? Quantify:
for a mag-24 arc at z=1 in DR10 depth, how many photons per pixel? Is Poisson
noise significant relative to sky noise?

**Q3.4** Gaussian PSF with sigma = psfsize_r / 2.355. Previous LLM said effect
is "a few percent to maybe 10%." Given main gap is ~70pp, **is this assessment
correct? Deprioritize PSF improvements?**

**Q3.5** Model 2 has fewer populated grid cells (209 vs 220). **Could this bias
the comparison?** Should we restrict both models to the same populated cells?

---

## SECTION 4: Injection Physics — Flux, Magnification, PSF

**Q4.1** Source mag U[23, 26] is UNLENSED. After SIE lensing, magnification can
be 5-20× for sources near the caustic. Lensed arc effective mag ~ 20-24.5.
**Is this physically correct?** What typical magnitudes and magnification factors
do real detected lenses have in DR10?

**Q4.2** Sersic profile normalization: `flux_nmgy_r` is total unlensed flux.
Shape is evaluated on source plane, mapped to image plane via ray-tracing. Code
multiplies by `flux_nmgy_r / (sum of source-plane shape)`. **Does this correctly
conserve flux?** Especially for high-magnification configs where the arc might be
partially outside the stamp?

**Q4.3** The injection engine uses the SAME PSF sigma for all three bands.
In reality, g-band PSF > r-band PSF > z-band PSF (atmospheric seeing is
wavelength-dependent). **Does the code account for band-dependent PSF?** If not,
g-band arcs are too sharp, z-band too blurred. Check `inject_sis_shear()`.

**Q4.4** For sources with large beta_frac (> caustic radius), SIE produces only
a single weakly-magnified image — not an arc. CNN wouldn't detect this. **What
fraction of injections produce single images vs arcs?** For SIE with q=0.7,
theta_E=1.5", the tangential caustic radius ≈ theta_E × (1-q)/(1+q) ≈ 0.26".
With beta_frac_range = (0.1, 1.0), how many are outside the caustic?

---

## SECTION 5: CNN Behavior — Why Does It Reject Injections?

**Q5.1** The CNN was trained on real candidates. **Could training inherently cause
it to reject synthetic injections?** If training positives have specific
statistical properties (host light profile, arc color, arc position, noise
correlation) that Sersic injections don't match, the model could learn to
discriminate "real positive" from "synthetic injection" even at high SNR.
**Is this the explanation for the 30% ceiling?**

**Q5.2** Could the difference between real and injected arcs' "observation
process" create a detectable statistical signature?
- DR10 sky subtraction might partially subtract extended arc light in real lenses
- Flat-fielding/scattered light present in real arcs, not synthetic
- Arc-region noise differs: real arc = Poisson + read; injection = host noise only
- **Could a CNN with AUC=0.9921 detect these subtle differences?**

---

## SECTION 6: Training Evaluation Concerns

### 6.1 Tier-A Evaluation Contamination

**Q6.1** `real_lens_scoring.py` filters `df[(df["split"] == "val") & (df["label"] == 1)]`.
This scores ALL val positives — Tier-A AND Tier-B. **There is no tier filter.**
If 73.3% recall includes Tier-B (noisy labels), the number is meaningless as
"real lens recall." **Verify: does the code filter by tier? What is recall on
Tier-A ONLY?**

**Q6.2** Even Tier-A lenses may be in the TRAINING split. Measuring recall on
training examples is circular. **How many Tier-A are in val vs train?** If
~130 in val (70/30 of 389), the 95% binomial CI for 73.3% of 130 is [65%, 80%].
**Is this reported?**

**Q6.3** A hostile referee: "Your 73% recall is on training data contaminated
with noisy labels. Injection completeness is on pristine synthetic data. The
numbers aren't comparable. The gap could be entirely label noise." **How to rebut?**

### 6.2 The Best Model Peaks at Epoch 1

**Q6.4** v4 finetune: best AUC=0.9921 at EPOCH 1 (out of 60). By epoch 60,
AUC dropped to 0.9794. 59/60 epochs made it WORSE. A referee: "Your 'training'
is 1 epoch of warmup from a pre-trained checkpoint." **How to frame this?**

**Q6.5** v2 peaks at epoch 19 (AUC=0.9915). v4 loads those weights and immediately
gets 0.9921 — only +0.0006. **Is this within noise? What is the standard deviation
of AUC across random seeds?**

**Q6.6** For v5 retrain: if it peaks at epoch 1 again, **how do we know the peak
in advance?** Picking best epoch by val AUC = selecting on val set. With 1432 val
positives and no test set, **is this selection biased?**

### 6.3 The Unweighted Loss

**Q6.7** All configs: `unweighted_loss: true`. 93:1 neg:pos ratio with mean
reduction → positives contribute ~1% of loss. **Does Paper IV handle class
imbalance differently?** Could this explain the AUC gap (0.9921 vs 0.9987)?

**Q6.8** The `sample_weight` column exists in manifest but is completely ignored
with `unweighted_loss: true`. **Intentional or oversight?** If we re-enable
weighted loss with Tier-B weight=0.5, would AUC improve?

### 6.4 Statistical Precision

**Q6.9** 200 injections/cell, true completeness ~3.5% → expected 7 detections/cell.
Bayesian binomial 95% CI for 7/200 ≈ [1.5%, 7.0%] — a factor of 4.7× range.
**Is 200 sufficient for scientifically useful completeness maps?**

**Q6.10** How many injections/cell needed to distinguish 3% from 10% completeness
at 95% confidence? **Should we increase to 500 or 1000?**

### 6.5 Missing Baselines

**Q6.11** A random classifier (p=0.5 for everything) at threshold p>0.3 has
100% "detection rate." Our CNN achieves 3.5%. **The model is actively REJECTING
injections.** It assigns p < 0.3 to 96.5% of injected images. **Why?**

**Q6.12** What is injection completeness using Paper IV's pre-trained model (if
available)? If Paper IV's model also achieves ~3%, the gap is fundamental to
Sersic injections. If much higher, the problem is in our training.

### 6.6 The AUC Metric

**Q6.13** AUC measures full ROC. For lens finding, only FPR < 0.1% matters.
**What is the partial AUC (pAUC) at FPR < 0.1%?**

**Q6.14** Is the model calibrated? If p=0.8, is true lens probability ~80%?
If not, threshold-based completeness (p>0.3, p>0.5) is arbitrary. **Has
calibration been assessed (reliability diagram)?**

---

## DELIVERABLES FOR THIS PROMPT

1. For Q2.1: concrete probability attribution to each cause (must sum to ~100%)
2. For Q2.2: a specific experimental design (script outline, data needed, runtime)
3. For Q4.4: calculate the fraction of injections outside the caustic
4. For Q6.1: verify from the code whether tier filtering exists
5. For Q6.11: explain WHY the model actively rejects injections
6. A ranked list of the 3 most likely explanations for the 70pp gap

**Be thorough and sincere. Show your math. Give concrete numbers.**
