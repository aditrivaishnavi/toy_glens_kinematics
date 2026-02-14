# Prompt 4 of 4: Honest Assessment + Roadmap + Code Review + Next Steps

**Attach:** `stronglens_calibration_for_review_20260213_v3.zip`
**Prior sessions:** Prompts 1-3 covered code audit, injection physics, training
evaluation, retrain decision, data pipeline, and hostile referee prep.

---

## CRITICAL: Read This First

Since Prompts 1-3, we have made **extensive code changes** based on both LLMs'
findings. The attached zip reflects the CURRENT state. **Do NOT propose fixes
for things already fixed** — instead, REVIEW the fixes for correctness. See
`CHANGES_SINCE_PROMPT_1.md` and `CHANGES_SINCE_PROMPT_3.md` for full inventories.

### Key Findings Carried Forward from Prompts 1-3

**From Prompt 1 (Code Audit):**
- Both LLMs: Annulus (20,32) contaminates galaxy light on 101x101 stamps. Impact
  is host-dependent (Re-dependent). Corrected (32.5,45) config exists.
- Both LLMs: 30+ scoring scripts lacked preprocessing consistency. FIXED via
  `PreprocessSpec` + `scoring_utils.load_model_and_spec()`.
- Both LLMs: Logger warning flooded (~152M times). FIXED with once-per-process flag.
- Both LLMs: Clumps params hardcoded despite YAML. FIXED — now explicit parameters.
- LLM1: v5 config missing freeze/warmup epochs. FIXED.
- LLM1: psfdepth interpretation potentially wrong (see Q_FU.1 below).

**From Prompt 2 (Injection Physics):**
- Both LLMs agree: beta_frac geometry explains ~50-55% of the 70pp gap.
  P(beta_frac < 0.55) = 29.5% matches the ~30% bright-arc ceiling.
- Both LLMs: source morphology realism (smooth Sersic) explains ~15-22%.
- LLM2: missing Poisson noise is significant for bright arcs (12x sky noise at
  mag 18). IMPLEMENTED in injection engine.
- LLM1: psfdepth interpretation flagged as potentially wrong — all reported arc
  SNR values may be unreliable (see Q_FU.1).
- LLM2: observation-process signatures (sky subtraction artifacts, noise
  correlation, flat-fielding) are detectable by the CNN. NOT addressed.
- Both LLMs: real-lens "recall" of 73.3% is NOT Tier-A-only. FIXED.
- LLM2: z-band PSF should be 0.94x r-band (Kolmogorov), not 1.0x. FIXED.

**From Prompt 3 (Retrain Decision):**
- Both LLMs: NO-GO on retraining until Q2.1-Q2.4 completed.
- Both LLMs: Not publishable in current form. Need: Tier-A holdout, one ceiling
  explanation, prior justification.
- Both LLMs: R_e range too narrow. Extended (0.05, 0.25) -> (0.05, 0.50). DONE.
- Both LLMs: Sersic n range too narrow (Herle finds CNN selects n >= 2.55 at
  8-sigma). Extended (0.7, 2.5) -> (0.5, 4.0). DONE.
- LLM1: torch/torchvision version mismatch causes "nms does not exist." FIXED
  via pinning torch==2.7.0, torchvision==0.22.0.
- LLM1: real_lens_scoring.py needs tier filtering + leakage guard. DONE.

---

## SECTION 1: Honest Overall Assessment

These are the most important questions. Be direct and concrete.

**Q1.1** **What is the single most likely explanation for the 70pp gap?** Not a
list — your BEST GUESS with probability. Apportion blame concretely:
- X% beta_frac geometry (area-weighted sampling produces non-arc-like configs)
- Y% source morphology (Sersic too smooth, missing knots/clumps)
- Z% missing observation-process signatures (noise correlation, sky-sub artifacts)
- W% evaluation metric mismatch (different populations being compared)
- V% something else (specify)
Must sum to ~100%.

**Q1.2** Both LLM1 and LLM2 from Prompt 2 assigned ~50-55% to beta_frac
geometry. Both predicted the bright-arc ceiling at beta_frac < 0.3 would jump
to 60-80%. **Do you agree with this prediction? What is YOUR estimate?**

**Q1.3** What is the probability that retraining with the corrected annulus will
materially improve selection function completeness (>2x improvement, i.e., from
3.5% to >7%)?

**Q1.4** **What is the ONE thing we are most likely wrong about** that would send
us back to the drawing board?

**Q1.5** If you were starting this project from scratch with our data and
infrastructure, **what would you do differently?**

**Q1.6** Is there a fundamental reason why Sersic injection-recovery CANNOT
achieve >30% completeness for a model trained on real lenses? If yes, **should
we abandon injection-recovery entirely** and switch to arc transplant (Model B)
or feature-space calibration (Model C)?

**Q1.7** At p>0.3, a random classifier detects 100% of injections. Our CNN
detects 3.5%. **The model actively REJECTS injections with high confidence.**
LLM2 Prompt 2 proposed three specific mechanisms:
1. Most injections too faint (60-70% of rejections)
2. Most bright injections lack arc morphology due to high beta_frac (20-25%)
3. Morphological/noise mismatch for the remaining bright + low-beta_frac (5-10%)

**Do you agree with this decomposition? Can you refine it?** Specifically: for
injections at mag 18-19 AND beta_frac < 0.3 (dramatic arcs, blindingly bright),
what fraction does the CNN still reject, and why?

**Q1.8** Given all fixes implemented (see CHANGES files): **what is the minimum
remaining set of actions before ANY pipeline result can be trusted?** Rank by
blast radius.

**Q1.9** Estimate total GPU-hours to produce publishable results from the
current state. **Is this a 1-week or 3-month project?**

**Q1.10** What cheap (<1 GPU-hour) experiments should we run BEFORE retraining,
ordered by information value? For each: what we learn, what result changes the
plan, how long it takes. Note: several of these experiments are already
implemented (see Section 3).

---

## SECTION 2: Follow-Up Questions from Prompts 1-3

These are unresolved issues or contradictions between the two LLMs.

### 2.1 The psfdepth Controversy (CRITICAL)

**Q_FU.1** LLM1 (Prompt 2) flagged that `estimate_sigma_pix_from_psfdepth()`
treats psfdepth as inverse variance (nmgy^-2), but claimed "Legacy Surveys
defines psfdepth_* as a 5-sigma PSF detection depth in AB magnitudes."

Our code documents: "psfdepth is inverse variance of PSF-flux (nmgy^-2), as
defined in the DR10 Tractor catalog schema." The code also notes that "DR10
brick-summary files define psfdepth differently."

**Who is right?** The DR10 Tractor catalog `psfdepth_r` column — is it:
(a) inverse variance of PSF photometry in nmgy^-2 (our assumption), or
(b) 5-sigma point-source depth in AB magnitudes (LLM1's claim)?

If (b), all our arc SNR values are wrong. If (a), LLM1 was confused between
Tractor catalog columns and brick-summary-file columns.

**Please verify by checking the actual Legacy Survey documentation:**
https://www.legacysurvey.org/dr10/catalogs/
https://www.legacysurvey.org/dr10/description/

### 2.2 Poisson Noise Mechanism

**Q_FU.2** LLM2 (Prompt 2) made a specific, testable prediction: "Missing
Poisson noise gets WORSE for brighter arcs. Brighter arcs are MORE anomalously
smooth, potentially causing the CNN to reject them even more confidently. This
explains why the ceiling doesn't improve with increasing brightness."

We have now IMPLEMENTED Poisson noise in the injection engine. **Review the
implementation in `dhs/injection_engine.py` (function `inject_sis_shear`,
`add_poisson_noise` parameter).** Is it correct? Does it properly model
photon shot noise for nanomaggy-unit images?

### 2.3 Linear Probe Experiment

**Q_FU.3** LLM1 (Prompt 2) recommended: "Train a simple linear probe
'real-positive vs injection' in embedding space. If it separates strongly
(AUC near 1), you have direct proof the CNN can tell them apart."

This has NOT been implemented. **Is this still a high-priority experiment?**
If yes, provide complete code that:
- Uses our existing `feature_space_analysis.py` embedding extraction
- Trains a logistic regression on penultimate-layer embeddings
- Reports AUC and identifies the most discriminative directions

### 2.4 Observation-Process Signatures

**Q_FU.4** LLM2 (Prompt 2) identified several observation-process signatures
the CNN could exploit to reject injections:
1. Sky subtraction artifacts (Tractor over-subtracts extended arc light)
2. Flat-fielding residuals (injected arc bypasses instrument response)
3. Noise correlation (coadd resampling creates spatially correlated noise;
   injected arc has no noise, breaking the correlation structure)

**How significant are these, individually and combined?** Are they addressable
without full image simulation? Would adding correlated noise to the injection
(matching the local covariance) help?

### 2.5 Disagreement on Poisson Noise Importance

**Q_FU.5** LLM1 assigned Poisson noise 3% of the gap. LLM2 assigned 10-15%.
LLM2's argument was quantitative (calculated Poisson/sky ratio = 12x at mag 18).
LLM1's argument was qualitative ("omitting it should make arcs easier, not harder").

**Who is right?** LLM2's argument seems stronger (bright arcs are anomalously
smooth = detectable statistical anomaly). But LLM1 argues the *absence* of noise
makes arcs *easier* to detect, not harder. **Resolve this contradiction.**

### 2.6 v2-to-v4 Improvement

**Q_FU.6** LLM2 (Prompt 2) calculated that v4's improvement over v2 (AUC
0.9915 -> 0.9921) is within noise (0.23 standard errors, p >> 0.05).
**Does this mean v4 is effectively the same model as v2?** Should we report
them as statistically indistinguishable and use v2 as the reference?

### 2.7 Weighted Loss

**Q_FU.7** LLM2 (Prompt 2) flagged that unweighted BCE with 93:1 imbalance is
"unusual for lens-finding CNNs. Most published lens finders use balanced
mini-batches or weighted loss." Paper IV's config is unknown.

Our v5 retrain config also uses `unweighted_loss: true`. **Should we experiment
with weighted loss?** If yes, what weighting scheme? Inverse frequency? Focal
loss? Down-weight Tier-B to 0.5?

---

## SECTION 3: Review of Implemented Changes

The following scripts and code changes have been implemented based on Prompt 1-3
findings. **Review each for correctness. Flag any bugs, logic errors, or
missing edge cases.** This is more valuable than providing new code.

### 3.1 New Diagnostic Scripts (Review for Correctness)

**(a) `scripts/annulus_comparison.py`** — Compares (20,32) vs (32.5,45) annulus
stats on val cutouts. Uses KS tests, correlation with PSF/depth, positive-vs-
negative breakdowns. **Is the statistical approach correct? Any missing checks?**

**(b) `scripts/mismatched_annulus_scoring.py`** — Scores injections/negatives
with v4 model but (32.5,45) preprocessing. **Is the experimental design valid
for answering "does annulus matter?"**

**(c) `scripts/split_balance_diagnostic.py`** — Reports positive spatial
clustering (HEALPix), PSF/depth KS tests across splits, Tier-A counts.
**Any missing diagnostics?**

**(d) `scripts/masked_pixel_diagnostic.py`** — Samples cutouts and reports
NaN/zero pixel fractions. **Is the >5% threshold reasonable?**

**(e) `scripts/feature_space_analysis.py`** — Extracts penultimate-layer
embeddings + per-layer FD. Now with multi-layer hooks. **Is the Frechet
distance computation correct? Any caveats for high-dimensional embeddings?**

**(f) `scripts/arc_morphology_statistics.py`** — Computes high-frequency power,
anisotropy, color-gradient coherence, local variance ratio. **Are these the
right pixel-level statistics to compare real vs injected arcs? Any missing?**

### 3.2 Code Changes (Review for Correctness)

**(g) `scripts/evaluate_parity.py`** — Added TPR at FPR=0.1% and FPR=1% via
ROC curve interpolation. **Is the interpolation approach correct? Any issues
with discrete ROC curves at very low FPR?**

**(h) `sim_to_real_validations/real_lens_scoring.py`** — Added `--tier-a-only`
flag + training-split leakage guard (prints Tier-A/B counts per split).
**Is the leakage guard sufficient? Any remaining contamination vectors?**

**(i) `dhs/injection_engine.py`** — Extended R_e default to (0.05, 0.50) and
Sersic n to (0.5, 4.0). Added literature-citing docstring. **Are these ranges
well-chosen? Should R_e extend further (to 1.0")?**

**(j) `tests/test_band_order.py`** — Three tests for band-order consistency.
**Are the tests sufficient to catch a band swap?**

**(k) `requirements.txt`** — Pinned torch==2.7.0, torchvision==0.22.0.
**Are these the correct compatible versions?**

### 3.3 Documentation (Review for Completeness)

**(l) `docs/MNRAS_SUPPLEMENTARY_TRAINING_LOG.md`** — Sections 13-18 added:
pre-retrain experiments, label noise estimation, success criteria + GO/NO-GO
tree, hostile-referee defense, literature comparison, prior justification.
**Review for scientific accuracy. Flag any incorrect literature citations or
numbers.**

---

## SECTION 4: Next Steps — Reassessing After Model 2's Failure

### 4.1 Model 2.5 (Real DR10 Blue Galaxies as Sources)

**Q4.1** Previous LLM proposed: select faint blue galaxies from DR10, extract
source-plane images, lens through SIE. Specifically:
- What color cuts, magnitude range, morphology types for source selection?
- Deconvolve PSF? Handle sky background?
- Current engine expects analytic Sersic — what code changes needed?
- **Is this the right next step given Model 2's failure?** Will real morphologies
  help if host type doesn't matter?

### 4.2 Model B (Arc Transplant)

**Q4.2** Fit smooth elliptical model to ~200 real lens cutouts, extract residual
arc, transplant onto non-lens host. **Practical with only ~200 lenses?** Label
circularity (arcs from training lenses used for testing)?

NOTE: We implemented `scripts/real_arc_morphology_experiment.py` which extracts
arc residuals via median-filter subtraction. **Review this implementation.**

### 4.3 Model C (Feature-Space Calibration)

**Q4.3** Extract CNN embeddings for real lenses vs injections, compute importance
weights w(x) = p_real(f(x)) / p_injection(f(x)), apply as correction. **Explain
with full mathematical formulation.** What embeddings? What distance metric? How
is correction applied? What assumptions? When does it fail?

### 4.4 Model D (Something New)

**Q4.4** Is there a Model D we haven't considered? For example:
- Could DR10 photometric pipeline effects (sky subtraction, flat-fielding) present
  in real arcs but not synthetic injections be addressed by injecting BEFORE
  the coadd pipeline?
- Could we test by injecting synthetic point sources and comparing CNN response
  to real stars?
- Any approach from the 2024-2026 literature we should adopt?
- LLM2 (Prompt 3) noted Zoobot outperforms purpose-built CNNs in Euclid. Should
  we consider foundation-model fine-tuning?

### 4.5 Priority Ordering

**Q4.5** Rank all next steps by expected information gain per GPU-hour:
1. Run Q2.1 (beta_frac cap) with existing scripts
2. Run Q2.2 (embedding analysis) with existing scripts
3. Run Q2.3 (annulus comparison) — CPU only
4. Run Q2.4 (mismatched scoring)
5. Linear probe experiment (Q_FU.3)
6. Model 2.5 (real DR10 source galaxies)
7. Arc transplant (Model B)
8. Feature-space calibration (Model C)
9. Retrain with annulus fix + extended priors
10. Something else

**For your #1 recommendation: state a SPECIFIC PREDICTION** — what completeness
improvement should we expect? What would you conclude if it doesn't materialize?

---

## SECTION 5: Code Deliverables (New Items Only)

We do NOT need code for items already implemented (see Section 3). We DO need
code for:

### 5.1 Linear Probe Script (if Q_FU.3 says yes)

**(a)** Complete script using existing embedding extraction + sklearn logistic
regression. Must work with the existing v4 checkpoint.

### 5.2 Your Top-Priority Next Step

**(b)** Whatever your #1 recommendation from Q4.5 is: provide complete working
code if it's something we don't already have. Include:
- All imports, all functions, complete `if __name__ == "__main__"` block
- Command-line argument parsing
- Data source URLs (if external data needed)
- Expected runtime on 1x GPU
- Expected output format

### 5.3 Updated Training Config (if retraining recommended)

**(c)** If retraining is recommended, should we use:
- `v5_annulus_fix.yaml` (from scratch with corrected annulus) or
- `v5ft_annulus_fix.yaml` (finetune from v4)?
- Should we enable weighted loss?
- Should we use the extended R_e and Sersic n ranges for the TRAINING data
  (not just the selection function grid)?

---

## SECTION 6: Publishability

**Q6.1** Given all the fixes and documentation we've added since Prompt 3:
is this paper publishable in MNRAS in its current form? Be honest. What is
STILL minimally needed?

**Q6.2** Provide a concrete paper outline: Title, abstract draft, section
structure, key figures, main claims. What can we claim NOW vs what needs
further experiments?

**Q6.3** What is the strongest version of this paper that can be written with
ZERO additional GPU time (using only existing results + the cheap CPU-only
experiments)?

---

## SECTION 7: Literature Verification

**Q7.1** Verify our claims against the actual literature. We documented extensive
comparisons in MNRAS_SUPPLEMENTARY_TRAINING_LOG.md Sections 17-18. **Are our
literature numbers correct?** Specifically:

- Herle et al. (2024): median selected n >= 2.55 at 8-sigma? R_S >= 0.178"?
- HOLISMOKES XI: brightness acceptance criteria (arc > 5-sigma sky AND > 1.5x
  lens flux AND mu >= 5)?
- Jacobs et al. (2019): 4.8% highest confidence grade?
- Euclid: ~70% completeness? Zoobot outperforms?
- Pearce-Casey: 77% TPR at 0.8% FPR?

**Q7.2** Any recent (2024-2026) papers on CNN lens-finder selection functions
we are STILL missing?

---

## FINAL INSTRUCTIONS

1. Answer EVERY numbered question (Q1.1-Q7.2, Q_FU.1-Q_FU.7). Do not skip any.
2. For Section 3 (code review): review each item and report PASS/FAIL/CONCERN.
3. For predictions, give concrete numbers.
4. For code deliverables: complete, runnable files only.
5. If you think the entire approach is wrong, say so and propose an alternative.
6. **CRITICAL: Resolve Q_FU.1 (psfdepth interpretation) definitively.** This
   affects the validity of ALL reported arc SNR values.
7. **Be thorough and sincere. Research the literature. Give concrete numbers.
   This is our final review before committing to the publication path.**
