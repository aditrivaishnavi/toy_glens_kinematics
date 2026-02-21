# Response to Pre-Submission Review Concerns

Thank you for the thorough and specific review. Below we address each concern point-by-point, citing the exact locations in the manuscript (v16) where each issue is handled. We ask that you assess whether these existing mitigations are adequate for submission, or whether specific additional text is needed.

---

## 1. Host Galaxy Confounding Factor (Linear Probe AUC = 0.997)

**Your concern:** The CNN's penultimate layer might cluster embeddings based on host galaxy type (massive elliptical vs random survey galaxy) rather than arc morphology, inflating the AUC.

**How the manuscript addresses this:**

We agree this is the most important methodological caveat, and the paper addresses it in four places:

1. **Abstract** (page 1): The headline AUC claim is explicitly hedged: *"consistent with a morphological mismatch between parametric and real arc morphology (though host-galaxy population differences may also contribute; see Section 5.4)."*

2. **Section 4.3** (page 6): The probe comparison paragraph opens by acknowledging *"only brightness is strictly controlled in this comparison; other properties (θ_E, PSF, depth, host galaxy type) are not matched to the Tier-A sample."* It then reports the Tier-A vs Tier-B control probe (AUC = 0.778 ± 0.062) and explicitly states: *"The true morphology-only contribution to the AUC lies between the host-controlled Tier-A vs Tier-B probe (0.778) and the Tier-A vs injection probe (0.997). The large gap between these two bounds indicates that injection-specific features contribute substantial additional separation beyond any host confound, but a fully host-matched injection experiment is needed for definitive decomposition."*

3. **Limitation 7** (Section 5.4, page 12): An entire paragraph is dedicated to this issue, explicitly calling for *"a fully host-matched injection experiment (matching hosts by colour, size, and surface brightness)"* as the definitive resolution.

4. **Conclusion item (v)** (page 13): The completeness map is framed as *"conditional on the stated parametric injection prior"* and the linear probe is proposed as a *"realism gate"* — a diagnostic tool, not a causal claim.

**Our position:** We do not claim to have decomposed the AUC into host vs morphology components. We provide an upper bound (0.997, includes both) and a partial lower bound (0.778, real hosts only), and explicitly state that the decomposition requires future work. The language throughout uses "consistent with" rather than "demonstrates that morphology is the sole cause." We believe this framing is honest and defensible. A referee may still request the host-matched experiment, but we have correctly identified it as an open question rather than claiming to have answered it.

**Question for you:** Is this level of hedging and the bounding argument (0.778 → 0.997) sufficient for submission, or do you think additional text is needed?

---

## 2. "We Already Knew Sérsic Was Too Simple"

**Your concern:** A reviewer familiar with HOLISMOKES XI might argue the community already recognises parametric models are inadequate, and ask why we didn't just inject real galaxy stamps.

**How the manuscript addresses this:**

The introduction (Section 1, page 1) explicitly positions against both prior works:

- On Herle et al. (2024): *"Their analysis was performed entirely in simulation, without comparison to real confirmed lenses."*
- On Cañameras et al. (2024): *"explicitly noted the inadequacy of Sérsic profiles, though they did not quantify the gap. Neither study measured the discrepancy between real and injected lenses directly in CNN feature space."*

The paper then states: *"In this work, to our knowledge, we provide the first measurement of this discrepancy directly in CNN feature space, combined with a controlled diagnostic experiment."*

The discussion (Section 5.1) makes the complementarity explicit: Herle et al. showed selection is biased in simulation; we show parametric injections are morphologically distinguishable from real lenses in feature space. Together the results establish that parametric injection-based selection functions are both biased and unreliable.

The "why not real stamps?" question is addressed in Future Directions (Section 5.5): adapting HUDF stamps to DESI DR10 requires solving the HST-to-DESI bandpass transformation, the 8.7x pixel scale difference, and PSF matching. This is non-trivial and is explicitly identified as forthcoming work. Crucially, we propose the linear probe AUC as the *quantitative gate* for evaluating whether real-stamp injections actually close the gap — which is itself a methodological contribution.

**Our position:** The novelty is not the qualitative observation that Sérsic is too simple (which is indeed known). The novelty is (a) the first quantitative measurement of the gap in CNN feature space using real confirmed lenses, (b) the controlled Poisson noise experiment that diagnoses the mechanism, and (c) the linear probe AUC as a reusable diagnostic tool for the community. We believe this is clearly articulated.

**Question for you:** Does the current positioning adequately distinguish our contribution from prior qualitative observations?

---

## 3. Annulus Normalisation Mismatch

**Your concern:** The annulus radii (20, 32) were tuned for 64x64 stamps but applied to 101x101 stamps, and reviewers hate known bugs.

**How the manuscript addresses this:**

- **Section 2.5** (page 3): Explicitly states the discrepancy, the geometrically optimal radii (32.5, 45.0), and why we retained the training-consistent annulus.
- **Appendix A** (page 13): Four diagnostic experiments demonstrating the effect is cosmetic:
  - The offset is 0.15 normalised units (1.5% of clip range)
  - The MAD is unchanged (KS test p = 0.648)
  - No correlation with PSF FWHM or depth
  - Mismatched scoring produces a non-significant 3.6 pp recall drop (p = 0.10)
- **Limitation 4** (Section 5.4): Explicitly listed.

**Our position:** The honest answer to "why not fix it and retrain?" is that retraining requires GPU resources and the effect is demonstrably cosmetic for the relative comparison between real and injected lenses (both processed through the same normalisation). The Appendix shows this rigorously.

**Question for you:** Is the Appendix A treatment sufficient, or should we add an explicit sentence stating the retraining cost justification?

---

## 4. Simplified Chromatic PSF

**Your concern:** The injection pipeline uses a single r-band PSF FWHM scaled by fixed factors for g and z, rather than true band-dependent PSFs.

**How the manuscript addresses this:**

**Limitation 3** (Section 5.4, page 12): A full paragraph acknowledges this, noting:
- Real observations have 10–20% chromatic seeing variation between bands
- This limitation is *"shared by most published injection-recovery analyses for ground-based surveys"* (citing Herle 2024)
- *"We have not quantified the contribution of PSF mismatch to the observed gap"*
- *"Using per-exposure PSFs from the imaging metadata could reduce the gap to some degree, and disentangling the PSF and morphology contributions is an important target for future work."*

Additionally, the closing paragraph of Limitations (added in v16) reinforces: *"our results should be interpreted as quantifying the realism gap of standard parametric injection pipelines, not isolating arc morphology as the sole causal factor."*

**Our position:** This is a field-wide limitation, not unique to our work. We acknowledge it honestly and scope our conclusions accordingly.

---

## 5. Small Tier-A Sample (n = 112)

**Your concern:** The recall metric and linear probe are based on only 112 spectroscopically confirmed lenses.

**How the manuscript addresses this:**

- **Table 2** (page 5): Wilson 95% CIs at every threshold (e.g., 89.3% [82.6%, 94.0%])
- **Limitation 6** (Section 5.4): *"our Tier-A sample contains only 112 lenses, yielding a 95 per cent confidence interval spanning approximately 11 percentage points on the recall. Forthcoming spectroscopic campaigns (DESI, 4MOST) will expand the confirmed lens sample by an order of magnitude."*
- **Section 4.3**: Notes the class imbalance in the probe (112 vs 500) and that the low fold-to-fold AUC standard deviation (0.003) indicates stability despite this imbalance.

**Our position:** We cannot manufacture more spectroscopically confirmed lenses. The CIs are honestly reported and the limitation is explicitly stated. The probe AUC of 0.997 ± 0.003 with five-fold CV on 112 samples is statistically robust — even with bootstrap resampling, an AUC this close to 1.0 would not collapse to chance.

---

## Summary

All five concerns raised in your review are explicitly addressed in the current manuscript, with appropriate hedging, quantitative bounds, and honest identification of future work. We do not believe any of these require additional text or experiments before submission, but we welcome your assessment of whether the existing treatment is adequate.

Does this response adequately address your concerns? Are there specific points where you think the manuscript's treatment falls short?
