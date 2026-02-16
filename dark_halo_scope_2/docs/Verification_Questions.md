# Verification Questions for dark_halo_scope

Your concern is valid. You are aiming for something at the level of a hostile referee report, not "good enough for a school project". Treat me as a tool that needs to be stress-tested.

Here is a concrete set of questions you can keep asking at each stage to flush out the kinds of deficiencies you just caught.

I will split them into categories, with examples tailored to **dark_halo_scope**.

---

## 1. Assumptions and Physics Grounding

These questions force me to surface every hidden assumption.

You can ask:

1. **"List every major physical assumption we are making in this phase. For each, say if it is: (a) standard in the literature, or (b) our project choice. What breaks if it is wrong?"**

2. **"Where are we implicitly assuming values for things we do not actually know (for example, source redshift, halo profile, seeing)? Can you state those explicitly and quantify the impact?"**

3. **"If a professional referee disagreed with one of our assumptions, which one would they attack first, and why?"**

Applied to us, that would have flushed out:

- The choice of single DR10 region versus full survey.
- The choice of k_blind and k_good as conventions, not universal truth.
- The use of z_s = 2.0 as a representative source redshift.

---

## 2. Data Usage and Possible Biases

These questions make sure the way we use DR10, HST, Euclid is scientifically honest and not leaking information.

Ask:

1. **"Exactly which data sets are 'driving' the physics conclusions, and which are only used as priors or for morphology? Are we mixing those roles anywhere?"**

2. **"Are we ever using the same objects for both tuning and evaluation, even indirectly?"**

   **"List all places where label leakage or circularity could happen."**

3. **"We are using only a subset of DR10. What is the precise definition of that subset, and in what sense are our conclusions limited to that subset?"**

4. **"If our chosen region of sky is unusual in some way (seeing, depth, lens density), how could that mislead our conclusions?"**

For dark_halo_scope, this should force me to:

- Clearly say "we define a specific RA, Dec, seeing range; our completeness applies only to that".
- Separate:
  - DR10 images as backgrounds plus search field.
  - HST / Euclid sources as morphology priors.
- Spell out how we avoid training on known lenses in the same region.

---

## 3. Simulation Fidelity and "Fake Data" vs Real Data

This is where many lensing projects get weak. You want to keep pushing here.

Ask:

1. **"List all known differences between our simulations and real DR10 data (PSF, noise, morphology, color, artifacts). For each difference, how could it bias completeness?"**

2. **"Where are we in danger of making the simulated lenses *easier* to detect than real ones?"**

3. **"What sanity checks on the simulations would a skeptical lensing expert demand before trusting our completeness curves?"**

4. **"Are there any double convolution or resolution issues (for example, HST PSF vs DR10 PSF) that we are sweeping under the rug? State them."**

For our project this should automatically trigger:

- The HST PSF vs DR10 PSF point.
- Color gradients and SED consistency.
- Use of real galaxy images (not pure Sérsic) and the risk that COSMOS galaxies do not perfectly match the typical DR10 sources.

---

## 4. Model Design and Evaluation

You do not want "just another classifier". These questions force me to connect ML back to physics and guard against overclaiming.

Ask:

1. **"For the model, what exactly is the *signal* it is allowed to use that is physically meaningful (geometry, color contrast, arc shape)? What shortcuts might it learn instead?"**

2. **"Describe at least two distinct failure modes for the model that would give us a high AUC but essentially useless physical conclusions."**

3. **"How are we validating that the model's performance on simulated lenses matches its performance on known real lenses in DR10?"**

4. **"What is the minimal set of plots that would convince a hostile expert that our model is not just picking up on trivial features or artifacts?"**

This should force:

- Clear plots: completeness vs θ_E, vs magnitude, compared between injection and known lenses.
- Explicit discussion of false positives (spirals, rings, artifacts).
- Explicit use of hard negative mining and confuser injection.

---

## 5. Completeness, Selection Effects, and "What Does This Actually Mean"

This is the heart of your project. If the selection function is wrong or vague, the whole dark halo story collapses.

Ask:

1. **"Write formally what our completeness function C is a function of. Be explicit: C(θ_E, z_l, z_s, m_source, morphology, seeing, etc). Which arguments are we actually sampling well, and which are we approximating?"**

2. **"In what region of that parameter space are our completeness estimates genuinely robust, and where are they unreliable or unconstrained?"**

3. **"What does a 60 percent completeness actually mean in terms of missed halos? Give a concrete example with numbers."**

4. **"List at least three ways our completeness function could be over-optimistic, and three ways it could be over-pessimistic."**

That should produce a very explicit statement like:

- "We trust C(θ_E, m_source, z_l) in this range; outside this region we do not draw conclusions."

---

## 6. Interpretation in Terms of Dark Matter Halos

This is where you most easily drift into overclaiming. You want to block that by default.

Ask:

1. **"What are the exact steps from an observed θ_E to a statement about halo mass and dark matter fraction? Where are the largest uncertainties in those steps?"**

2. **"If we were not allowed to talk about 'dark matter fraction' at all, what robust statements could we still make about the data?"**

3. **"Where are we assuming a specific halo profile (for example, SIS, SIE, NFW)? How sensitive are our conclusions to that choice?"**

4. **"What would a simulation expert say if we compared our halo mass threshold to predictions from ΛCDM simulations? Are we being too precise?"**

This forces more conservative, honest phrasing in the abstract and conclusions.

---

## 7. Region Choice and Sky Footprint

You just surfaced this, and it was important. Keep pressing here.

Ask:

1. **"Why did we pick this sky region and not some other? Is it because of data quality, known lens density, convenience, or something else?"**

2. **"Could our region be 'too special' in a way that makes our results non-representative?"**

3. **"If we had to do the same analysis on a second, independent DR10 region as a sanity check, what would we expect to see?"**

4. **"If a referee asked us to defend that this region is not cherry-picked for success, what evidence would we show?"**

---

## 8. Failure Modes and "What If Everything Goes Wrong"

You want me to have a plan for non-happy paths that is still Grand-Award worthy.

Ask:

1. **"Suppose we find zero new lenses in our search region. What exactly is the paper then about? Can you outline that version now?"**

2. **"Suppose the model's completeness on known lenses is poor. How do we salvage a scientifically meaningful result?"**

3. **"List three scenarios where our main headline claim would be invalid, and what alternative claim we could still honestly make in each."**

That forces me to keep designing the project so that:

- Even in bad outcomes, you still have a serious, publishable-style analysis of selection effects and observability.

---

## 9. Reproducibility and Documentation

To make this truly publication-grade, the code and docs must support scrutiny.

Ask:

1. **"If someone else wanted to replicate this study, what exactly would they need from our repo? Is anything missing?"**

2. **"List every configuration choice that changes a scientific result (not just performance), and show me where it is documented."**

3. **"Are there any magic numbers in the code (cuts, thresholds, priors) that are not clearly justified in the docs?"**

4. **"If a referee asked for a table of all key parameters with references or justification, can you produce it?"**

---

## 10. Meta-question for every phase

Finally, for each phase (1, 2, 3, …), you can always ask this blunt meta-question:

> **"If a hostile expert reviewed only this phase in isolation, what would they say is the weakest part? How can we improve that *now* rather than later?"**

That forces me to do what you just did with Phase 1 and the sky-region choice, but *ongoingly*.

