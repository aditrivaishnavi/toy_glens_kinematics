# Prompt 11: Strategic Questions — Answers

## Q1: Journal Strategy

**Recommendation: MNRAS.**

MNRAS is the optimal choice for three reinforcing reasons. First, it has an established precedent for exactly this type of work — Herle et al. (2024) published their characterization of injection biases for CNN lens finders in MNRAS, which means your referees will be drawn from a pool already familiar with the problem space. Second, MNRAS has no page charges, which matters for a paper that will be figure- and table-heavy (4 figures, 3 tables, and an appendix). Third, your paper is fundamentally a methods-validation contribution: you are not discovering new lenses or measuring cosmological parameters, but providing the community with diagnostic tools and a cautionary measurement. MNRAS has long been the home for such work in gravitational lensing.

A&A would be a reasonable second choice given the HOLISMOKES series, but A&A's referee pool skews more toward the Euclid/HSC community. Your work is firmly grounded in DESI DR10, and MNRAS is the natural home for DESI-adjacent analyses. ApJ would cost over $2000 in page charges for a ~12-page paper and provides no compensating prestige advantage for a methods paper. AJ is viable but slightly lower visibility.

## Q2: Paper Title

My recommended title is:

> **The morphological barrier: quantifying the injection realism gap for CNN strong lens finders in DESI Legacy Survey DR10**

This title works because "morphological barrier" is specific and memorable (gives the paper an identity that will be cited), "injection realism gap" names the phenomenon you measured, and the survey name anchors it to a concrete dataset.

Two alternatives:

> **Parametric injections are not realistic enough: a controlled experiment on CNN strong lens selection functions with DESI DR10**

This is more direct and would appeal to referees who value clarity over elegance. The downside is that it reads as a negative result, which could discourage some readers.

> **Diagnosing the sim-to-real gap in CNN strong lens selection functions: evidence from Poisson noise falsification in DESI DR10**

This foregrounds the Poisson experiment, which is the paper's most novel methodological contribution. The risk is that it sounds overly narrow if readers do not immediately understand why Poisson noise matters.

## Q3: Does the Paper Stand Without a HUDF Pilot?

**Yes, it stands.** The paper's contribution is diagnostic, not prescriptive. You are answering the question "are parametric injections realistic enough?" — not "here is a realistic injection pipeline." These are different papers.

The defense has three layers. First, the Poisson falsification experiment is a complete scientific result: you made a prediction from first principles (photoelectron budget), tested it with a controlled experiment (gain sweep), and obtained a clear answer (the gap is morphological, not textural). Adding a HUDF pilot does not strengthen this conclusion; it would begin answering a different question. Second, a hasty pilot would actually weaken the paper. HUDF lacks DESI g-band, has 8.7× finer pixels requiring PSF reconvolution, and needs SED-dependent color transformations. Any shortcut in these steps would allow a referee to argue you are measuring PSF artifacts rather than morphological realism. Third, the paper already proposes a concrete, measurable threshold (linear probe AUC < 0.7) that future work can test. This is better than a preliminary pilot that might not meet it.

Frame the future work section as: "We propose the linear probe AUC as a quantitative realism gate. The natural next step is replacing parametric Sersic sources with real galaxy stamps from deep imaging (HUDF, GOODS/CANDELS), following the approach of Cañameras et al. (2024) for HSC. This is the subject of forthcoming work."

## Q4: The Single Biggest Weakness a Referee Will Attack

**The critique:** "You tested one model architecture on one survey with one set of hyperparameters. How do you know the morphological barrier is fundamental and not an artifact of your specific CNN? Perhaps a model trained with data augmentation that includes Poisson noise, or a different architecture (e.g., Vision Transformer), would close the gap."

This is the strongest attack because it is partly correct. You have shown that *this* CNN distinguishes injections from real lenses, but you have not shown that *all* CNNs would. A referee familiar with the ML literature will know that representation learning is architecture-dependent.

**The defense has three parts.** First, the linear probe result (AUC = 0.996) is measured on the penultimate features of a standard EfficientNetV2-S, which is representative of modern CNN architectures. The features that separate real from injected lenses are not exotic learned representations — they are basic morphological properties (smooth vs. structured arcs) that any sufficiently expressive network would learn. The separation occurs at mid-level features (Fréchet distance jumps at features_3, the 4th of 8 blocks), not at the final classification layer, meaning it reflects genuine input-space differences rather than classifier-specific decision boundaries.

Second, the Poisson falsification experiment is architecture-independent. The per-pixel photoelectron calculation shows that Sersic arcs at mag 21 have ~1 e⁻/pixel — the noise is comparable to the signal regardless of what classifier you use. This is a property of the injections, not the model.

Third, training with Poisson noise augmentation would not close the gap. The problem is not that the CNN has never seen noisy arcs; it is that the CNN has never seen *smooth* arcs (the Sersic model lacks the spatially coherent substructure of real lensed galaxies). Augmenting with noise makes noisy injections look more like noisy real arcs, but it does not give smooth Sersic models the star-forming clumps, caustic crossings, and multiple-image structures of real sources.

Include a paragraph in Section 5 (Discussion) that addresses this head-on: "Our results are measured on a single architecture (EfficientNetV2-S). However, the morphological barrier we identify is a property of the injected sources, not the classifier. The per-pixel photoelectron analysis (Section 4.3) depends only on the survey gain and source flux, not on the model. The linear probe separation occurs at mid-level CNN features (Section 4.2) corresponding to texture and shape — properties that any vision model with sufficient capacity would encode. Testing on additional architectures is a useful cross-check but is unlikely to alter the fundamental conclusion that parametric Sersic sources lack the morphological complexity of real lensed galaxies."
