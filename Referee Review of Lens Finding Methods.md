# **Referee Report: The Morphological Barrier: Quantifying the Injection Realism Gap for CNN Strong Lens Finders in DESI Legacy Survey DR10**

## **1\. Introduction and Strategic Context**

### **1.1 The Epistemological Crisis in Automated Discovery**

The transition of observational cosmology from a discipline of hand-crafted catalogs to one of industrial-scale automation has precipitated a fundamental epistemological crisis, nowhere more acute than in the domain of strong gravitational lens finding. As the community prepares for the deluge of data from Stage IV dark energy experiments—most notably the *Euclid* mission and the Vera C. Rubin Observatory’s Legacy Survey of Space and Time (LSST)—the reliance on Convolutional Neural Networks (CNNs) to identify rare, topologically complex systems has become absolute. However, the scientific utility of these discovered samples is strictly bounded by our ability to characterize the selection function: the probability ![][image1] that a given physical system will be recovered by the pipeline.1

For the past decade, the standard operating procedure has been to calibrate these opaque deep learning models using "injection-recovery" simulations. In this paradigm, synthetic lenses are generated using analytic parametric profiles—typically Smooth Sérsic sources lensed by Singular Isothermal Ellipsoids (SIE)—and injected into real survey footprints. The recovery rate of these injections is then taken as the ground truth selection function of the survey.3 The manuscript under review, "The morphological barrier: quantifying the injection realism gap for CNN strong lens finders in DESI Legacy Survey DR10," essentially dismantles this paradigm. By rigorously quantifying the divergence between the feature-space representations of real lenses and these parametric simulations, the authors expose a catastrophic "sim-to-real" gap that threatens to bias population statistics derived from future surveys if left unaddressed.

The implications of this work extend beyond the specific technicalities of the DESI Legacy Survey. They strike at the heart of the machine learning application in astrophysics: the assumption that a model trained or calibrated on simplified physics can generalize to the chaotic reality of the sky. The authors demonstrate that for strong lensing, this assumption is not merely inexact—it is structurally flawed. The identification of a "morphological barrier"—a fundamental deficit in the complexity of the source model that cannot be masked by noise augmentation—marks a turning point in the field. It suggests that the era of parametric simulations for calibration is ending, and a new era of data-driven or generative modeling must begin.

### **1.2 The Specific Context of DESI DR10**

The choice of the DESI Legacy Imaging Surveys (Data Release 10\) as the testbed for this analysis is strategic and highly relevant.3 With approximately 14,000 to 20,000 square degrees of coverage in ![][image2] bands, DESI DR10 represents the state-of-the-art in pre-LSST ground-based imaging. The survey’s depth (r-band 5$\\sigma$ point source depth ![][image3]) and median seeing (![][image4] arcsec) present a challenging environment for lens finding, where the distinction between a compact arc and a myriad of diverse contaminants (tidal tails, ring galaxies, edge-on spirals) is subtle.3

In this regime, the classifier must rely on high-frequency morphological cues rather than gross topology. This makes the fidelity of the training and calibration data paramount. If the synthetic training set lacks the subtle texture of real star-forming galaxies—the "clumpiness," the spiral arms, the asymmetric dust lanes—the network will learn a simplified decision boundary that fails to capture the true diversity of the lens population. This manuscript provides the first quantitative measurement of this failure mode in the specific context of wide-field ground-based data, contrasting starkly with previous works that operated largely within the safe confines of simulation-to-simulation comparisons.1

## **2\. Theoretical Framework and Methodological Audit**

### **2.1 The EfficientNetV2-S Architecture and Training Dynamics**

The authors employ an EfficientNetV2-S architecture 3, a choice that reflects a mature understanding of the trade-offs between model capacity and computational efficiency. Unlike earlier generations of lens finders that relied on massive ResNet-50 or Inception hierarchies, EfficientNetV2 utilizes Neural Architecture Search (NAS) to optimize the balance of depth, width, and resolution. With approximately 20-22 million parameters, the "Small" (S) variant is well-suited for the scale of the training set (451,681 cutouts).3

The training regimen described is rigorous. The use of a two-phase training schedule—an initial aggressive learning rate (![][image5]) followed by a fine-tuning phase with cosine decay (![][image6])—is consistent with best practices for transfer learning from ImageNet weights.3 The final validation AUC of 0.9921 represents a highly converged model, suggesting that any subsequent failures in realism generalization are due to data properties, not under-fitting.

Crucially, the authors address the risk of spatial data leakage with a level of rigor often missing in similar studies. By recomputing HEALPix indices (NSIDE=128) and verifying that the 112 validation Tier-A lenses share *zero* pixels with the 277 training lenses, they ensure that the reported recall metrics are true measures of generalization to unseen sky regions.3 This spatial disjointness is vital because the background noise properties and PSF characteristics in survey data are spatially correlated; a simple random split would likely result in an overestimation of performance.

### **2.2 The Preprocessing Pipeline: A Forensic Analysis**

The preprocessing pipeline, detailed in Section 2.5 and Appendix A of the manuscript, utilizes a "raw\_robust" normalization scheme. This involves subtracting the median and dividing by the Median Absolute Deviation (MAD) of a specific pixel annulus.3 The choice of robust statistics (median/MAD) over mean/std is appropriate for astronomical data, which is often non-Gaussian due to cosmic rays and bright sources.

However, the manuscript candidly admits a "known issue": the annulus radii (![][image7]) were inherited from a legacy ![][image8] pipeline and are suboptimal for the ![][image9] cutouts used here.3 The geometric optimal would be (![][image10]). The authors perform a forensic impact assessment of this discrepancy, finding a median offset of 0.15 normalized units. While they argue this is "cosmetic," it is worth noting that in the faint-end regime (magnitude 24-25), a 0.15$\\sigma$ shift in the background zero-point could subtly influence the activation of the initial convolutional layers. Nevertheless, because both the real lenses and the injections are subjected to this *same* suboptimal normalization, the relative comparison—the "realism gap"—remains internally consistent and valid. The transparency regarding this artifact is commendable and adds to the credibility of the study.

### **2.3 The Injection Engine: Parametric Orthodoxy vs. Physical Reality**

The injection pipeline described in the text and the accompanying injection\_priors.yaml serves as the representation of the "Parametric Orthodoxy".3 It generates synthetic lenses using the standard ingredients found in the literature:

* **Lens Mass:** Singular Isothermal Ellipsoid (SIE) with external shear (![][image11]).  
* **Source Light:** Sérsic profile (![][image12]) with optional Gaussian "clumps" (probability 0.6).  
* **Ray Tracing:** Inverse ray-shooting on a ![][image13] supersampled grid.

The inclusion of clumps is a notable attempt to break the symmetry of the Sérsic model. Specifically, the pipeline adds 1-4 Gaussian blobs with flux fractions of 15-45%.3 This is a sophisticated touch compared to purely smooth simulations, yet the central finding of the paper is that even this "clumpy" parametric model fails to fool the CNN. This failure is significant. It implies that "clumpiness" as mathematically defined (Gaussian perturbations) is distinct from "clumpiness" as physically manifested (star-forming knots, HII regions, spiral arm fragments) in the feature space of the network.

The pipeline correctly handles the flux calibration in nanomaggies (AB zeropoint 22.5) and applies the PSF convolution *before* noise addition, ensuring physical correctness.3 The decision to inject into real survey cutouts (rather than blank noise) ensures that the "host galaxy" noise and neighbors are realistic; the only synthetic element is the arc itself. This isolation of the arc's realism is what makes the subsequent Poisson experiment so powerful.

## **3\. The Injection Realism Gap: Quantifying the Deficit**

### **3.1 The Headline Divergence**

The manuscript presents a stark contrast in performance metrics that defines the central problem.

* **Real Lens Performance:** On the sample of 112 spectroscopically confirmed (Tier-A) lenses, the model achieves a recall of **89.3%** (95% Wilson CI: \[82.6%, 94.0%\]) at a threshold of ![][image14].3  
* **Injection Completeness:** On the grid of 110,000 parametric injections covering the full parameter space, the marginal completeness is only **3.41%**.3

At first glance, one might argue these numbers are incomparable—the Tier-A sample is biased towards bright, obvious lenses, while the injection grid includes faint, unobservable systems (down to mag 26). However, the authors dismantle this counter-argument using a **Linear Probe** analysis.3

### **3.2 The Linear Probe as a Diagnostic Tool**

To test whether the CNN detects injections and real lenses using the same features, the authors trained a logistic regression classifier on the penultimate feature layer (1280 dimensions) to distinguish "Real" from "Simulated."

* **Result:** The linear probe achieved an Area Under the Curve (AUC) of **0.996 ![][image15] 0.004**.3

This is a devastating result for the parametric injection paradigm. An AUC of 0.5 would indicate that the simulations are indistinguishable from reality in the network's eyes. An AUC of \~1.0 means they are linearly separable—they occupy completely disjoint manifolds in the feature space. The network effectively sees two different classes of objects: "Real Lenses" and "Parametric Arcs." It has learned to classify both as "positive," but via different neural pathways.

The authors further support this with **Fréchet Distance** measurements. The distance between real and simulated embeddings is small (0.21) at early layers (edges, gradients) but explodes to 63.58 at mid-level layers (features\_3).3 This localizes the divergence to the texture and morphology abstraction layers. The CNN sees the edges of the simulation just fine, but the "texture" of the light profile is fundamentally alien compared to the real galaxies it was trained on.

## **4\. The Poisson Falsification Experiment: A Deep Dive**

The most scientifically significant contribution of this manuscript is the falsification of the "Noise Texture Hypothesis." This hypothesis posits that the realism gap exists because parametric injections are "too smooth"—they lack the pixel-level Poisson shot noise that characterizes real photon arrivals. If this were true, adding Poisson noise to the injections should make them look more "real" and increase detection rates.

### **4.1 Experimental Design**

The authors constructed a rigorous paired experiment to test this 3:

* **Baseline:** Standard injection (no shot noise added to the arc flux).  
* **Poisson (![][image16]):** Poisson noise added corresponding to the approximate DESI DR10 gain of 150 electrons per nanomaggy.5  
* **Control (![][image17]):** A validation run with effectively infinite gain to verify the code path introduces zero noise in the limit.

### **4.2 The Verdict: Noise Degrades Detection**

The results contradict the texture hypothesis entirely. Adding physically correct Poisson noise **degraded** the marginal completeness from 3.41% to 2.37%—a relative loss of \~30%.3 The significance of this drop is overwhelming (![][image18], ![][image19]).

We can analyze the "Bright Arc" results (Table 5 in manuscript, data from bright\_arc\_all\_conditions.json) to understand the mechanism bin-by-bin 3:

**Table 1: Detection Rates (![][image20]) for Baseline vs. Poisson Conditions**

| Magnitude Bin | Baseline Rate | Poisson Rate (g=150) | Difference | Median Arc SNR (Base) |
| :---- | :---- | :---- | :---- | :---- |
| **18.0 \- 19.0** | 17.0% | 14.5% | \-2.5 pp | 1556.2 |
| **19.0 \- 20.0** | 24.5% | 18.0% | \-6.5 pp | 671.5 |
| **20.0 \- 21.0** | 27.5% | 25.5% | \-2.0 pp | 250.1 |
| **21.0 \- 22.0** | 35.5% | 33.5% | \-2.0 pp | 101.4 |
| **22.0 \- 23.0** | 31.0% | 29.5% | \-1.5 pp | 39.3 |
| **23.0 \- 24.0** | 24.0% | 17.5% | \-6.5 pp | 15.6 |
| **24.0 \- 25.0** | 8.5% | 6.0% | \-2.5 pp | 6.2 |

Data Source: 3

In every single magnitude bin, the Poisson condition underperforms the Baseline. The degradation is particularly severe in the transition zones (mag 19-20 and 23-24).

### **4.3 The Photoelectron Budget Analysis**

The authors provide a compelling physical explanation based on the photoelectron budget.3 For a faint arc (mag \~21) spread over ![][image21] pixels, the flux is roughly 1 electron per pixel (assuming gain ![][image22]).

* The shot noise is ![][image23].  
* The Signal-to-Noise Ratio (SNR) per pixel is ![][image24].

Without Poisson noise, the arc is a smooth, coherent ridge of intensity (say, 0.007 nmgy) sitting atop the background. The CNN can integrate this coherent structure along its curve. When Poisson noise is added, this ridge shatters into a disconnected scatter of bright and dark pixels. The "spatial coherence"—the primary feature the CNN uses to identify the arc shape—is destroyed.

This finding falsifies the idea that "missing noise" is the problem. In fact, "missing noise" was the only thing keeping the parametric injections detectable at all\! The smoothness of the parametric model was an *unrealistic advantage* that compensated for its lack of morphological structure. When that advantage is removed (by adding noise), the emptiness of the underlying model (simple ellipses) is revealed, and detection collapses.

### **4.4 The "Poisson \+ Clipping" Interaction**

A further nuance is revealed in the interaction with the clipping threshold. The registry notes that extending the clip range to ![][image25] (from ![][image26]) drastically improves bright arc detection in the baseline case (mag 18-19: 17.0% ![][image27] 30.5%).3 However, combining Poisson noise with clip20 is catastrophic. At magnitude 21-22, clip20 alone yields 40.5%, while Poisson \+ clip20 yields 24.0%—a massive 16.5 percentage point deficit.3

This suggests that the wider dynamic range, while beneficial for signal, also preserves high-sigma noise spikes from the Poisson distribution. These spikes, which were previously clipped, are interpreted by the network (trained on clip10) as artifacts or non-lens features. This highlights the extreme sensitivity of CNNs to the specific tail distributions of pixel values, further emphasizing the need for perfect consistency between training and simulation domains.

### **4.5 Validation of the Controls**

Crucially, the authors validated their Poisson implementation with a "Gain Sweep" control run (D05). By setting the gain to ![][image28] (effectively infinite), they recovered the baseline completeness of 3.41% exactly.3 This confirms that the code logic is correct—the degradation at ![][image16] is a physical consequence of the variance, not a software bug or a normalization error. This level of rigor allows the community to accept the negative result with high confidence.

## **5\. The "Bright Arc" Analysis: A Study in SNR**

The bright\_arc\_all\_conditions.json data allows for a granular analysis of the detection efficiency as a function of signal strength.

### **5.1 The Peak Sensitivity Window**

Across all conditions, the detection rate peaks in the **21.0 \- 22.0 magnitude bin**.3

* **Baseline Peak:** 35.5%  
* **Poisson Peak:** 33.5%  
* **Clip20 Peak:** 40.5%

This bin corresponds to a median arc SNR of ![][image29].3 This "Goldilocks zone" represents the sweet spot where the arc is bright enough to stand out against the background but not so bright that it triggers the clipping ceiling or resembles a spiral arm of the host galaxy.

### **5.2 The High-SNR Failure Mode**

Interestingly, detection rates *drop* for the very brightest arcs (mag 18-19).

* **Baseline:** 17.0% (at SNR \~1556)  
* **Clip20:** 30.5% (at SNR \~1556)

The fact that clip20 nearly doubles the recovery rate here indicates that the standard clip10 preprocessing is aggressively truncating the signal of these bright arcs, rendering them flat and featureless ("plateaued"). The CNN likely interprets these flat-topped features as artifacts or saturated stars rather than lenses. This is a critical operational insight: the preprocessing parameters chosen for general sensitivity may be actively suppressing the strongest signals.

### **5.3 The Faint-End Drop-off**

The performance collapses rapidly beyond magnitude 23\.

* **Mag 23-24 (SNR \~15):** 24.0% (Baseline)  
* **Mag 24-25 (SNR \~6):** 8.5% (Baseline)  
* **Mag 25-26 (SNR \~2):** 1.0% (Baseline)

This steep decline confirms that the "completeness" of the survey is heavily luminosity-dependent. The marginal completeness figure of 3.41% is dominated by the vast volume of faint, undetectable sources in the prior volume (![][image30]). This underscores the necessity of reporting completeness as a function of magnitude (or SNR) rather than a single scalar value.

## **6\. Comparative Contextualization with Recent Literature**

To fully appreciate the significance of this work, it must be placed in dialogue with the two key contemporaneous studies referenced: Herle et al. (2024) and Cañameras et al. (2024).

### **6.1 Comparison with Herle et al. (2024)**

Herle et al. (2024) performed a systematic characterization of CNN selection biases using *simulated* Euclid-like data.1 They identified strong biases favoring larger Einstein radii and higher Sérsic indices (![][image31]).

* **Convergence:** The current manuscript confirms these geometric biases. The completeness vs. ![][image32] relation (Table 3 in text) shows a clear peak at ![][image33] arcsec, with degradation at both compact (![][image34]) and extended (![][image35]) scales.3  
* **Divergence:** However, Herle et al. operated entirely within the simulation domain. They could characterize *internal* biases (e.g., "the model prefers parameter A over B"), but they could not measure the *external* validity gap (e.g., "the model prefers real lenses over simulations"). The current manuscript bridges this gap. By comparing Tier-A recall (Real) vs. Injection Completeness (Sim), it adds the critical "Sim-to-Real" axis that was missing in Herle et al., showing that even when geometric parameters are optimal, the *morphological* mismatch causes detection failure.

### **6.2 Comparison with Cañameras et al. (2024) (HOLISMOKES XI)**

Cañameras et al. (2024) took a pragmatic approach for the HSC survey, explicitly noting the "inadequacy of Sérsic profiles".9 Instead of parametric models, they used real galaxy stamps from the Hubble Ultra Deep Field (HUDF) as sources for their injections.

* **Validation:** The current manuscript provides the rigorous statistical justification for the decision made by Cañameras et al. By proving that parametric injections are 99.6% separable from real lenses (Linear Probe AUC), the authors validate the intuition that "real stamps" are non-negotiable. The "Morphological Barrier" identified here explains *why* the Cañameras approach is superior: real HUDF stamps inherently contain the clumps, asymmetries, and high-frequency textures that survive Poisson noise, whereas Sérsic profiles do not.  
* **Synthesis:** The logical progression for the field is now clear. The methodology of this manuscript (Linear Probe validation) should be applied to the pipeline of Cañameras et al. If real-stamp injections are truly realistic, the Linear Probe AUC between them and real lenses should drop from 0.996 to ![][image36].

## **7\. Broader Implications for Survey Cosmology**

### **7.1 The Selection Function as a Lower Bound**

The authors argue that their measured completeness map ![][image37] should be interpreted as a "conservative lower bound".3 This reasoning is sound. Since the parametric injections are *harder* to detect than real lenses (due to the lack of high-contrast substructure that survives noise), the true selection function for real galaxies is likely *higher* than what the simulations suggest.

* **Implication:** If we use this parametric selection function to correct volume estimates, we will *over-correct* (dividing by a too-small ![][image38]), leading to an *overestimation* of the lens number density.  
* **Consequence:** This is "safe" for setting upper limits but dangerous for precision cosmology (e.g., measuring the velocity dispersion function). It confirms that parametric calibration is insufficient for the percent-level precision required by Euclid/LSST.

### **7.2 The "Realism Gate" Protocol**

The authors propose the "Linear Probe AUC" as a standard "Realism Gate" for future surveys. This is a highly actionable and valuable recommendation. Before any simulation pipeline is used for science, it must pass the "Turing Test" of the Linear Probe: a simple classifier should not be able to distinguish the simulations from the real validation set. This metric is cheap to compute, model-agnostic, and provides an objective stop/go criterion for pipeline development.

## **8\. Conclusions and Recommendations**

The manuscript "The morphological barrier" is a landmark study that rigorously quantifies a systemic failure mode in modern astronomical computer vision. By combining a large-scale survey application (DESI DR10) with controlled experiments (Poisson falsification), the authors have elevated the discussion of "sim-to-real gaps" from anecdotal intuition to hard statistical fact.

### **8.1 Key Findings Summary**

1. **The Gap is Massive:** There is an 86-percentage point deficit between the recall of real Tier-A lenses (89.3%) and the marginal completeness of parametric injections (3.41%).  
2. **Texture is Secondary:** The "missing noise texture" hypothesis is falsified. Adding realistic Poisson noise reduces detection rates further (![][image18] degradation), proving that parametric models rely on unrealistic smoothness to be detected at all.  
3. **Morphology is Primary:** The barrier is morphological complexity. Real lenses possess structural features (clumps, gradients) that are robust to noise; parametric profiles do not.  
4. **Separability:** The feature space of the CNN allows for trivial separation of Real vs. Sim lenses (AUC 0.996), rendering parametric selection functions intrinsically biased.

### **8.2 Final Recommendation**

The manuscript should be accepted for publication. Its methodology is robust, its controls are exemplary (Gain ![][image28]), and its conclusions are vital for the community. It effectively closes the door on "Parametric Sérsic" simulations as a viable calibration strategy for next-generation surveys and points the way toward "Real Stamp" or "Generative" (GAN/Diffusion) approaches. The proposed "Linear Probe AUC" should be adopted as a standard metric for evaluating the fidelity of all future lens simulation pipelines.

## ---

**Appendix: Summary of Statistical Data**

The following tables summarize the key quantitative results extracted from the manuscript and supporting data files, supporting the conclusions drawn in this report.

**Table A1: Poisson Falsification Experiment (Grid Level)** Comparing marginal completeness across 110,000 injections 3

| Metric | Baseline (No Noise) | Poisson (g=150) | Z-Score (Diff) | P-Value |
| :---- | :---- | :---- | :---- | :---- |
| **Completeness (![][image20])** | 3.41% | 2.37% | 14.6 | ![][image39] |
| **Completeness (![][image20])** | 2.75% | 1.80% | \- | \- |
| **Completeness (![][image20])** | 2.26% | 1.37% | \- | \- |

**Table A2: Feature Space Diagnostics** Metrics characterizing the separation between real and simulated lenses 3

| Diagnostic Metric | Value | Interpretation |
| :---- | :---- | :---- |
| **Linear Probe AUC** | **![][image40]** | Near-perfect linear separability. |
| **Fréchet Dist (Block 0\)** | 0.21 | Low-level features (edges) are indistinguishable. |
| **Fréchet Dist (Block 3\)** | 63.58 | High-level features (texture/morphology) diverge. |
| **Median Score (Real)** | 0.995 | Model is highly confident on real lenses. |
| **Median Score (Sim)** | 0.110 | Model is equivocal on parametric sims. |

**Table A3: The "Poisson \+ Clipping" Interaction (Mag 21-22)** Demonstrating the non-linear degradation when combining noise with wider clipping 3

| Condition | Detection Rate (p\>0.3) | Relative to Baseline |
| :---- | :---- | :---- |
| **Baseline** | 35.5% | \- |
| **Clip20 Only** | 40.5% | \+5.0 pp (Improvement) |
| **Poisson Only** | 33.5% | \-2.0 pp (Degradation) |
| **Poisson \+ Clip20** | 24.0% | **\-11.5 pp** (Interaction Failure) |

*Note: The combined effect is significantly worse than the sum of the parts, indicating that clip20 allows Poisson noise spikes to act as "hard negatives" for the classifier.*

#### **Works cited**

1. \[2301.13230\] Strong lensing selection effects \- arXiv, accessed February 14, 2026, [https://arxiv.org/abs/2301.13230](https://arxiv.org/abs/2301.13230)  
2. Selection functions of strong lens finding neural networks | Monthly Notices of the Royal Astronomical Society | Oxford Academic, accessed February 14, 2026, [https://academic.oup.com/mnras/article/534/2/1093/7755422](https://academic.oup.com/mnras/article/534/2/1093/7755422)  
3. mnras\_merged\_draft\_v1.tex  
4. Selection functions of strong lens finding neural networks | Request, accessed February 14, 2026, [https://www.researchgate.net/publication/384361850\_Selection\_functions\_of\_strong\_lens\_finding\_neural\_networks](https://www.researchgate.net/publication/384361850_Selection_functions_of_strong_lens_finding_neural_networks)  
5. LS (Legacy Surveys) \- Astro Data Lab, accessed February 14, 2026, [https://datalab.noirlab.edu/data/legacy-surveys](https://datalab.noirlab.edu/data/legacy-surveys)  
6. Legacy Survey Files, accessed February 14, 2026, [https://www.legacysurvey.org/dr10/files/](https://www.legacysurvey.org/dr10/files/)  
7. Multimodal Flare Forecasting with Deep Learning \- arXiv, accessed February 14, 2026, [https://arxiv.org/html/2410.16116v1](https://arxiv.org/html/2410.16116v1)  
8. Automatic Classification of Spectra with IEF-SCNN \- MDPI, accessed February 14, 2026, [https://www.mdpi.com/2218-1997/9/11/477](https://www.mdpi.com/2218-1997/9/11/477)  
9. HOLISMOKES \- XIII. Strong-lens candidates at all mass scales and their environments from the Hyper-Suprime Cam and deep learning \- AIR Unimi, accessed February 14, 2026, [https://air.unimi.it/retrieve/cb3205d4-fe23-4f88-bd47-8dd0d3962c30/aa50927-24.pdf](https://air.unimi.it/retrieve/cb3205d4-fe23-4f88-bd47-8dd0d3962c30/aa50927-24.pdf)  
10. HOLISMOKES. XI. Evaluation of supervised neural networks for strong-lens searches in ground-based imaging surveys \- Semantic Scholar, accessed February 14, 2026, [https://www.semanticscholar.org/paper/41fbe16e4a74494a0bc7325274d47877103f2a55](https://www.semanticscholar.org/paper/41fbe16e4a74494a0bc7325274d47877103f2a55)

[image1]: <data:image/png;base64,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>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC4AAAAYCAYAAACFms+HAAABpUlEQVR4Xu2WzSsGQRzHf3ktL+XgwEE5c3BUIrk6uDi4kaOLlLjISXJSSnHgRrn4C1yeg3LlJFyclEjykpC377edfXb2Z9bu7HPTfurTznxnZp9pd2fmESkoKCgo+G/0wj04qBsyUAN3YYep98Ad2FTu4ccUXLXq/bDFqpd5gOumvA0/4XfUnErYl9dDOAk7rdyHcwnG0XfYB69jPQyv8EplHMQnmAW+oTFT5rg1q+w78WqJxpNG+GXVy0xLcPNWK+NgZl1W9hcj5jou/hNN40MHIa6nMufIsnAh+cYl8aYDG/7Qi8qeTO4Lx5R0mINaeKtDjevHmO2rrA1OqEzDcUM6VMxLsAMlwTa93qrgncpkSeJPlwuB9W4rI+Enxe/fxaikv6VwPfGNJsH2G3M9ho+m3GB3CpmF93BFkr/vBXgqwTbnYlOCLTWNS0lecGew2ZS5MXAe3A6d+7fmWdwTJ8MSHFKVknT/zJTk98rlTWdUFsKDoVLq4ZYOfeEkT0yZr4j1o6g5xgBc1GEOeCJXDE+lDXgAl2FdvDlGuw5ykve/S0GBLz8Uc15C8J2A7AAAAABJRU5ErkJggg==>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADcAAAAXCAYAAACvd9dwAAACOklEQVR4Xu2WP0hVcRTHT5lBIkYYJlTgaCTZELiIhAYRSFEaUkNDZhAVNPhnc5GGIFzEwSmQhqgGh6JyaXEQxamShqghKHQpUiwwtXM85+c79/t+9/kc3hL3A1/ePd9zfu/357573iXKyMj4n5hkbbB+sK5DLtDB+kNa94K1O5kuSBfrF+nYa5ArKTJhuV1fsXgxl97kCavbxfOkdRXOS0Nql1w8y5pyccl4xZoA7yXpws87T2JRoNHi786LcZm07gD44tWAt8Ux1lE0I+xCA/hLOlGn846bt+C8Vda6i5tIa+acF+MDJQ8lIN5rNOtIEzLRil3f9AWOfaxTaAKHWePgnabtFx6e0UOYAPCOB6K+P73AW9KTPQL+b4iL5Q3pxA2YMG6R5q9iIsIXimyCUjZ3Dg2jmrVGuUGi5kRFcZSRjpWHPsYwaadcZtVDLkYr6fcddF6teXmbK4Y9aOwAuduFfo6BM6SL68NEhE+sny6WOVI394ByyWeQQ1rQKIA8/E/RLEDqAiPcYX1jPbZYxskvLcE0a8jFY6SFl5wXGGDtRTOF56z74H21z/2kc/S4nBA2t11HjiHj8hrhQzSMd6QDelkXWO9ZHxMV6fSTNgmPdMFRux4h/W7pzp7YnbtL+QcqNXLQgXvm5YEDPVWsR6RvBDcgl0Yb5RaJOms1lRZ75G1FvEHnnTTP18rzL7Ffj8QXXVwycENe/t3xhHnylxMawm2XD8yw2sGTziv1n+1TmlFGRkZGxo75B3I5mdfAZL6oAAAAAElFTkSuQmCC>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC0AAAAXCAYAAACf+8ZRAAABd0lEQVR4Xu2VO0sDURCFR3xgGhuxU1CEiHaCWIjYWwn+ARtJI3aCjZ2Vjz9gZWEj+AMsxU60VERsEhCxs1B84PuMs9dMxuVu7obt7geHZM6dyZ7cTe4SRSKRUDqgN2tm0AdVoW/oGio3LhfHFclFnZplHDpS9SLJ/LLyCueBwkK/kvSXlOf94qPQgDVTaLOGh9DQ+/S/PzX0YGJ+Qc/J+4puUPAOTFjTQ2hoyxTJ/KZd4LAW/l29Q/3GfzF1Fq2EniSZPbYLzKw1EnqhT6rfHtZ0Q0c2eUNvQXskswtmrWn46MpD3tCOTpL5mvF/2aD6bh6YNcuMNTy0GppxuUa0eQKtq3qHpGleeY5VqMuaHkJDP0Gnxvsg+Yw1bW7rQnFO0rwCzUEXJA+NEHyhu6ElVfN/yO2qxnl8JP/h27keaBe6JHk6heKO0DRcmDHjDamar89eVXmF8QjdQTeJbqF7aFj18N07UzXTTnKsctBa8nqoGyKRSCQfP4fCZHP3pESXAAAAAElFTkSuQmCC>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGAAAAAXCAYAAAD0v0pBAAAC9klEQVR4Xu2YSQhOURTHjyFDSciwYIEiNhRJiRVFFiJhZ8NCoSwkKWVhxwJFiYViYZ6jiLIQWQgZIlYohEIZM5y/8+73nXu+e9/wTRbur/69d/93fPd89777PqJE4j9hOOuGNRPd4zelAPwzDrPWUUEAJrPekUTqJmuQn53LHNZ3kroXTZ6jN+seSZnXrLF+dteZzzpiTcVm1kfWZ9YKk1eFwawZVBCAVaxdKn2IZKKmKC/GPNZ+lR5HUtfyg9VDpT+wVqp0N9jA+kQyPuion13jIeuySt9nXVfpKjzNrrkBcAMq8kKEyhxn7VXps6yJKu0I1e0WsQAMpPC44Old4UuO8LxgT3YFuQF4QY2dVgmA3a5OsQ6oNMpgH9RgNZRpf7o1AkywRgliAbhD4XHB0yu9DFeVXpKsetwXsomkQ+yTRbhArTae5lLmueUIbpPss0Xg3fHLmgrbV1liAYj98GJ+WbAjRFeAZiFJRzttRoRZVB8clh+uA7wSgisDPWZt8XLz6UPhhw95ZelmAHBAeZ/pm8nz2E4yqJ+s2SYvj7XkT/BWP/sv/cgvg8FUwQah2clwoP4xa1J8omN+RxhJ0tl5mxHgHOtKdr+D6gNdXitBNCbzsO+PVmVwzKuCC4JrqxXQxglrUnyiY37HKNPheGos0z/ztI/7oSoNrmX+MOPn0ZMa224WtHHSmhRvP+a3BWw5+4znOpxpfM0D1kFrUj0IABMcGzj8jdaM4CYf9FL3zYL6p61J9e8EC7xH1mwHiykcXefhYR1LWCNU+gzJR0oI3Z5t2wF/lDUDhI6srQYBdTF+y1IKtwtvqjXbBRrvq9KTMu+C8twk6MHhiBiaxGesZSqNvR5HUc0Ckl9bGUITAvSqqIJ7l8S+bpGHfwcc2zKvYwwhOWtDb0k62+2VELBk1xvPvbChN9l1kVdCeEKShyMorrf87ChzrWHAD2OaNSOsIXk+fBQ9J/kAfUVyfNa4LRRjvMv6Sq2/9BOJRCKRSCQSNf4Al9rwTU3rEm0AAAAASUVORK5CYII=>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEcAAAAXCAYAAABZPlLoAAACDklEQVR4Xu2XzStFQRjGX8lCIkQsUGx8/BX8B5byF1giFqIsfGRtZyVFsrJQyka2bKQoGxQRRSHf3+/TOXPMec2cj+veu2B+9XTPPDPnvNNz75mZS+RwWPhgLbIGWAes4nD3/+bT1warJNzlQDCx9LGWWG1+u4U1z+oNRuSXcoqeeAdrn7wxC6IvDbh/kLXJahB9AeP0/RNT2g6NyD3VrGMKz8FEP3lrhaKH7GPj0O97ZzVq7YBR1jRrjjXCKgx3552ocOC3Grwprf0YIxPLrFNpAgSCn2omdEtDUMQqlWYMtnA6yew/k9mPoozC92DXutXaAcOUeTg1rBtp+mAHeJFmAmzhrJPZPyKzH0UBa0xr37G6tHbAEGuCvAKz/udMaEQ0deQ9XAfBvAovKbZwrsns75HZjwNvzBbrkDUp+gKwyK0JD8X0ZOOoZ9371wjmTetLiy0cm79DZj9n2CYShQroN8EAW+0zMvu7ZPazAt4/Cba2tAUryXuVbDtCUmzh2NYcHP1NflbAg68MXpqCKhhQxXrQ+tJiq42Nw+RnslslBg/GSVF6SQtW0M/FFwGpNSgtUbXh44uQ3orwsga+ZZxQFe3kFWzWPBs4LzxJ0wcByV0sCVHhnJO3dStqyRuL81TOwGulJgU1hbutxP3/wgEQISUBWzUW3RNfuIYnuWRdsFYp3VwdDofD4fhjfAHO7ZqLHl0dUQAAAABJRU5ErkJggg==>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAJYAAAAYCAYAAAAYuwRKAAAEK0lEQVR4Xu2aaYgURxTHnxqPRIWgaIIaQQVjiBeKEFBwRSRk8QIVD4SAH0KCIKigYAjkixC8QFG/eB8RFEXB+0BFETwQbzQBYyAqJsQLFRUT9f331dt587Z79po9ZrZ+8Gf6/at6put1d1V19RBFIpFIJBKJRCK5Ocd6x7rPGu7KlBGsWyT1fnVlEWEA6yFJjs6yPs4uLucISZ3HrBmurChoxXpj4iUkDT5jPDCH9dbE35PUi2RATlaYeCtJjgYZD8BrGbanhvifTHFx8Jy1zHl/kzS2o/EQf2Fi9X5xXlMG+fA3m/cOsvaYGOwnqTPW+QWNNryT8WYGb3uIx4fY85qS/abKXaqYD39h/Rfiicb7Mni4ocsZSHLHa9eG8bKQ7mJ004ud9zNJQ5eG+HiIPXco2c8FhlQMGQpyN93ExcQCkvyUGq8ra7OJQQlJvYtqYH5ymuTqQ8FLVheSi6u6CW9MPCI5/tYhfhJizw1K9tP4jjWKdZ1kCH7BakYykV1n6hUD2ssv9wUJHCap21eNE+Hzh1DQLcTY/j1s55P+rC0pwl2wibWBtZ61lrVGdqsW3UmOf7XxECddQFco2U9D6+ocDgwN29NCXB+0p4r5y2ce8QC0g/U/a6Qr87Qgaf8Fa/4UPv+g7AS3MdsK7vpCAO1AQi1Ygki6gNDzJPlpfBs+sQ96eqWd2a4Nbb3RwGDYQ1v3+QIDRrnyIdCDnU9604DufoI3GyHPWHO9SelzrNuU7FcG9inxZh6oybHUNTimtOPCVAI9WyrYcbg364AerEXVVFXB0D3GeRgawI+UnJyaPBXquk2+GUcyNFcFzB19nipTVcDQ54dNvbCGOX8na6Hz/rLBFMqdqAMkK7EfhLgf6xDJRHYbybBzL5Q1FHtZXzlvUpCCNnYwsXrY1zKPMm1NQlfuc4EJLxZon1LmhBwl2VefvjEX3B229eTl6h3qGoxISb+vHuZSCnKEebnlE9Yqa1yjil+mdGb1IhlDcUcBXJV9KHuftP3rA8wT/YlRfWjqPSBZXlA+JamjJxrMCh6G1DRQftKbhpskr44U1Mdk+yOSXJcE/xhJ76c0ZA4VHIM+SQM8bMFD56JgMu/zrPra1Ctb8FppjQR8o7HUjzUOxZfXJ75xVp5/SZ7okCiU98wuLuMOSU7SSNsPYJHW/66N07aT4oYAPTpee0HIFY4pqxcKXpqam3qV8g3rN+fhS7AGBgaTDJXFRE1PMpY47BMShmf9LswvMadT7G/gVchVEzcJXrE+I3myUmxSLpMMkxuNV8hgKPAT2KoymmS4U9Dz4ZEdYA5zImyXkry0/TzEp1iTw/b88Fn0zGadp8ziKda47F2JVXsks7fxChk8GdUGvKC9RLL67/9yguUN/BVlCMnv7Ao+JvJ/UvwrT1GTr8XOSCQSiUQiTZP3fHoyEDaQNpEAAAAASUVORK5CYII=>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAD4AAAAXCAYAAABTYvy6AAABo0lEQVR4Xu2WSytFURTHt4ESGRqbKSUzA4/BYWBCGZBioDuiFF9BMvAFfAMZmCpTecwkeQykGHmUDDwmZlirvY72/Z+9z9mDdZXsX/1z1m+du6x9XQdjEonEf6Kdckv5opxCz8c7ZRLlL6G2a83YIc1Sr1DefrpFFoy93zuswdSM0q4d0mhxHNecEM8mMKzBqO7qe2Er1C6v8tU7LEAvCmARRQDVXVleynW/sb8/IeaM/agx3mEBrikDKIV9ygTKAGq7dovcopxT2igb4nzwQyKnMKyCG0oG7sDEz1DddUYkvviT8gHuidLk1IVhEfCTeFiuDynTTq8K1V3HRD66ktgTn8P3LTs1UxgWCR/+hDKLjQpUd+0UuelKYkd8n9T4jjKFYZHw310+/Cg2KlDfleU2uF3xXVIfQfgnxv0rqWPhQ0/J9QVlxOnFoLorS37wuJyJD8HfxPsuluAeOocPn4ErQ3XXHmm4cL0KzmXQ2HvmsRHgmDKOUuAn9BDKAOq7LhnbzP//Xa9v1/FCeaDcUe6lriJDAayhKKHRuyYSiUTiT/ENzWWoW+oR//oAAAAASUVORK5CYII=>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFIAAAAXCAYAAACYuRhEAAACe0lEQVR4Xu2YPWxOURjHHxERhoaEiEE7qaiP+AhDY6KTdDJYjJaWEESIhMnmY+lgIRaDSLeuxjKQDiWCqEQEra+JBDHg+Tvn9D3vv+e5PacRJTm/5El7fs+9p7f/+/bceypSqVQqlRJOag2yjDin9Unri9ZB6jE3tNaznAfGWBDDWj+1HmstpR7zjkXMTa3v4iZDHWpvT4MfdDsaP9K6G43BHa0f0pqrp73917gnrWtApVgkrtfpxwv9ePX0EY5x75vmmoEVZIekJ4FbxlLZJvMbZOCEpK8bjGq9JndJ7ONHxO7NwAoy3BUG7hpLmVuQWFaaWMUig6Yg4a+Q6/U+xR8JEj41ieXnEuQucctFim6tNywzsIIMf8ZnyXd5v488+G+CBHu0npFbpzVFLhcryC3iPPoxK70/TR4UB3mYpdiBWT4EuYEbGfRpTfjvEeLbqFeKFeRucf4o+eXeXyUPioM8wlLswCwfgtzEjUxCmO+5UYgVJJYK+GPkV3h/njwoDpLvErACs3wIcjM3MsH5+CS+4EYhVpALxPkz5Nd4f4A8KA6S7xL4LOlJ4J6wlFaQWItKwblhTdyp9TzqlWIFCeCtpza/S4LiII+zVPZLehK47SylFeRWbswCjp8khzD5AZRLU5DYODwkd0rs47ODDOsDXkpToDcQjS96l2KvuF4/NxrYqPWSpWeH1lOWGVwQdx14GjN43eLrx3iIXAC7NvQXcyOAveYHcW/5r/xX7CmxbYxZIm6i+1oPtL6JW2ti8HDAufFccFZAMZdZEGtZNIBrxxobXwd+R+z/Y8In8Ja4c663t3/zVdrnwvvsR5n9fw2VSqVSqVT+OX4BoPXMhn+ZYGwAAAAASUVORK5CYII=>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAE0AAAAXCAYAAABOHMIhAAADH0lEQVR4Xu2XWaiNURTHV4Z4UIYXyVDkQVzEgxJFGZNQlChDngx59UAekJQ8oEjxJFJIyBB5kLlQXlCmMg8RkjEu63/X3ues8/+G8917KfH96t85+7/W3mef/e3pEykpKSn5NxmqeqP6qbqi6lIbrnBaLOetahHFshimOqMarmqj6qfarDrsk1pAO7G+MO9U81RdVZ1Vs8T6W5RdYu0+UvWhWIXFqq2uvEesEv6kB1778H1OKL+qhjMZL5brVaRePb5K+qDxb0G9ajKyaVRNdGXUnezKFWLDed5JSc6M42I508hnxojl4gmuVXWqDbeIlWIzgfsN4G1QbVeNo1ge6yTZ3tQUr4knkgzwoH0PZUz1yKDgvXReGqNVa9hsBW3FBuygJPsN0rwioN4tNsX83mwyeIpInOK8nqrdrgzGiuVdJ58ZJb930D6Hzz8xaNh7Gfh++0owQyxpCwdSOCWW28ABYqRqr1juPtUn1bmajOIsF+sjyBu0O6qbqktiqwSHRj1Q7yibYj62p1Q2qfarfkj9vQBLBI1d5UAKOFAekoe658krgj9A8gatgyufCF4eONWRc4gDYv5dNhksRSQe44ADS6TesszjhdT/IwyuEp6sQWMGiOWt5gCBHD7sQOEHjMSsDmHaY0a2hrNi7XcnPwtccRaSlzVoWAWeOItuk88gB1sOA38nm1iObMZBw8nnQUfXk/eYykzaA7gWvI7kZ4ErBJ62V7ynxTK4FzzfLq448C44Lw3kZJ2es70xM5j8p6Lnn9oK1RJXBpgp21wZnV3mygDtHCDvW/A9Q1QTyMvjmSTbwFXkI3m4nCJvLvm4JXgwYNzeiBSvCZh+40Tn4WEDjeBgiAPJmuTyojfQeTfErh2RHmI5C5wHYt2ivJdkPu5T98n7ItVrSgSvVai71Hndgudn6QfJ2LuRjNcH6LVYRT97AA+UF/aMyHRJP1HjU4wzDK9WzA6xfacvBwi8t+JCjUs5tobnYks1Ml/sN56Gz4suFhmsesCmVF/5jogN7OXa8N/JRqm+35YUBDOxpBmsEnu5L2kG/dkoKfl/+QVE9/Q6zu5dsgAAAABJRU5ErkJggg==>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAHsAAAAYCAYAAADap4KLAAAEYElEQVR4Xu2ZXahVRRTHV2pqKX5jRRqIpvigL4JEElZEBZkvhgoRF+xBERRUFBT1PpSSKCJCT/WgECIYWWmGlCCKCkFQIH5k9iB+QNm3aVlq639nD2edv7PmnHOV493H84M/98x/zZ5z9szeM2vmirRp06ZNGdigulUoxWOqXmy2CNPZKCvPqcazSeyRykCfpBgYrDrPZguxVdXBZtnoI2EAL3KAQB3U9fDe9nmqS6r/VO9QrBHWqP5QXVW9SbFavC/h951TPUEx8JvqLdVoCff4lOrrqhqBX1UT2SwTr0l+agaos5ZNw/eqpWwqu6T6IfpA9Ysp18sJ1RemfFx1xJRz3FS9aMq4z5dNOXqsJVU1AkMl3089ng8lvHW4iWcoFsETncPrgNRskOrsHIMk3T68IWwSeFv52hkJD+WVqu2q+RRj8PDMYrMsYKBHSrjhCxSLcOdY3lVdYVPZIunr4GEmqJdvxG8H03MO1MGswMDHlB1BH9TLJtU/bFoeV+1QLeKA+G9Ts4gdib+pTkWnfMKmAde8zaZyXdLted/j4dX3fAviX7IpwUfCFfnXfK4Fdhzu9x6Vyg+DLleH5RSVm018qrG+4vdNM7HojyPPgmueZFP8wfB8D6++51sQ/5RNCf7npow3FTPU76qdRfxpE2eS37tY9bPqwaIcn4o4hWCN6Ft8zoEfEm8OQjseh9jIgEx5c/G5v4S2efuUvDED4rwuA28wPN/Dq+/5Eez3Ef+IAxL8M6aMLP8lU54qoc4w41kQG5MymWdVB4vPeBBqgTftgCk/LCFJmGs8SwcbGXCTdqBSHXiDygzXj6TaAp7v4dX3fAviH7MpwT/MJoE66J8UiNWdZKLyQtUIDiTw1pMfVJ+R10iiAbiz1hUeMlPwgmp9JZyE24jEDJ+pZ5AsXn3PtyC+n00J/numHGdeS659+K+y6ZFriMHJlQf2vrEtiNfbHDgxs+tWxP42HCxgJsmBur3ZlLAvTt0jPMxM9fKn+O2kTvIsqONl43OKz28U5eWVcBe5MYI/mU0P7FtXs9lkcKqVGsizUukM72YtqDOWTeVRSV8Pbxl5uQOb2eK3M4W8VVTGQPO1cT2OIG9BGTsmC7zvyIsglspTknAS1CwwJZ9WPS/+WhwTGwjbp1qgXiebBXiDkd1GXpHbO3934e0j34L4AlPeWHgWvEDwsDxGkGDBQ/IZwUzBR6Hc1t6EFxkufixJQ5XvInEQIXuowMRt2AoOJNgm4WzZAw8M3hAkRGhzYHW4a0B+KmIeD0mIf6X6VvW36oGqGiKTJOQxDPIOXIuzAjwQx6rDXSB3Qh1svfD3mtzefgQvzF9seuDN+ZHNJvGIhLV0JgcS9GPDYYDkB6pe0NFlAAkzHqy6wPqEaaiVwD87ap0p50Cy+DqbPZCGH2z81weH+60Eti4NdQLRSHZ+L0FSO4rNHN6+uezgRAnJX3fw1seeRKfc2ezVckyQcgxcd4j78jZt2tw3/A8xEz4B4BWc9AAAAABJRU5ErkJggg==>

[image12]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGgAAAAYCAYAAAAWPrhgAAADmklEQVR4Xu2ZSawMURSGjzmIOcQULFhYSMQQCxEbC7EQkSBiYdoYIpInFuaHhQgLgoUEz0osLEiQGFPGYGEIMWxIENPCPEfC+d+t2+/W3/d2l+73upvuL/lTff57qurePtVV91aL1KhR4/9jtqNqYJj8Y+P9rRoeqxroIWasN7ihUkGBfKxTfVR9VS2ktlzcUk1QtVcNVK1VvU9k/D1tJdxPH0NV18Xsc5baLBEblYpv4PdVZ5z4nuqKE4ewX6Srn4mMwvgh/n76mCjJ3JEUWyI2WpLWqgWqxY66JDLCcOe7ejwArzubHh6p9qi2qvpTWyGsVj0Vf598IA/jd8FFco28iOIWYbSYDq0Xc6UMVvVV9VK1cvJywQO/7fEAvH1serjARhG0EVOcI+LvE9NHTB62Lqdj3yWiOMMI1QFV5zjGlV6v2iHml5CWXaq9bBYAdxwxeyDkM81ZoG/xNm2BcKH68hok248obqSD6oRqjpgdNqsOxm0bYi8NuA2dZ7NA+JyhQoR85qbql5hxvlE9STanZplqWvw5bYGOij8Pt1z2I4obORdvbYE2Om34JfFBQlxiowj4nKFChHzmO8XYp5BZHIprSVugi+LP2ynGH+B4kfM5w8p4i1kSH2ipxwuBvDF5lPZ2yecMFSLk5+OwmP3sLT0NXNC0BTok/rzdYnzMMi2R8zkLJF8l70PspwF5U/MI65A08DlDhQj5+agXs98s8kNghT+PvLQFCj2D9ku2H1GcAMmTPd4q8kI8ZKMIuOOfPB6A94BNwlfE7bE3ifwQW8Tcwl3ZdZCNQ4wXk1fULA5XEifPcLyeYn6SueinOsZmgXBfZno8AA/TehesUVyQgzcJLlh/8PGwFJhLXi5eSPYxwFjVKPKQN528z6q35EUUZ7gr2Sc77niv3IYcYJqN1zHFwn0B8BY58bbYc3kXe0scD32qc2L7ZqHB8QA8COucNIRu//Y4LqfEzCItWA8iZ4jjgYjiDNgZA3FpJ00nS7Nat2Aaar9MzFA6JZtTwQMEHcX4eKF4R8zMjBe+WM89Jg+cFLMv3uNhuzzZ3MgaMROl+dxAYG34WvVc9Uz1UsztzoLpM8Sgz1+k6dnlu71GbLQkg1QrxDyQN4lZX/V2E3LgK1ApmKIax2YJidioVMpVILyzKycRG5VKOQqEvyOa4/lZDBEblQoK1C1WqcAstFxgfYixXuaGSgXPLqtqAG/9q2m8NWqUiD/sWgJlM/SC4wAAAABJRU5ErkJggg==>

[image13]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABkAAAAXCAYAAAD+4+QTAAAA5UlEQVR4Xu2TsQ5BMRiF/3ewEqsH8BJGVhuJWAxGHsA7SOwWQgwkJAxCJBK72WKwmjl/bhu9B/d2cAfJ/ZIvaU+bnqRNRVL+hTx8cPhrtCDRkiW8S4IlWTiFN/le0uKAKHLA2IOjStpwyKGhDDccuoxhzoyjSpQOnFBWgVvKQmTg2pnHlShdCa5W0YKds/YRPtCnRNGiPTzwAtOHBcp8S+rwDOe8wCwkuEtX+090PHhtDaEF9g2acOaseRH3GWvy/shaZN/Ii6iSKlxxaGjAEYfMCV7hxajjY2iHSI/mTImDlJTkeAI/mDlzbGp0ygAAAABJRU5ErkJggg==>

[image14]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADwAAAAYCAYAAACmwZ5SAAACIUlEQVR4Xu2VzUtVQRjG38IwF0aLTGgluXAlLUJcuTAiXFiIRC5ctlD8E0r3pX+AGwmRaFNBez82UX5AUH6FIoYfSLrQTMlSIZ+Hma7je2buvedcgoucH/w4Z553zjkz954zI5KSkpLyf+iBP+Ev+FjVsnEJfoJ/4SZsPFsuTubhsNOehR+cdogrcNVp3xIz8bdOVjBNOigQDpqD1DC7qkPFmJh+DU7Gtu9+ibkG9+ArXUjIZ/EPkNmADhWdYvpVOJl3wjWwX05/wdvwHazL9MhNOfwOP8ILqhYH7wAlnGejTMw1k27IwU3DO7a4Altt7QgO2vN8KRHzzX0T88C4hCYWykNUiem/pnJ5DS/C+2I63HRqz2yWlFG4C6/rQhZCEwvlPp7DF2L6P1U16bbHCYne8I0nS8JL+AdW64KH0MRCeS54zYEOCQtfPVmSh2h6xdynXhc8hJ4ZynPBN4zXdegCwzZPtqGyOAzBQzGLYr5wxfdNzPeHaDhWrkEuI2Ku5THDPRu6tNvsssrzgTffFrNlxeWRRMdCmHH3cHmi2r63YN1m3LIyjNuw2ba5iLHNh+cLV+c5uARLVS0ueoB9NnPZsVmXkx3DB06bsE/kG2a4CGfs+T68caZHGG49/BXf60IB/Ns/p+AX+Fuie3stXFYZ2RJz7YI9ck4RWOArnISHOih2+Aro1+Vc80NOv4VKVTuX3BWzSrdI9DtJSUkpbk4AO0qQirmvcLIAAAAASUVORK5CYII=>

[image15]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAXCAYAAADUUxW8AAAAQUlEQVR4XmNgGAXIgBGIQ9EFiQWcQFyCLkgs4AHiUnRBYgEvw6DX/J9ELAvRhhsQbTM2MNw1o4cmISwN0TYK6AcAKikdhdQm6wYAAAAASUVORK5CYII=>

[image16]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADkAAAAWCAYAAAB64jRmAAAA00lEQVR4XmNgGAWjYBSMglEwCkYBKcAciJcDsT26xHABH4F4MpQ9B4j/AvF/hPTQBz+A+CmaGMiDS9DE0MFiPHghEC8A4nlAPBeIZwOxCljXAIA8BoiHRJDEmKFiWkhiQxqAPIOeLEuxiA1pAPLMNzSxz1BxQqCLRKwN0UZ/APLMfixia9DEhjRoZkCNtX9Q/oCFOq1AMRC/B+J2hmGYH7GBLwzDzJOgvPgTTQzkwQI0sSENQB66AGXzQvnHENLDA3AD8TQg3gXErUDMhio9CoYEAABaRTYkC2MkMgAAAABJRU5ErkJggg==>

[image17]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEYAAAAYCAYAAABHqosDAAACBklEQVR4Xu2XSygFURjHvyiJCLFkZUW22LGQLOwsLJRY2Epkw1JSkgVl4xFRNhI2ImWhSFFWNnYepRAir7y+v3MmZz4zd+69rvtw51f/nPP77ozjczozl8gnqRnmZAhXyXnjvHKq7aX/TxunifNB9sbkchqMOeplxjxpkI1p1s7ikbNnzBMCrwVPkPojjzlFomYhGyNBfUjKeGSX1GKtuPHOqTXm+GydMbeAz5RSU0KBf0dc0knui+6jn7V6BwfgsqTUvEiRCARqDPyhlKR8oYPLFg5cGeNFY+xIBWeeUyULMcCrMRtSkvIjDi5HuDtOC6eV087ptVUFt5xRPcahhue828KigVdjVqQk5Vf1uIdzw7kktTuetLfua8b1cY2LzoTDBXPCSWYDZIYzzZniTHLGOcVfVwWHW2NSSHmn7Q9/JGW4YCvhhvmGS9UOp3ascGsMgF+SkpTfkjJcrO1k0u3goo1XY9akJOWxMyMCbvYgHA4nt0WZDIaYUnVZUHg1xu2p1ChluOBmmw5uQbhoE6gxaIqslTu4XyFflvBGiXko/92/ADsM6yiQBSaPVC3dcNjl+8Y8InRxrjkDFPvzBW+j55xTzon+eUHqSWdSQ2qdy6TWvmMvR557im1j4gKcLc/CoSkdwiUdaMKBHuOLFubb3+XkBV/HxzjrnH5Omr3s4+PjEzafApSYhWk16eoAAAAASUVORK5CYII=>

[image18]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEYAAAAYCAYAAABHqosDAAAB00lEQVR4Xu2XzSsFURjGn3xkochO2fgblIX1XVjYSFlYyNeK8gfwDyg2IkU+ImUlCzs7xdKCssPGN1mgEBLv65wxZ945c+fM5t4p51dPM+8zz9Q977nz3rmAx+Px/G/upJFAFelbmik0k26h7tuOXsonh1AfNpAL73DPMvOkT6PeIu0Yda7hXXRZ7DjpHG5ZpoB4lutH4eUWl8ZUQjVlE+nZAM6dCK9R1BHaSBukCl3zcSS8XHJcGvOmj66NqYPKzeq6nVQdXo4zgOhz3QH13JaTtMaMkjr1uWtjhqFyY1D31JJ2SR9GJsKpqHkwNQjPxnoRrZFWSSukZdIiqeb3LjfSGnNvnLs2ZgHh5ptwfSS8GAekJmmWgWKNkYPStTFTUDkeGSZf2k+EBxk/h3kgqTE9pH7huTamDyo3JPxn7VvhXeApn4XJjMrS9KTGTJD2hIL3mKBOoh4qx7PG5EX7MV6lgYRgCUlqjI1r2LOtpBbhcW7J4sXuv0T4VbohHevzXjNUBvahPofLwH6CZWGwL3jQ4nFdMA1ePP9UM/zeEgyhrr9E6eF3E/4Pwxt2QboiPSA+F5hpqP9TQZY31nzNmNOSzECt80wfu6OXPR6Px+Px5JwfZouSGDJo4d0AAAAASUVORK5CYII=>

[image19]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFAAAAAYCAYAAABtGnqsAAACg0lEQVR4Xu2YO2gVQRSGT3w/EHwU4iMBFRQbER+IICIqwSKGKD4KETsLKztBray0TUQrUbAQMQSxELGwkKCBBAstlAiCiFooIj4SUXycn3MmzD135mbXXTdXmA9+duY/c3d2z52dmV2iRKJk3pn6F9Y11mXWJVUiwn7Wb6++mvWc9YB1X3XLiycMZ6k2gSe8Mvhg6k3PFNZ3a3rMZt0luelBVkttOBev9egn0Oc0a7I1m5VnJDfiFGIJSWym1hdofdJYi+zsYs3Scqy/t9aYCLZZYxw+UfyGvrKuG2+I9c2rX2WNNpDjglcO9XeOdcqaVXKHZP7AKMlDowTCP2C8k+rnYQPrnif8HselXht4aBdlFesia67W17NusjaOtcjPVJJH8QVruollJZbArST+FuMfUX++8fMQ6g9eqzUdmHgfs7aTNHzJ2quxH6wrWs7KPNZ71kMqNqmDWAKPk/jrjO+2IZuMn4VFrI8k147jSi+Gc7o5so4bJBPvbpKGy72YXdYbsYxk/umzgQLEEniGxF9j/D3qHzL+PwXLMxig+ovtDXgWPO6/WOdtoARiCTxK4q81/j71dxi/EtDx04AXugGfdpI2mMDLJpZANwduNv5h9bHFqRx0fDDgvTFeDDcSe2ygALEEYlGCX8YqXApuFPlgHoE3w/jj0UayR8PjX5RYAgH8buPdVr9ysGKi4w6tY1EJ/cN5mEMyevttIAcjFE9IaLSh7nYQlYKOh1lPtIxPOItrWvw9eH98RHL+aSYW4zPJq9MrFd5TsRFf4Tci+cz0U4+4bmxvJgR0XsXSX2RENy2dVP8oJHKAXTcSeIy10MQSGdhJsgp3UfFXr0Qikfif+APC3qQDN9LNrwAAAABJRU5ErkJggg==>

[image20]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADUAAAAWCAYAAABg3tToAAABNUlEQVR4XmNgGAWjYBSMglFAHvBAFxgOQASIPwPxMnSJgQbqQDwdiAWgfGMg3gDEpnAVhAEvEL8A4mNAzIgmR3cAcsAlIHYC4v9A/BCIg6Byv4F4AZRNLGAB4itAfB+IOdHk6AZWAzETEPsyQDylhCTXARUjF+wF4o9ALIYuQWtQA6VPMGB6YA0WMXLAEiD+CcTK6BK0BiDHX8ciRg1PdTFAzDFHl6A1AFkajkXsGZoYKWAREP9igBREdAduDJgxEg0V40ATJwbsAeJ3DJDifsDAcQaIB3ygfFDBAeKHwVUQBqBS7yoQ3wFidjS5AQEgD9wC4stQ9hcglkJRgRuAiu0nQHwYXWKgAcgjoORGDghBFxgMwI8BMz8NefCBAeKpLCAWR5MbssCFAVL6BTAMgjbbKMABAHtvOBn4qBGVAAAAAElFTkSuQmCC>

[image21]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACgAAAAXCAYAAAB50g0VAAABpUlEQVR4Xu2UvyuFYRTHD0mJLGRB3WIxKIONKFlMJOxKyq4MfqRMyqIYDLLIQrESKRlYlPxI8g9YlPKj/D6nc957z/P1uu5yB3o/9e0+53vO857nPr3nJUpI+N9Msz5Z76yJMJWmnnVMWrcLubyyx+pw8Tjr3sVCO+nBIpogzhuFrFs0SZtXQzziYuGFdQRemgZWLZoxFKAB9FD8TYiXsnWVxfLr2TE/IGXmB+vJ1sO+wFHCakYTKCV9hqjOPPnzvvEUxBErFOPLwZB91iurBvxniH9ikzKHPGE9hGnashyySDF+FxpGBekERo1ErUFFds4p3NvocgfmIfOkvn9Xc6IIjV94ZPXb+poyhywzb81iZIHU/9Zv1hKidcghbWgAq6xD8OTm5dkXFv/0Di5TjC9jPePiJdKiXudFjLGK0QRkbzQcnknKNG+xdU5TPIeGcUZaPMrqJn2nroKKeGRPH5rMEOvGxXGXIMN0B17WGyknHf1L0ga5MEAxt0Dq+V7brDcXy/dValLOyxuDpM3kRuQQsq4MKpRT0oHaIK3pDNMJCQl/my8pVGxtbYkgjQAAAABJRU5ErkJggg==>

[image22]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADEAAAAXCAYAAACiaac3AAABzUlEQVR4Xu2WPSiFURjHH0KUDKy+YpKBwSgLi0mZjZJNomRTDBQrmSiZTJQiZDHIiKRsJBP5KPLt/3TO0fM+99xz7ytXhvdXv7rP/5zz9j73Pfe8lyghIeGvKYAvOrSswllYD/NgE9yAzXKSZYbMda5hqxrLGafwU+hjj6Jz2OXIDMMdnBD1E5wUdc65p/RN7MJBuASH1JijnVLXl3uybxpglQ498KPPllAT25T5WryFfOs565FBrQ0/4KP93CcnCEpgiw4DhJrYosxN8NpXHZLJD2XAN6/hR82LK1XO+zEOoSY24QCZb3uRzLx+OcFmfA0N55F76ZSFoAK+k1ngjHsyhJpYg2OiLiYzt0NkXN+I2uHuJxZ8VP6EUBM+9M3x51tROzhP2UFTdoBdUWOaNh0ECDVRqAPyN/EsagfnZzLYh+OiniczqVtkjhFYpMMA6ZqoJpOvq9zXhG89Z3MymJaF4IjM5GHYBY/JvMTikK6JOjK53P8MZ2+i5je6Xs8nGmeRJxn6ZsvgAjyBvWosG9yR7YPzfFGP2ox/4BLOGkV9QP4T69d5gFfwwnpJ5pTh/0kOboCPVz7K3bYpFeOOGjJjO/CczLUSEhIS/jlf8NKIbH9rj/EAAAAASUVORK5CYII=>

[image23]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAKQAAAAYCAYAAAB0vVZPAAADxUlEQVR4Xu2ZW8gNURTHl/uL+6U8oZTcr6UUOQglL4qSB0/kkgfxIk/zRPFCkkK+XJIHuUSh1IdE4skDeSGX5B7iAYX1b80+s2edfebMHJ85M+fbv/p3Zq+1Z5+1Z6+Z2XsPkacdqbD+pJSnxUxgdVLtwKRVGShLnB7mOCvQxjYj0AZPMVnEOq+NbYZ/OpaIp6xp2thmBNrgKR5vKT4P7Bl3027LB32Nu0tD0tNxMCX78+YkyXy+W7KL5ALcIhmUa3F3lSINWDMEqjyC9YKKsyi7zfpNUSwT4+7uwXjWddYlVm+qPzA9WPe1sUS4+mRTr9+tYCblmJCzWMO1sYW8Z82jaDAOhsebqzWEjaxlylYmAm1QlDYhB7KWkgyOrUYsIPmTn6xf4fH8WI0IJEge3GNtCY8/W3bX4LxW5TKh++LC1edWkSohR1MUtEsboqpOvmgDyar2E6u/sudxYZCIR8Ljqaztlu8ZSQyjLFseMTULYqtoo0Wa2M04pmUfyXVKq4dyWioaJuQAkgpbLds7kgloGjCwg7QxZDLVJvfIWI10BCTnblN2F6Y/s8PyOVavyE1DSfx4nRvuWMfNgoWTSydINuQ7WMdYRym6WZKoUPy6uahn1yS1kTcmISdphwHOC8q2LrS3GiQg4ugMf6E5sRq1oM4BVdbYA7SJtdjyFQ3Tdxf17JoiJuQU7QCYH7oCPUNuexIXKer4fuWz6UPxJ1ZaHpG0/Uo7LFZT7T6iLoNVJG2dInkbFB1XQgWqnITr/FZhEhJTqRqukjtQ2OyFQCO+sdaHx0g2bLOgDdeXkcvakJIlJHNStIu9RQ02Wh8rGzq/Q9kMZpBc/e9H8l8drCfKV489GZUFV5y6nITr/CQw1fiRQR/ktFSYhJyuHeA01QaKbRvYsOJOQ1/Wcm0MQaBoay3J0wtfTfD0bRbsKaK95xRflAC0u0LZrpDE5wJzO7R1U9nRb/uadMX88l+pkMTUGZaDqicdWRPyf2IScoZ2gGEUDxSf1VBeY9ka0Shxx7DOsh6wFsZdmcET11xcbOsYdpIsXmzGkdTDbz3gryjbG5JtIHzZwescXzyKgJ1UWZOrSAlppon1HmI0l6KA8XoaEncXjpUUxXuIpIN4bdh8J5lyfCSZQ96Nu6vg05oG7Y7VxgJQoeyJhWuAm+tlKBxnmYp1Jbi58YbEGgCx4Bc2vO1KDeZ3hykaHAhbPV0FBg1fnAx7reNWkzUhPTmB+SPuLAyO+RrTleAJe4NkZV8kKtrgKQ7Y+MYr2+PxeDwej8eTH38BO988CQ+2tNwAAAAASUVORK5CYII=>

[image24]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAB4AAAAXCAYAAAAcP/9qAAAA30lEQVR4XmNgGAWjgHLAAsS/0AVpCW4A8X8kTHfwiYEIizWBWBZdEAtgRBfAA/BarMAAkfwHxN+g7DRkBUiAE4hN0AXxALwWgyxEB/uB+DcQy6CJf0fjEwJ4LfZEF4ACYSD+y4CaSGxQVBAGeC0mBoCyBTmAoMWdDAhfrUaTQwd26AJ4AF6LTwBxMxJ/JgNEcRCSGAyUAzEbuiAegNfiHnQBKLjMANFUAsT+QHyFAVIwkALwWozPB3xAPB+IrwFxCpocMQCWPekGPgPxcyB+DMVPgfgdECsjKxoFo2D4AACRzzkRWz6VJwAAAABJRU5ErkJggg==>

[image25]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACIAAAAXCAYAAABu8J3cAAABZUlEQVR4Xu2UPS8FQRSGj6AgSAQJEXRqkYiPqPwEicoP0ImIj0h0GoVKoZFIlCJ+hUInIaJWaHS0vs975+x19p1du3vrfZI32fPM7M65s3NXpKamddo0KyxzONR8aD415zTmudT8aB413TSWS5dmi2UGb5ppux6QsBDi6TQ3bnW71SPNGf/Qo9lmScxqXjS9zs1IWOTWuWvNs6vBkcQNZ4KHFzWCV4KHPZDnXcH1iavBgvlCyjQC8N77yPlGktew/zfcYML8MvmIso0w8xIWQINgyurN5ozAkPldL5NfUDZj4bZMMP7t6iVz686BfvOn5CNa2ZErzRe5SQkLbpAfNH9APqJqI2uaV5YSvkdYcI88dhV+lXxElUbmNE/ksIi/zvvXFH5LyjYyrLljKelGcGbuXQ12JD0nlzKNdEh8kJPcuHmL5jyoj8lFDynKaLitceJ5LAmfiWQHLjTvmrP0cE1NTTV+AWPrbc/TsEzZAAAAAElFTkSuQmCC>

[image26]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACIAAAAXCAYAAABu8J3cAAABIElEQVR4XmNgGAWjgHzACMSh6IIEwBl0ATSwGoj/A/E1IOZCk8MJOIG4BF0QCzjJADEchrEBVgaInByUzwzlS8JV4AE8QFyKLogHFDHgdshhIH6CJtbDgFs9CuBloJ5DQOLT0MSsoOIEAbUcAouGGjRxeah4EJo4BqCWQwwYIOIgeWQgChUvRxaEJTRisSxEGwrA5RAnBoh4Hpq4IFR8Npo4BqBWiKgxQMQL0MRFoOLNaOIYgFoOAZVHIPFKNHFQqILEo9HEMQC1HAICIHFcuYZgWUJNh/wD4ktoYmUMuNWjAFId0sUAMRiUG9CBDQOmpSD+JDQxsCApWBqiDQx+AfELBkjJ+RhKvwbixUhqQAAWAisZIHrmoUqPglEwCkgDADPoXc3Z6HkgAAAAAElFTkSuQmCC>

[image27]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAXCAYAAADpwXTaAAAAVUlEQVR4XmNgGAWjgKpgL7oAJeAfugAlwAaIy9AFKQHngNgcXRAETMjEt4B4HwMa8CMTX4NiFgYKwUQg9kYXJAcoAnEnuiC54BO6ACXgMLrAKBhuAACnlhESw2iRqwAAAABJRU5ErkJggg==>

[image28]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACIAAAAXCAYAAABu8J3cAAABXUlEQVR4Xu2Uuy4FURSGl0RBCCLexGuo1ULjEoJoFErNcYkH0IuIUqVSKCgEER6ARpC4JOJS8K/s2bLmn70U9qGQ+ZIvZ/a/Jnv/mTkZkX/OAGxQ1gbv4QdcpFnT6YWz8ECqRY7M9TvcNOtfY0+qRfRJdBbX88X6W+bgGIeGBfgIn+EIzSKpIpYd+MqhsgHfJLRUx8vjL87hrlmfwX2zjmiRJQ4NekYHh4xXpEvSj1OzHsq0yAplkQfYzWEKr8ix+EXWKdMiq5Qpp7C1uN6ygxRekfjamFSuRdYo09c/DIck/AdfStMEuUX0j3wHb+GTyeN90W0zS6I3TXAo1QMjXp6NbjrJofgHenk2uukUh+If6OXZ6KbTHEp436kDNbvgsBnoxjMcgkHxi/RzmEufhI2XeVCgs1Gz1q9nqtyP0Y/LDbyCl8XvtYTPvqVdwsGH8ETCt6CldEdNTc0f8QlXj2gIoKSjfgAAAABJRU5ErkJggg==>

[image29]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADEAAAAXCAYAAACiaac3AAABjklEQVR4Xu2VzSsFURjGHyVlIyU2qFts7CzsiI2NlfI3SNmru7OxUjaKhYXsbFjYklJWLEWyJuwkn+X7fb0zzZm395wZcm8W51dPd87vnJm5z9zuGSASidSbRsqLlg69lEPKJ2VXzWlaIevqxhnkhmksRpCf61djpp1ynvjQtWrKHfw3Zj+tHP9qB8qllCrRR+nW0qBBiwC+Eh0Qz58uO4m3CJaoQCY/KE/J8ZS7wKGZMqBlAF+JWdh+DbZngiX4y2v2KK+ULuWf1bgIX4kt2H4ZtmeCJca0SGijvCM7mTOUW1GMr8Q+bL8I8Z16AgUlysBb5W/wlViH7Zcg3rpfYYl5ZIs21JxmWIsAvhK+/8QqbM8ES/CWNueMVyCLJxyXUqU0aRnAV2IQ4v9sd1rQIuEYctIMZZxyAnmJ/QRfCcZ6UA+UG+VSgiVCT7YFsu2dUibVXBnSLdtim/LmjPn9w2srjnMJlqgF95RrykWSS8gT7nEXEUeUR8om5AuO5qe/uaVcIbsWH7OLRCKRf8wX4Jt/euNxdPwAAAAASUVORK5CYII=>

[image30]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAD4AAAAYCAYAAACiNE5vAAACHUlEQVR4Xu2Wz0tUURzFj4VmhTsD0QikNhmCErQKBWnRKmgThouilgaChJuIEGsvtBISW/UvlLSKoja1iBYhJPZDXBQpCClYaOfr992ZO9+5z3nvTUrC/cBh3pxz5737nfvjXSASiUT+P25Sv6gt6pnJ0pC2+5p31LXk+iC0INGhUotqPmGPCz9njTo5QP2mTnjeUWhRG57nc576gj0uvIn6TL2iGiqjQlxEeYR9Qp7jD3ULKXkPNQX994QW6i41isoOj1GPqQ7Py4KM1HtqgTpssrxMU53GSyv8I9WMlMIbqZfUJWg4AS1OuJd4XdQcdE21J97xpE1enlKrVJsNCnIE2p9vxu+lJpPrYOEvks/r0HC8HG2PvHhfPU8Q77bx8jIDnYbdNsjJGgJFkU3vOli4TGlBRtSGNwKe20nPGL8oD6D367NBBkagv5U++byBDpojWLhDgtfGk5G2P5AZYb16GIbe76oNanAWlaPqOInyUnXULFx2Teu5peB7P41XhPvQew3YIAOt1Lrx3Bq/At2zfH2HPst9LzGYBBbx+gPeheTaPjwLsivLu/i0DTIi03rFmgj33/EEKfkHVAeh9S2HAefdoU55WS3kaCkdPmaDnMjzQwpNe8dzaBt5rVYgu+tD4z2ilo0nyDSXm1y2QQA5A7yl5qHv03oZQnXBTrNeO4f08Qe1CF0KS9Az/q4jR9Z/cWKLRCKRSGQH/gLUQop8K/IMgQAAAABJRU5ErkJggg==>

[image31]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAD4AAAAYCAYAAACiNE5vAAACFUlEQVR4Xu2WTUhUURTHj4FaWChKlCAJLZMWtUoS2rSQQFqJixZtg5biwoU06Sbog8BwFSJolLQJsl1BJQlRiyAQdSNuRKOC8KPog/qfuffSef/e1LznVC7uD34M73/uDPe8+87MiEQikcj/5Tn8DpfhCaqV4gh8CI/CHfAgvA7v2UXblRr4xVxfEXcDnpmsFCfFrbW+SayoIKc42CLr8Bplq+KaaKKc0SfjAbwJB+HuZLmytMFv8BIXchJOaq/Jzvts0mRpdMCLHKZxGI7COn+9BxbEzYXOSBZa4Ud4hwsZ0fm8TFlBXONXKWeOSxmN14p7LM6I+9AhOOZrF3yWh3pxc6UzWUW1vLwXtx/d8+9oh7fErb0NN+HTxArwyL+Gxu2d0pPP23igGs7DRbiTalk4IG4vI1xIQZ+WJcr0vdM26POvs/Jrk2GmKoXe9bfinoas6D50HPOyIiV60XCGsg8+rxQ6959gCxf+wBrs5TAjj8X1so/yYtiZkvVTlocn8B1s5EIZLMAuysbpmtF984G99Fli3Hp8aOk2mW74hqmVg872nLiN65+RPNyHxyjTfakBbURH0qL7vkvZZ58neJ0STplM56Ncwre5nvJWGJCfJ8fuMutCdshkr8T9pAWaxa05a7IiX+EwZXpi4UMbqJbGfrgBJ7iQE27WajkNX1CmhC/rcNL6N/avcI6DSCQSiUT+DT8As8aB2Mvb7AkAAAAASUVORK5CYII=>

[image32]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABUAAAAYCAYAAAAVibZIAAABHElEQVR4Xu2SO0pDURRFTyRE66QQbATRzhEES38oIna2WqSWTCADSB17CxGsxcJKC8dgI+IIBEv/7p1zI+ftJPBywS4LNo+77rmb9zObMoIWcqIylznkB5lH1pC34nYeLGzKejasJ+YK+RDH0g1xpWmYFxyIp2uLK821eUFkObkt8aXh4S9kJq0ryGnyWVTNDz8htyF02aXH5ocXxdM9iDtHPpFX5AK5RO7MZ4/CnPWSjPCD0S2IJ/T74jbNv8EfXRsu5Z28iBsQZ/fSdTW4PrtWHKzLOrJuxb1xc310UB9vwD3yiHSQG/N3PBaWsOwbWZG9CGe2w3opXZ+Dm5hRj8tf8lBlWXZsuLRm/otl8W5eGMNXxcKzMDfln/gF+jZFTRSJBFoAAAAASUVORK5CYII=>

[image33]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEcAAAAYCAYAAACoaOA9AAACmElEQVR4Xu2XS6hNYRTHF3kPlEcpl67EQJSJCcnEmyRGBiYkEyWJZGTglnvzGHl0R5KkSBnIQAkDJSN5F0nKDGFA3tbfWvuctf/n29u5595up87+1b97vt+37tl7f/t73CtSUVExTOzU7GHZ6YzT/NFM0yzVfMt3ty0XxO77q6aH+v7Hcc13zXuxZy4EF1hC7bGh3Y7gHqf654Xe/lXvLuWT5AcTg9sb2jWuaX6Qw4VWkmsnDmgeaEYGt1/svk8El2K5WF1kcsLJFJebyMPtJTcUzNPMZJlgBAvintg9niQP1/CQBJZSqgZuaxTXXUbmuFtNfjDMEvvO35ov/hmbf4rxmkUsiQmaK5KfOdgGmhkc9PNKAfAPWWCdZhfBGzvtfijBoDC3xG5yBnms/1Y4KHbfu7iDQM1nlmK+du1RLl5pboc0M/oDZS0LB8saLye7JlJ6cpSA3/3AMkFRXe65t3uju9ZtwD0jhyPzp9guf1FzSXNHrHZbqBsMeFmt8kLzjmUBuOePLMV8bYafchHBxgw3nTyA30huldge1Qx9Un87l6mPWcaiBLy4JyxLwPVTf8fBY5D/cdRFBDMjNeVArN3gPxcEVwZOl8Oh3S/2fZuDy8AxPYZlAbs1N8jxMzHZC2LgzmSN9S4ykme9s0LyfUV1RRxj4TwS+659YrPyseZ5rqIYzK7zLKVxFh2idurAwUEENzpKfmBeNhl3NS/FLoQ3hak8EMpmwkTNWc1TzQ7qK6JL6jOAcyTUXXWHPTICNz+070viBMNgoBAb0Vzqi6BmTWjP9p+vgxtOsDfwoGRZHOomad6KvYBIt1jtTc0bsZqWwRcxOF22sOw01knj4GCZ4GjvaLL/Q2KwBDEw50JdRUVFRUWL/AXIsbkVrntdkwAAAABJRU5ErkJggg==>

[image34]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEcAAAAYCAYAAACoaOA9AAACSklEQVR4Xu2XvUscYRDGx48YBRGSgIFYCKIgmL9A7DQqioidhY0GrcV/wMYiICmjYmcRhFRRxEIUFLSRgCFFUiiJHxEUVExAxCQmmcd5V8e5PW53704O3B88HPPMwO47+34dUUxMzB0xwBq05n2nmPWP9ZTVwLq4nc55Dq0RgNesX6xjkjEnBY2pN/FDFeciH0ne01MYfrBGVHzOeqXia+ZYv42Hh70w3l3Rbo0UzFK45jRSYv1jH4+eOLPL+PCGjJdtJkk+Uq1NpCBsc7CU/Orh9Whj3pmaaue1GD9bLJJM83KbCEjY5qDWrhQA/5M1Lln5Ls5jjTk/mxSyPrN2WCUmF5YozflpTRIfe88VeEEYX1nLSvDCPCwMZSQnywe6+SDpEqU5J9YkM+4+F1RepwV4X4z3lvWHZPpPs96xVkhqe1VdMipYZyQDyTRRmnNqTRL/rxe8cYYGGzO8Z8YH8DuN10yyR6UCmyyaO2ETGSBKc/zucfA3vWDUGRrMDL8pB3Rth/t9rrwgeDNoxibSIEpz/OrhjXsB7hO6yPesdzTR7VyyuqCUsvZZ6ySHQDqkas6wif0OHLwDvAfatAO2y8ZjjbVF8qAFkj0oExSwNljfSP7CRGGV5N39bvTvSXLYIzXw6lSMj5RwgqEZKMRGVGNyGtS0qrjK/W4rL12w1LCkcTENAo7dA9Z31h7JTDxivVQ1j5yPU1JTSTKmJdYuSU1k7DQEuAp0WzMD9Fsjl2mjxOYUkZw+9xrvf4gWliAaM6XqYmJiYmIi8h9wnaI3MdID4QAAAABJRU5ErkJggg==>

[image35]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEcAAAAYCAYAAACoaOA9AAACgElEQVR4Xu2XS6hOURTHF8mzlMeNKEoMFJkpuiR55ZFkwpCBlJKUZGRyRQkTj4EkSUopAyEjijIwYCDJ66YMlLwm8rb+1t7X2v/vnu+eF311z6/+nW//1/rO3me/zj4iDQ0N/4ltql1sDnZGqn6pJqm6VV/ScMdyQazdn1U9FGvHK9Va1RhVl2q76muS4UAFC6k8wpU7EbRxYvg9L5R//A23Bbms4UlG4KrqG3lIXk5eFeazUZG9qgeqoc7bI9buo87LIuYdU82lWB8TxBI3kA9vN3lVwKj0qu6ohqShUtwTa+Nx8uMsGIg8OXJNWhNnBm8l+XWAkX6oeqkaRbEijFZdlnTmYBuotXOQhHUaK8Gongz+v+a66pNqMgdKsk+s3Ts40A/I+6C6q3qi+piGRYaJJb1Q3XLK2/t1cVb1Xdqs/Zygze/YzICfD/9LvK3BmO7N4D0mD69MPAB6+KLqkuq2WO4Wl1eFA2L3W8yBHDxVvWWzAHiVo+5F0TgRDA82ZnhTyAfw15O3QmyPqgMsB9SxmQMDgIF7xGZBlojVfSoah4PhwczImpo+d124znFeWXB4w72XciAHO1U3yeNnYn5Kaw4OhPD6DpJrghEZT2XPMkljWXlFOCN2vprNgZxg+Z1nU1pn0X4qo+2vyTsU/Gne5AfmZRPBrv5MrCKMFKZyWW6o3osd28syVay9/emgy7sSPOyRkY3B9yCH99k/nYEAptosinmQs8qVZ4Rrr/PagSPCfdVzse+4qmAD5k6JWuDyxonNkrHOA0fEct+E6+k0XAzcgMFRYBObGeDzoY6TccexWlo7B58DeLUPavAZz9MWSxAdc87lNTQ0NDSU5DcjIKluYR9JuwAAAABJRU5ErkJggg==>

[image36]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC0AAAAXCAYAAACf+8ZRAAABZUlEQVR4Xu2WvyuFURjHH5QoiUlWow2TH6VksphYDAaL8jcYWC3+AKsMFouJQWTBQInZRDJRkhLfx3lP93m/933cxFHqfOo7nM95zj3PPfe+516RTCbzHZaRR+QZWaA5jwFkHxlEmpE+ZB3ZsUWpuEL2zPgSOTZjj0nknXJfqkhEp4TNGHVdLIlxZBfZQFaRjvJ0Os7Fb1qb+YoxZIXlXxA/VsbzllH5h00PI5sS6rYkPMSHpQrDrdRe9IHmmAkWhNec5y16a9yQ0zVH5D6f8ikzHpFQ2G1cZAZpYkl4zXm+EXdSsc5eTRG9I7VwzrhW5M2MPbzmPN+IAwnreqwcsgPiVGqbaVrK05U8SXVz6q5ZElVv7KxwbeR/lVmp31hRZw9Im1gyY0Vrtsm9Fj45usmiGa8VzhJPtd84veP12ov0SqiZNy4Z7RI2O0EukBepf4CnJXz9GP0LoGvjCetPeyaTyfyQDyZqYeCaAb3uAAAAAElFTkSuQmCC>

[image37]: <data:image/png;base64,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>

[image38]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAYCAYAAAAlBadpAAAAsUlEQVR4XmNgGJZAGogLgHgmECshiVshsTHAYiD+D8S3gdgbiFWBeBoQPwdiS6gcVgCS+AfE/OgSQFDJAJG/hC4BAn8Y8JgKBSD5IHTBD1AJTnQJNIBhuC5U8Ba6BBaAofkvVBCbPwkCkEYME4kFZGtmZoBofIkugQVgtYAYmy2AOAFdEATuMkA0g1yBDYDEX6ELIgOQZlAiQTfACIhfo4lhBbsZEF74CqVTUVSMgqEIAG1gK0HBSgf2AAAAAElFTkSuQmCC>

[image39]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADkAAAAUCAYAAAA3KpVtAAAAjklEQVR4Xu3UMQqDQBRFUTUpFCRNwCYbcC+uwz2JuzB1VhAEu7QhhUWagIWVKbwWNm8BIfP5F04zr5xhosjzPM/7jyo9sFSLBaUOFrphQqFD6B3xwAuZbMF3whs9EtmC74IZVx0stX0mXzQ6WGy/0U4Hi+UYcUcsm7kOGPBEKpvJtif8wVkHi9V64Hm/aQWboxIg+IwoAQAAAABJRU5ErkJggg==>

[image40]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGIAAAAUCAYAAAB23ujSAAAClElEQVR4Xu2YTahNURTHl++k5GPwfJvJSChJPiaGknxOkJmJFIWBYmAiA+UpAyWSkEwoUUYYKKEMFFFe+QhhQAnJx/q/tfa96/yde+95d/Leyf7V6tz/f6+9zz77nLP3Plckk8lkMpnOjNO4qfFH477GsGJxWxZpvBare4DKIlfFcp5ojKSyyGaNH2K5O6isKj/ZqAPTxS56rOvJroc3MlpzXiw3cUrjQ9BggljOXNdTXY9uZDR5p3Ev6C8aS4KuSuxTbfiqcYm8BxrfyWNmiV3wWvLhrSfNA/NY4xd5d6SYh3ah9wWvKny+WoBObyJvv/vtOCGWM418eJ9JPwsaHHU/Ar2VvAWkq8JtD3lWiHV6Gfnb3J9EfqRPLAdTTwReGojZ/vtRs7ifg+6n6eqwa0yLWJ9Wu98ttbsRu8Q6vZD8je4vJj9yRiyn7I2IA4Hf/EacdT9NYe9d79TYojFTbGq86+UDpXY34pBYp+eRn+Zn7GBakRb1deSX3QheD367v9d1qnO7kWFvRrxZZXySZt0q8dSqDT22i3VwPvkb3F9JPnNNLC/x0nX00q5simtMg7fcW+5eqjPHdYLbqko3dQaVtEbwFhGLJnwMYifWiD1puCnY8pYN3iiNyxpvNFZpnBbLSd8Tfa6Zsraq0E2dQWWMWKe72TW1AvUusEngWyG2f4R04r+5EQCdPk7edfcjWMB7yEPOjaAxxXG9FyUedG/QaT2YETwAD/UHCp+vFpQ9/dBxEU4DVZZ3LmgswseCBt+kWO8k6QT++ngb9FKxvBHBq0pZ+7XgotjOBkdcBLa1zBWNPeSlQX3ux93F4n7Gi5V9FDtHuyccX/TIxV8bOE4sFv/DK2k+IFXioVXLZDKZTAf+Apzj0AVAvhrQAAAAAElFTkSuQmCC>