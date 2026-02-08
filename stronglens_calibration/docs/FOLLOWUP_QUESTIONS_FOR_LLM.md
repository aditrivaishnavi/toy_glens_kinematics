# Follow-up Questions for LLM - Complete Implementation Clarifications

**Context**: We've committed to Option 1 (Hybrid): real-image detector + calibrated injection selection function. Before we execute, we need clarification on implementation details where your recommendations conflict with our existing code or where specifics are needed.

**Goal**: MNRAS/ApJ-level publication with genuine novelty via selection function + failure mode analysis.

---

## SECTION A: Negative Sampling Strategy

### A1. Two-Pool Design
You recommend two pools:
- **N1**: Deployment-representative (same population as 43M deployment scan)
- **N2**: Hard confusers (rings, spirals, mergers, edge-on disks, blue clumpy star-formers)

Our current code uses a single `is_control` pool (50% of samples are controls from the same LRG without injection).

**Questions:**
1. What ratio should N1:N2 be in the final training set?
2. For the 100:1 overall negative:positive ratio per (nobs_z, type) bin, how does this break down? Is it 90:10 N1:N2? 80:20?
3. Should the control samples (same galaxy, no injection) be:
   - A separate third pool?
   - Part of N1?
   - Deprecated entirely in favor of N1+N2?
4. For N2 (hard confusers), should we:
   - Curate manually (labeled contaminant catalog)?
   - Sample automatically via morphology/color cuts?
   - Both (curated for specific categories, auto-sampled for prevalence)?

### A2. Defining Hard Confusers
5. What specific criteria define each confuser category for automatic sampling?
   - Rings: color cuts? Tractor shape parameters?
   - Spirals: TYPE=EXP + specific colors?
   - Mergers: how to identify from Tractor catalogs?
   - Edge-on disks: axis ratio threshold?
   - Blue clumpy star-formers: color cuts (g-r < X)?
6. If we don't have labeled contaminant catalogs, can we approximate with Tractor parameters?

### A3. Exclusion Radius
7. You recommend 5" exclusion around known/candidate lenses. Is this:
   - From the lens centroid?
   - From the edge of the arc (centroid + θ_E)?
8. Should we exclude any galaxy within 5" of ANY of the 5,104 candidates, or just the 434 Tier-A?
9. What about galaxies that are in the same brick as a known lens but >5" away?

### A4. Galaxy Selection Criteria
10. For N1 (deployment-representative), should we match Paper IV's deployment population exactly?
    - TYPE in [SER, DEV, REX, EXP] (exclude PSF)?
    - z < 20.0?
    - Any maskbit exclusions?
11. Should we apply LRG-like color cuts to better match positive host properties, or keep it broad?
12. What maskbit values should exclude a galaxy from being a negative?

### A5. De-duplication
13. How should we handle galaxies that appear in multiple sweep files (overlap regions)?
    - Take first occurrence?
    - Random selection with seed?
    - Does this matter for our purposes?

---

## SECTION B: Spatial Splits

### B1. HEALPix Configuration
14. What nside should we use?
    - nside=64 gives ~0.84 deg² cells, ~50,000 cells full-sky
    - nside=128 gives ~0.21 deg² cells, ~200,000 cells full-sky
    - Trade-off: larger cells = more galaxies per cell but fewer cells; smaller cells = better spatial granularity but sparser
15. For 70/15/15 train/val/test split, approximately how many HEALPix cells should be in each split to ensure statistical power?
16. Should we stratify the HEALPix cell allocation by observing conditions (so each split has similar PSF/depth distribution)?

### B2. Split Assignment
17. Should the split be deterministic (hash of HEALPix cell ID) or random with seed?
18. How do we ensure positives are also in the correct spatial split? (Cross-match by HEALPix cell?)

---

## SECTION C: Label Handling

### C1. Implementation Choice
19. Should we implement tier-based labels as:
    - **Label smoothing**: target = weight (e.g., 0.95 for confident, 0.75 for probable)?
    - **Sample weights**: in the loss function?
    - **Both**: label smoothing + sample weights?
20. What's the recommended approach for binary cross-entropy with weighted/smoothed labels?

### C2. Weight/Label Values
21. You suggested confident=0.95, probable=0.7-0.8. What values exactly?
    - Tier-A (434 spectroscopically graded "confident"): 0.95?
    - Tier-B (4,666 "probable"): 0.75?
    - Literature confirmed (SLACS, BELLS, SL2S): 1.0?
22. Should weights vary within Tier-B based on any other criteria (e.g., discovery paper, visual grade)?

### C3. SLACS/BELLS Handling
23. The lessons_learned document notes SLACS/BELLS have low DR10 visibility (arcs often not visible). Should we:
    - Include them in training with full weight (label=1.0)?
    - Down-weight them since the arc signal may not be learnable?
    - Exclude from training and use only for evaluation?
    - Use them for "stress test" evaluation only (report separately)?

### C4. Paper IV's Training Set
24. Paper IV used 1,372 confirmed lenses. Do we have access to this exact list?
25. If not, should we:
    - Reconstruct from SLACS+BELLS+SL2S+SWELLS+GALLERY (we have ~93)?
    - Use our 434 Tier-A as proxy?
    - Something else?

### C5. Training/Evaluation Separation
26. What's the cleanest way to ensure no overlap between training positives and evaluation anchors?
    - Spatial separation (different HEALPix cells)?
    - Explicit ID exclusion list?
    - Both?

---

## SECTION D: DESI Spectroscopic Catalogs

### D1. Availability
27. Are the DESI DR1 spectroscopic lens catalogs publicly available?
    - Single Fiber Search (arXiv:2512.04275, ~4000 candidates)
    - Pairwise Spectroscopic Search (arXiv:2509.16033)
28. If available, what's the data access URL/format?
29. Do we need DESI collaboration membership to access?

### D2. Usage
30. How many of these have spectroscopic CONFIRMATION vs just "spectroscopically selected"?
31. Should we use them as:
    - Independent validation set (no training)?
    - Silver positives for training with lower weight?
    - Something else?
32. How do we handle overlap with our imaging candidates? (Crossmatch and exclude from training?)

---

## SECTION E: Injection Realism

### E1. Source Magnitude Prior
33. What distribution should we sample source magnitudes from?
    - Empirical from deep field (which field? Depth?)?
    - Parametric (what functional form and parameters)?
    - Tied to observed arc brightness in Tier-A anchors?
34. What magnitude range? (Faint limit where arcs become invisible in DR10?)

### E2. Source Morphology
35. What source morphology model should we use?
    - Smooth Sersic (simplest)?
    - Clumpy (more realistic for high-z star-forming)?
    - COSMOS morphologies (if available)?
36. Should we bracket with multiple morphologies as a sensitivity test?
37. Our COSMOS bank has specific morphologies - are these appropriate for DR10-visible arcs?

### E3. PSF Model
38. Our code uses Moffat PSF. What beta parameter?
    - From DR10 PSF metadata?
    - Fixed (e.g., beta=3.5)?
39. Should PSF be evaluated at cutout center or averaged over cutout?

### E4. Noise Model
40. You said use "measured background statistics from the same cutout (robust MAD-based noise)". Should we:
    - Measure from outer annulus (r > 20 pixels)?
    - Measure from inverse variance map if available?
    - What if the outer annulus has sources?

### E5. Inner Image Handling
41. You said we need "an explicit, reviewable choice" for inner images. Recommend:
    - A) Include inner images with realistic visibility (may be buried in noise)
    - B) Explicitly suppress inner images with justified argument
    - C) Run both as ablation
    - Which should be the primary?

### E6. Acceptance Thresholds
You specified diagnostics but not exact thresholds. For implementation:

42. **Arc Annulus SNR matching**:
    - "Within a factor of ~2 in median" - is this 0.5x to 2x, or within ±2 of absolute value?
    - "Similar tails (especially low-SNR)" - how do we quantify? KS test? 10th percentile within X%?
43. **Noise histogram**:
    - What KS test p-value threshold? (p > 0.05? p > 0.1?)
44. **Color distribution**:
    - What tolerance on (g-r, r-z) medians? ±0.1 mag? ±0.2 mag?
    - Should we match full distribution or just moments?
45. **GO/NO-GO decision**:
    - All diagnostics must pass?
    - Weighted score?
    - What if one passes marginally?

---

## SECTION F: Cutouts

### F1. Size
46. Our Phase 4c uses 64×64 pixels (16.4" at 0.256"/pixel). You suggested 101×101.
    - Should negatives match existing positives (64×64)?
    - Or use larger (101×101) for more context?
47. If we change to 101×101, do we need to regenerate all positive cutouts for consistency?

### F2. Bands
48. Should we require all three bands (g, r, z) or allow missing bands?
49. How do we handle cutouts where one band has NaN but others are OK?

---

## SECTION G: Training Configuration

### G1. Architecture
50. ResNet18 vs EfficientNet-B0 - which should be our primary baseline?
51. For fair comparison with Paper IV, should we implement their exact ensemble (ResNet + EfficientNet + meta-learner)?
52. Or is a single architecture cleaner for selection function interpretation?

### G2. Hyperparameters
53. What's minimally sufficient for a "defensible" baseline?
    - Number of epochs?
    - Samples per epoch (given we have 400K+ negatives)?
    - Learning rate schedule?
54. Batch size given 24GB GPU VRAM? (For 64×64 3-channel images)

### G3. Augmentation
55. What augmentations are appropriate?
    - Rotation (90°, 180°, 270°)?
    - Flips (horizontal, vertical)?
    - What about scaling/brightness - risk introducing shortcuts?

---

## SECTION H: Selection Function

### H1. Grid Design
56. What axes should the selection function grid cover?
    - θ_E: what range and resolution? (0.5-3.0" in 0.25" steps?)
    - PSF: what range and resolution? (0.8-2.5" in 0.3" steps?)
    - Depth (psfdepth_r): what range and binning?
    - Host type: grouped (SER/DEV vs REX/EXP) or separate?
    - Source magnitude: include as axis?
57. How many total grid cells is reasonable?

### H2. Statistical Power
58. What's the minimum number of injection points per grid cell?
    - 100? 1000? Depends on target precision?
59. How do we handle cells with very few injections?
    - Hierarchical/Bayesian smoothing?
    - Mark as "insufficient data"?
    - Merge with adjacent cells?

### H3. Uncertainty
60. Bootstrap or Bayesian binomial for uncertainty?
61. How many bootstrap iterations?
62. What confidence level for error bars (68%? 95%)?

---

## SECTION I: Milestone Sequencing

### I1. Dependency Order
63. Your milestones suggest M1 (train baseline) in Week 1-2. But to train, we need:
    - Negative samples (requires EMR job)
    - Proper spatial splits (HEALPix)
    - Label handling for candidates
    
    Should we:
    - A) Build EMR job first → Generate negatives → Then train baseline
    - B) Use existing Phase 4c controls as temporary negatives → Train quick baseline → Then build proper EMR job
    - C) Something else?

64. What's the critical path? What can we do in parallel?

---

## SECTION J: Paper Strategy

### J1. Figures
65. What figures are "must-have" for MNRAS acceptance?
    - Completeness surface (θ_E × PSF)?
    - Failure mode gallery?
    - Injection realism validation plots?
    - Calibration curves?
    - What else?

### J2. Claims
66. What claims should we explicitly AVOID to prevent reviewer rejection?
    - "We found new lenses" (without confirmation)?
    - "Our model is better than Paper IV" (without fair comparison)?
    - What else?

### J3. Novelty Framing
67. How exactly should we frame "following Paper IV methodology" as not being replication?
68. What's the one-sentence novelty statement for the abstract?

### J4. Journal Choice
69. MNRAS vs ApJ vs AAS Journals - any preference given our contribution type?

---

## SECTION K: Computational Constraints

### K1. Feasibility Check
70. We have single-GPU (24GB) + EMR for data processing. Is anything in the plan unrealistic for this budget?
71. Approximate EMR cost estimate for:
    - Generating 500K negative cutouts?
    - Running injection grid (how many million injections)?

---

## SECTION L: What We're Missing

### L1. Gap Check
72. Given everything above, what critical question are we NOT asking that would cause reviewer rejection?
73. What's the single highest-risk failure mode in our plan?
74. If you had to cut scope to hit a deadline, what would you defer to "future work" vs "must have for this paper"?

---

## Summary: Top Priority Questions

If you can only answer some questions, prioritize these (implementation blockers):

1. **A1.1-A1.4**: N1:N2 ratio and control sample handling
2. **B1.14-B1.16**: HEALPix nside and stratification
3. **C1.19-C2.21**: Label implementation and exact weights
4. **D1.27-D1.29**: DESI spectroscopic catalog access
5. **E6.42-E6.45**: Injection acceptance thresholds
6. **F1.46-F1.47**: Cutout size decision
7. **G1.50-G2.54**: Architecture and training config
8. **H2.58-H2.59**: Minimum injections per grid cell
9. **I1.63**: Milestone sequencing (EMR first vs quick baseline)
10. **L1.72-L1.74**: Critical gaps and scope prioritization

---

## End of Questions

Please provide as much detail as possible. We want to implement correctly the first time and avoid EMR reruns.

