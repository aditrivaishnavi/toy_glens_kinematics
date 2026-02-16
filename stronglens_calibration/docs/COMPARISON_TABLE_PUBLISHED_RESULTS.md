# Comparison with Published CNN Lens Finder Results

**Date:** 2026-02-14  
**Purpose:** Contextualize our results against published work for the MNRAS paper discussion section.

---

## Table 1: CNN Lens Finder Performance Comparison

| Study | Survey | Architecture | Source Model | Real Lens Recall | Injection Completeness | Sim-to-Real Gap Measured? |
|-------|--------|-------------|--------------|-----------------|----------------------|--------------------------|
| **This work** | DESI DR10 (g/r/z) | EfficientNetV2-S | Sersic + clumps | **89.3%** (112 Tier-A, p>0.3) [82.6%, 94.0%] | ~3.5% marginal (no Poisson), ~5-7% (with Poisson, D03 pending) | **Yes** (linear probe AUC=0.991) |
| Herle et al. (2024) | Simulated Euclid-like | Multiple CNNs | Parametric Sersic | N/A (simulated only) | Characterizes selection function biases; completeness depends on theta_E and source properties | No (quantifies selection bias, not realism gap) |
| HOLISMOKES XI (Canameras et al. 2024) | HSC PDR2 (g/r/i) | ResNets, AlexNet, G-CNN | Real galaxy stamps (HUDF) | TPR0 = 10-40% (zero FP on 189 confirmed lenses) | Not reported as injection-recovery grid | No (focuses on TPR at fixed FP) |
| Euclid Prep. XXXIII (2024) | Simulated Euclid | Classical CNN, Inception, ResNet | Parametric (simulated) | N/A (simulated only) | ~90% on clear lenses, ~75-87% on faint arcs | No |
| Huang et al. (2019) | DECaLS (g/r/z) | ResNet | N/A (candidate search) | Not formally reported | Not reported | No |
| Jacobs et al. (2019) | DES (g/r/i/z) | CNN | Parametric | Not formally reported | Not formally reported | No |

---

## Table 2: Key Methodological Differences

| Aspect | This Work | Herle et al. (2024) | HOLISMOKES XI | Euclid Prep. XXXIII |
|--------|-----------|--------------------|--------------|--------------------|
| **Training data** | Real DR10 cutouts (451k) | Simulated images | Real HSC images | Simulated Euclid images |
| **Source injection model** | Sersic + Gaussian clumps | Parametric Sersic | Real HUDF galaxy stamps | Parametric |
| **Poisson noise on injections** | Optional (added in D03) | Not specified | Included (real stamps) | Included (simulation) |
| **Band-dependent PSF** | Single r-band PSF | Yes | Yes | Yes |
| **Correlated noise** | No | Survey-dependent | Real survey noise | Simulated |
| **Selection function output** | C(theta_E, PSF, depth) grid | Selection bias characterization | TPR at fixed FPR | Accuracy metrics |
| **Realism validation** | Linear probe AUC in CNN features | None | Not applicable (real stamps) | None |
| **Confirmed lens sample** | 112 Tier-A (spectroscopic) | N/A (simulated) | 189 confirmed | N/A (simulated) |

---

## Table 3: What Makes Our Contribution Novel

| Claim | Evidence | Comparison |
|-------|----------|------------|
| First quantitative measurement of injection realism gap in CNN feature space | Linear probe AUC = 0.991 +/- 0.010 | No prior work has measured this |
| Injection completeness is a conservative lower bound | AUC = 0.991 means CNN trivially distinguishes injections from real lenses | Herle et al. characterize selection biases but do not test against real lenses |
| Source texture (Poisson noise) is a major contributor to the gap | +17.5pp detection improvement at key magnitudes | Not tested in prior ground-based work |
| Lensing geometry (beta_frac) contributes to but does not dominate the gap | Restricting to [0.1,0.55] roughly doubles detection, but ceiling remains ~35% | Herle et al. characterize theta_E bias but not beta_frac |
| UMAP visualization directly shows distinct manifolds | Real lenses and injections barely overlap in 2D projection | No prior visualization of this kind |

---

## Discussion Points for Paper

### Why HOLISMOKES XI uses real stamps and we use parametric
HOLISMOKES XI (Canameras et al. 2024) used 1,574 real galaxy stamps from the HUDF with
spectroscopic redshifts as source-plane galaxies. This avoids the parametric realism
problem entirely but requires high-resolution source stamps and careful redshift matching.
Our parametric Sersic approach is more common in the field (easier to implement, no
external high-resolution data needed) but our linear probe result demonstrates its
fundamental limitation. We propose the linear probe AUC as a quantitative gate that
parametric injection pipelines should pass before their completeness estimates are trusted.

### Why direct completeness comparison is inappropriate
Our marginal completeness (~3.5%) cannot be directly compared to the ~90% completeness
reported by Euclid Prep. XXXIII because:
1. Euclid simulations are fully synthetic (no real survey artifacts)
2. The "clear lens" category in Euclid prep. is pre-selected for high contrast
3. Our metric covers the FULL parameter space (many configurations produce faint/unresolvable arcs)
4. Our injection model lacks the realism of the training data (the entire point of our paper)

The appropriate comparison is: what happens when you use the SAME injection pipeline to
measure completeness AND compare it against real-lens performance? This is what we do,
and the gap (89.3% real recall vs ~3.5% injection completeness) is the novel finding.

### Herle et al. (2024) vs this work
Herle et al. characterize HOW selection functions are biased (toward larger theta_E,
concentrated sources) but work entirely in simulation. They do not test whether their
simulated lenses look like real lenses in CNN feature space. Our linear probe result
provides a complementary measurement: not just that selection is biased, but that the
bias is so strong that injections are almost perfectly separable from real lenses.
Together, the two results tell a complete story: selection functions are both biased
(Herle) and unreliable unless realism-validated (this work).
