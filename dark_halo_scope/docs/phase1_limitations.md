# Phase 1 Limitations

*Known approximations in the analytic observability window and how they will be addressed in later phases.*

---

## Overview

Phase 1 provides a **geometric upper bound** on lens detectability based purely on the seeing-limited resolution of DR10. This document explicitly lists the limitations of this approach and explains why they are acceptable at this stage.

> **Key statement**: Phase 1 ignores flux limits; it is a geometric upper bound on detectability.

---

## Limitation 1: No Flux / Surface-Brightness Limit

### What's Missing

Phase 1 only encodes **resolution** (whether the arc is geometrically resolvable), not whether the arc is **bright enough** to stand out above:
- Photon noise
- Sky background
- Lens galaxy light (which dominates at small θ_E)

### Why This Is Acceptable

Phase 1 is explicitly a "seeing-wall only" analysis. It answers:
> "If the arc were infinitely bright, would we be able to resolve it?"

Magnitude dependence will naturally enter in the **injection–recovery completeness curves** (Phases 4–6), where we inject arcs of varying brightness into real noisy backgrounds and measure detection rates.

### How It Will Be Addressed

| Phase | Action |
|-------|--------|
| Phase 4 | Inject arcs with realistic magnitudes (m_source = 22–25) |
| Phase 5 | Train detector on varying arc brightness |
| Phase 6 | Measure completeness as function of (θ_E, m_source) |

The final selection function will be `C(θ_E, z_l, m_source)`, not just `C(θ_E, z_l)`.

---

## Limitation 2: Single Representative Seeing

### What's Missing

The θ_E–z_l and mass windows are drawn for **one representative FWHM** (1.25" in r-band). Real DR10 has a distribution of seeing conditions:
- Good conditions: FWHM ~ 1.0"
- Median: FWHM ~ 1.25"
- Poor conditions: FWHM ~ 1.5"

### Why This Is Acceptable

We already mitigate this with the **seeing-sensitivity plot** (`phase1_seeing_sensitivity.png`), which shows how the detection thresholds shift across the realistic FWHM range.

The qualitative picture remains the same:
- Blind zone shifts by ±0.1–0.2"
- The "seeing wall" concept is robust

### How It Will Be Addressed

| Phase | Action |
|-------|--------|
| Phase 3 | Store per-object PSF FWHM in background HDF5 |
| Phase 4 | Use actual PSF when injecting arcs |
| Phase 5–6 | Completeness naturally reflects true seeing distribution |

Injection tests on real cutouts with their own PSFs will fold in the true spread.

---

## Limitation 3: Source Redshift Distribution

### What's Missing

The mass windows use a **single z_s per panel** (z_s = 1.5, 2.0, or 2.5) rather than a realistic source redshift distribution P(z_s).

The mapping from θ_E to M(<θ_E) depends on z_s:
```
M(<θ_E) = f(θ_E, z_l, z_s)
```

Different z_s values give different masses for the same θ_E.

### Why This Is Acceptable

We have already demonstrated (in `phase1_mass_zs_comparison.png`) that the **qualitative conclusion is robust** across z_s = 1.5–2.5:
- Only high-mass halos (log M > 11) are visible
- The "blind" vs "good" boundary shifts by ~0.3 dex, not orders of magnitude

For Phase 1's purpose of establishing the "observability window," this is sufficient.

### How It Will Be Addressed

When discussing halo masses in later phases, we can either:

1. **Stay in θ_E space** — report selection function as C(θ_E, z_l) without converting to mass
2. **Show a mass band** — display allowed M(<θ_E) range based on plausible z_s ∈ [1.5, 2.5]
3. **Use photo-z priors** — for real lens candidates, fold in P(z_s | photometry)

| Phase | Action |
|-------|--------|
| Phase 6 | Report selection function primarily in θ_E space |
| Phase 7 | For known lenses with spectroscopic z_s, compute precise masses |
| Phase 8 | Present mass estimates with z_s uncertainty bands |

---

## Summary Table

| Limitation | Impact | Mitigation in Phase 1 | Resolution in Later Phases |
|------------|--------|----------------------|---------------------------|
| No flux limit | Overestimates detectability for faint arcs | Stated as "geometric upper bound" | Injection–recovery with varying m_source |
| Single FWHM | Ignores seeing variation | Sensitivity plot shows FWHM dependence | Per-object PSF in simulations |
| Single z_s | Mass estimates are approximate | 3-panel z_s comparison | Work in θ_E space or show mass bands |

---

## Key Takeaway

> **None of these are project-killers.** They are clearly marked approximations that will be tightened by later phases.

Phase 1 establishes the **physics foundation**:
- The seeing wall is real and quantifiable
- Only massive halos (θ_E > ~1") are geometrically observable
- The qualitative picture is robust to parameter variations

Phases 4–6 will layer **empirical calibration** on top of this analytic framework.

---

*Last updated: December 2024*

