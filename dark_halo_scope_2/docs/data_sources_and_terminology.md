# Dark Halo Scope – Data Sources and Terminology

## 1. Project Overview

**Core goal**

Quantify **which dark matter halos are actually visible as strong galaxy–galaxy lenses** in DESI Legacy Imaging DR10, given its real image quality, and how this observational "window" maps onto halo mass and redshift.

We do this by:

1. Building a **physically motivated parent sample** of massive foreground galaxies in DR10 (potential lenses).
2. Injecting **realistic lensed background galaxies** (from HST/COSMOS) into real DR10 cutouts, with real DR10 PSFs.
3. Training and calibrating a **lens-detection model** using these injections.
4. Performing a **selection-function analysis**: completeness as a function of Einstein radius, lens redshift, source properties, and (approximately) halo mass.
5. Running the trained model on the **actual DR10 parent sample** to:
   - Recover known lenses (validation).
   - Identify new strong lens candidates (discovery).
   - Place real and simulated systems on the "observability map".

The project is **physics-first**: ML is a calibrated instrument, not the end goal.

---

## 2. Key Terms and How They Map to Data

### 2.1 Physical roles

- **Lens / Deflector / Foreground galaxy**
  - A massive galaxy and its dark matter halo at redshift \( z_l \) that lenses background sources.
  - In this project:
    - Comes from **DR10** catalogs (bright, red, LRG-like galaxies).
    - Has:
      - DR10 photometry (g, r, z magnitudes).
      - Approximate redshift (photo-z or spec-z).
      - Per-object PSF / seeing information.
      - DR10 image cutout in g, r, z bands.

- **Source galaxy / Background galaxy**
  - A more distant galaxy at redshift \( z_s > z_l \) whose light is bent into arcs/rings.
  - In this project:
    - Morphology and small-scale structure come from **HST/COSMOS** images (space-based, high resolution).
    - Colors and brightness are adjusted to mimic typical **blue, star-forming galaxies** at \( z_s \sim 1.5–2.5 \).

- **Einstein radius \( \theta_E \)**
  - Angular radius of the ring/arc pattern for a strong lens.
  - Directly determined by the lens mass distribution, distances \( D_l, D_s, D_{ls} \), and thus related to halo mass.
  - In this project:
    - A primary parameter on the **observability map**.
    - Converted to enclosed mass \( M(<\theta_E) \) given assumed cosmology and a representative \( z_s \).

---

### 2.2 Project-specific sampling terms

- **Parent sample (foreground galaxies)**
  - A curated set of **DR10 galaxies** that are plausible lenses.
  - Expected size: **~20,000–50,000** massive red galaxies (LRG-like).
  - Used for:
    - Creating real DR10 cutouts (backgrounds, noise, neighbors).
    - Providing the actual targets for final lens search.

- **Injection sample / injection set**
  - The subset of the parent sample into which we **inject synthetic lensed sources** for completeness tests.
  - Example scale:
    - 10,000–20,000 DR10 galaxies.
    - With ~1–3 distinct injected lens configurations per galaxy.
  - Used for:
    - Quantifying **selection function / completeness** in parameter space.

- **Source library (real-galaxy templates)**
  - A collection of **real galaxies from HST/COSMOS** used as background sources in the simulations.
  - Expected size: **hundreds to a few thousand** unique galaxies.
  - Each source is reused many times with different:
    - Lens mass parameters (e.g., \( \theta_E \), ellipticity, shear).
    - Redshifts and brightness scalings.
    - DR10 foregrounds and PSFs.

- **Cutout**
  - A small postage-stamp image centered on a DR10 galaxy, typically **g, r, z** bands stacked or as separate channels.
  - Before injection:
    - "Plain" DR10 galaxy plus neighbors and real noise.
  - After injection:
    - Same DR10 cutout **plus** a synthetic lensed arc/ring created by ray-tracing a COSMOS source through a lens model and convolving with the DR10 PSF.

Mapping summary:

> **Physics:** halo + source galaxy  
> → **Data:** DR10 foreground cutout + HST/COSMOS source image  
> → **Simulation:** lensed source (via SIE+shear) convolved with DR10 PSF and injected into DR10 cutout.

---

## 3. Data Sources and Their Roles

### 3.1 DESI Legacy Imaging Surveys DR10 (ground-based survey)

**Role:**  
Main survey for **foreground lenses and discovery**.

**What DR10 provides:**

- **Tractor catalogs**
  - Positions (RA, Dec).
  - Fluxes/magnitudes in g, r, z.
  - Shape parameters and size estimates.
  - Object classification flags.
  - Per-object PSF information (FWHM per band, or proxies via survey-CCDs).
- **Imaging**
  - g, r, z band images at ~0.262″/pixel with typical seeing FWHM ~1.0–1.5″.
  - Used to build cutouts for:
    - Injection simulations.
    - Final lens-search inference.

**Use in this project:**

- Define the **parent sample** of possible lenses (massive red galaxies).
- Provide **real backgrounds** and **real PSFs** for injection–recovery.
- Supply the **data where real strong lenses are actually discovered**.

---

### 3.2 HST/COSMOS (space-based source morphology library)

**Role:**  
Provide **realistic high-resolution galaxy morphologies** for background sources in simulations.

**What COSMOS provides:**

- HST/ACS imaging (e.g., F814W) over ~2 deg² with:
  - Pixel scale ~0.05″.
  - PSF FWHM ~0.06–0.1″ (space-based; atmosphere-free).
- COSMOS/COSMOS2020 catalogs:
  - Photometric redshifts.
  - Multi-band photometry.

**Use in this project:**

- Select a **source library** of real galaxies (star-forming, clumpy, varied).
- In the simulator:
  - Treat COSMOS postage stamps as **intrinsic surface brightness distributions** (via "INTERPOL" / "RealGalaxy" type models).
  - Ray-trace each source through a parametric lens mass model (SIE+shear).
  - Convolve the resulting arcs with the **DR10 PSF** to match ground-based resolution.
  - Inject into real DR10 cutouts.

This combination yields **physically realistic, survey-matched lensed arcs** for training and completeness tests.

---

### 3.3 Strong-lens catalogues (for validation and cross-matching)

**Role:**  
Provide **known strong lenses** to validate the method and quantify recovery.

Relevant sources:

- Published DR10/DR9 strong-lens searches (e.g., Huang, Storfer, Inchausti and collaborators).
- These catalogues typically provide:
  - Positions (RA, Dec).
  - Basic classification / confidence grades.
  - Sometimes lens redshift and Einstein radius estimates.

**Use in this project:**

- Cross-match with the DR10 **parent sample**.
- Measure:
  - Recovery rate of known lenses by the trained detector.
  - Dependence of recovery on θ_E, redshift, brightness.
- Use recovered known lenses as anchor points on the observability map.

---

### 3.4 Optional / stretch datasets

These are **not required** for the core DR10 project but may be used as enhancements if time permits.

- **Euclid Q1 VIS imaging**
  - High-resolution space-based imaging over ~63 deg².
  - Possible uses:
    - Extra source morphologies (small subset) for diversity.
    - Cross-survey comparison in overlapping DR10–Euclid regions:
      - Show that Euclid finds small-θ_E lenses that DR10 + our pipeline cannot, confirming the seeing wall.

- **Other HST fields or COSMOS-based generative models**
  - Additional morphological diversity for the source library.
  - Only if the core pipeline is already robust.

---

## 4. Phase Mapping (High-Level)

- **Phase 1 – Analytic Observability**
  - Inputs: DR10 seeing, cosmology, assumed θ_E thresholds, simple z_s assumptions.
  - Output: Analytic "seeing wall" and mass–redshift observability maps.

- **Phase 2 – Parent Sample & Metadata (DR10)**
  - Inputs: DR10 Tractor catalogs, DR10 seeing/PSF metadata.
  - Output: Clean parent sample of ~20k–50k potential deflectors with all needed metadata (RA, Dec, mags, z_l, PSFs, flags).

- **Phase 3 – Cutouts & PSFs**
  - Inputs: Parent sample, DR10 imaging.
  - Output: g/r/z cutouts and per-object PSF models for each parent galaxy.

- **Phase 4 – Real-Galaxy Injection Engine (HST + DR10)**
  - Inputs: DR10 cutouts + PSFs, COSMOS source library, lens mass models.
  - Output: Large set of realistic injected-lens cutouts with known parameters.

- **Phase 5–6 – Detector Training & Injection–Recovery**
  - Inputs: Injected cutouts (positives), real DR10 cutouts (negatives + hard negatives).
  - Output: Trained detector; completeness and purity as functions of θ_E, z_l, source properties.

- **Phase 7 – Real DR10 Lens Search & Validation**
  - Inputs: Trained detector, full DR10 parent sample, known lens catalogs.
  - Output: Recovery stats for known lenses; list of new high-quality candidates.

- **Phase 8 – Interpretation & Observability Map**
  - Inputs: Analytic Phase-1 maps, empirical completeness from injections, real lens sample.
  - Output: Final "dark halo observability window" plots and scientific conclusions.

---

## 5. Ground vs Space: Atmospheric Effects

- **Ground-based (DR10)**
  - Suffers from **atmospheric seeing**:
    - PSF FWHM ~1.0–1.5″.
    - Smears small Einstein radii and fine structure.
  - This sets the **fundamental resolution limit** ("seeing wall") we quantify in Phase 1.

- **Space-based (HST/COSMOS and optionally Euclid VIS)**
  - No atmospheric seeing; PSF FWHM ~0.05–0.1″.
  - Provides sharp, clumpy galaxy morphologies for **source templates**.
  - In overlap regions, Euclid/HST can reveal lenses that DR10 cannot, independently confirming the limits measured in this project.

In our simulations, we **start** with space-quality source structure (HST), then **apply** the DR10 PSF and noise, so that the detector and completeness maps are matched to the actual DR10 data, not to idealized images.

---

