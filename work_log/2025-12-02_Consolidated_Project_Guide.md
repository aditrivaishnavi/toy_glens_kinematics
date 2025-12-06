# Comprehensive Project Guide: Detecting Dark Matter Subhalos Using Galaxy Kinematics

*Consolidated on: December 2, 2025*

---

## Background: What Are We Trying to Do?

### The Scientific Problem

**Dark matter subhalos** are small clumps of dark matter that orbit within larger dark matter halos around galaxies. They are predicted by cosmological simulations but are extremely difficult to detect directly because dark matter doesn't emit light.

**Strong gravitational lensing** happens when a massive foreground galaxy (the "lens") bends the light from a background galaxy (the "source") so much that we see multiple distorted images, often forming arcs or rings. If a dark matter subhalo sits near one of these bright arcs, it will cause a **small local perturbation** in the arc's shape and velocity pattern.

### Our Hypothesis

We believe that using **both the brightness map (flux) AND the velocity map (kinematics)** of a lensed galaxy should help detect subhalos better than using brightness alone. The velocity field (how fast different parts of the galaxy are moving toward or away from us) might reveal subtle gravitational effects that flux alone misses.

### What We're Building

A machine learning pipeline that:
1. Takes real galaxy data from the MaNGA survey
2. Simulates what these galaxies would look like if they were gravitationally lensed (with and without subhalos)
3. Trains neural networks to distinguish lenses with subhalos from those without
4. Tests whether adding velocity information improves detection accuracy

---

## Source Data: MaNGA Survey

### What is MaNGA?

**MaNGA** stands for **Mapping Nearby Galaxies at Apache Point Observatory**. It's part of the Sloan Digital Sky Survey (SDSS) and uses a technique called **Integral Field Spectroscopy** to observe nearby galaxies.

### How Does MaNGA Work?

Traditional telescopes take a single image of a galaxy. MaNGA is different—it uses fiber bundles that capture **spectra at thousands of points across the galaxy simultaneously**. This means for each pixel-like region (called a "spaxel"), we get a full spectrum of light.

From these spectra, scientists extract:
- **Emission line fluxes** — how bright specific spectral lines are (we use Hα, the red hydrogen line)
- **Emission line velocities** — how fast the gas is moving toward or away from us (Doppler shift)
- **Stellar velocities** — how fast the stars are moving
- **Velocity dispersions** — how "turbulent" the motion is

### What Files Do We Use?

| File Type | Contents | Purpose |
|-----------|----------|---------|
| **MAPS files** | 2D maps extracted from spectra | Our main input: contains flux and velocity at each spatial position |
| **dapall catalog** | Summary statistics for all MaNGA galaxies | Used to rank and select good galaxy candidates |
| **drpall catalog** | Data reduction metadata | Contains physical properties like galaxy size, inclination |

### What Do We Extract from MAPS Files?

Each MAPS file contains 35 emission lines. We specifically use **Hα (H-alpha)**, which is:
- The brightest emission line in star-forming galaxies
- Index 24 in the MaNGA data arrays
- Measured in units of 10⁻¹⁷ erg/s/cm²/spaxel (brightness) and km/s (velocity)

We extract two 2D maps:
1. **EMLINE_GFLUX** — Gaussian-fitted emission line flux (brightness)
2. **EMLINE_GVEL** — Gaussian-fitted emission line velocity (motion toward/away from us)

**Important:** We use **Hα gas velocities** from EMLINE_GVEL (index 24) for the kinematic channel, **not stellar velocities** (STELLAR_VEL). Gas velocities trace the motion of ionized hydrogen in star-forming regions, which typically shows cleaner rotation patterns than stellar velocities in disk galaxies.

---

## Complete Pipeline Walkthrough

### Step 1: Ranking Galaxies by Kinematic Quality

**What this step does:**
Not all MaNGA galaxies are suitable for our experiment. We need galaxies that are:
- Rotating disks (so the velocity field shows a clear pattern)
- Well-observed (low mask fraction)
- Bright in Hα emission (so we have strong signal)

**How it works:**
The script `rank_manga_disks.py` reads the survey catalogs and computes a ranking score based on:

| Metric | What It Measures | Why It Matters |
|--------|------------------|----------------|
| `HA_GSIGMA_1RE` | Gas velocity dispersion within 1 effective radius | Lower is better—indicates coherent rotation rather than chaotic motion |
| `STELLAR_SIGMA_1RE` | Stellar velocity dispersion | Lower values indicate disk-dominated (not bulge-dominated) galaxies |
| `EMLINE_GFLUX_1RE[24]` | Hα flux brightness | Higher is better—more signal to work with |
| Inclination | How tilted the galaxy is relative to our line of sight | Moderate inclination (not face-on, not edge-on) gives best velocity measurement |

**Output:** `manga_disk_candidates.csv` — a ranked list of galaxy identifiers with URLs to download the full data files.

---

### Step 2: Downloading Selected Galaxies

**What this step does:**
Downloads the actual MAPS data files for our top-ranked candidate galaxies from the SDSS archive.

**How it works:**
The script `download_manga_maps_from_csv.py` reads the ranked CSV file and downloads FITS files (a standard astronomy file format) to `data/maps/`.

**Output:** 10 MAPS files, each about 30-50 MB, named like `manga-8593-12705-MAPS-HYB10-MILESHC-MASTARHC2.fits.gz`

The naming convention is: `manga-{PLATE}-{IFU_DESIGN}-MAPS-{MODEL_TYPE}.fits.gz`
- **PLATE** — the observation plate number
- **IFU_DESIGN** — the fiber bundle identifier
- Together they form the "plateifu" identifier like "8593-12705"

---

### Step 3: Batch Quality Inspection

**What this step does:**
Opens each downloaded MAPS file and computes detailed quality metrics to decide which galaxies are actually usable for our experiments.

**How it works:**
The script `batch_inspect_maps.py` extracts the Hα flux and velocity maps, then computes:

| Metric | What It Measures | Good Values |
|--------|------------------|-------------|
| `vel_grad` | Approximate velocity gradient across the galaxy (km/s per pixel; MaNGA pixel scale = 0.5 arcsec/pixel) | > 2.5 indicates strong rotation |
| `frac_valid_in_1.5Re` | Fraction of pixels with valid data within 1.5 effective radii | > 0.60 means good coverage |
| `mask_fraction` | Fraction of all pixels that are masked (invalid) | < 0.70 is acceptable |
| `flux_max` | Maximum brightness value | > 50 indicates detectable emission |
| `flux_vel_corr` | Correlation between flux and velocity | Should be near 0 for nice rotating disks |

**Flags computed:**
- `flag_low_rotation` — True if velocity gradient is too weak
- `flag_heavily_masked` — True if too many pixels are invalid
- `flag_low_flux` — True if emission is too faint

**Output:** `maps_quality_summary.csv` — detailed metrics for each galaxy, plus a `usable` column indicating whether each galaxy passes our quality thresholds.

---

### Step 4: Lensability Scoring

**What this step does:**
Assigns a numerical "lensability score" (0-100) and tier (elite/good/borderline/reject) to each galaxy based on how suitable it is for lensing simulations.

**How it works:**
The script `score_lensability.py` uses a scoring function from `src/utils/lensibility.py`:

**Hard rejection criteria (any of these = rejected):**
- Valid fraction within 1.5 effective radii < 60%
- Velocity gradient < 2.5 km/s per pixel (equivalent to ~5 km/s per arcsec at MaNGA's 0.5″/pixel scale)
- Mask fraction > 70%
- Maximum flux < 50 units

**Point-based scoring (when not rejected):**
- Higher velocity gradient → more points
- Higher valid fraction → more points
- Reasonable velocity amplitude → more points
- Low mask fraction → more points
- Near-zero flux-velocity correlation → more points

**Tiers:**
| Tier | Score Range | Meaning |
|------|-------------|---------|
| Elite | > 80 | Excellent for all experiments |
| Good | 60-80 | Suitable for most purposes |
| Borderline | 40-60 | Use with caution |
| Reject | < 40 or hard fail | Do not use |

**Output:** `maps_lensability_scored.csv` — original metrics plus `lens_score`, `lens_tier`, `lens_hard_reject`, and `lens_notes` columns.

---

### Step 5: Preparing Machine-Learning-Ready Tensors

**What this step does:**
Converts the variable-sized MAPS data into fixed-size, normalized tensors that can be fed directly into neural networks.

**How it works:**
The script `prep_source_maps.py` performs these operations for each galaxy:

#### 5.1 Extraction
- Opens the FITS file
- Extracts Hα flux map (index 24 from EMLINE_GFLUX cube)
- Extracts Hα velocity map (index 24 from EMLINE_GVEL cube)
- Extracts corresponding mask arrays

#### 5.2 Masking
- Identifies pixels where the mask = 0 (valid data)
- Sets invalid pixels to zero
- For velocity, requires BOTH flux mask AND velocity mask to be valid

#### 5.3 Flux Normalization
Goal: Map flux values to the range [0, 1]

Method:
1. Find the 5th percentile value (p5) and 99.5th percentile value (p99.5) of valid pixels
2. Subtract p5 from all values
3. Divide by (p99.5 - p5)
4. Clip to [0, 1]

This robust normalization prevents extreme outliers from dominating.

#### 5.4 Velocity Normalization
Goal: Map velocity values to approximately [-1, 1], centered at the systemic velocity

Method:
1. Find the median velocity of valid pixels (this is approximately the galaxy's systemic velocity)
2. Subtract the median (now 0 represents systemic velocity)
3. Find the 95th percentile of absolute deviations from median
4. Divide by this scale factor
5. Clip to [-1, 1]

This centers the velocity field and scales it consistently across galaxies.

#### 5.5 Resampling
Goal: Convert variable-sized maps to a fixed 64×64 grid

Method: Bilinear interpolation using scipy's zoom function

#### 5.6 Stacking
Combine into a single tensor of shape (2, 64, 64):
- Channel 0: Normalized flux
- Channel 1: Normalized velocity

#### 5.7 Compactness Metrics
Computed on the final resampled flux map:
- `total_flux` — sum of all pixel values
- `flux_frac_central_5x5` — fraction of flux in the central 25 pixels
- `r_half_pix` — radius (in pixels) containing 50% of total flux
- `r80_pix` — radius containing 80% of total flux

**Outputs:**
- `data/source_tensors/source_tensor_{plateifu}.npy` — the 2-channel tensor
- `data/source_tensors/previews/preview_{plateifu}.png` — visual sanity check
- `data/source_tensors/compactness_metrics.csv` — metrics for all processed galaxies

---

### Step 6: Toy Lensing Demonstration

**What this step does:**
Verifies that our gravitational lensing simulation works correctly by applying a simple lens model to one of our galaxy tensors.

**How it works:**
The script `sis_demo_lens.py` and the centralized module `src/glens/lensing_utils.py` implement a **Singular Isothermal Sphere (SIS)** lens model.

#### What is an SIS lens?
The SIS is the simplest realistic model for a gravitational lens. It assumes the lensing mass (typically a galaxy) has a density profile that falls off as 1/r². This produces:
- An **Einstein ring** when the source is perfectly aligned behind the lens
- **Arcs** when slightly misaligned

#### The Lens Equation
For each pixel position θ in the image plane (what we observe), we trace back to find where it came from in the source plane (the unlensed galaxy):

```
β = θ - α(θ)
```

Where:
- **θ** = position in image plane (angular coordinates)
- **α** = deflection angle (how much the light is bent)
- **β** = corresponding position in source plane

For SIS: `α = θ_E × θ/|θ|`
- **θ_E** = Einstein radius (determines lens strength, typically ~1 arcsecond for galaxy lenses)

#### Ray Tracing Both Channels
**Critical insight:** The same deflection applies to both flux and velocity because gravitational lensing only bends light paths—it doesn't change the light's properties.

Process:
1. Build a grid of image-plane coordinates [-1, 1] × [-1, 1]
2. Compute deflection at each point using SIS formula
3. Calculate source-plane positions β = θ - α
4. Sample the source flux at positions β using bilinear interpolation
5. Sample the source velocity at positions β using bilinear interpolation
6. Apply flux-based masking: set velocity to 0 wherever lensed flux is below threshold (0.1)

#### Why Flux-Based Velocity Masking?
In real observations, we can only measure velocity where there's detectable emission. The "velocity = 0" can mean two things:
- The gas is actually at systemic velocity
- There's no gas there (masked region)

By masking velocity based on flux, we make "velocity = 0" unambiguously mean "no measurement here."

**Output:** Visual demonstration showing:
- Original flux and velocity maps
- Lensed flux (Einstein ring)
- Lensed velocity ("kinematic horseshoe" pattern)

---

### Step 7: Baseline CNN — Lensed vs Unlensed Classification

**What this step does:**
Trains a simple neural network to distinguish unlensed galaxies from lensed ones. This is a **sanity check** to verify the entire pipeline works.

**Why this is expected to be easy:**
An unlensed galaxy looks like a blob or disk. A lensed galaxy looks like a ring or arc. These are dramatically different morphologies that any image classifier should detect trivially.

**How it works:**
The script `toy_cnn.py` implements:

#### Dataset Generation
For each training sample:
1. Randomly pick a galaxy from the training set
2. Flip a coin: 50% unlensed, 50% lensed
3. If lensed: apply SIS lens with random Einstein radius (0.3-0.7)
4. Apply random augmentations (rotations, flips, small noise)
5. Return the 2-channel tensor with label (0=unlensed, 1=lensed)

#### Neural Network Architecture
```
Input: (batch, 2, 64, 64)  # 2 channels, 64×64 pixels

Conv2d(2→16, kernel=3×3) → ReLU → MaxPool(2×2)
Conv2d(16→32, kernel=3×3) → ReLU → AdaptiveAvgPool → Flatten
Linear(32→2)  # 2 output classes

Output: logits for [unlensed, lensed]
```

#### Training Setup
- Loss function: Cross-entropy
- Optimizer: Adam with learning rate 0.001
- Train set: 4 galaxies, 4000 synthetic samples
- Validation set: 1 galaxy, 800 samples
- Test set: 1 galaxy, 800 samples
- Epochs: 10

#### Galaxy Splits (by plateifu)
| Set | Galaxies |
|-----|----------|
| Train | 8593-12705, 8993-12705, 11982-9102, 10500-12703 |
| Validation | 8652-12703 |
| Test | 9487-9102 |

This split is **by galaxy** to prevent data leakage—the model never sees augmented versions of test galaxies during training.

**Results:**
- Training accuracy: 100%
- Validation accuracy: 100%
- Test accuracy: 100%

**Interpretation:** As expected, this task is trivial. The model easily distinguishes ring morphology from blob morphology. This confirms the pipeline is working but is NOT a scientific result.

---

### Step 8: Subhalo Detection Experiments

**What this step does:**
The main scientific experiment: Can a neural network distinguish a **smooth gravitationally lensed galaxy** from one with a **dark matter subhalo** causing a local perturbation?

**How it works:**
The script `subhalo_cnn.py` and `src/glens/lensing_utils.py` implement:

#### Adding a Subhalo
A subhalo is modeled as a second, smaller SIS lens added to the main lens:

```
Total deflection = Main SIS deflection + Subhalo SIS deflection
```

The subhalo is placed "on-arc" — just outside the Einstein ring where its effect will be most visible:
- Radial position: `r = θ_E_main × (1 + offset)` where offset ∈ [0.1, 0.3], placing the subhalo at 110%-130% of the Einstein radius (i.e., 10-30% beyond the ring)
- Azimuthal position: random angle around the ring

#### Physics Parameters (Configurable via Command Line)

| Parameter | Meaning | Default | Experiment Values |
|-----------|---------|---------|-------------------|
| `theta_E_main` | Main lens Einstein radius | 0.5 | Fixed at 0.5 |
| `theta_E_sub_factor` | Subhalo strength as fraction of main | 0.1 | 0.1 (weak) or 0.3 (strong) |
| `psf_sigma` | Gaussian blur simulating telescope optics (pixels) | 1.0 | 0.0 (none) to 1.0 |
| `flux_noise_sigma` | Gaussian noise added to flux channel | 0.08 | 0.0 to 0.08 |
| `vel_noise_sigma` | Gaussian noise added to velocity channel | 0.03 | 0.0 to 0.03 |
| `flux_mask_threshold` | Flux below which velocity is masked | 0.1 | Fixed at 0.1 |

#### Point Spread Function (PSF)
Real telescopes don't produce perfectly sharp images—light from each point is spread into a small blur. We simulate this with Gaussian convolution applied to both flux and velocity maps.

#### Channel Mode Ablations
To understand which information source is important, we train three variants:
- `both` — CNN sees flux and velocity channels
- `flux_only` — CNN sees flux channel, velocity channel is zeroed
- `vel_only` — CNN sees velocity channel, flux channel is zeroed

#### Experiment Results

**Easy Case (Strong subhalo, no blur, no noise):**
| Channel Mode | Training Accuracy | Validation Accuracy | Test Accuracy |
|--------------|-------------------|---------------------|---------------|
| Both channels | 100% | 100% | 100% |
| Flux only | 100% | 100% | 100% |
| Velocity only | 53-60% | 48-54% | 52-59% |

**Moderate Case (Strong subhalo, some blur and noise):**
| Channel Mode | Training Accuracy | Validation Accuracy | Test Accuracy |
|--------------|-------------------|---------------------|---------------|
| Both channels | ~100% | ~100% | ~100% |
| Flux only | ~100% | ~100% | ~100% |
| Velocity only | 53-57% | 48-54% | ~58% |

**Hard Case (Weak subhalo, realistic blur and noise):**
| Channel Mode | Training Accuracy | Validation Accuracy | Test Accuracy |
|--------------|-------------------|---------------------|---------------|
| Both channels | 55-58% | 48-55% | ~50-55% |

---

### Step 9: Visual Debugging

**What this step does:**
Generates diagnostic images to visually verify that subhalo perturbations are being simulated correctly and to understand why velocity-only detection fails.

**How it works:**
The script `debug_subhalo_samples.py` creates a 3×2 panel showing:

| Row | Flux Column | Velocity Column |
|-----|-------------|-----------------|
| Row 1 | Smooth SIS lens (no subhalo) | Smooth velocity pattern |
| Row 2 | SIS + subhalo (no blur/noise) | Perturbed velocity pattern |
| Row 3 | SIS + subhalo (with blur/noise) | Degraded perturbed pattern |

**What the debug images reveal:**
- In the clean case, the subhalo creates a visible "kink" or distortion in the arc
- After applying PSF blur and noise, this distortion becomes much harder to see
- The velocity perturbation is even more subtle than the flux perturbation
- With realistic degradation, the velocity signal is essentially invisible

---

## Key Scientific Findings

### What We Learned

1. **Flux morphology dominates subhalo detection**
   - Even with moderate blur and noise, the neural network can detect subhalos using only the flux channel
   - The morphological distortion (the "kink" in the arc) is the primary signal

2. **Velocity alone cannot detect subhalos in this setup**
   - Velocity-only experiments hover around 50% accuracy (random chance for binary classification)
   - This is true even with zero noise and strong subhalos
   - The kinematic perturbation from a subhalo is too subtle relative to the overall velocity pattern

3. **Adding velocity to flux doesn't improve performance**
   - Flux+velocity performs the same as flux-only
   - The velocity channel provides no additional discriminative information in this experiment

4. **The experiment is now in a "scientifically honest" regime**
   - We've confirmed that our simulations work correctly
   - We've identified the limitations of velocity-based detection
   - Future work should explore scenarios where velocity might actually help

---

## File Summary

### Scripts (in `src/scripts/`)

| Script | Purpose |
|--------|---------|
| `rank_manga_disks.py` | Rank MaNGA galaxies by kinematic quality using catalog data |
| `download_manga_maps_from_csv.py` | Download MAPS files from SDSS archive |
| `batch_inspect_maps.py` | Compute quality metrics for downloaded galaxies |
| `score_lensability.py` | Assign lensability scores and tiers |
| `prep_source_maps.py` | Convert MAPS to normalized 2-channel tensors |
| `sis_demo_lens.py` | Demonstrate SIS lensing on a single galaxy |
| `toy_cnn.py` | Train lensed vs unlensed classifier (sanity check) |
| `subhalo_cnn.py` | Train subhalo detector with configurable physics |
| `debug_subhalo_samples.py` | Generate visual diagnostics for subhalo experiments |

### Core Modules (in `src/`)

| Module | Purpose |
|--------|---------|
| `glens/lensing_utils.py` | Centralized lensing physics (deflection, sampling, PSF, noise) |
| `utils/lensibility.py` | Lensability scoring function |
| `data_io/fits_loader.py` | Generic FITS file loading |
| `data_io/manga_extractor.py` | MaNGA-specific data extraction |

### Data Files (in `data/`)

| File/Directory | Contents |
|----------------|----------|
| `maps/` | Downloaded MaNGA MAPS FITS files |
| `source_tensors/` | Processed 2-channel tensors (.npy files) |
| `source_tensors/previews/` | Visual sanity check images |
| `source_tensors/compactness_metrics.csv` | Compactness metrics for each galaxy |
| `maps_quality_summary.csv` | Quality metrics from batch inspection |
| `maps_lensability_scored.csv` | Lensability scores and tiers |
| `usable_maps_index.txt` | List of 8 usable MAPS filenames |
| `manga_disk_candidates.csv` | Ranked list of all MaNGA disk candidates |
| `debug_subhalo/` | Diagnostic images from debugging script |

### Trained Models (in `models/`)

| Model | What It Classifies | Configuration |
|-------|-------------------|---------------|
| `toy_cnn.pt` | Unlensed vs lensed | Sanity check (trivial) |
| `subhalo_cnn.pt` | Smooth vs subhalo | Default settings |
| `subhalo_cnn_easy_flux_only.pt` | Smooth vs subhalo | Strong subhalo, no noise, flux only |
| `subhalo_cnn_easy_vel_only.pt` | Smooth vs subhalo | Strong subhalo, no noise, velocity only |
| `subhalo_cnn_easy_flux_vel.pt` | Smooth vs subhalo | Strong subhalo, no noise, both channels |
| `subhalo_cnn_mid_noise_flux_only.pt` | Smooth vs subhalo | Strong subhalo, moderate noise, flux only |
| `subhalo_cnn_mid_noise_vel_only.pt` | Smooth vs subhalo | Strong subhalo, moderate noise, velocity only |
| `subhalo_cnn_mid_noise_flux_vel.pt` | Smooth vs subhalo | Strong subhalo, moderate noise, both channels |

---

## Recommended Next Steps

Based on the work log conclusions:

1. **Design a "velocity matters" experiment**
   - Create scenarios where flux morphology is deliberately similar between classes
   - Only the velocity field differs
   - This would isolate whether velocity can carry discriminative information at all

2. **Improve subhalo physics**
   - Replace simple SIS subhalo with more realistic truncated NFW profile (using lenstronomy library)
   - Model velocity dispersion enhancement near subhalo
   - Use higher resolution (128×128) and smaller PSF

3. **Document and consolidate**
   - Write up current findings for project reports
   - Create clear figures showing the experimental progression
   - Lock in achievements before expanding scope

---

## Glossary

| Term | Definition |
|------|------------|
| **Dark matter subhalo** | A small clump of dark matter orbiting within a larger dark matter halo |
| **Gravitational lensing** | Bending of light by gravity, causing background objects to appear distorted |
| **Einstein ring** | A circular arc formed when source, lens, and observer are perfectly aligned |
| **Einstein radius (θ_E)** | The angular radius of the Einstein ring; characterizes lens strength |
| **SIS (Singular Isothermal Sphere)** | A simple lens model with density ∝ 1/r² |
| **MaNGA** | Mapping Nearby Galaxies at Apache Point Observatory survey |
| **MAPS file** | 2D data product from MaNGA containing flux, velocity, and other maps |
| **Plateifu** | Unique galaxy identifier in MaNGA (plate number + fiber bundle ID) |
| **Hα (H-alpha)** | Hydrogen emission line at 656.3 nm; bright in star-forming regions |
| **Spaxel** | Spatial pixel in integral field spectroscopy data |
| **PSF (Point Spread Function)** | How a telescope blurs point sources; we model as Gaussian |
| **Flux** | Brightness or intensity of light |
| **Velocity field** | Map of line-of-sight velocities across a galaxy |
| **Ray tracing** | Computing where light rays originate by reversing through the lens equation |

---

*This guide consolidates all project documentation as of December 2, 2025.*

