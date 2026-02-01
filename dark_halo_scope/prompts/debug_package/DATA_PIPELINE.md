# Data Pipeline Documentation

## DECaLS DR10 Overview

**Source**: DESI Legacy Imaging Surveys Data Release 10
- URL: https://www.legacysurvey.org/dr10/
- Coverage: ~20,000 sq deg of sky
- Bands: g, r, z (optical)
- Pixel scale: 0.262 arcsec/pixel
- Typical PSF FWHM: 1.0-1.5 arcsec

**Key Data Products Used**:
- `sweep` catalogs: Galaxy positions, photometry, morphology
- `coadd` images: Stacked images for cutout extraction
- `psfsize_*`: PSF FWHM per band at each position

---

## Phase 3: Parent Sample Selection

**Goal**: Select Luminous Red Galaxies (LRGs) suitable for lens injection

**Selection Criteria**:
```python
# Morphology: Extended sources only
type != 'PSF'  # Not point sources

# Brightness: Visible but not saturated
r_mag > 17 and r_mag < 21

# Color: Red sequence (LRG-like)
g - r > 0.5
r - z > 0.2

# Quality: Good photometry
nobs_g >= 1 and nobs_r >= 1 and nobs_z >= 1
fracmasked_g < 0.3  # Not too masked

# Footprint: In DECaLS south
dec < 32  # Southern survey
```

**Output**: ~2M galaxies with positions, photometry, PSF info

---

## Phase 4a: Injection Manifest Creation

**Goal**: Define which synthetic lenses to inject into which galaxies

**Injection Grid (v4_sota)**:
```python
theta_e_arcsec = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]  # Einstein radius
src_dmag = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]  # Source magnitude below lens
src_reff_arcsec = [0.1, 0.2, 0.3, 0.5]  # Source effective radius
src_e = [0.2, 0.4, 0.6]  # Source ellipticity
shear = [0.0, 0.05, 0.1]  # External shear
```

**Control Strategy (THE PROBLEM)**:
```python
# PAIRED CONTROLS - Current implementation
# For each positive (galaxy + lens), control = same galaxy without lens
control_frac = 0.5  # 50% of samples are controls
# Control uses SAME galaxy position, just with theta_e = 0 (no lens)
```

**Why Paired Controls Are Problematic**:
- Model sees: Galaxy A with lens, Galaxy A without lens
- Model can learn: "Is there extra flux in the center?" 
- Model should learn: "Is there an arc-shaped morphology?"

**Better Approach (Not Yet Implemented)**:
```python
# UNPAIRED CONTROLS
# Controls come from DIFFERENT galaxies, matched on PSF/depth
# Galaxy A → positive (with lens)
# Galaxy B → control (no lens, but similar observing conditions)
```

---

## Phase 4c: Stamp Injection

**Goal**: Inject synthetic lensed arcs into real galaxy cutouts

**Injection Physics**:
```python
# Lens Model: Singular Isothermal Ellipsoid (SIE)
# Source: Sersic profile galaxy

def inject_lens(galaxy_cutout, params):
    # 1. Create source galaxy (Sersic profile)
    source = sersic_profile(reff=params.src_reff, ellip=params.src_e, mag=params.src_dmag)
    
    # 2. Apply gravitational lensing (SIE model)
    lensed_arcs = sie_lens(source, theta_e=params.theta_e, shear=params.shear)
    
    # 3. Convolve with PSF (GAUSSIAN - another limitation)
    lensed_arcs = convolve_gaussian(lensed_arcs, fwhm=psf_fwhm)
    
    # 4. Add to galaxy cutout
    result = galaxy_cutout + lensed_arcs
    return result
```

**PSF Modeling Issue**:
- Current: Gaussian PSF approximation
- Reality: DECaLS PSFs have wings (better modeled by Moffat)
- Impact: Synthetic arcs may look different from real arcs

**Output**: 64x64 pixel stamps stored as Parquet
```python
# Parquet schema (simplified)
stamp_npz: bytes  # Compressed numpy array (64, 64, 3) for g,r,z
is_control: int  # 0 = positive (lens injected), 1 = control
theta_e_arcsec: float  # Einstein radius (0 for controls)
psf_fwhm_used_r: float  # PSF FWHM used for injection
psfsize_r: float  # Actual PSF FWHM from catalog
arc_snr: float  # Signal-to-noise of injected arc
# ... many more columns
```

---

## Dataset Statistics (v4_sota)

**Training Set**:
- Total: ~1.3M samples
- Positives: ~650K (50%)
- Controls: ~650K (50%)

**Validation Set** (used in training):
- Total: 128,000 samples
- Positives: 62,910 (49%)
- Controls: 65,090 (51%)

**Injection Parameter Distribution**:
```
theta_e/PSF ratio (resolvability):
  < 0.5 (unresolved): ~60% of positives
  0.5-1.0 (marginally resolved): ~25%
  > 1.0 (well resolved): ~15%
```

**This is another problem**: 60% of positives have Einstein radius smaller than PSF FWHM, making the arc undetectable by morphology.

---

## Key Issues Identified in Data Pipeline

### Issue 1: Paired Controls
- Controls are same galaxy as positives without injection
- Model can learn shortcuts instead of arc morphology

### Issue 2: Gaussian PSF
- Real PSFs have extended wings (Moffat profile)
- Synthetic arcs convolved with Gaussian look different from real

### Issue 3: 60% Unresolved Lenses
- Most injections have theta_e < PSF FWHM
- These look like point sources, not arcs
- Model cannot learn arc morphology from these

### Issue 4: No Hard Negative Mining
- Controls are random galaxies, not confusing ones
- Real false positives will be galaxies that LOOK like lenses
- Need to mine hard negatives after initial training

---

## Files in This Package Related to Data

- `spark_phase4_pipeline.py` - Main data generation code
- See `build_grid()` function for injection parameters
- See `inject_sie_stamp()` for lens injection physics
- See `add_control_params()` for control strategy

