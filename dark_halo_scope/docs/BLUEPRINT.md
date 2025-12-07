# Dark Halo Scope: Technical Blueprint

> **Working Title**: Which Dark Matter Halos Are Actually Visible?  
> **Subtitle**: The Strong Lensing Selection Function of DESI Legacy DR10

*This document serves as the complete project specification.*

---

## Table of Contents

1. [Project Definition and Success Criteria](#0-project-definition-and-success-criteria)
2. [Overall Phase Structure](#1-overall-phase-structure)
3. [Phase A – Repo and Layout](#2-phase-a--repo-and-layout)
4. [Phase 1 – Analytic Observability](#3-phase-1--analytic-observability)
5. [Phase 2 – DR10 Parent LRG Sample](#4-phase-2--dr10-parent-lrg-sample)
6. [Phase 3 – Background HDF5 Dataset](#5-phase-3--background-hdf5-dataset)
7. [Phase 4 – Simulation Engine (LensInjector)](#6-phase-4--simulation-engine-lensinjector)
8. [Phase 5 – Detection Operator + Active Learning](#7-phase-5--detection-operator--active-learning)
9. [Phase 6 – Injection Recovery and Selection Function](#8-phase-6--injection-recovery-and-selection-function)
10. [Phase 7 – DR10 Application and Candidates](#9-phase-7--dr10-application-and-candidates)
11. [Phase 8 – Optional Mass and f_DM](#10-phase-8--optional-mass-and-f_dm)
12. [Writing Phase](#11-phase-w--writing-and-figures)

---

## 0. Project Definition and Success Criteria

### Core Deliverable

A **quantitative, empirically calibrated selection function** for galaxy-scale strong lenses in DESI DR10, expressed primarily in terms of:

| Observable | Symbol | Description |
|------------|--------|-------------|
| **Einstein radius** | θ_E | Angular scale of lensing effect (arcseconds) |
| **Lens redshift** | z_l | Distance to the lensing galaxy |

### Methodology

The selection function is derived by:

1. **Combining analytic theory** with DR10 observing conditions
2. **Injecting realistic, clumpy lensed sources** into real DR10 LRG cutouts with correct PSFs and SEDs
3. **Using a single ML detector** as a calibrated instrument to measure completeness and purity
4. **Anchoring results** with known lenses and (potentially) new candidates

### Minimum Viable Success

> **If all else fails, you still have a strong physics project from the selection function alone.**

The analytic + simulation-based selection function is the core deliverable. Real lens discovery is a bonus.

---

## 1. Overall Phase Structure

Phases are ordered for **early risk reduction** — killing the hardest uncertainties first:

| Phase | Name | Goal | Risk Level |
|-------|------|------|------------|
| **A** | Repo Setup | Clean project structure | Low |
| **1** | Analytic Observability | θ_E physics without data | Low |
| **2** | DR10 Parent Sample | LRG catalog and selection | Medium |
| **3** | Background HDF5 | Cutouts with PSF/metadata | Medium-High |
| **4** | LensInjector | Realistic arc injection | **High** |
| **5** | Detection Operator | ML model + active learning | Medium |
| **6** | Injection Recovery | Selection function extraction | Medium |
| **7** | DR10 Application | Known lenses + candidates | Low |
| **8** | Mass Modeling | Optional f_DM estimates | Low |
| **W** | Writing | Integrated documentation | Ongoing |

### Priority Order

Get through **Phases 1, 2, 3, and minimal Phase 4 early** — these derisk the hardest parts:
- Data access and download infrastructure
- Realistic simulation pipeline

---

## 2. Phase A – Repo and Layout

### Goal

A clean, extensible project structure.

### Directory Structure

```
dark_halo_scope/
├── env/                          # Environment files (conda/pip)
│   └── requirements.txt
├── data/
│   ├── raw/                      # Downloaded data
│   ├── processed/                # Cleaned/prepared data
│   └── sims/                     # Generated simulations
├── src/
│   ├── physics/
│   │   ├── __init__.py
│   │   ├── lens_equations.py     # θ_E calculations
│   │   └── observability_maps.py # Seeing-limited regions
│   ├── dr10/
│   │   ├── __init__.py
│   │   ├── query_catalog.py      # Tractor catalog queries
│   │   ├── download_cutouts.py   # Image retrieval
│   │   └── build_background_hdf5.py
│   ├── sims/
│   │   ├── __init__.py
│   │   ├── cosmos_loader.py      # COSMOS/HST sources
│   │   ├── lens_injector.py      # Core injection engine
│   │   ├── preprocess.py         # Raw → ML-ready
│   │   └── build_sim_datasets.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── detector.py           # ResNet-18 + metadata
│   │   ├── train_detector.py
│   │   └── eval_detector.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── injection_recovery.py
│   │   ├── selection_function.py
│   │   └── real_dr10_analysis.py
│   └── utils/
│       ├── __init__.py
│       ├── plotting.py
│       └── io.py
├── notebooks/
│   ├── 01_analytic_observability.ipynb
│   ├── 02_dr10_background_qc.ipynb
│   ├── 03_sim_qc.ipynb
│   └── 04_injection_recovery_plots.ipynb
├── docs/
│   └── BLUEPRINT.md              # This file
└── README.md
```

### Environment

| Requirement | Version |
|-------------|---------|
| Python | 3.10+ |
| numpy | latest |
| scipy | latest |
| pandas | latest |
| astropy | latest |
| h5py | latest |
| matplotlib | latest |
| torch | latest |
| torchvision | latest |
| lenstronomy | latest |

---

## 3. Phase 1 – Analytic Observability

### Goal

Get the θ_E and seeing physics clear and coded — **no DR10 data required yet**.

### Implementation

#### `src/physics/lens_equations.py`

```python
def theta_E_SIS(sigma_v: float, z_l: float, z_s: float, cosmo) -> float:
    """
    Compute Einstein radius for a Singular Isothermal Sphere.
    
    Parameters
    ----------
    sigma_v : float
        Velocity dispersion in km/s
    z_l : float
        Lens redshift
    z_s : float
        Source redshift
    cosmo : astropy.cosmology
        Cosmology object
    
    Returns
    -------
    theta_E : float
        Einstein radius in arcseconds
    """
    pass

def sigma_v_from_Mhalo(M_halo: float, z_l: float) -> float:
    """
    Simple scaling relation from halo mass to velocity dispersion.
    """
    pass
```

#### `src/physics/observability_maps.py`

```python
# Representative DR10 FWHM values (arcseconds)
DR10_FWHM = {
    'g': 1.3,  # typical seeing
    'r': 1.2,
    'z': 1.1
}

def confusion_regions(theta_E: float, FWHM: float) -> str:
    """
    Classify observability regime.
    
    Returns
    -------
    str : One of 'blind', 'low_trust', 'good'
    
    Thresholds:
    - blind: θ_E < 0.5 × FWHM
    - low_trust: 0.5 × FWHM ≤ θ_E < 1.0 × FWHM
    - good: θ_E ≥ 1.0 × FWHM
    """
    pass

def evaluate_observability_grid(
    sigma_v_range: np.ndarray,
    z_l_range: np.ndarray,
    z_s_range: np.ndarray,
    cosmo
) -> dict:
    """
    Compute θ_E and classification over a parameter grid.
    """
    pass
```

### Notebook: `01_analytic_observability.ipynb`

Required plots:

1. **θ_E vs σ_v** for several (z_l, z_s) combinations
2. **θ_E vs z_l** for fixed σ_v and multiple z_s values
3. **Shaded regions** showing:
   - 🔴 Blind (θ_E < 0.5 × FWHM)
   - 🟡 Low trust (0.5–1.0 × FWHM)
   - 🟢 Good (θ_E > FWHM)

### Milestone Checklist

- [ ] `lens_equations.py` implemented and tested
- [ ] `observability_maps.py` implemented
- [ ] Notebook with all required plots
- [ ] Markdown summary stating: *"θ_E and z_l are our primary observables; mass inferences will assume z_s or a simple P(z_s)."*

---

## 4. Phase 2 – DR10 Parent LRG Sample

### Goal

Get a fixed, reproducible parent sample of likely lens galaxies.

### Implementation

#### `src/dr10/query_catalog.py`

```python
def query_dr10_region(
    ra_min: float, ra_max: float,
    dec_min: float, dec_max: float,
    output_path: str = "data/raw/dr10_catalog_region.parquet"
) -> pd.DataFrame:
    """
    Query DR10 Tractor catalog for a rectangular region.
    """
    pass

def apply_lrg_cuts(catalog: pd.DataFrame) -> pd.DataFrame:
    """
    Apply LRG-like selection cuts.
    
    Cuts:
    - TYPE in {SER, DEV, REX, EXP} (not PSF)
    - r < 20.0–20.5 (bright)
    - g−r and r−z color cuts for red galaxies
    """
    pass
```

### Output Files

| File | Contents |
|------|----------|
| `data/raw/dr10_catalog_region.parquet` | Full catalog query result |
| `data/processed/parent_sample.parquet` | LRG-selected sample (~30k–50k objects) |

### Notebook: `02_dr10_background_qc.ipynb`

Required content:

- Color–magnitude diagrams (g−r vs r, r−z vs r)
- Object count summary
- Confirmation of red, massive population
- Distribution of morphological types

### Milestone Checklist

- [ ] `query_catalog.py` implemented
- [ ] `parent_sample.parquet` created
- [ ] QC notebook complete with color-magnitude plots
- [ ] Confirmed ~30k–50k LRG candidates

---

## 5. Phase 3 – Background HDF5 Dataset

### Goal

Build the background dataset **once**, in raw flux units, with PSF and metadata.

### Key Design Decision

> **Injection happens in raw flux units. Preprocessing is applied uniformly AFTER injection.**

This ensures simulated and real data go through identical processing.

### Implementation

#### `src/dr10/download_cutouts.py`

```python
def download_cutouts(
    parent_sample: pd.DataFrame,
    output_dir: str = "data/raw/cutouts/",
    size_pix: int = 128,
    pixel_scale: float = 0.262  # arcsec/pixel
) -> None:
    """
    Download g, r, z cutouts for each object in parent sample.
    
    Cutout size: 128×128 pixels = 33.5 arcsec
    """
    pass
```

#### `src/dr10/build_background_hdf5.py`

```python
def build_background_hdf5(
    cutout_dir: str,
    parent_sample: pd.DataFrame,
    output_path: str = "data/processed/dr10_backgrounds_raw.h5"
) -> None:
    """
    Build HDF5 with raw backgrounds + metadata.
    
    HDF5 structure:
    - /images_raw: (N, 3, 128, 128) float32, raw flux units
    - /psf_fwhm: (N, 3) float32, FWHM per band
    - /meta: (N, K) float32, catalog metadata
    - /ids: (N,) object IDs
    """
    pass
```

### HDF5 Schema

| Dataset | Shape | Type | Description |
|---------|-------|------|-------------|
| `/images_raw` | (N, 3, 128, 128) | float32 | Raw g, r, z flux |
| `/psf_fwhm` | (N, 3) | float32 | Per-band seeing FWHM |
| `/meta` | (N, K) | float32 | Catalog properties |
| `/ids` | (N,) | int64 | Object identifiers |

### Milestone Checklist

- [ ] `download_cutouts.py` implemented
- [ ] `build_background_hdf5.py` implemented
- [ ] `dr10_backgrounds_raw.h5` created
- [ ] Notebook extended with example raw cutouts and PSF distributions

---

## 6. Phase 4 – Simulation Engine (LensInjector)

> ⚠️ **This is the critical engine room — highest risk phase.**

### Goal

Produce realistic lens injections into DR10 backgrounds with:
- Clumpy COSMOS sources
- Per-band PSFs
- Blue SEDs for sources

### 6.1 COSMOS Loader

#### `src/sims/cosmos_loader.py`

```python
class COSMOSLoader:
    """
    Load and serve COSMOS/HST galaxy images as lensing sources.
    
    Attributes
    ----------
    pixel_scale : float
        Native COSMOS pixel scale (~0.05 arcsec)
    images : np.ndarray
        Preloaded source images
    clumpiness : np.ndarray
        Optional precomputed clumpiness metric
    """
    
    def __init__(self, cosmos_path: str):
        pass
    
    def get_random_source(self) -> tuple[np.ndarray, float]:
        """Return (image, clumpiness_score)."""
        pass
```

### 6.2 LensInjector Class

#### `src/sims/lens_injector.py`

```python
@dataclass
class InjectionParams:
    theta_E: float          # Einstein radius (arcsec)
    z_l: float              # Lens redshift
    z_s: float              # Source redshift
    m_source_r: float       # Source r-band magnitude
    ellipticity: tuple      # (e1, e2)
    shear: tuple            # (gamma1, gamma2)
    source_position: tuple  # (x, y) in source plane

class LensInjector:
    """
    Inject lensed arcs into raw DR10 cutouts.
    
    Key requirements:
    - Operates on raw DR10 cutouts (flux units)
    - Uses per-band PSF FWHM
    - Applies SED-based magnitude offsets for blue sources
    """
    
    def __init__(
        self,
        pixel_scale: float = 0.262,
        zero_point: float = 22.5
    ):
        self.pixel_scale = pixel_scale
        self.zero_point = zero_point
    
    def mag_to_flux(self, mag: float) -> float:
        """Convert magnitude to DR10-like flux (nanomaggies)."""
        return 10 ** ((self.zero_point - mag) / 2.5)
    
    def get_sed_offsets(self, z_source: float) -> dict:
        """
        Get band offsets for a blue star-forming source.
        
        Returns
        -------
        dict : {'g': Δm_g, 'r': 0.0, 'z': Δm_z}
        """
        pass
    
    def inject_arc(
        self,
        background_cutout: dict,  # {'g': arr, 'r': arr, 'z': arr}
        psf_fwhm: dict,           # {'g': fwhm, 'r': fwhm, 'z': fwhm}
        source_image: np.ndarray, # High-res COSMOS source
        params: InjectionParams,
        source_scale: float       # Source pixel scale (arcsec)
    ) -> tuple[dict, InjectionParams]:
        """
        Inject a lensed arc into the background cutout.
        
        Steps:
        1. Randomize SIE ellipticity and shear
        2. Sample source position (reject if magnification too low)
        3. Build lenstronomy ImageData/PSF
        4. For each band:
           a. Compute target_mag = m_source_r + Δm_band
           b. Convert to target flux
           c. Normalize source and scale to flux
           d. Generate lensed arc with lenstronomy
           e. Add to background_cutout[band]
        5. Return injected cutout + parameters
        """
        pass
    
    def check_arc_visibility(
        self,
        arc_flux: np.ndarray,
        threshold: float = 100.0
    ) -> bool:
        """Check if injected arc has sufficient total flux."""
        return arc_flux.sum() > threshold
```

### 6.3 Building Simulated Datasets

#### `src/sims/build_sim_datasets.py`

```python
def build_simulation_grid(
    theta_E_range: np.ndarray,
    z_l_range: np.ndarray,
    z_s_range: np.ndarray,
    m_source_range: np.ndarray,
    n_sources_per_point: int = 5
) -> list[dict]:
    """Generate parameter grid for simulations."""
    pass

def build_sim_hdf5(
    backgrounds_h5: str,
    cosmos_loader: COSMOSLoader,
    injector: LensInjector,
    param_grid: list[dict],
    output_path: str,
    include_negatives: bool = True
) -> None:
    """
    Build simulation HDF5 with injected lenses.
    
    HDF5 structure:
    - /images_raw: (N, 3, 128, 128) raw flux
    - /labels: (N,) 0=non-lens, 1=lens
    - /theta_E: (N,)
    - /z_l: (N,)
    - /z_s: (N,)
    - /m_source_r: (N,)
    - /clumpiness: (N,)
    """
    pass
```

### Output Files

| File | Contents |
|------|----------|
| `data/sims/dr10_sims_raw_train.h5` | Training simulations |
| `data/sims/dr10_sims_raw_test.h5` | Test simulations |

### Notebook: `03_sim_qc.ipynb`

Required visualizations:

1. **RGB composites** of single injected lens on LRG
2. **Grid of lenses** across θ_E and m_source
3. **Confuser examples** (non-lenses that look suspicious)

Visual checks:
- [ ] LRG appears red
- [ ] Arc appears blue and clumpy
- [ ] Seeing blur consistent with DR10

### Milestone Checklist

- [ ] `cosmos_loader.py` implemented
- [ ] `lens_injector.py` implemented and tested
- [ ] `build_sim_datasets.py` implemented
- [ ] Training and test HDF5 files created
- [ ] QC notebook with visual verification

---

## 7. Phase 5 – Detection Operator + Active Learning

### Goal

Train a single detector D and refine via hard negatives.

### 7.1 Preprocessing

#### `src/sims/preprocess.py`

```python
def preprocess_raw_to_ml(
    images_raw: np.ndarray,  # (N, 3, H, W)
    sky_estimates: np.ndarray = None
) -> np.ndarray:
    """
    Convert raw images to ML-ready format.
    
    Steps:
    1. Sky subtraction
    2. Robust scaling (percentile-based)
    3. Clipping to [-3, 10] or similar
    4. Add radial channel R
    
    Returns
    -------
    images_preproc : np.ndarray
        Shape (N, 4, H, W) with channels [g, r, z, R]
    """
    pass

def build_radial_channel(shape: tuple) -> np.ndarray:
    """Create normalized radial distance channel."""
    pass
```

### 7.2 Detector Model

#### `src/models/detector.py`

```python
class LensDetector(nn.Module):
    """
    ResNet-18 based lens detector with metadata fusion.
    
    Architecture:
    - ResNet-18 trunk (4 input channels: g, r, z, R)
    - Metadata MLP
    - Combined head with:
      - P_lens: sigmoid output for classification
      - θ̂_E: regression output for Einstein radius
    """
    
    def __init__(
        self,
        n_meta_features: int = 8,
        pretrained_trunk: bool = False
    ):
        super().__init__()
        # ResNet-18 with modified first conv
        # Metadata MLP
        # Combined head
    
    def forward(
        self,
        images: torch.Tensor,   # (B, 4, 128, 128)
        metadata: torch.Tensor  # (B, n_meta)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        p_lens : (B, 1) probability of being a lens
        theta_E_pred : (B, 1) predicted Einstein radius
        """
        pass
```

### 7.3 Training Pipeline

#### `src/models/train_detector.py`

```python
def train_detector(
    train_h5: str,
    val_h5: str,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3
) -> None:
    """
    Train lens detector on simulated data.
    
    Loss: BCE for classification + MSE for θ_E regression
    """
    pass

def active_learning_round(
    detector: LensDetector,
    real_backgrounds_h5: str,
    n_candidates: int = 1000,
    output_path: str = "data/processed/confuser_candidates.csv"
) -> pd.DataFrame:
    """
    Select high P_lens candidates from real data for manual labeling.
    """
    pass
```

### Milestone Checklist

- [ ] Preprocessing pipeline implemented
- [ ] Detector model defined
- [ ] Training script working on simulated data
- [ ] Good performance on simulated test set
- [ ] Active learning round completed
- [ ] Confusers added and model retrained

---

## 8. Phase 6 – Injection Recovery and Selection Function

### Goal

Extract **completeness** and **purity** as functions of (θ_E, z_l, m_source).

### Implementation

#### `src/analysis/injection_recovery.py`

```python
def compute_completeness(
    predictions: np.ndarray,  # P_lens scores
    labels: np.ndarray,       # True labels
    threshold: float = 0.5
) -> float:
    """Fraction of true lenses detected."""
    pass

def bin_completeness(
    predictions: np.ndarray,
    labels: np.ndarray,
    theta_E: np.ndarray,
    m_source: np.ndarray,
    theta_E_bins: np.ndarray,
    m_source_bins: np.ndarray
) -> np.ndarray:
    """
    Compute completeness in (θ_E, m_source) bins.
    
    Returns
    -------
    C : np.ndarray
        Shape (n_theta_bins, n_m_bins)
    """
    pass
```

#### `src/analysis/selection_function.py`

```python
def combine_analytic_and_empirical(
    analytic_regions: dict,  # From Phase 1
    empirical_C: np.ndarray, # From injection recovery
    theta_E_grid: np.ndarray,
    z_l_grid: np.ndarray
) -> np.ndarray:
    """
    Combine analytic blind/low-trust/good with empirical completeness.
    
    Returns
    -------
    selection_map : np.ndarray
        Combined selection function C(θ_E, z_l)
    """
    pass

def selection_function_to_Mhalo(
    selection_map: np.ndarray,
    theta_E_grid: np.ndarray,
    z_l_grid: np.ndarray,
    z_s_assumed: float,
    cosmo
) -> np.ndarray:
    """
    Map selection function to halo mass space.
    
    ⚠️ Only valid for assumed z_s — label clearly!
    """
    pass
```

### Notebook: `04_injection_recovery_plots.ipynb`

Required figures:

1. **Heatmap**: C(θ_E, m_source_r)
2. **Heatmap**: C(θ_E, z_l) with analytic regions overlaid
3. **Central physics figure**: Selection function with clear legend

### Milestone Checklist

- [ ] Injection recovery computed
- [ ] Binned completeness maps generated
- [ ] Selection function combined with analytic regions
- [ ] Core physics figure produced

---

## 9. Phase 7 – DR10 Application and Candidates

### Goal

Anchor selection function to reality with known lenses and candidate search.

### Implementation

#### `src/analysis/real_dr10_analysis.py`

```python
def apply_detector_to_dr10(
    detector: LensDetector,
    backgrounds_h5: str,
    output_path: str = "data/processed/dr10_predictions.parquet"
) -> pd.DataFrame:
    """Apply trained detector to all LRG backgrounds."""
    pass

def cross_match_known_lenses(
    predictions: pd.DataFrame,
    known_lens_catalogs: list[str],
    match_radius: float = 2.0  # arcsec
) -> pd.DataFrame:
    """Cross-match predictions with known lens catalogs."""
    pass

def compute_known_lens_recovery(
    matched: pd.DataFrame,
    threshold: float = 0.5
) -> dict:
    """
    Compute recovery fraction of known lenses.
    
    Returns
    -------
    dict : {'n_known': int, 'n_recovered': int, 'fraction': float}
    """
    pass
```

### Deliverables

1. **Candidate list**: Top 200–300 by P_lens, manually graded
2. **Figure**: θ_E–z_l selection map with:
   - Background completeness shading
   - Analytic blind regions
   - Known lenses (circles)
   - New candidates (stars)

### Milestone Checklist

- [ ] Detector applied to all DR10 LRGs
- [ ] Cross-match with known lens catalogs
- [ ] Recovery fraction computed
- [ ] Top candidates inspected and graded
- [ ] Summary figure produced

---

## 10. Phase 8 – Optional Mass and f_DM

> *If time allows*

### Goal

Illustrative mass estimates for a small subset of lenses.

### Implementation

```python
def fit_sie_model(
    image: np.ndarray,
    psf: np.ndarray,
    z_l: float,
    z_s_assumed: float
) -> dict:
    """
    Fit SIE lens model using lenstronomy.
    
    Returns
    -------
    dict : {
        'theta_E_fit': float,
        'M_tot_within_theta_E': float,
        'ellipticity': tuple
    }
    """
    pass

def compute_f_DM_range(
    M_tot: float,
    stellar_mass_estimate: float,
    z_s_range: tuple
) -> tuple:
    """
    Compute dark matter fraction range.
    
    f_DM = (M_tot - M_stellar) / M_tot
    
    Returns range for different z_s assumptions.
    """
    pass
```

### Milestone Checklist

- [ ] SIE fitting implemented for subset
- [ ] M_tot computed for multiple z_s values
- [ ] f_DM ranges presented as illustrative (not definitive)

---

## 11. Phase W – Writing and Figures

> *Integrated throughout the project*

### Key Figures to Produce

| Figure | Phase | Description |
|--------|-------|-------------|
| Analytic observability map | 1 | θ_E vs z_l with seeing limits |
| Color-magnitude diagram | 2 | LRG parent sample selection |
| Example cutouts | 3 | Raw DR10 backgrounds |
| Injection gallery | 4 | Lenses across parameter space |
| Detection ROC | 5 | Classifier performance |
| **Selection function** | 6 | **Core physics result** |
| Known lens recovery | 7 | Validation plot |
| Candidate gallery | 7 | New lens candidates |

### Documentation Deliverables

- [ ] README with project overview
- [ ] Method description for each phase
- [ ] Clear statement of assumptions and limitations
- [ ] Reproducibility instructions

---

## Summary Table

| Phase | Key Output | Risk | Status |
|-------|------------|------|--------|
| A | Project structure | Low | ⬜ |
| 1 | Analytic θ_E maps | Low | ⬜ |
| 2 | parent_sample.parquet | Medium | ⬜ |
| 3 | dr10_backgrounds_raw.h5 | Medium | ⬜ |
| 4 | LensInjector + sims | **High** | ⬜ |
| 5 | Trained detector | Medium | ⬜ |
| 6 | **Selection function C(θ_E, z_l)** | Medium | ⬜ |
| 7 | Known lens validation | Low | ⬜ |
| 8 | f_DM estimates (optional) | Low | ⬜ |

---

*Blueprint version: 1.0*  
*Created: December 2025*

