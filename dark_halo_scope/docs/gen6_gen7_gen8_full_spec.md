# Gen6 / Gen7 / Gen8 / Uber: Complete Specification and Code Package

## Purpose

This document provides a full specification of all model generation approaches for gravitational lens detection, enabling the LLM reviewer to assess which path is the most realistic and promising for original research.

---

## Overview: Model Generation Progression

| Generation | Source Model | Key Innovation | Data Requirement | Status |
|------------|--------------|----------------|------------------|--------|
| **Gen5** | COSMOS HST morphologies | Real galaxy shapes via lenstronomy | COSMOS bank (done) | ✅ Complete |
| **Gen5-Prime** | COSMOS + paired controls | Shortcut mitigation (hard neg + core dropout) | Paired parquet (done) | 📋 Ready |
| **Gen6** | Deep ground-based cutouts | Same PSF/noise regime as target survey | Deep field FITS | ⚠️ Needs deep data |
| **Gen7** | Hybrid Sersic + clumps | Controlled realism without HST artifacts | None (procedural) | ✅ Code ready |
| **Gen8** | Any source + domain randomization | Artifacts: cosmic rays, PSF anisotropy, jitter | None (procedural) | ✅ Code ready |
| **Uber** | Mixture of above | Combined robustness | All of above | ✅ Code ready |

---

## Gen5: COSMOS-Based Sources (Current Baseline)

### Description
Uses real HST COSMOS galaxy morphologies as lensed sources. Templates are downsampled from HST resolution (~0.03"/pix) to ground-based (0.262"/pix) and lensed using lenstronomy.

### Strengths
- Real galaxy morphologies (irregular shapes, clumps, spirals)
- Validated with SLACS/BELLS anchors
- Production data exists

### Weaknesses
- HST-specific artifacts may transfer (sharpness, noise patterns)
- Sources are at HST depth, not ground-based depth
- Core leakage shortcut identified (AUC=0.90 on central pixels)

### Current Results
- Synthetic test AUC: 0.895
- SLACS/BELLS recall: ~4.4% at threshold 0.5 (before remediation)

### Code Location
```
dark_halo_scope/emr/gen5/spark_phase4_pipeline_gen5.py
  - render_cosmos_lensed_source() lines 157-260
  - Uses lenstronomy ImageModel for PSF convolution
```

---

## Gen5-Prime: Shortcut Mitigation (Research Plan 1)

### Description
Gen5 architecture + paired counterfactual controls + azimuthal-shuffle hard negatives + core dropout/masking.

### Key Changes from Gen5
1. **Paired Controls**: Each positive has a matched control from same sky position
2. **Hard Negatives**: Azimuthal-shuffle destroys arc morphology while preserving radial profile
3. **Core Dropout**: Randomly mask central 5-pixel radius during training
4. **Leakage Gates**: Evaluation suite to verify model doesn't rely on shortcuts

### Data
- `s3://darkhaloscope/phase4_pipeline/phase4c/v5_cosmos_paired/`
- Train: ~250k pairs, Val: ~25k pairs, Test: ~25k pairs

### Novelty Claim
"Diagnose and mitigate a core-leakage shortcut in lens finding with paired counterfactual controls, morphology-preserving hard negatives, and explicit leakage gates."

### Detailed Plan
See: `research_plan_1_shortcut_mitigation.md`

---

## Gen6: Deep Ground-Based Sources

### Description
Use cutouts from **ground-based deep fields** (e.g., HSC Ultra-Deep, Legacy Survey deep regions) as source templates instead of HST COSMOS. This eliminates the HST-to-ground domain gap.

### Scientific Rationale
HST templates at 0.03"/pix contain morphological detail that disappears when downsampled to 0.262"/pix, but PSF and noise characteristics still mismatch. Ground-based deep cutouts have:
- Same PSF regime as target survey (seeing-limited, ~1" FWHM)
- Same noise characteristics (sky background, read noise)
- Naturally matched texture statistics

### Implementation

```python
# dhs_gen/deep_sources/deep_source_bank.py

def build_deep_source_bank(
    fits_dir: str,          # Directory containing deep field cutouts
    out_npz: str,           # Output bank file
    n_sources: int,         # Number of sources to extract
    stamp_size: int,        # Output stamp size (e.g., 96)
    src_pixscale_arcsec: float,    # Source pixel scale (e.g., 0.168 for HSC)
    target_pixscale_arcsec: float, # Target pixel scale (0.262 for DR10)
) -> None:
    """
    Builds a compact NPZ bank of deep ground-based galaxy cutouts.
    
    Workflow:
    1. Read FITS files from fits_dir
    2. Background subtract using border pixels
    3. Resample to target pixel scale
    4. Pad/crop to stamp_size
    5. Save to NPZ with metadata
    """
```

### Data Requirements
| Survey | Pixel Scale | Depth | Availability |
|--------|-------------|-------|--------------|
| HSC Ultra-Deep | 0.168"/pix | ~27 mag | Public, ~10 sq deg |
| Legacy DR10 Deep | 0.262"/pix | ~25 mag | Need to identify deep regions |
| DES Deep Fields | 0.263"/pix | ~26 mag | Public |

### Usage in Stage 4c
```python
# In spark_phase4_pipeline_gen6.py

# Load deep source bank
deep_bank = np.load("deep_bank_20k_96px.npz")
templates = deep_bank["images"]  # (N, 96, 96)

# Select template deterministically
idx = hash(task_id) % len(templates)
template = templates[idx]

# Convert to surface brightness for lenstronomy
from dhs_gen.utils import to_surface_brightness
template_sb = to_surface_brightness(template, pixscale_arcsec=0.262)

# Use with lenstronomy INTERPOL light model
kwargs_source = [{"image": template_sb, "center_x": src_x, "center_y": src_y}]
```

### Strengths
- No HST-to-ground domain gap
- Sources have realistic ground-based noise/PSF already baked in
- May improve sim-to-real transfer

### Weaknesses
- **BLOCKER**: Need to source/prepare deep cutouts (not yet done)
- Deep fields have limited area (~10-100 sq deg vs ~15,000 sq deg DR10)
- May have survey-specific artifacts

### Risk Assessment
| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Deep data unavailable | Medium | Skip Gen6, focus on Gen7/8 |
| Deep cutouts too noisy | Low | Quality filtering in bank building |
| Survey-specific artifacts | Medium | Use multiple surveys for diversity |

---

## Gen7: Hybrid Sersic + Clumps

### Description
Procedurally generate source templates as **smooth Sersic profile + explicit Gaussian clumps + color gradient**. This provides controlled realism without importing any survey-specific artifacts.

### Scientific Rationale
Real galaxies are not smooth Sersics - they have:
- Star-forming clumps
- Dust lanes
- Color gradients (blue outskirts, red center)

By adding these features procedurally with tunable parameters, we can:
1. Create more realistic sources than pure Sersic
2. Ablate the effect of each realism knob
3. Avoid importing HST/other survey artifacts

### Implementation

```python
# dhs_gen/hybrid_sources/sersic_clumps.py

def generate_hybrid_source(
    key: str,                           # Deterministic hash key (task_id)
    H: int = 96, W: int = 96,          # Template size
    re_pix: float = 6.0,               # Effective radius in pixels
    n_sersic: float = 1.0,             # Sersic index (1=exponential, 4=de Vaucouleurs)
    q: float = 0.8,                    # Axis ratio
    n_clumps_range: Tuple[int, int] = (2, 6),  # Number of clumps
    clump_sigma_pix_range: Tuple[float, float] = (0.8, 2.0),  # Clump size
    clump_flux_frac_range: Tuple[float, float] = (0.05, 0.25), # Clump brightness
    gradient_strength: float = 0.2,    # Color gradient amplitude
    normalize_sum: bool = True,
) -> Dict[str, object]:
    """
    Generate a hybrid source template.
    
    Components:
    1. Base smooth Sersic profile with random orientation
    2. N clumps placed at ~1 Re with random positions/sizes/brightnesses
    3. Linear color gradient (bluer outskirts)
    
    Returns:
        {"img": np.ndarray (H,W), "meta": dict with generation params}
    """
```

### Usage in Stage 4c
```python
# In spark_phase4_pipeline_gen7.py

from dhs_gen.hybrid_sources.sersic_clumps import generate_hybrid_source

# Generate source for this task
result = generate_hybrid_source(
    key=task_id,
    H=96, W=96,
    re_pix=src_reff_pix,
    n_sersic=1.0,  # Exponential disk typical for lensed sources
    n_clumps_range=(2, 8),
    gradient_strength=0.15,
)
template = result["img"]

# Convert to surface brightness and use with lenstronomy
template_sb = to_surface_brightness(template, pixscale_arcsec=0.262)
```

### Configuration Space

| Parameter | Range | Effect |
|-----------|-------|--------|
| n_sersic | 0.5-4.0 | Profile concentration (0.5=flat, 4=cuspy) |
| n_clumps | 0-10 | Irregularity level |
| clump_flux_frac | 0.01-0.40 | Clump brightness relative to total |
| gradient_strength | 0-0.5 | Color gradient amplitude |

### Ablation Experiment
Train models with:
- **Gen7a**: Sersic only (no clumps, no gradient)
- **Gen7b**: Sersic + clumps
- **Gen7c**: Sersic + clumps + gradient (full Gen7)

Compare performance on:
- Synthetic test set
- Real anchor lenses
- Hard negatives (ring galaxies, spirals)

### Strengths
- No data dependency (fully procedural)
- Tunable realism for ablation studies
- No survey-specific artifacts
- Fast to generate (no I/O)

### Weaknesses
- Still idealized compared to real galaxies
- Missing complex morphologies (mergers, tidal features)
- Clump distribution is simplistic

---

## Gen8: Domain Randomization

### Description
Apply **realistic imaging artifacts** on top of any source model (Gen5/6/7) to improve robustness to real-world variations.

### Scientific Rationale
Real survey images contain artifacts that simulations often miss:
- PSF anisotropy (elliptical, varying across field)
- Low-frequency background residuals (imperfect sky subtraction)
- Cosmic rays
- Saturation spikes from bright stars
- Astrometric jitter (imperfect alignment)

By adding these during training, the model becomes robust to their presence in real data.

### Implementation

```python
# dhs_gen/domain_randomization/artifacts.py

@dataclass
class ArtifactConfig:
    # PSF anisotropy
    enable_psf_anisotropy: bool = True
    psf_e_sigma: float = 0.05          # Ellipticity scatter
    
    # Background residuals
    enable_bg_plane: bool = True
    bg_plane_amp: float = 0.02         # Amplitude as fraction of MAD
    
    # Cosmic rays
    enable_cosmic_rays: bool = True
    cosmic_rate: float = 0.15          # Probability per stamp
    cosmic_amp: float = 8.0            # Amplitude in MAD units
    cosmic_length_pix: Tuple[int, int] = (8, 24)
    
    # Saturation spikes
    enable_sat_wings: bool = True
    sat_rate: float = 0.08
    sat_amp: float = 12.0
    
    # Astrometric jitter
    enable_astrom_jitter: bool = True
    jitter_sigma_pix: float = 0.25


def apply_domain_randomization(
    img: np.ndarray,              # Input image (2D)
    key: str,                     # Deterministic hash key
    psf_fwhm_pix: Optional[float] = None,
    psf_model: str = "moffat",
    moffat_beta: float = 3.5,
    cfg: ArtifactConfig = ArtifactConfig(),
    max_kernel_size: int = 63,    # Prevent kernel > stamp size
) -> Dict[str, object]:
    """
    Apply domain randomization artifacts to an image.
    
    Order of operations:
    1. Add background plane
    2. Apply PSF with anisotropy (if psf_fwhm provided)
    3. Add cosmic ray (probabilistic)
    4. Add saturation wings (probabilistic)
    5. Apply astrometric jitter
    
    Returns:
        {"img": np.ndarray, "meta": dict with artifact params}
    """
```

### Artifact Profiles

| Profile | Cosmic Rate | Sat Rate | BG Amp | Jitter σ | Use Case |
|---------|-------------|----------|--------|----------|----------|
| **none** | 0 | 0 | 0 | 0 | Baseline |
| **mild** | 0.10 | 0.05 | 0.01 | 0.2 | Typical survey |
| **strong** | 0.25 | 0.15 | 0.05 | 0.5 | Stress test |

### Usage in Stage 4c
```python
# In spark_phase4_pipeline_gen8.py

from dhs_gen.domain_randomization.artifacts import (
    apply_domain_randomization, 
    ArtifactConfig
)

# Apply artifacts after arc injection
cfg = ArtifactConfig(
    cosmic_rate=0.15,
    sat_rate=0.08,
    jitter_sigma_pix=0.3,
)

result = apply_domain_randomization(
    img=stamp_with_arc,
    key=task_id,
    psf_fwhm_pix=psf_fwhm_pix,
    cfg=cfg,
)
final_stamp = result["img"]
```

### Ablation Experiment
Train models with:
- **Gen8a**: Gen5 + no artifacts (baseline)
- **Gen8b**: Gen5 + mild artifacts
- **Gen8c**: Gen5 + strong artifacts

Compare:
- Performance degradation on clean test set
- Performance improvement on noisy test set
- Robustness to unseen artifact types

### Strengths
- Improves robustness without new data
- Tunable artifact intensity
- Can be applied on top of any source model
- Deterministic given task_id

### Weaknesses
- May degrade performance on clean data
- Artifacts are stylized, not exact survey replication
- Need to tune artifact rates for each survey

---

## Uber: Combined Mixture

### Description
Combine Gen5 (COSMOS), Gen6 (deep), Gen7 (hybrid), and Gen8 (domain randomization) in a weighted mixture for maximum robustness.

### Implementation

```python
# dhs_gen/uber/mixer.py

DEFAULT_UBER = MixerConfig(
    source_modes=["cosmos", "deep", "hybrid"],
    source_probs=[0.34, 0.33, 0.33],        # Even split
    artifact_profiles=["none", "mild", "strong"],
    artifact_probs=[0.15, 0.70, 0.15],      # Mostly mild
    salt="uber_v1",
)

def assign_modes(task_id: str, cfg: MixerConfig) -> Dict[str, str]:
    """
    Deterministically assign source_mode and artifact_profile based on task_id.
    """
    sidx = categorical_from_hash(task_id, np.array(cfg.source_probs), salt=cfg.salt + "_src")
    aidx = categorical_from_hash(task_id, np.array(cfg.artifact_probs), salt=cfg.salt + "_art")
    return {
        "source_mode": cfg.source_modes[sidx],
        "artifact_profile": cfg.artifact_profiles[aidx]
    }
```

### Workflow for Uber
1. **First**: Run ablations for Gen6, Gen7, Gen8 separately
2. **Evaluate** each on locked real-lens anchor set
3. **Only then**: Create Uber mix with weights based on ablation results
4. **Track** per-row columns (source_mode, artifact_profile) for stratified analysis

### Why This Order Matters
- Uber without ablations is **not publishable** - reviewers will ask "which component helped?"
- Ablations provide interpretable results for paper
- Uber provides practical final model

---

## Comparison Matrix

| Aspect | Gen5 | Gen5-Prime | Gen6 | Gen7 | Gen8 | Uber |
|--------|------|------------|------|------|------|------|
| **Source** | COSMOS HST | COSMOS HST | Ground deep | Procedural | Any | Mixed |
| **Artifacts** | None | None | None | None | Synthetic | Mixed |
| **Data Needed** | COSMOS bank | Paired parquet | Deep FITS | None | None | All |
| **Novel Claim** | Real morphologies | Shortcut mitigation | Domain matching | Controlled realism | Artifact robustness | Combined |
| **Readiness** | ✅ Done | ✅ Ready | ⚠️ Need data | ✅ Ready | ✅ Ready | ⏳ After ablations |

---

## Recommended Path for Publication

### Option A: Gen5-Prime Only (Fastest)
**Timeline:** 2-3 weeks

**Paper claim:** "Shortcut-aware training with paired counterfactual controls, morphology-preserving hard negatives, and explicit leakage gates for robust strong lens detection."

**Experiments:**
- Ablation: baseline vs +paired vs +hardneg vs +core-dropout vs full
- Leakage gates pass/fail
- SLACS/BELLS anchor performance

**Risk:** Modest novelty, may be seen as "engineering" rather than "research"

### Option B: Gen5-Prime + Gen7/Gen8 Ablation (Recommended)
**Timeline:** 4-6 weeks

**Paper claim:** "Controlled realism and domain randomization for simulation-to-real transfer in strong lens detection, with rigorous shortcut mitigation."

**Experiments:**
- Gen5-Prime as baseline
- +Gen7 (hybrid sources): Does procedural realism help?
- +Gen8 (domain randomization): Do artifacts help?
- Stratified analysis by source_mode and artifact_profile

**Advantage:** Clear ablation story, publishable regardless of which helps

### Option C: Full Ablation Suite + Uber (Most Thorough)
**Timeline:** 6-8 weeks

**Paper claim:** "A comprehensive study of simulation realism for deep learning lens detection: source diversity, artifact injection, and shortcut mitigation."

**Experiments:**
- All of Option B
- +Gen6 if deep data available
- Uber ensemble with calibration
- Selection function analysis per generation
- Bootstrap confidence intervals

**Risk:** Scope creep, but strongest paper

---

## Blockers and Decision Points

### Decision 1: Do we have deep cutout data for Gen6?
- **If YES:** Include Gen6 in ablation suite
- **If NO:** Skip Gen6, proceed with Gen7+Gen8

### Decision 2: Which publication path?
- Option A is safe but modest
- Option B is recommended balance
- Option C is most thorough but risky

### Decision 3: Timeline constraints?
- If 2 weeks: Option A only
- If 4-6 weeks: Option B
- If 8+ weeks: Option C

---

## Code Package Summary

### Directory Structure
```
dhs_gen6_gen7_gen8_uber_code_fixed/
├── dhs_gen/
│   ├── __init__.py
│   ├── utils.py                          # Common utilities
│   ├── deep_sources/                     # Gen6
│   │   ├── __init__.py
│   │   ├── deep_source_bank.py          # Bank builder
│   │   └── deep_source_sampler.py       # Sampler for Stage 4c
│   ├── hybrid_sources/                   # Gen7
│   │   ├── __init__.py
│   │   └── sersic_clumps.py             # Hybrid generator
│   ├── domain_randomization/             # Gen8
│   │   ├── __init__.py
│   │   └── artifacts.py                 # Artifact injection
│   ├── uber/                             # Uber mixer
│   │   ├── __init__.py
│   │   └── mixer.py
│   └── validation/
│       ├── __init__.py
│       └── quality_checks.py            # QA utilities
├── tests/
│   ├── test_artifacts.py
│   ├── test_hybrid.py
│   └── test_utils_and_decode.py
├── README.md
├── SPARK_INTEGRATION.md
├── INDEPENDENT_REVIEW.md                 # Review status
└── pyproject.toml
```

### Test Status
All 8 tests pass after bug fixes:
```
1. bilinear_resample... PASSED
2. decode_stamp_npz... PASSED
3. hybrid_source... PASSED
4. domain_randomization... PASSED
5. to_surface_brightness... PASSED
6. mixer... PASSED
7. deep_source imports... PASSED
8. kernel max_size... PASSED
```

---

## Questions for LLM Reviewer

1. **Research viability:** Given the current state, which path (A/B/C) is most likely to result in a publishable paper with original contribution?

2. **Gen6 necessity:** Is Gen6 (deep ground-based sources) essential for the paper, or can we make a strong contribution with Gen7+Gen8 alone?

3. **Ablation priority:** If time is limited, which ablation is highest priority?
   - Gen7 (hybrid sources) vs baseline
   - Gen8 (artifacts) vs baseline
   - Full (Gen7+Gen8) vs components

4. **Novelty assessment:** For each approach, what is the honest assessment of novelty?
   - Gen5-Prime shortcut mitigation: Novel methodology or engineering?
   - Gen7 procedural realism: Novel or obvious?
   - Gen8 domain randomization: Novel in lens-finding context?

5. **Publication venue:** Given the expected contributions, what venue is realistic?
   - Top ML venue (NeurIPS, ICML): Requires significant novelty
   - Astrophysics journal (MNRAS, ApJ): Requires validated results
   - Applications venue (AISTATS, ML4Astro): Middle ground

6. **Critical experiments:** What experiments would make the difference between "engineering report" and "research contribution"?

---

## Immediate Next Steps

1. **Confirm deep data availability** for Gen6 decision
2. **Run Gen5-Prime baseline** with full shortcut mitigation
3. **Prepare Gen7 manifest** (no data dependency)
4. **Prepare Gen8 artifact config** (tune rates for DR10)
5. **Design ablation experiment** based on LLM guidance
