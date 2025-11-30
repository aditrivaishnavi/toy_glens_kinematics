# Project Plan: Multi-Modal Subhalo Detection via Lensed Kinematics

> End-to-end plan for detecting dark-matter subhalos using combined flux + velocity data from lensed galaxies.

---

## Table of Contents

1. [Goal and Success Criteria](#1-goal-and-success-criteria)
2. [High-Level Pipeline](#2-high-level-pipeline)
3. [Detailed Phase-by-Phase Plan](#3-detailed-phase-by-phase-plan)
4. [Risks and Kill-Switch Checks](#4-risks-and-kill-switch-checks)
5. [Summary](#5-summary)

---

## 1. Goal and Success Criteria

### Scientific Goal

Demonstrate, on realistic simulated strong-lens data, that a **multi-modal model (flux + kinematics)** can detect dark-matter subhalo perturbations more reliably than image-only methods, under instrument conditions similar to MaNGA / MUSE / JWST IFU.

### Engineering/ISEF Goal

Produce a complete, reproducible pipeline that:

1. Starts from **real IFU data** (MaNGA MAPS).
2. Builds a **simulation engine** for lensed images + velocity fields.
3. Trains **two ML models**:
   - **Baseline**: image-only (flux).
   - **Multi-modal**: flux + velocity.
4. Shows clear, **quantitative improvement** (ROC/AUC, recall at fixed FPR) plus visual examples of subhalo perturbations.
5. Documents **limitations and failure modes**.

---

## 2. High-Level Pipeline

Think of four layers:

### Layer 1: Real Data Layer
```
MaNGA MAPS files → cleaned flux + velocity tensors
```

### Layer 2: Source-Plane Model Layer
- **Option A**: Analytic rotating disks
- **Option B**: MaNGA-derived maps as templates
- **Option C**: Hybrid analytic + MaNGA

### Layer 3: Lensing + Subhalo Simulation Layer
Using lenstronomy (+ optional Paltas) to impose:
- Macro lens (host galaxy)
- Subhalo(s)
- PSF, noise, pixel scale
- Apply same mapping to both channels: flux and velocity
- Enforce flux-weighted validity mask for velocity

### Layer 4: ML Layer
- Dataset builder → PyTorch Dataset
- Baseline CNN (flux only)
- Dual-stream CNN (flux + vel)
- Training, evaluation, ablation, robustness tests

**Each layer should be testable independently.**

---

## 3. Detailed Phase-by-Phase Plan

### Phase 0 – Clean Up What You Already Have (✅ Mostly Done)

#### Current State

You have:
- Scripts to rank MaNGA galaxies (`dapall` + `drpall`)
- A download script for MAPS files
- `batch_inspect_maps.py` that computes rotation quality, mask fraction, valid fraction in 1.5 Rₑ, etc.
- `prep_source_maps.py` that:
  - Loads MAPS
  - Extracts Hα flux and gas velocity
  - Normalizes them, resamples to 64×64
  - Saves tensors `source_tensor_<plateifu>.npy` and preview PNGs

#### Action Items

**1. Freeze a "v1 data contract" for source tensors:**

| Attribute | Value |
|-----------|-------|
| File naming | `data/source_tensors/source_tensor_<plateifu>.npy` |
| Shape | `(2, 64, 64)` |
| Channel 0 | Normalized Hα flux in `[0, 1]` |
| Channel 1 | Normalized gas velocity in `[-1, 1]`, with 0 meaning systemic |

Include a sidecar JSON or CSV with:
- `plateifu`, `inclination`, `gradient`, `mask_fraction`
- Flux normalization constants (`p5`, `p99.5`)
- Velocity scale parameters (`median`, `scale`)

**2. Decide canonical training subset:**

Based on previews and qualitative ranking:
- **Hero galaxy**: `8593-12705` (cleanest rotation)
- **Supporting galaxies**: `8993-12705`, `8652-12703`, `12071-12702`, `10500-12703`
- **Exclude for now**: `9487-9102`, `11982-9102`, `11013-6101` (too patchy/noisy)

Create a text file:
```
data/canonical_sources.txt
8593-12705
8993-12705
8652-12703
12071-12702
10500-12703
```

This makes future scripts deterministic.

---

### Phase 1 – Source-Plane Models (Analytic vs MaNGA vs Hybrid)

You need a clear design for how a "source galaxy" (light + velocity) is represented before lensing.

#### Option A – Analytic Rotating Disk (Simplest, Fully Controlled)

Implement a module: `src/sources/analytic_disk.py`

**Inputs:**
- Grid size (e.g., 128×128 in source plane for oversampling)
- Physical parameters:
  - Inclination `i`, position angle `PA`
  - Scale radius `R_s`, maximum rotation speed `V_max`
  - Systemic velocity `V_sys`, intrinsic dispersion `σ_0`

**Output:**
- 2D surface-brightness map (flux) on source grid
- 2D velocity field on source grid

**Implementation Steps:**
1. Build a coordinate grid `(x, y)` in units of kpc or arcsec
2. Rotate into galaxy frame by PA
3. Project into disk plane using inclination (z-component suppressed)
4. Use a simple rotation curve:
   ```
   V_c(R) = V_max * (1 - exp(-R / R_turn))
   ```
5. **Flux**: Sérsic or exponential disk:
   ```
   I(R) ∝ exp(-R / R_s)
   ```
   Optionally add a bulge: second Sérsic with smaller `R_s` and different index.

6. **Velocity**: Line-of-sight velocity:
   ```
   V_los = V_c(R) * sin(i) * cos(θ) + V_sys
   ```

7. **Normalize**:
   - Rescale flux to `[0, 1]`
   - Compute velocity median and scale → map to `[-1, 1]` for storage

**Pros**: Fully under your control; easy to vary.
**Cons**: Not as "realistic" as MaNGA.

#### Option B – MaNGA-Based Templates (Already Built)

You already have normalized flux+velocity 64×64 tensors.

**Option B1: Direct source-plane template**
- Treat `source_tensor_<plateifu>.npy` as the source-plane flux+velocity grid
- Upsample to a higher-res grid for lensing (e.g., 128×128) using bilinear interpolation

**Option B2: Warp/rotate/scale MaNGA templates**
- Apply random rotations, flips, mild scaling to create variety
- **Note**: Apply the same transform to both flux and velocity
- Rotating 90° flips the velocity gradient direction; that's fine if consistent

**Pros**: Realistic clumpiness; directly from data.
**Cons**: Limited variety, residual noise; less physical control.

#### Option C – Hybrid (Analytic Disk Shaped to Match MaNGA Stats)

Optional and more advanced:
- Fit an analytic disk (Option A) to a given MaNGA template's:
  - Hα half-light radius
  - Inclination estimate from ellipse fit
  - Peak rotation speed
- Use the analytic model for lensing but tune parameters from MaNGA

**Recommendation**: Implement A + B, not C, and describe C as future work.

---

### Phase 2 – Lensing + Subhalo Simulator

Create `src/sim/lens_simulator.py`

#### 2.1 Coordinate System and Pixel Scale

Choose one standard:
- **Image-plane grid**: 64×64 pixels
- **Pixel scale**: e.g., 0.05"/pixel (HST-like) or 0.1" (JWST/NIRCam coarse)
- **Einstein radius**: θ_E ~ 1.0" (20 pixels at 0.05"/pix)

Hardcode for Milestone 2; justify in writeup.

#### 2.2 Macro Lens

Use lenstronomy's PEMD or SIE as the main lens potential.

**Parameters**: Einstein radius, ellipticity, orientation, external shear.

```python
def get_macro_lens_model(theta_E, e1, e2, shear, center=(0, 0)):
    # returns lenstronomy lens_model and kwargs_lens
```

#### 2.3 Subhalos

**Option 1** – Manually inject a single SIS or NFW clump at random position near Einstein ring:
```python
def sample_single_subhalo(mass, r_offset, angle):
    # convert to lens strength; return as extra lens component
```

**Option 2** – Use Paltas to draw many subhalos from a mass function (if time allows).

For ISEF feasibility, **Option 1 is enough**:
- You know exactly where the subhalo is and what mass → perfect labels
- You can control its distance to the bright arcs

#### 2.4 Ray Tracing Flux and Velocity Consistently

**This is critical.**

**Implementation sketch:**

1. **Source-plane grid**:
   - Have `source_flux` and `source_vel` on a fine grid `(Ns, Ns)` (e.g., 128×128)
   - Coordinate range: e.g., `[-Rs, Rs]` where `Rs` ~ few kpc, mapped to arcsec via angular diameter distance

2. **Image-plane grid**:
   - For each image pixel `(i, j)` define angular coordinates `(θ_x, θ_y)` in arcsec via pixel scale

3. **Lens equation**:
   ```python
   beta_x, beta_y = lens_model.ray_shooting(theta_x, theta_y, kwargs_lens)
   ```
   where `(beta_x, beta_y)` are source-plane coordinates.

4. **Interpolation**:
   - Map `(beta_x, beta_y)` into source-grid pixel indices
   - Use bilinear interpolation to sample:
     ```python
     flux_img = sample(source_flux, beta_x, beta_y)
     vel_img = sample(source_vel, beta_x, beta_y)
     ```

5. **Flux-weighted masking (fixing the "zero ambiguity")**:
   ```python
   mask = flux_img > flux_threshold  # e.g., 0.05 of max
   vel_img[~mask] = 0.0
   ```
   Optionally store mask as third channel.

6. **PSF + noise**:
   - Convolve `flux_img` and `vel_img` with a PSF kernel (Gaussian or real instrument PSF)
   - Add:
     - Poisson noise to flux
     - Gaussian noise to velocity where mask is True
   - Ensure noise levels approximate MaNGA/MUSE SNR

7. **Downsampling velocity resolution (optional)**:
   - Compute lensing on a finer grid
   - Block-average velocity to coarser grid (e.g., 32×32) and upsample back to 64×64
   - Keep flux at 64×64

---

### Phase 3 – Dataset Builder

Create `src/datasets/subhalo_dataset.py`

**Core idea**: Each sample is:
- **Input**: Flux-only image `(1×64×64)` or flux+vel tensor `(2×64×64)`
- **Labels**:
  - `label_subhalo_present` (0/1)
  - Optional: `(mass, x, y)` of subhalo for regression

**Implementation steps:**

1. **Config file (YAML/JSON)** describing simulation ranges:
   - Subhalo masses (e.g., 10⁷–10⁹ M☉)
   - Number of samples per mass bin
   - Range of positions (within ±0.2" of Einstein ring)
   - Noise levels, PSF FWHM
   - Choice of source model (analytic vs MaNGA templates)

2. **Sample generation function**:
   ```python
   def generate_sample(with_subhalo: bool) -> dict:
       # choose a source:
       #   - randomly pick analytic or one of canonical MaNGA templates
       # choose lens macro params (sample from small range)
       # if with_subhalo:
       #   sample subhalo mass and position, add to lens model
       # run ray-tracing to produce flux_img, vel_img
       # return { 'flux': flux_img, 'vel': vel_img, 'label': int(with_subhalo), 'meta': {...} }
   ```

3. **Offline generation or on-the-fly**:
   - **Option 1**: Pre-generate N samples and save as `.npz` or `.hdf5` (safer for training reproducibility)
   - **Option 2**: Generate on the fly inside `__getitem__` (simpler code, more CPU cost)

   **Recommendation**: Option 1 is safer.
   
   Script `make_dataset.py`:
   - Generates, say, 10k train + 2k val + 2k test samples
   - Stores them in `data/sim_dataset/train_*.npz`, etc.

---

### Phase 4 – Models and Training

Create `src/models/`:
- `baseline_cnn.py` – single-stream flux-only model
- `dual_stream_cnn.py` – two-branch model (flux and vel)

#### 4.1 Baseline Model (Image-Only)

**Architecture** (simple, robust):
```
Conv(1→16, 3×3, padding=1), ReLU
MaxPool(2×2)
Conv(16→32, 3×3, padding=1), ReLU
MaxPool(2×2)
Conv(32→64, 3×3, padding=1), ReLU
GlobalAvgPool → 64-dim
Linear(64→32), ReLU
Linear(32→2)  # classes: no-subhalo / subhalo
```

**Train with**:
- `CrossEntropyLoss`
- `Adam(lr=1e-3)`
- Batch size 64 (or less if memory limited)

#### 4.2 Dual-Stream Model (Flux + Velocity)

Two branches with shared structure, then fusion:
```
Branch A: same as baseline but input=flux (1 channel)
Branch B: same as baseline but input=velocity (1 channel)
Outputs: feat_A (64-d), feat_B (64-d)
Fusion: concat → 128-d
Head: Linear(128→64), ReLU, Linear(64→2)
```

Training setup identical to baseline.

#### 4.3 Training Protocol

Create `train_baseline.py` and `train_dual_stream.py` scripts:
- Load dataset splits
- Train for N epochs (e.g., 20–50)
- For each epoch, log:
  - Training loss, accuracy
  - Validation loss, accuracy, ROC AUC

**Key metrics for your story**:
- `AUC_baseline` vs `AUC_dual`
- Recall at fixed FPR (e.g., FPR = 5%)
- Performance vs subhalo mass bin (10⁷, 10⁸, 10⁹ M☉)

**Expectations (honest)**:
- At high masses (~10⁹ M☉), both models should do well.
- At low masses (~10⁷ M☉), both may struggle; you want an observable uplift from dual-stream, not miracles.

---

### Phase 5 – Stress Tests and Robustness

To make this "grand-prize strong," you need non-cherry-picked tests.

#### Domain Shift Across Source Morphology

- Train on a subset of analytic disks + a couple of MaNGA templates
- Test on:
  - Different analytic parameters (different inclinations)
  - Different MaNGA templates not seen in training

#### Noise and PSF Variation

- Train with one PSF and SNR level
- Test with:
  - Worse seeing (broader PSF)
  - Lower SNR (stronger noise)
- Show that dual-stream model degrades less severely or remains more robust

#### Velocity Degradation Test (Realism)

- Generate a test set where velocity maps are degraded to a coarser resolution (e.g., 32×32 underlying, then upsampled)
- Compare baseline vs dual-stream performance
- **Goal**: Show kinematics still provides signal even when low-res

#### Ablations

Train dual-stream model with:
1. **Randomized velocities** (destroy physical correlation) → performance should drop to baseline
2. **Only velocity, no flux** → degrade to show both are needed

---

### Phase 6 – Application to Quasi-Real Data

This is where MaNGA comes back visibly into the story.

1. Pick 1–2 MaNGA sources (e.g., `8593-12705`, `8652-12703`)
2. Construct a "near-real" test case:
   - Use the actual MaNGA flux+velocity as source
   - Set lens mass parameters to mimic a plausible strong lens (Einstein radius ~1")
   - Inject a subhalo at known position
   - Generate lensed multi-channel image
3. Run both models on these examples:
   - Show predicted probability of subhalo presence
   - If you implement localization head later, show heatmap/saliency overlay
4. **Document gaps**:
   - Point out that MaNGA kinematics is not actually observed through the lens in reality
   - Real IFU lens samples are rarer (MUSE, KCWI), but your pipeline is ready for them

---

### Phase 7 – Documentation, Figures, and Narrative

You will need:

1. **Flowchart** of the full pipeline (MaNGA → source tensors → simulation → ML)
2. **Example maps** (the ones you already generated) with annotations
3. **One slide or figure contrasting**:
   - Baseline ROC vs dual-stream ROC
   - Detection rate vs subhalo mass
4. **Explicit limitations section**:
   - Kinematics modeled analytically or via MaNGA, not full hydro sim
   - Lens and subhalo models simplified (SIE + SIS)
   - Domain gap to real IFU strong lenses

---

## 4. Risks and Kill-Switch Checks

### Ray-Tracing Implementation Complexity

**Risk**: lenstronomy integration for 2 channels may take longer than expected.

**Mitigation**:
- First implement your existing toy SIS lens (pure numpy) on the 2-channel tensors to validate pipeline
- Only then swap in lenstronomy (same interface)

### Kinematic Modeling Realism

**Risk**: Building a fully physical kinematic model is too heavy.

**Mitigation**:
- Start with analytic disks (Option A)
- Use MaNGA only for visual comparison and 1–2 "near-real" demos

### Signal Might Be Too Weak at Low Masses

**Risk**: No clear gain from dual-stream at 10⁷–10⁸ M☉.

**Mitigation**:
- Tune scenario where subhalo effect is deliberately stronger (e.g., place it close to bright arc, restrict to masses ≥10⁸ M☉)
- Clearly state that the project explores potential of the method and that extrapolation to fainter masses is future work

### Time

If lenstronomy + Paltas combo proves too heavy:
- Use lenstronomy alone with single subhalo
- Skip population-level realism; still scientifically meaningful as a feasibility study

---

## 5. Summary

**Big picture, in brutally honest terms:**

| Component | Reality |
|-----------|---------|
| **MaNGA** | Not replaced, but used primarily to validate preprocessing and provide realistic example sources |
| **Sim engine** | lenstronomy + simple subhalo is where the actual labeled training data must come from |
| **Kinematics** | Will almost certainly require your own analytic or MaNGA-derived modeling, not something Paltas magically supplies |
| **ML** | Two models (flux-only vs dual-stream), evaluated on controlled simulations and a few quasi-real cases |

If you follow the phases above:
- Each step is implementable with your current codebase and environment
- The main heavy lift is writing the lensing simulator carefully and verifying it with simple tests before going large-scale

---

## Quick Reference: File Locations

```
data/
├── maps/                          # Downloaded MaNGA MAPS files
├── source_tensors/                # Prepared (2, 64, 64) tensors
│   └── previews/                  # Visual sanity checks
├── canonical_sources.txt          # List of hero + supporting galaxies
├── usable_maps_index.txt          # Output from batch_inspect_maps.py
├── maps_quality_summary.csv       # Full quality metrics
└── sim_dataset/                   # Generated training data (Phase 3)
    ├── train_*.npz
    ├── val_*.npz
    └── test_*.npz

src/
├── sources/
│   └── analytic_disk.py           # Phase 1 Option A
├── sim/
│   └── lens_simulator.py          # Phase 2
├── datasets/
│   └── subhalo_dataset.py         # Phase 3
├── models/
│   ├── baseline_cnn.py            # Phase 4
│   └── dual_stream_cnn.py         # Phase 4
└── scripts/
    ├── prep_source_maps.py        # Phase 0 (done)
    ├── batch_inspect_maps.py      # Phase 0 (done)
    ├── make_dataset.py            # Phase 3
    ├── train_baseline.py          # Phase 4
    └── train_dual_stream.py       # Phase 4
```

---

*Last updated: November 2024*

