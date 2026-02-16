# Core Leakage Investigation - Raw Data and Code

## Problem Statement

A Logistic Regression classifier using only the central 10x10 pixels achieves AUC = 0.90 distinguishing `stamp` (with injected arc) from `ctrl` (without arc).

You previously suggested 3 potential issues to investigate. Below are the raw test results and code.

---

## Test 1: PSF Convolution Unit Test

### Code (from spark_phase4_pipeline_gen5.py lines 879-889)

```python
def _fft_convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """FFT-based convolution for small stamps (e.g., 64x64)."""
    ih, iw = img.shape
    kh, kw = kernel.shape
    if kh > ih or kw > iw:
        raise ValueError(f"Kernel {kernel.shape} larger than image {img.shape}")
    pad = np.zeros((ih, iw), dtype=np.float32)
    pad[:kh, :kw] = kernel.astype(np.float32)
    pad = np.fft.ifftshift(pad)  # move kernel center to (0,0)
    out = np.fft.ifft2(np.fft.fft2(img.astype(np.float32)) * np.fft.fft2(pad)).real
    return out.astype(np.float32)
```

### Test: Impulse Response

```python
# Create impulse at (32, 32)
img = np.zeros((64, 64), dtype=np.float32)
img[32, 32] = 1.0

# Convolve with PSF kernel (FWHM=5 pix, kernel shape 19x19)
out = _fft_convolve2d(img, kernel)

# Find peak
peak = np.unravel_index(np.argmax(out), out.shape)
```

**Result:**
```
Impulse at: (32, 32)
Peak after convolution: (9, 9)
Flux sum before: 1.0
Flux sum after: 1.0
```

### Test: Compare to Alternative Implementation

```python
def _fft_convolve2d_alt(img, kernel):
    """Alternative implementation using roll instead of ifftshift."""
    ih, iw = img.shape
    kh, kw = kernel.shape
    pad = np.zeros((ih, iw), dtype=np.float32)
    pad[:kh, :kw] = kernel
    oy, ox = kh // 2, kw // 2
    pad = np.roll(np.roll(pad, -oy, axis=0), -ox, axis=1)
    out = np.fft.ifft2(np.fft.fft2(img) * np.fft.fft2(pad)).real
    return out.astype(np.float32)

out_alt = _fft_convolve2d_alt(img, kernel)
peak_alt = np.unravel_index(np.argmax(out_alt), out_alt.shape)
```

**Result:**
```
Peak (original code): (9, 9)
Peak (alternative code): (32, 32)
```

### Test: Where is kernel center after ifftshift?

```python
# 19x19 kernel placed at [0:19, 0:19]
pad = np.zeros((64, 64), dtype=np.float32)
pad[:19, :19] = kernel

# Before ifftshift
print("Before ifftshift, kernel max at:", np.unravel_index(np.argmax(pad), pad.shape))

# After ifftshift
pad_shifted = np.fft.ifftshift(pad)
print("After ifftshift, kernel max at:", np.unravel_index(np.argmax(pad_shifted), pad_shifted.shape))
```

**Result:**
```
Before ifftshift, kernel max at: (9, 9)
After ifftshift, kernel max at: (41, 41)
```

---

## Test 2: Arc Position in Production Data

### Code

```python
# Load actual production data sample with theta_E ~ 2"
sample = df[(df["theta_e_arcsec"] >= 1.8) & (df["theta_e_arcsec"] <= 2.2)].iloc[0]

stamp_data = np.load(io.BytesIO(sample["stamp_npz"]))
ctrl_data = np.load(io.BytesIO(sample["ctrl_stamp_npz"]))

diff = stamp_data["image_r"] - ctrl_data["image_r"]

# Find arc peak
peak_actual = np.unravel_index(np.argmax(diff), diff.shape)
```

### Simulate what buggy vs correct PSF would produce

```python
# Render simple lensed arc (Sersic n=1 source through SIS lens)
arc_no_psf = render_arc(theta_e_pix=7.6)

# Convolve with original code
arc_buggy = _fft_convolve2d(arc_no_psf, kernel)
peak_buggy = np.unravel_index(np.argmax(arc_buggy), arc_buggy.shape)

# Convolve with alternative code
arc_alt = _fft_convolve2d_alt(arc_no_psf, kernel)
peak_alt = np.unravel_index(np.argmax(arc_alt), arc_alt.shape)
```

**Result:**
```
theta_E: 2.0 arcsec = 7.6 pix
PSF FWHM: 1.38 arcsec, sigma = 2.24 pix

Peak locations:
  Original code simulation: (2, 7)
  Alternative code simulation: (25, 30)
  Actual production data: (32, 21)

Distance from actual data peak:
  Original code: 33.1 pix
  Alternative code: 11.4 pix
```

### Quadrant flux distribution in actual data

```python
q1 = np.sum(diff[0:32, 0:32])   # Top-left
q2 = np.sum(diff[0:32, 32:64])  # Top-right
q3 = np.sum(diff[32:64, 0:32])  # Bottom-left
q4 = np.sum(diff[32:64, 32:64]) # Bottom-right
```

**Result:**
```
Top-left (0-32, 0-32):     0.7642
Top-right (0-32, 32-64):   0.4803
Bottom-left (32-64, 0-32): 1.0264
Bottom-right (32-64, 32-64): 0.6877
Total: 2.9585
```

---

## Test 3: Cutout Alignment

### Code

```python
def phase_correlation_shift(img1, img2):
    f1 = np.fft.fft2(img1)
    f2 = np.fft.fft2(img2)
    cross_power = (f1 * np.conj(f2)) / (np.abs(f1 * np.conj(f2)) + 1e-10)
    correlation = np.fft.ifft2(cross_power).real
    max_idx = np.unravel_index(np.argmax(correlation), correlation.shape)
    shift_y = max_idx[0] if max_idx[0] < 32 else max_idx[0] - 64
    shift_x = max_idx[1] if max_idx[1] < 32 else max_idx[1] - 64
    return shift_y, shift_x

# Test on 50 samples, using outer region (mask center to avoid arc)
shifts = []
for i in range(50):
    stamp_r = stamp_data["image_r"]
    ctrl_r = ctrl_data["image_r"]
    mask = np.ones_like(stamp_r)
    mask[16:48, 16:48] = 0
    shift_y, shift_x, _ = phase_correlation_shift(stamp_r * mask, ctrl_r * mask)
    shifts.append((shift_y, shift_x))
```

**Result:**
```
Y-shift: mean=0.0, std=0.0
X-shift: mean=0.0, std=0.0
```

### Shift sensitivity test

```python
core_slice = (slice(27, 37), slice(27, 37))
core_diff_orig = np.mean(stamp_r[core_slice] - ctrl_r[core_slice])

for dy, dx in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,0), (0,1), (1,-1), (1,0), (1,1)]:
    ctrl_shifted = ndimage.shift(ctrl_r, (dy, dx), mode="nearest")
    core_diff = np.mean(stamp_r[core_slice] - ctrl_shifted[core_slice])
```

**Result:**
```
Shift (-1, -1): core_diff = 0.03093
Shift (-1,  0): core_diff = 0.03415
Shift (-1,  1): core_diff = 0.04091
Shift ( 0, -1): core_diff = 0.01090
Shift ( 0,  0): core_diff = 0.01627  <-- unshifted
Shift ( 0,  1): core_diff = 0.02582
Shift ( 1, -1): core_diff = -0.00597
Shift ( 1,  0): core_diff = 0.00131
Shift ( 1,  1): core_diff = 0.01365
```

---

## Test 4: Deflection Unit Consistency

### Code

```python
def deflection_sis(x, y, theta_e, eps=1e-12):
    r = np.sqrt(x * x + y * y) + eps
    ax = theta_e * x / r
    ay = theta_e * y / r
    return ax, ay
```

### Test: SIS deflection magnitude

```python
theta_e = 2.0  # arcsec
for r in [1.0, 2.0, 5.0, 10.0]:
    ax, ay = deflection_sis([r], [0.0], theta_e)
    alpha_mag = np.sqrt(ax[0]**2 + ay[0]**2)
```

**Result:**
```
r = 1.0 arcsec: |alpha| = 2.0 arcsec
r = 2.0 arcsec: |alpha| = 2.0 arcsec
r = 5.0 arcsec: |alpha| = 2.0 arcsec
r = 10.0 arcsec: |alpha| = 2.0 arcsec
```

### Test: Source plane at Einstein radius

```python
x_img, y_img = 2.0, 0.0  # Point at theta_E
ax, ay = deflection_sis([x_img], [y_img], theta_e=2.0)
beta_x = x_img - ax[0]
beta_y = y_img - ay[0]
```

**Result:**
```
Image plane: (2.0, 0.0) arcsec
Deflection: (2.0, 0.0) arcsec
Source plane: (0.0, 0.0) arcsec
```

---

## Previous Diagnostic Results

### Outer annulus check

```
Outer annulus (r >= 20 pix) mean diff: 0.00004
Core (r < 5 pix) mean diff: 0.015
Ratio: 363x
```

### Regression analysis

```
core_offset vs theta_E: R² = 0.10, slope negative
core_offset vs total_arc_flux: R² = 0.52, slope positive
core_offset vs arc_snr: R² = 0.40, slope positive
```

### Stratified by θ_E

```
theta_E < 0.75": 41% of arc flux in core
theta_E 0.75-1": 30% of arc flux in core
theta_E 1-1.5": 21% of arc flux in core
theta_E 1.5-2": 15% of arc flux in core
theta_E >= 2": 7.7% of arc flux in core
```

---

## Questions

1. The current `_fft_convolve2d` code produces peak at (9, 9) when input is impulse at (32, 32). The alternative implementation produces peak at (32, 32). Which behavior is correct for FFT-based convolution?

2. The production data shows arc peak at (32, 21), which is closer to the alternative implementation result (25, 30) than the current code result (2, 7). What does this indicate about how the production data was generated?

3. Given all the data above, what is causing the core leakage (AUC = 0.90 from central 10x10 pixels)?

4. Should we proceed with training, and if so, with what approach?

---

## Attached Files

- `psf_convolution_test.py` - PSF convolution unit test
- `alignment_test.py` - Alignment test code
- `deflection_unit_test.py` - Deflection unit test
- `diagnostic_test1_outer_annulus.py` - Outer annulus check
- `diagnostic_test2_regression.py` - Regression analysis
- `diagnostic_test3_stratified.py` - Stratified analysis
- `diagnostic_test4_radial_profile.py` - Radial profile analysis
- `spark_phase4_pipeline_gen5.py` - Full injection pipeline
