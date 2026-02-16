# PSF Convolution Bug - Unexplained Discrepancy

## Background

You previously identified a PSF convolution bug in `_fft_convolve2d`. I confirmed the bug exists and fixed it. However, I found a discrepancy I cannot explain.

---

## The Bug (Confirmed)

The buggy code:
```python
def _fft_convolve2d(img, kernel):
    ih, iw = img.shape
    kh, kw = kernel.shape
    pad = np.zeros((ih, iw), dtype=np.float32)
    pad[:kh, :kw] = kernel.astype(np.float32)
    pad = np.fft.ifftshift(pad)  # BUG
    out = np.fft.ifft2(np.fft.fft2(img) * np.fft.fft2(pad)).real
    return out.astype(np.float32)
```

### Unit test results:
```
Original source (Sersic at center):
  Peak at: (32, 32)
  Center value: 1.0

After BUGGY PSF convolution:
  Peak at: (9, 9)
  Center (32,32) value: 0.0

Flux by quadrant:
  Top-left (0-32, 0-32):     19.84
  Top-right (0-32, 32-64):   0.18
  Bottom-left (32-64, 0-32): 0.18
  Bottom-right (32-64, 32-64): 0.006
```

The bug shifts a centered source from (32,32) to (9,9). Flux concentrates in top-left quadrant.

---

## Where the Bug Exists

1. **`spark_phase4_pipeline_gen5.py` in git** (before my fix) - lines 995-1006
2. **`spark_phase4_pipeline_moffat.py` on S3** (from Feb 1) - same bug

---

## The Discrepancy

### Production data timeline:
- `v5_cosmos_production` stamps created: **2026-02-04 15:26 UTC**
- `spark_phase4_pipeline_moffat.py` on S3: **2026-02-01**
- That script contains the buggy `ifftshift` code

### What the production data shows:

```python
# Sample with theta_E = 2.0"
diff = stamp - ctrl

# Arc position in actual data:
Peak at: (32, 21)

# Quadrant flux distribution:
Top-left (0-32, 0-32):     0.7642
Top-right (0-32, 32-64):   0.4803
Bottom-left (32-64, 0-32): 1.0264
Bottom-right (32-64, 32-64): 0.6877
```

### Expected if buggy code was used:

- Arc should be shifted to top-left corner (around (9, 9))
- Flux should be heavily concentrated in top-left quadrant

### What we observe:

- Arc peak at (32, 21) - near center, not corner
- Flux distributed across all quadrants

---

## What I Cannot Explain

If the production data was generated using the buggy `_fft_convolve2d`:
- The arcs should be shifted ~23 pixels toward (0,0)
- They're not

Possibilities I considered:
1. **Different code path used** - maybe production didn't use `_fft_convolve2d`?
2. **EMR used different code** - local code differed from git?
3. **Different pipeline entirely** - not the gen5 pipeline?

---

## Code Path Analysis

In `spark_phase4_pipeline_gen5.py`:

```
render_lensed_source() 
  -> calls _convolve_psf() at line 744
  
_convolve_psf()
  -> calls _fft_convolve2d() at line 907
```

So if `render_lensed_source` was used, it would call the buggy code.

But there's also:
```
_convolve_gaussian() at line 802  # Separable convolution - CORRECT
```

This is used by `inject_sis_stamp()` at line 999 (deprecated function).

---

## Questions

1. **Can you explain** why production data shows correctly positioned arcs if the buggy code was used?

2. **What should I check next** to understand which code path actually generated the data?

3. **Given this uncertainty**, what is your recommendation for:
   - Proceeding with training on this data?
   - Regenerating data with the fixed code?

4. **Does this affect your interpretation** of the Test A results (arc-only core fraction)?

---

## Test A Results (for reference)

| θ_E bin | Arc-Only | Diff |
|---------|----------|------|
| < 0.75" | 0.716 | 0.411 |
| 0.75-1" | 0.518 | 0.332 |
| 1-1.5" | 0.293 | 0.237 |
| 1.5-2" | 0.148 | 0.168 |
| ≥ 2" | 0.079 | 0.080 |

For θ_E ≥ 2", arc-only and diff match well.

---

## Attached

- `psf_convolution_test.py` - Unit tests showing the bug
- `spark_phase4_pipeline_gen5.py` - Pipeline with fix applied
