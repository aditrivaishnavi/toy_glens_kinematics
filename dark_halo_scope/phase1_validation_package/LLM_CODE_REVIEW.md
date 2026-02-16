# Critical Code Review: LLM Reference Implementation

Reviewing the ~600-line reference implementation for subtle bugs.

---

## Issue 1: CRITICAL - `azimuthal_shuffle_delta` Destroys Color Coherence

**Location:** `azimuthal_shuffle_delta()` function

**Code:**
```python
perm = rng.permutation(idxs)
# shared permutation across channels to keep color coherence
out_flat[:, idxs] = flat_delta[:, perm]
```

**Problem:** The comment says "shared permutation across channels" but the indexing is WRONG.

- `out_flat` has shape `(C, H*W)`
- `idxs` are indices into the flattened spatial dimension
- `perm` is a permutation of those indices
- `out_flat[:, idxs] = flat_delta[:, perm]` assigns values from `perm` positions to `idxs` positions

**This is actually CORRECT** - all channels use the same `perm` so color coherence IS preserved. ✓

**Wait, let me re-check...** Actually there's a subtle issue:
- `flat_delta[:, perm]` reads from permuted positions
- But `out_flat[:, idxs]` writes to original positions

This means pixel at position `idxs[i]` gets value from position `perm[i]`. Since `perm` is a permutation of `idxs`, this is a valid shuffle within the radial bin. **CORRECT** ✓

---

## Issue 2: CRITICAL - `_radius_maps` Center Calculation

**Location:** `_radius_maps()` function

**Code:**
```python
def _radius_maps(h: int, w: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    dy = yy - cy
    dx = xx - cx
```

**Problem:** For even-sized stamps (64x64), center is at (32, 32), but pixel (31, 31) and (32, 32) are equidistant from true center (31.5, 31.5).

**Impact:** Minor asymmetry. For our 64x64 stamps, this is acceptable but worth noting.

**Verdict:** ACCEPTABLE ✓ (consistent with our existing code)

---

## Issue 3: POTENTIAL BUG - `CoaddCutoutProvider.fetch()` WCS Pixel Order

**Location:** `CoaddCutoutProvider.fetch()`

**Code:**
```python
xpix, ypix = wcs.world_to_pixel_values(float(ra), float(dec))
xpix = int(round(float(xpix)))
ypix = int(round(float(ypix)))
# ...
cutout = img[y0:y1, x0:x1]  # Note: img[y, x] order!
```

**Problem:** WCS returns (x, y) but numpy arrays are indexed as [row, col] = [y, x]. 

**Check:** The code does `img[y0:y1, x0:x1]` which is correct for numpy's row-major order.

**But wait:** FITS data from astropy is typically in (NAXIS1, NAXIS2) = (x, y) order. Need to verify if `hdul[1].data` has shape (naxis2, naxis1) or (naxis1, naxis2).

**Astropy convention:** `hdul[1].data` has shape `(NAXIS2, NAXIS1)` = `(ny, nx)`, so indexing as `[y, x]` is CORRECT. ✓

---

## Issue 4: BUG - `_fill_mask_from_outer` Uses Wrong Replacement

**Location:** `_fill_mask_from_outer()` function

**Code:**
```python
def _fill_mask_from_outer(img: np.ndarray, mask: np.ndarray, outer_mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = img.copy()
    for c in range(img.shape[0]):
        pool = img[c][outer_mask]
        if pool.size == 0:
            continue
        n = int(mask.sum())
        out[c][mask] = rng.choice(pool, size=n, replace=True)
    return out
```

**Problem:** This samples FROM the outer region and places INTO the masked region. But `rng.choice(pool, size=n)` returns a 1D array, and `out[c][mask]` expects a 1D array of size `mask.sum()`.

**Check:** `mask.sum()` gives the number of True values, `rng.choice(..., size=n)` returns array of size n. These should match.

**Verdict:** CORRECT ✓

---

## Issue 5: CRITICAL BUG - `gaussian_blur_torch` Kernel Not Normalized After Clipping

**Location:** `gaussian_blur_torch()` and `_gaussian_kernel_1d()`

**Code:**
```python
def _gaussian_kernel_1d(sigma: float, truncate: float = 3.0, device: Optional[torch.device] = None) -> torch.Tensor:
    if sigma <= 0:
        return torch.tensor([1.0], dtype=torch.float32, device=device)
    radius = int(math.ceil(truncate * float(sigma)))
    xs = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    k = torch.exp(-(xs * xs) / (2.0 * float(sigma) * float(sigma)))
    k = k / (k.sum() + 1e-12)
    return k
```

**Check:** The kernel IS normalized (`k / k.sum()`). **CORRECT** ✓

---

## Issue 6: POTENTIAL BUG - `Preprocess6CH` Creates Residual on Batch Dim 1

**Location:** `Preprocess6CH.__call__()`

**Code:**
```python
def __call__(self, x_np: np.ndarray) -> torch.Tensor:
    raw = robust_mad_norm_outer_np(x_np, self.outer_mask, clip=self.clip)  # (3,H,W)
    t = torch.from_numpy(raw).unsqueeze(0)  # (1,3,H,W)
    t = t.contiguous()
    blur = gaussian_blur_torch(t, sigma=self.resid_sigma_pix)
    resid = (t - blur).squeeze(0)  # (3,H,W)
    raw_t = t.squeeze(0)
    x6 = torch.cat([raw_t, resid], dim=0)  # (6,H,W)
    return x6
```

**Check:** 
- Input: `x_np` is `(C,H,W)` = `(3,64,64)`
- After unsqueeze: `(1,3,64,64)`
- After blur: `(1,3,64,64)`
- After squeeze: `(3,64,64)`
- After cat on dim=0: `(6,64,64)`

**Verdict:** CORRECT ✓

---

## Issue 7: BUG - `PairedParquetDataset` Missing Required Column

**Location:** `PairedParquetDataset._build_index()`

**Code:**
```python
cols = ["stamp_npz", "ra", "dec", "brickname", "theta_e_arcsec", "psfsize_r"]
if "psfdepth_r" in self.dataset.schema.names:
    cols.append("psfdepth_r")
if "arc_snr" in self.dataset.schema.names:
    cols.append("arc_snr")
if "row_id" in self.dataset.schema.names:
    cols.append("row_id")
else:
    cols.append("brickname")  # fallback - BUT BRICKNAME ALREADY IN COLS!
```

**Problem:** If `row_id` is not in schema, it adds `brickname` again, which is already in `cols`. This would cause issues when building the dict.

**Actually:** When building the dict `{c: table[c][i].as_py() for c in cols}`, duplicate keys just overwrite. Not a crash, but wasteful.

**Better fix:**
```python
if "row_id" not in self.dataset.schema.names:
    # brickname already in cols, no need to add
    pass
```

**Verdict:** MINOR BUG - causes redundant work but not crash ✓

---

## Issue 8: CRITICAL BUG - `GateRunner._theta_aware_core_mask` Returns Empty for Small θ

**Location:** `GateRunner._theta_aware_core_mask()`

**Code:**
```python
def _theta_aware_core_mask(self, theta_pix: float, psf_fwhm_pix: float, k: float = 1.5) -> np.ndarray:
    rc = max(theta_pix - k * psf_fwhm_pix, 0.0)
    return self._r < float(rc)
```

**Problem:** For θ_E = 5 pixels and PSF FWHM = 5 pixels:
- `rc = max(5 - 1.5*5, 0) = max(-2.5, 0) = 0`
- Returns `r < 0` which is an EMPTY MASK!

**This is intentional** - for unresolved lenses, there IS no distinct core. The code handles this later:

```python
# If core radius is 0 for most samples, core mask is empty -> fallback to fixed r<2
if core_m.sum() < 8:
    core_m = make_circular_mask(STAMP_SIZE, STAMP_SIZE, r0=2.0)
```

**Verdict:** HANDLED CORRECTLY ✓

---

## Issue 9: BUG - `_extract_region_stats` Doesn't Handle Empty Regions

**Location:** `_extract_region_stats()`

**Code:**
```python
def _extract_region_stats(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    feats = []
    for c in range(x.shape[0]):
        v = x[c][mask]
        feats.extend([
            float(np.mean(v)),
            float(np.std(v)),
            # ...
        ])
    return np.array(feats, dtype=np.float32)
```

**Problem:** If `mask` is all False, `v` is empty array. Then:
- `np.mean([])` returns `nan` with warning
- `np.std([])` returns `nan` with warning
- `np.percentile([], 25)` raises error!

**This WILL crash** for empty masks.

**Fix needed:**
```python
def _extract_region_stats(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    feats = []
    for c in range(x.shape[0]):
        v = x[c][mask]
        if v.size == 0:
            feats.extend([0.0] * 7)  # 7 stats per channel
            continue
        feats.extend([
            float(np.mean(v)),
            # ...
        ])
    return np.array(feats, dtype=np.float32)
```

**Verdict:** BUG - needs fix ⚠️

---

## Issue 10: BUG - `_radial_profile_features` Empty Bin Handling

**Location:** `_radial_profile_features()`

**Code:**
```python
for b in range(nbins):
    m = (r >= edges[b]) & (r < edges[b + 1])
    v = x[c][m]
    feats.append(float(np.mean(v)) if v.size else 0.0)
```

**Check:** This DOES handle empty bins correctly with `if v.size else 0.0`. **CORRECT** ✓

---

## Issue 11: CRITICAL - Model Signature Mismatch

**Location:** `GateRunner._predict_model()`

**Code:**
```python
def _predict_model(self, model, x6, psf_fwhm_pix, psfdepth_r, device, batch_size=128):
    # ...
    meta = torch.stack([psf_fwhm_pix[i:i+batch_size], psfdepth_r[i:i+batch_size]], dim=1).to(device)
    logit = model(xb, meta)
```

**Assumption:** Model takes `(x, meta)` where `meta` is `(B, 2)` tensor with `[psf_fwhm_pix, psfdepth_r]`.

**Our model:** Takes `meta` as `(B, 2)` with `[psfsize_r, psfdepth_r]` (arcsec, not pixels!).

**MISMATCH!** The code passes `psf_fwhm_pix` (in pixels) but our model expects `psfsize_r` (in arcsec).

**Fix needed:**
```python
# Convert back to arcsec for model
psf_fwhm_arcsec = psf_fwhm_pix * PIX_SCALE_ARCSEC
meta = torch.stack([psf_fwhm_arcsec[i:i+batch_size], psfdepth_r[i:i+batch_size]], dim=1).to(device)
```

**Verdict:** BUG - needs fix ⚠️

---

## Issue 12: POTENTIAL - `PairedMixCollate` RNG Not Per-Worker Safe

**Location:** `PairedMixCollate.__init__()`

**Code:**
```python
self.rng = np.random.default_rng(int(seed))
```

**Problem:** In multi-worker DataLoader, each worker shares the same RNG state initially. This means:
- Worker 0 and Worker 1 may generate identical sequences
- Need per-worker seeding

**Fix needed:**
```python
def __call__(self, batch):
    # Get worker info for per-worker randomness
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        worker_seed = self.base_seed + worker_info.id
        rng = np.random.default_rng(worker_seed)
    else:
        rng = self.rng
    # Use rng instead of self.rng
```

**Verdict:** BUG in multi-worker setting ⚠️

---

## Issue 13: Memory - `GateRunner.run()` Loads All Pairs into RAM

**Location:** `GateRunner.run()`

**Code:**
```python
for pos, ctrl, meta in pairs:
    pos_list.append(pos)
    ctrl_list.append(ctrl)
    # ...
```

**Problem:** For `max_pairs=2000`, this stores 2000 * 2 * (3 * 64 * 64 * 4 bytes) = ~200 MB. Acceptable but could be streamed.

**Verdict:** ACCEPTABLE ✓ (for max_pairs=2000)

---

## Issue 14: BUG - `decode_stamp_npz` Doesn't Handle Compressed NPZ

**Location:** `decode_stamp_npz()`

**Code:**
```python
def decode_stamp_npz(blob: bytes) -> np.ndarray:
    bio = io.BytesIO(blob)
    with np.load(bio) as z:
        arrs = [z[k].astype(np.float32) for k in NPZ_KEYS]
    x = np.stack(arrs, axis=0)
```

**Check:** `np.load()` handles both `.npz` and `.npz` (compressed) transparently. **CORRECT** ✓

But wait - our pipeline stores stamps as gzip-compressed NPZ. Let me check if `np.load` handles this...

**Test:** `np.load(io.BytesIO(gzip_compressed_bytes))` will FAIL because np.load doesn't auto-decompress gzip.

**Our actual data:** Uses `gzip.compress(npz_bytes)` before storing in Parquet.

**Fix needed:**
```python
def decode_stamp_npz(blob: bytes) -> np.ndarray:
    bio = io.BytesIO(blob)
    try:
        with np.load(bio) as z:
            arrs = [z[k].astype(np.float32) for k in NPZ_KEYS]
    except:
        # Try gzip decompression
        import gzip
        bio = io.BytesIO(gzip.decompress(blob))
        with np.load(bio) as z:
            arrs = [z[k].astype(np.float32) for k in NPZ_KEYS]
    x = np.stack(arrs, axis=0)
    return x
```

**Verdict:** LIKELY BUG depending on our data format ⚠️

---

## Issue 15: CRITICAL - `make_hardneg` May Produce Out-of-Range Values

**Location:** `make_hardneg()`

**Code:**
```python
def make_hardneg(ctrl: np.ndarray, pos: np.ndarray, nbins: int, rng: np.random.Generator) -> np.ndarray:
    delta = (pos - ctrl).astype(np.float32)
    shuf = azimuthal_shuffle_delta(delta, nbins=nbins, rng=rng)
    return (ctrl + shuf).astype(np.float32)
```

**Problem:** After shuffling, `ctrl + shuf` might exceed the original data range. If `ctrl` is already near the clip limit (±10), adding shuffled delta could push it further.

**Impact:** During normalization, values outside [-clip, clip] get clipped. The hard negative might have different clipping pattern than original, potentially leaking information.

**Mitigation:** Apply clipping AFTER hard negative construction (which the preprocessing does). **ACCEPTABLE** ✓

---

## Summary of Bugs Found

| Issue | Severity | Status |
|-------|----------|--------|
| 9. Empty region stats crashes | HIGH | Needs fix |
| 11. Model meta units mismatch (pix vs arcsec) | HIGH | Needs fix |
| 12. RNG not worker-safe | MEDIUM | Needs fix |
| 14. Gzip decompression missing | MEDIUM | Needs verification |
| 7. Duplicate brickname in cols | LOW | Minor |

---

## Recommended Fixes Before Using

```python
# Fix Issue 9: Empty region handling
def _extract_region_stats(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    feats = []
    for c in range(x.shape[0]):
        v = x[c][mask]
        if v.size == 0:
            feats.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            continue
        feats.extend([
            float(np.mean(v)),
            float(np.std(v)),
            float(np.median(v)),
            float(np.percentile(v, 25)),
            float(np.percentile(v, 75)),
            float(np.max(v)),
            float(np.min(v)),
        ])
    return np.array(feats, dtype=np.float32)

# Fix Issue 11: Meta units
# In GateRunner._predict_model:
psf_fwhm_arcsec = psf_fwhm_pix * PIX_SCALE_ARCSEC
meta = torch.stack([psf_fwhm_arcsec[i:i+batch_size], psfdepth_r[i:i+batch_size]], dim=1).to(device)

# Fix Issue 12: Per-worker RNG
# In PairedMixCollate.__call__:
worker_info = torch.utils.data.get_worker_info()
if worker_info is not None:
    rng = np.random.default_rng(self.base_seed + worker_info.id + int(time.time() * 1000) % 10000)
else:
    rng = self.rng

# Fix Issue 14: Gzip handling
def decode_stamp_npz(blob: bytes) -> np.ndarray:
    bio = io.BytesIO(blob)
    try:
        with np.load(bio) as z:
            if NPZ_KEYS[0] in z:
                arrs = [z[k].astype(np.float32) for k in NPZ_KEYS]
            else:
                raise KeyError("Expected keys not found")
    except Exception:
        import gzip
        try:
            decompressed = gzip.decompress(blob)
            bio = io.BytesIO(decompressed)
            with np.load(bio) as z:
                arrs = [z[k].astype(np.float32) for k in NPZ_KEYS]
        except Exception as e:
            raise ValueError(f"Cannot decode stamp_npz: {e}")
    x = np.stack(arrs, axis=0)
    if x.shape[-2:] != (STAMP_SIZE, STAMP_SIZE):
        raise ValueError(f"Unexpected stamp shape: {x.shape}")
    return x
```
