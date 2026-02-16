# Plan B: Contract Definitions

This document defines all contracts between phases to ensure consistency.

**CRITICAL**: All code MUST import from `shared/` module. Do NOT redefine constants, schemas, or utility functions elsewhere.

## 1. Constants Contract (`shared/constants.py`)

### Image Dimensions
| Constant | Value | Description |
|----------|-------|-------------|
| `STAMP_SIZE` | 64 | Stamp size in pixels |
| `STAMP_SHAPE` | (3, 64, 64) | Expected shape (C, H, W) |
| `NUM_CHANNELS` | 3 | Number of bands (g, r, z) |
| `PIXEL_SCALE_ARCSEC` | 0.262 | DECaLS pixel scale |

### Core/Outer Definitions
| Constant | Value | Description |
|----------|-------|-------------|
| `CORE_RADIUS_PIX` | 5 | r < 5 is "core" |
| `CORE_SIZE_PIX` | 10 | 10x10 box for LR gate |
| `OUTER_RADIUS_PIX` | 20 | r > 20 is "outer" for normalization |

### Gate Thresholds
| Gate | Threshold | Direction | Description |
|------|-----------|-----------|-------------|
| `auroc_synth_min` | 0.85 | > | Minimum AUROC on synthetic test |
| `core_lr_auc_max` | 0.65 | < | Maximum core LR AUC (shortcut gate) |
| `core_masked_drop_max` | 0.10 | < | Maximum drop when core masked |
| `hardneg_auroc_min` | 0.70 | > | Minimum AUROC on hard negatives |

### Catalog Requirements
| Constant | Value | Description |
|----------|-------|-------------|
| `MIN_ANCHORS` | 30 | Minimum anchor lenses |
| `MIN_RING_CONTAMINANTS` | 50 | Minimum ring galaxies |
| `MIN_SPIRAL_CONTAMINANTS` | 50 | Minimum spirals |
| `MIN_MERGER_CONTAMINANTS` | 30 | Minimum mergers |
| `MIN_THETA_E_ARCSEC` | 0.5 | Minimum Einstein radius |

---

## 2. Schema Contract (`shared/schema.py`)

### Parquet Schema
Required columns for training parquet files:

| Column | Type | Description |
|--------|------|-------------|
| `stamp_npz` | bytes | NPZ blob with injected stamp |
| `ctrl_stamp_npz` | bytes | NPZ blob with control (no injection) |

Optional columns:
- `task_id`, `brickname`, `ra`, `dec`
- `theta_e_arcsec`, `arc_snr`, `src_z`, `lens_z`
- `psf_fwhm_arcsec`, `bandset`

### NPZ Format
Primary format (multi-band):
```python
{
    "image_g": np.ndarray,  # (64, 64) float32
    "image_r": np.ndarray,  # (64, 64) float32
    "image_z": np.ndarray,  # (64, 64) float32
}
```

Legacy format (single key):
```python
{"img": np.ndarray}  # (64, 64) or (C, 64, 64)
```

### Anchor Schema
| Column | Type | Required |
|--------|------|----------|
| `name` | str | Yes |
| `ra` | float | Yes |
| `dec` | float | Yes |
| `theta_e_arcsec` | float | Yes |
| `source` | str | Yes |
| `tier` | str | No |

### Contaminant Schema
| Column | Type | Required |
|--------|------|----------|
| `name` | str | Yes |
| `ra` | float | Yes |
| `dec` | float | Yes |
| `category` | str | Yes |

Valid categories: `ring`, `spiral`, `merger`, `spike`

### Batch Schema
Training batch dictionary:

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `x` | (B, 3, 64, 64) | torch.Tensor | Input images |
| `y` | (B,) | torch.Tensor | Labels (0.0 or 1.0) |
| `theta_e` | (B//2,) | torch.Tensor | Optional: Einstein radii |
| `arc_snr` | (B//2,) | torch.Tensor | Optional: Arc SNR |

---

## 3. Utility Function Contract (`shared/utils.py`)

These are the ONLY implementations. Do NOT reimplement elsewhere.

### `decode_stamp_npz(blob: bytes) -> Tuple[np.ndarray, str]`
- Input: Raw bytes from parquet `stamp_npz` or `ctrl_stamp_npz` column
- Output: (array shape (C, H, W), bandset string like "grz")
- Handles gzip compression

### `validate_stamp(stamp: np.ndarray, name: str) -> Dict[str, bool]`
- Checks: shape, NaN, Inf, variance, range
- Returns dict with `valid` key

### `robust_normalize(img: np.ndarray, outer_radius: int, clip_sigma: float) -> np.ndarray`
- Normalizes using outer annulus median/MAD
- Returns clipped normalized image

### `azimuthal_shuffle(diff: np.ndarray, n_bins: int, seed: Optional[int]) -> np.ndarray`
- Shuffles pixels within radial bins
- Preserves radial profile, destroys morphology

### `apply_core_dropout(img: np.ndarray, radius: int, fill_mode: str) -> np.ndarray`
- Masks central pixels
- fill_mode: "outer_median", "zero", or "noise"

### `create_radial_mask(height: int, width: int, radius: float, inside: bool) -> np.ndarray`
- Creates circular boolean mask

---

## 4. Cross-Phase Contract

### Phase 0 → Phase 1
- Phase 0 validates data exists and meets schema
- Phase 1 expects parquet files at `{parquet_root}/{split}/*.parquet`
- Phase 1 expects columns defined in `PARQUET_SCHEMA`

### Phase 1 → Phase 4
- Phase 1 produces checkpoint at `checkpoints/{variant}/best_model.pt`
- Phase 1 produces validation JSON at `results/{variant}_validation.json`
- Phase 4 aggregates from `checkpoints/**/validation.json`

### Data Flow
```
parquet files → decode_stamp_npz → validate_stamp → robust_normalize → DataLoader → batch
```

---

## 5. Validation Rules

### Before Training
1. All Phase 0 checks must pass
2. Data loader produces valid batches (see `validate_loader`)
3. Model forward/backward pass works (see `validate_model`)

### During Training
1. Loss is finite (no NaN)
2. AUROC is tracked every epoch
3. Gate metrics evaluated at end of training

### After Training
1. All gates must pass (see `GATES` thresholds)
2. Validation JSON must be saved
3. Model checkpoint must be saved

---

## 6. Adding New Constants/Schemas

When adding new constants or schemas:

1. Add to `shared/constants.py` or `shared/schema.py`
2. Add validation in `validate_constants()` if applicable
3. Update this document
4. Update `tests/test_contracts.py` with new test
5. Run contract tests: `python tests/test_contracts.py`

**Never add magic numbers directly in phase code.**
