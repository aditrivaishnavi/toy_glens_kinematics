"""
Constants for Plan B codebase.

All magic numbers are defined here. Do NOT use hardcoded values elsewhere.

Lessons Learned:
- L1.6: PSF kernel exceeding stamp size - need explicit limits
- L21: Core brightness shortcut - consistent core radius
"""
from dataclasses import dataclass
from typing import Tuple


# =============================================================================
# IMAGE DIMENSIONS
# =============================================================================

STAMP_SIZE: int = 64  # Pixels
STAMP_SHAPE: Tuple[int, int, int] = (3, 64, 64)  # (C, H, W)
NUM_CHANNELS: int = 3  # g, r, z bands
PIXEL_SCALE_ARCSEC: float = 0.262  # DECaLS pixel scale


# =============================================================================
# CORE DEFINITIONS
# =============================================================================
# These define what "core" means consistently across all code

CORE_RADIUS_PIX: int = 5  # r < 5 pixels is "core"
CORE_SIZE_PIX: int = 10  # 10x10 central box for LR gate
CORE_SLICE = slice(27, 37)  # For 64x64: center 10x10 is [27:37, 27:37]


def get_core_slice(stamp_size: int = STAMP_SIZE, core_size: int = CORE_SIZE_PIX):
    """Get slice for core region given stamp and core sizes."""
    center = stamp_size // 2
    half = core_size // 2
    return slice(center - half, center + half)


# =============================================================================
# NORMALIZATION
# =============================================================================

OUTER_RADIUS_PIX: int = 20  # r > 20 pixels is "outer" for normalization
CLIP_SIGMA: float = 5.0  # Clip normalized values to [-5, 5]
MAD_TO_STD: float = 1.4826  # Conversion factor


# =============================================================================
# TRAINING DEFAULTS
# =============================================================================

DEFAULT_BATCH_SIZE: int = 128
DEFAULT_LR: float = 1e-4
DEFAULT_WEIGHT_DECAY: float = 1e-4
DEFAULT_EPOCHS: int = 50
DEFAULT_EARLY_STOPPING_PATIENCE: int = 15


# =============================================================================
# MITIGATION DEFAULTS
# =============================================================================

DEFAULT_HARD_NEGATIVE_RATIO: float = 0.4
DEFAULT_CORE_DROPOUT_PROB: float = 0.5
DEFAULT_CORE_DROPOUT_RADIUS: int = 5
AZIMUTHAL_SHUFFLE_BINS: int = 8


# =============================================================================
# GATE THRESHOLDS
# =============================================================================
# These are the exit criteria for training

@dataclass(frozen=True)
class GateThresholds:
    """Gate thresholds - MUST NOT be modified after Phase 0."""
    
    # Minimum AUROC on synthetic test
    auroc_synth_min: float = 0.85
    
    # Maximum Core LR AUC (shortcut gate)
    core_lr_auc_max: float = 0.65
    
    # Maximum drop when core is masked
    core_masked_drop_max: float = 0.10
    
    # Minimum AUROC on hard negatives
    hardneg_auroc_min: float = 0.70


GATES = GateThresholds()


# =============================================================================
# VALUE RANGES
# =============================================================================
# For data validation

VALUE_RANGE_MIN: float = -1e6
VALUE_RANGE_MAX: float = 1e6
NORMALIZED_VALUE_MIN: float = -50.0
NORMALIZED_VALUE_MAX: float = 50.0
VARIANCE_MIN: float = 1e-10  # Minimum variance (detect flat images)


# =============================================================================
# ANCHOR/CONTAMINANT REQUIREMENTS
# =============================================================================

MIN_ANCHORS: int = 30
MIN_RING_CONTAMINANTS: int = 50
MIN_SPIRAL_CONTAMINANTS: int = 50
MIN_MERGER_CONTAMINANTS: int = 30
MIN_THETA_E_ARCSEC: float = 0.5


# =============================================================================
# VALIDATION
# =============================================================================

def validate_constants():
    """Validate constants are self-consistent."""
    assert STAMP_SIZE > 0, "STAMP_SIZE must be positive"
    assert STAMP_SHAPE == (NUM_CHANNELS, STAMP_SIZE, STAMP_SIZE)
    assert CORE_RADIUS_PIX < STAMP_SIZE // 2, "Core radius too large"
    assert OUTER_RADIUS_PIX > CORE_RADIUS_PIX, "Outer radius must exceed core"
    assert OUTER_RADIUS_PIX < STAMP_SIZE // 2, "Outer radius too large"
    assert 0 < GATES.auroc_synth_min < 1, "Invalid auroc threshold"
    assert 0 < GATES.core_lr_auc_max < 1, "Invalid core LR AUC threshold"
    assert 0 < GATES.core_masked_drop_max < 1, "Invalid drop threshold"
    assert 0 < GATES.hardneg_auroc_min < 1, "Invalid hardneg threshold"
    

# Run validation on import
validate_constants()
