"""
Schema definitions for Plan B codebase.

This module defines the expected schema for all data files.
All code MUST validate against these schemas.

Lessons Learned:
- L9.1: Always verify data format matches actual data
- L7.2: Handle missing bands gracefully
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
import numpy as np


# =============================================================================
# PARQUET SCHEMA
# =============================================================================

@dataclass
class ParquetSchema:
    """
    Schema for paired training parquet files.
    
    Required columns MUST exist.
    Optional columns may be absent.
    """
    
    # Required columns - MUST exist
    required_columns: Set[str] = field(default_factory=lambda: {
        "stamp_npz",       # bytes: NPZ blob with injected stamp
        "ctrl_stamp_npz",  # bytes: NPZ blob with control (no injection)
    })
    
    # Optional metadata columns - may be absent
    optional_columns: Set[str] = field(default_factory=lambda: {
        "task_id",         # str: Unique identifier
        "brickname",       # str: DESI brick name
        "ra",              # float: Right ascension
        "dec",             # float: Declination
        "theta_e_arcsec",  # float: Einstein radius in arcsec
        "arc_snr",         # float: Arc signal-to-noise ratio
        "src_z",           # float: Source redshift
        "lens_z",          # float: Lens redshift
        "psf_fwhm_arcsec", # float: PSF FWHM
        "bandset",         # str: Band set (e.g., "grz")
    })
    
    # NPZ key schema
    npz_band_keys: List[str] = field(default_factory=lambda: [
        "image_g", "image_r", "image_z"
    ])
    npz_legacy_key: str = "img"
    
    # Expected data types
    column_types: Dict[str, str] = field(default_factory=lambda: {
        "stamp_npz": "bytes",
        "ctrl_stamp_npz": "bytes",
        "task_id": "str",
        "brickname": "str",
        "ra": "float64",
        "dec": "float64",
        "theta_e_arcsec": "float64",
        "arc_snr": "float64",
    })
    
    def validate_dataframe(self, df, strict: bool = False) -> Dict[str, Any]:
        """
        Validate a DataFrame against this schema.
        
        Args:
            df: pandas DataFrame
            strict: If True, fail on missing optional columns
        
        Returns:
            Dict with validation results
        """
        results = {
            "valid": True,
            "missing_required": [],
            "missing_optional": [],
            "extra_columns": [],
            "type_mismatches": [],
        }
        
        columns = set(df.columns)
        
        # Check required columns
        missing_required = self.required_columns - columns
        if missing_required:
            results["valid"] = False
            results["missing_required"] = list(missing_required)
        
        # Check optional columns
        missing_optional = self.optional_columns - columns
        results["missing_optional"] = list(missing_optional)
        if strict and missing_optional:
            results["valid"] = False
        
        # Check for extra columns (informational)
        known = self.required_columns | self.optional_columns
        extra = columns - known
        results["extra_columns"] = list(extra)
        
        return results


# Default schema instance
PARQUET_SCHEMA = ParquetSchema()


# =============================================================================
# ANCHOR SCHEMA
# =============================================================================

@dataclass
class AnchorSchema:
    """Schema for anchor (real lens) catalog."""
    
    required_columns: Set[str] = field(default_factory=lambda: {
        "name",            # str: Unique identifier
        "ra",              # float: Right ascension
        "dec",             # float: Declination
        "theta_e_arcsec",  # float: Einstein radius
        "source",          # str: Discovery source (e.g., "SLACS", "Jacobs")
    })
    
    optional_columns: Set[str] = field(default_factory=lambda: {
        "tier",            # str: "A" or "B"
        "arc_visible",     # bool: Arc visible in DR10
        "z_lens",          # float: Lens redshift
        "z_source",        # float: Source redshift
    })
    
    def validate_dataframe(self, df) -> Dict[str, Any]:
        """Validate anchor DataFrame."""
        results = {"valid": True, "errors": []}
        
        columns = set(df.columns)
        missing = self.required_columns - columns
        if missing:
            results["valid"] = False
            results["errors"].append(f"Missing columns: {missing}")
        
        return results


ANCHOR_SCHEMA = AnchorSchema()


# =============================================================================
# CONTAMINANT SCHEMA
# =============================================================================

@dataclass
class ContaminantSchema:
    """Schema for contaminant (non-lens) catalog."""
    
    required_columns: Set[str] = field(default_factory=lambda: {
        "name",      # str: Unique identifier
        "ra",        # float: Right ascension
        "dec",       # float: Declination
        "category",  # str: "ring", "spiral", "merger", etc.
    })
    
    valid_categories: Set[str] = field(default_factory=lambda: {
        "ring", "spiral", "merger", "spike"
    })
    
    def validate_dataframe(self, df) -> Dict[str, Any]:
        """Validate contaminant DataFrame."""
        results = {"valid": True, "errors": []}
        
        columns = set(df.columns)
        missing = self.required_columns - columns
        if missing:
            results["valid"] = False
            results["errors"].append(f"Missing columns: {missing}")
        
        if "category" in columns:
            categories = set(df.category.unique())
            invalid = categories - self.valid_categories
            if invalid:
                results["errors"].append(f"Invalid categories: {invalid}")
        
        return results


CONTAMINANT_SCHEMA = ContaminantSchema()


# =============================================================================
# STAMP SCHEMA
# =============================================================================

@dataclass
class StampSchema:
    """Schema for decoded stamp arrays."""
    
    expected_shape: tuple = (3, 64, 64)
    dtype: np.dtype = np.float32
    
    # Value constraints
    min_value: float = -1e6
    max_value: float = 1e6
    min_variance: float = 1e-10
    
    def validate_array(self, arr: np.ndarray, name: str = "stamp") -> Dict[str, Any]:
        """
        Validate a stamp array.
        
        Returns dict with validation results.
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
        }
        
        # Shape check
        if arr.shape != self.expected_shape:
            if arr.shape[0] == 3 and arr.ndim == 3:
                results["warnings"].append(
                    f"{name} shape {arr.shape} != expected {self.expected_shape}"
                )
            else:
                results["valid"] = False
                results["errors"].append(
                    f"{name} shape {arr.shape} incompatible with expected {self.expected_shape}"
                )
        
        # NaN check
        if np.isnan(arr).any():
            results["valid"] = False
            results["errors"].append(f"{name} contains NaN")
        
        # Inf check
        if np.isinf(arr).any():
            results["valid"] = False
            results["errors"].append(f"{name} contains Inf")
        
        # Range check
        if arr.min() < self.min_value or arr.max() > self.max_value:
            results["valid"] = False
            results["errors"].append(
                f"{name} range [{arr.min():.2e}, {arr.max():.2e}] exceeds limits"
            )
        
        # Variance check
        if arr.std() < self.min_variance:
            results["warnings"].append(f"{name} has very low variance")
        
        return results


STAMP_SCHEMA = StampSchema()


# =============================================================================
# BATCH SCHEMA
# =============================================================================

@dataclass
class BatchSchema:
    """Schema for training batches."""
    
    required_keys: Set[str] = field(default_factory=lambda: {
        "x",  # Tensor: (B, C, H, W)
        "y",  # Tensor: (B,) or (B, 1)
    })
    
    optional_keys: Set[str] = field(default_factory=lambda: {
        "theta_e",
        "arc_snr",
        "n_hard_negatives",
    })
    
    def validate_batch(self, batch: Dict) -> Dict[str, Any]:
        """Validate a training batch."""
        import torch
        
        results = {"valid": True, "errors": []}
        
        # Check required keys
        missing = self.required_keys - set(batch.keys())
        if missing:
            results["valid"] = False
            results["errors"].append(f"Missing keys: {missing}")
            return results
        
        x = batch["x"]
        y = batch["y"]
        
        # Check x shape
        if x.dim() != 4:
            results["valid"] = False
            results["errors"].append(f"x must be 4D, got {x.dim()}D")
        elif x.shape[1] != 3 or x.shape[2] != 64 or x.shape[3] != 64:
            results["valid"] = False
            results["errors"].append(f"x shape {x.shape} invalid")
        
        # Check y shape
        if y.dim() not in [1, 2]:
            results["valid"] = False
            results["errors"].append(f"y must be 1D or 2D, got {y.dim()}D")
        
        # Check no NaN/Inf
        if not torch.isfinite(x).all():
            results["valid"] = False
            results["errors"].append("x contains NaN/Inf")
        
        # Check labels are binary
        unique = set(y.cpu().numpy().flatten().tolist())
        if not unique.issubset({0.0, 1.0}):
            results["valid"] = False
            results["errors"].append(f"y has non-binary values: {unique}")
        
        return results


BATCH_SCHEMA = BatchSchema()
