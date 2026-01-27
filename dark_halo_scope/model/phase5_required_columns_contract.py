"""Phase 5 required-columns contract for Dark Halo Scope.

Purpose
- Prevent Phase 5 data loading/scoring code from accidentally dropping Stage 4c columns
  required for Phase 5 completeness surfaces (parallel to Phase 4d).

If Stage 4c schema changes, regenerate and bump CONTRACT_VERSION.
"""

from __future__ import annotations
from typing import Iterable, List, Sequence, Set, Any

CONTRACT_NAME = 'dark_halo_scope_phase5_required_columns'
CONTRACT_VERSION = '1.0.0'
SOURCE_PIPELINE_FILE = 'spark_phase4_pipeline.py'

STAGE4C_METRICS_SCHEMA_COLUMNS: List[str] = ['task_id', 'experiment_id', 'selection_set_id', 'ranking_mode', 'selection_strategy', 'region_id', 'region_split', 'brickname', 'ra', 'dec', 'stamp_size', 'bandset', 'config_id', 'theta_e_arcsec', 'src_dmag', 'src_reff_arcsec', 'src_e', 'shear', 'replicate', 'is_control', 'task_seed64', 'src_x_arcsec', 'src_y_arcsec', 'src_phi_rad', 'shear_phi_rad', 'src_gr', 'src_rz', 'psf_bin', 'depth_bin', 'ab_zp_nmgy', 'pipeline_version', 'psfsize_r', 'psfdepth_r', 'ebv', 'stamp_npz', 'cutout_ok', 'arc_snr', 'total_injected_flux_r', 'bad_pixel_frac', 'wise_brightmask_frac', 'metrics_ok', 'psf_fwhm_used_g', 'psf_fwhm_used_r', 'psf_fwhm_used_z', 'psf_source_g', 'psf_source_r', 'psf_source_z', 'metrics_only', 'lens_model', 'lens_e', 'lens_phi_rad', 'magnification', 'tangential_stretch', 'radial_stretch', 'expected_arc_radius', 'physics_valid', 'physics_warnings']
PHASE5_REQUIRED_COLUMNS: List[str] = ['task_id', 'experiment_id', 'selection_set_id', 'ranking_mode', 'selection_strategy', 'region_id', 'region_split', 'brickname', 'ra', 'dec', 'pipeline_version', 'config_id', 'stamp_npz', 'stamp_size', 'bandset', 'ab_zp_nmgy', 'lens_model', 'is_control', 'theta_e_arcsec', 'src_dmag', 'src_reff_arcsec', 'psf_fwhm_used_r', 'psfsize_r', 'psfdepth_r', 'bad_pixel_frac', 'wise_brightmask_frac', 'cutout_ok']
PHASE5_OPTIONAL_COLUMNS: List[str] = ['arc_snr', 'total_injected_flux_r', 'magnification', 'psf_fwhm_used_g', 'psf_fwhm_used_z', 'psf_source_g', 'psf_source_r', 'psf_source_z', 'psf_bin', 'depth_bin', 'metrics_ok', 'metrics_only', 'src_e', 'shear', 'replicate', 'task_seed64', 'src_x_arcsec', 'src_y_arcsec', 'src_gr', 'src_rz', 'src_phi_rad', 'shear_phi_rad', 'lens_e', 'lens_phi_rad', 'tangential_stretch', 'radial_stretch', 'expected_arc_radius', 'physics_valid', 'physics_warnings', 'ebv']

def _as_set(cols: Iterable[str]) -> Set[str]:
    return set(cols)

def assert_no_duplicate_columns(columns: Sequence[str]) -> None:
    seen = set()
    dups = []
    for c in columns:
        if c in seen:
            dups.append(c)
        else:
            seen.add(c)
    if dups:
        raise ValueError(f"Duplicate columns detected: {sorted(set(dups))}")

def assert_subset(subset: Sequence[str], superset: Sequence[str], *, name: str) -> None:
    missing = sorted(_as_set(subset) - _as_set(superset))
    if missing:
        raise ValueError(f"{name}: missing columns: {missing}")

def assert_phase5_required_columns_present(df, *, strict_stage4c: bool = False) -> None:
    """Assert Phase 5 required columns exist in a Spark DataFrame."""
    cols = list(df.columns)
    assert_no_duplicate_columns(cols)
    assert_subset(PHASE5_REQUIRED_COLUMNS, cols, name="PHASE5_REQUIRED_COLUMNS")
    if strict_stage4c:
        extra = sorted(_as_set(cols) - _as_set(STAGE4C_METRICS_SCHEMA_COLUMNS))
        if extra:
            raise ValueError(
                "strict_stage4c=True but df contains columns not in Stage 4c schema: " + str(extra)
            )

def assert_columns_present_in_parquet(spark, parquet_path: str, *, strict_stage4c: bool = False) -> None:
    df = spark.read.parquet(parquet_path)
    assert_phase5_required_columns_present(df, strict_stage4c=strict_stage4c)

def select_phase5_columns(df, *, keep_optional: bool = True) -> Any:
    required = [c for c in PHASE5_REQUIRED_COLUMNS if c in df.columns]
    optional = [c for c in PHASE5_OPTIONAL_COLUMNS if (keep_optional and c in df.columns)]
    return df.select(*(required + optional))
