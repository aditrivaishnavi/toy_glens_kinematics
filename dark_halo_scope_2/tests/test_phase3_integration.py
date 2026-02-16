"""
Integration tests for Phase 3 sweep processing.

These tests use real FITS files and Phase 2 outputs to verify the full
processing pipeline works correctly without Spark.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from emr.phase3_core import (
    LRG_VARIANTS,
    BASELINE_VARIANT,
    parse_s3_uri,
    nanomaggies_to_mag,
    get_col,
    compute_lrg_flags,
    process_sweep_chunk,
    build_brick_to_modes,
)
from astropy.io import fits


# ============================================================================
# FITS File Inspection Tests
# ============================================================================

class TestFITSInspection:
    """Tests that inspect real FITS files to understand their structure."""
    
    def test_fits_column_names(self, sample_sweep_path):
        """Inspect FITS column names to check case and content."""
        with fits.open(sample_sweep_path, memmap=True) as hdul:
            names = list(hdul[1].columns.names)
            
            print(f"\nFITS columns ({len(names)} total):")
            for name in sorted(names):
                print(f"  {name}")
            
            # Check critical columns exist
            lower_names = [n.lower() for n in names]
            assert "ra" in lower_names, "RA column not found!"
            assert "dec" in lower_names, "DEC column not found!"
            assert "brickname" in lower_names, "BRICKNAME column not found!"
            assert "type" in lower_names, "TYPE column not found!"
            assert "flux_r" in lower_names, "FLUX_R column not found!"
            assert "flux_z" in lower_names, "FLUX_Z column not found!"
            assert "flux_w1" in lower_names, "FLUX_W1 column not found!"
    
    def test_fits_brickname_format(self, sample_sweep_path):
        """Inspect FITS brickname format (case, encoding)."""
        with fits.open(sample_sweep_path, memmap=True) as hdul:
            data = hdul[1].data
            names = list(data.columns.names)
            
            # Get bricknames
            brick_col = get_col(data, names, ["brickname"])
            
            # Sample first 10 unique bricknames
            unique_bricks = np.unique(brick_col[:1000])[:10]
            
            print(f"\nSample bricknames from FITS:")
            for b in unique_bricks:
                b_str = str(b).strip()
                print(f"  '{b_str}' (len={len(b_str)}, type={type(b)})")
            
            # Check format
            for b in unique_bricks:
                b_str = str(b).strip()
                # Bricknames should be lowercase like "0001m442"
                assert b_str == b_str.lower() or b_str == b_str.upper(), \
                    f"Mixed case brickname: {b_str}"
    
    def test_fits_type_values(self, sample_sweep_path):
        """Inspect FITS TYPE column values."""
        with fits.open(sample_sweep_path, memmap=True) as hdul:
            data = hdul[1].data
            names = list(data.columns.names)
            
            # Get types
            type_col = get_col(data, names, ["type"])
            
            # Get unique types
            unique_types, counts = np.unique(type_col, return_counts=True)
            
            print(f"\nTYPE values in FITS:")
            for t, c in sorted(zip(unique_types, counts), key=lambda x: -x[1]):
                print(f"  {t}: {c}")
            
            # PSF should exist and be uppercase
            type_strs = [str(t) for t in unique_types]
            assert "PSF" in type_strs or "PSF " in type_strs, \
                f"PSF type not found. Types: {type_strs}"
    
    def test_fits_flux_statistics(self, sample_sweep_path):
        """Check flux statistics to verify data quality."""
        with fits.open(sample_sweep_path, memmap=True) as hdul:
            data = hdul[1].data
            names = list(data.columns.names)
            
            flux_r = np.asarray(get_col(data, names, ["flux_r"]), dtype=np.float64)
            flux_z = np.asarray(get_col(data, names, ["flux_z"]), dtype=np.float64)
            flux_w1 = np.asarray(get_col(data, names, ["flux_w1"]), dtype=np.float64)
            
            # Statistics
            print(f"\nFlux statistics:")
            print(f"  FLUX_R: min={flux_r.min():.2f}, max={flux_r.max():.2f}, "
                  f"median={np.median(flux_r):.2f}, n_positive={np.sum(flux_r > 0)}")
            print(f"  FLUX_Z: min={flux_z.min():.2f}, max={flux_z.max():.2f}, "
                  f"median={np.median(flux_z):.2f}, n_positive={np.sum(flux_z > 0)}")
            print(f"  FLUX_W1: min={flux_w1.min():.2f}, max={flux_w1.max():.2f}, "
                  f"median={np.median(flux_w1):.2f}, n_positive={np.sum(flux_w1 > 0)}")
            
            # There should be many objects with positive flux
            assert np.sum(flux_r > 0) > 1000
            assert np.sum(flux_z > 0) > 1000
            assert np.sum(flux_w1 > 0) > 1000


# ============================================================================
# Brickname Comparison Tests
# ============================================================================

class TestBricknameComparison:
    """Tests comparing bricknames between FITS and Phase 2 CSVs."""
    
    def test_brickname_case_matches(
        self, 
        sample_sweep_path, 
        phase2_regions_bricks_csv
    ):
        """
        CRITICAL: Verify brickname case matches between FITS and Phase 2.
        
        If FITS uses uppercase and Phase 2 uses lowercase (or vice versa),
        the brick matching will fail silently.
        """
        if not phase2_regions_bricks_csv.exists():
            pytest.skip("Phase 2 regions bricks not found")
        
        # Get bricknames from FITS
        with fits.open(sample_sweep_path, memmap=True) as hdul:
            data = hdul[1].data
            names = list(data.columns.names)
            fits_bricks = get_col(data, names, ["brickname"])
            fits_brick_set = set(str(b).strip() for b in np.unique(fits_bricks)[:100])
        
        # Get bricknames from Phase 2
        phase2_bricks = pd.read_csv(phase2_regions_bricks_csv)
        phase2_brick_set = set(phase2_bricks["brickname"].unique()[:100])
        
        print(f"\nSample FITS bricknames: {list(fits_brick_set)[:5]}")
        print(f"Sample Phase 2 bricknames: {list(phase2_brick_set)[:5]}")
        
        # Check if any overlap
        overlap = fits_brick_set & phase2_brick_set
        print(f"Overlap: {len(overlap)} bricks")
        
        # Check case consistency
        fits_sample = list(fits_brick_set)[0] if fits_brick_set else ""
        phase2_sample = list(phase2_brick_set)[0] if phase2_brick_set else ""
        
        fits_is_lower = fits_sample == fits_sample.lower()
        phase2_is_lower = phase2_sample == phase2_sample.lower()
        
        print(f"FITS brickname case: {'lower' if fits_is_lower else 'UPPER'}")
        print(f"Phase 2 brickname case: {'lower' if phase2_is_lower else 'UPPER'}")
        
        assert fits_is_lower == phase2_is_lower, \
            "Case mismatch between FITS and Phase 2 bricknames!"
    
    def test_sweep_contains_target_bricks(
        self, 
        sample_sweep_path, 
        phase2_regions_summary_csv,
        phase2_regions_bricks_csv
    ):
        """
        Check if the sample sweep contains any of our target bricks.
        """
        if not phase2_regions_bricks_csv.exists():
            pytest.skip("Phase 2 data not found")
        
        # Build brick_to_modes
        regions_summary = pd.read_csv(phase2_regions_summary_csv)
        regions_bricks = pd.read_csv(phase2_regions_bricks_csv)
        
        brick_to_modes = build_brick_to_modes(
            regions_summary,
            regions_bricks,
            variant="v3_color_relaxed",
            ranking_modes=["n_lrg", "area_weighted", "psf_weighted"],
            num_regions=100,  # Use more regions to increase coverage
            max_ebv=0.12,
            max_psf_r_arcsec=1.60,
            min_psfdepth_r=23.6,
        )
        
        target_bricks = set(brick_to_modes.keys())
        print(f"\nTarget bricks: {len(target_bricks)}")
        
        # Get bricknames from sweep
        with fits.open(sample_sweep_path, memmap=True) as hdul:
            data = hdul[1].data
            names = list(data.columns.names)
            fits_bricks = get_col(data, names, ["brickname"])
            sweep_bricks = set(str(b).strip() for b in np.unique(fits_bricks))
        
        print(f"Sweep bricks: {len(sweep_bricks)}")
        
        # Check overlap
        overlap = target_bricks & sweep_bricks
        print(f"Overlap: {len(overlap)} bricks")
        
        if len(overlap) > 0:
            print(f"Sample overlapping bricks: {list(overlap)[:5]}")
        else:
            print("WARNING: No overlap between sweep and target bricks!")
            print(f"Sweep name: {sample_sweep_path.name}")
            print(f"Sample target bricks: {list(target_bricks)[:10]}")
            print(f"Sample sweep bricks: {list(sweep_bricks)[:10]}")


# ============================================================================
# Full Processing Pipeline Tests
# ============================================================================

class TestProcessSweepChunk:
    """Tests for the full sweep processing pipeline."""
    
    def test_process_with_synthetic_brick_mapping(self, sample_sweep_path):
        """Test processing with a synthetic brick mapping that guarantees matches."""
        # First, get some real bricknames from the sweep
        with fits.open(sample_sweep_path, memmap=True) as hdul:
            data = hdul[1].data
            names = list(data.columns.names)
            fits_bricks = get_col(data, names, ["brickname"])
            unique_bricks = list(set(str(b).strip() for b in np.unique(fits_bricks)[:10]))
        
        print(f"\nUsing bricks from sweep: {unique_bricks}")
        
        # Create a synthetic brick_to_modes that WILL match
        brick_to_modes: Dict[str, List[Tuple[str, int, int]]] = {}
        for i, brick in enumerate(unique_bricks):
            brick_to_modes[brick] = [("test_mode", 999, i + 1)]
        
        # Process the sweep
        rows = list(process_sweep_chunk(
            str(sample_sweep_path),
            brick_to_modes,
            chunk_size=100000,
        ))
        
        print(f"Processed {len(rows)} LRG rows")
        
        # We should get SOME rows since we're matching real bricks
        assert len(rows) > 0, "No LRGs found even with matching bricks!"
        
        # Verify row structure
        if rows:
            row = rows[0]
            assert "RA" in row
            assert "DEC" in row
            assert "BRICKNAME" in row
            assert "is_lrg_v3" in row
            assert "phase3_ranking_mode" in row
            assert row["phase3_ranking_mode"] == "test_mode"
            
            # Verify LRG flags are correct
            assert row["is_lrg_v3"] == True, "LRG should pass v3 cut"
    
    def test_process_with_real_phase2_mapping(
        self,
        full_sweep_path,
        phase2_regions_summary_csv,
        phase2_regions_bricks_csv,
    ):
        """
        INTEGRATION TEST: Process a real sweep with real Phase 2 data.
        
        This should produce LRGs if everything is working correctly.
        The deprecated script found 560 LRGs in sweep-000m035-005m030.fits.
        """
        if not phase2_regions_summary_csv.exists():
            pytest.skip("Phase 2 data not found")
        if not full_sweep_path.exists():
            pytest.skip("Full sweep file not found")
        
        # Build brick_to_modes from real Phase 2 data
        regions_summary = pd.read_csv(phase2_regions_summary_csv)
        regions_bricks = pd.read_csv(phase2_regions_bricks_csv)
        
        print(f"\n=== Integration Test ===")
        print(f"Sweep: {full_sweep_path.name}")
        print(f"Regions summary: {len(regions_summary)} regions")
        print(f"Regions bricks: {len(regions_bricks)} brick mappings")
        
        # Apply same quality cuts as EMR job
        print(f"\nApplying quality cuts...")
        
        brick_to_modes = build_brick_to_modes(
            regions_summary,
            regions_bricks,
            variant="v3_color_relaxed",
            ranking_modes=["n_lrg", "area_weighted", "psf_weighted"],
            num_regions=100,
            max_ebv=0.12,
            max_psf_r_arcsec=1.60,
            min_psfdepth_r=23.6,
        )
        
        print(f"brick_to_modes: {len(brick_to_modes)} unique bricks")
        
        if len(brick_to_modes) == 0:
            pytest.fail("brick_to_modes is empty after quality cuts!")
        
        # Check if sweep contains any target bricks
        with fits.open(full_sweep_path, memmap=True) as hdul:
            data = hdul[1].data
            names = list(data.columns.names)
            fits_bricks = get_col(data, names, ["brickname"])
            sweep_bricks = set(str(b).strip() for b in np.unique(fits_bricks))
        
        target_bricks = set(brick_to_modes.keys())
        overlap = target_bricks & sweep_bricks
        
        print(f"Sweep contains {len(sweep_bricks)} unique bricks")
        print(f"Overlap with target: {len(overlap)} bricks")
        
        if len(overlap) == 0:
            print("\nNo overlap - this sweep doesn't contain target bricks")
            print(f"Sample target bricks: {list(target_bricks)[:5]}")
            print(f"Sample sweep bricks: {list(sweep_bricks)[:5]}")
            pytest.skip("Sweep doesn't contain any target bricks")
        
        # Process the sweep
        print(f"\nProcessing sweep...")
        rows = list(process_sweep_chunk(
            str(full_sweep_path),
            brick_to_modes,
            chunk_size=100000,
        ))
        
        print(f"Found {len(rows)} LRG rows")
        
        # Count by mode
        if rows:
            mode_counts = {}
            for row in rows:
                mode = row["phase3_ranking_mode"]
                mode_counts[mode] = mode_counts.get(mode, 0) + 1
            
            print("LRGs by mode:")
            for mode, count in sorted(mode_counts.items()):
                print(f"  {mode}: {count}")
        
        # This is the key assertion
        assert len(rows) > 0, "No LRGs found in sweep!"
    
    def test_lrg_selection_counts(self, sample_sweep_path):
        """
        Test LRG selection counts to verify thresholds are working.
        """
        # Get all objects from the sweep that pass basic filters
        with fits.open(sample_sweep_path, memmap=True) as hdul:
            data = hdul[1].data
            names = list(data.columns.names)
            
            # Get fluxes
            flux_r = np.asarray(get_col(data, names, ["flux_r"]), dtype=np.float64)
            flux_z = np.asarray(get_col(data, names, ["flux_z"]), dtype=np.float64)
            flux_w1 = np.asarray(get_col(data, names, ["flux_w1"]), dtype=np.float64)
            
            # Get types
            typ = get_col(data, names, ["type"])
            type_arr = np.array(typ).astype(str)
            
            # Filters
            non_psf = type_arr != "PSF"
            good_flux = (flux_r > 0) & (flux_z > 0) & (flux_w1 > 0)
            keep = non_psf & good_flux
            
            print(f"\nTotal objects: {len(flux_r)}")
            print(f"Non-PSF: {np.sum(non_psf)}")
            print(f"Good flux: {np.sum(good_flux)}")
            print(f"Both: {np.sum(keep)}")
            
            # Compute magnitudes for objects that pass
            mag_z = nanomaggies_to_mag(flux_z)
            mag_r = nanomaggies_to_mag(flux_r)
            mag_w1 = nanomaggies_to_mag(flux_w1)
            
            r_minus_z = mag_r - mag_z
            z_minus_w1 = mag_z - mag_w1
            
            # Compute LRG flags
            flags = compute_lrg_flags(mag_z, r_minus_z, z_minus_w1)
            
            # Count LRGs per variant
            print(f"\nLRG counts by variant:")
            for v in LRG_VARIANTS:
                lrg_mask = flags[v.name] & keep
                print(f"  {v.name}: {np.sum(lrg_mask)}")
            
            # v3 should have a reasonable count
            v3_count = np.sum(flags["v3_color_relaxed"] & keep)
            assert v3_count > 0, "No v3 LRGs found in sweep!"


# ============================================================================
# Debug Tests
# ============================================================================

class TestDebugScenarios:
    """Debug tests to investigate specific failure scenarios."""
    
    def test_empty_brick_to_modes_scenario(
        self,
        phase2_regions_summary_csv,
        phase2_regions_bricks_csv,
    ):
        """
        Debug scenario: What if quality cuts filter out ALL regions?
        """
        if not phase2_regions_summary_csv.exists():
            pytest.skip("Phase 2 data not found")
        
        regions_summary = pd.read_csv(phase2_regions_summary_csv)
        
        # Check distribution of quality metrics
        print(f"\n=== Quality Metric Distributions ===")
        print(f"median_ebv: min={regions_summary['median_ebv'].min():.4f}, "
              f"max={regions_summary['median_ebv'].max():.4f}, "
              f"median={regions_summary['median_ebv'].median():.4f}")
        print(f"median_psf_r_arcsec: min={regions_summary['median_psf_r_arcsec'].min():.2f}, "
              f"max={regions_summary['median_psf_r_arcsec'].max():.2f}, "
              f"median={regions_summary['median_psf_r_arcsec'].median():.2f}")
        print(f"median_psfdepth_r: min={regions_summary['median_psfdepth_r'].min():.2f}, "
              f"max={regions_summary['median_psfdepth_r'].max():.2f}, "
              f"median={regions_summary['median_psfdepth_r'].median():.2f}")
        
        # Count regions passing each cut individually
        n_total = len(regions_summary)
        n_pass_ebv = (regions_summary['median_ebv'] <= 0.12).sum()
        n_pass_psf = (regions_summary['median_psf_r_arcsec'] <= 1.60).sum()
        n_pass_depth = (regions_summary['median_psfdepth_r'] >= 23.6).sum()
        
        print(f"\nRegions passing each cut:")
        print(f"  ebv <= 0.12: {n_pass_ebv}/{n_total} ({100*n_pass_ebv/n_total:.1f}%)")
        print(f"  psf <= 1.60: {n_pass_psf}/{n_total} ({100*n_pass_psf/n_total:.1f}%)")
        print(f"  depth >= 23.6: {n_pass_depth}/{n_total} ({100*n_pass_depth/n_total:.1f}%)")
        
        # Count regions passing all cuts
        all_cuts = (
            (regions_summary['median_ebv'] <= 0.12) &
            (regions_summary['median_psf_r_arcsec'] <= 1.60) &
            (regions_summary['median_psfdepth_r'] >= 23.6)
        )
        n_pass_all = all_cuts.sum()
        print(f"  ALL cuts: {n_pass_all}/{n_total} ({100*n_pass_all/n_total:.1f}%)")
        
        assert n_pass_all > 0, "All regions filtered out by quality cuts!"
    
    def test_sweep_brick_coverage(
        self,
        full_sweep_path,
        phase2_regions_bricks_csv,
    ):
        """
        Debug: Check which sweep files contain which bricks.
        """
        if not full_sweep_path.exists():
            pytest.skip("Full sweep not found")
        if not phase2_regions_bricks_csv.exists():
            pytest.skip("Phase 2 bricks not found")
        
        # Get bricks from sweep
        with fits.open(full_sweep_path, memmap=True) as hdul:
            data = hdul[1].data
            names = list(data.columns.names)
            fits_bricks = get_col(data, names, ["brickname"])
            sweep_bricks = set(str(b).strip() for b in np.unique(fits_bricks))
        
        # Get Phase 2 bricks
        phase2_bricks = pd.read_csv(phase2_regions_bricks_csv)
        all_phase2_bricks = set(phase2_bricks["brickname"].unique())
        
        print(f"\n=== Brick Coverage Analysis ===")
        print(f"Sweep: {full_sweep_path.name}")
        print(f"Sweep bricks: {len(sweep_bricks)}")
        print(f"Phase 2 bricks: {len(all_phase2_bricks)}")
        
        overlap = sweep_bricks & all_phase2_bricks
        print(f"Overlap: {len(overlap)}")
        
        if overlap:
            print(f"Sample overlapping bricks: {list(overlap)[:10]}")
            
            # For overlapping bricks, check which regions they belong to
            for brick in list(overlap)[:3]:
                regions = phase2_bricks[phase2_bricks["brickname"] == brick]["region_id"].tolist()
                print(f"  {brick} -> regions {regions}")

