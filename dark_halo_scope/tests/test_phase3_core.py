"""
Unit tests for Phase 3 core functions.

Tests the pure functions extracted from spark_phase3_define_fields_and_build_parent.py
to identify bugs in region selection, brick mapping, and LRG extraction.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from emr.phase3_core import (
    LRG_VARIANTS,
    BASELINE_VARIANT,
    LRGVariant,
    get_variant_by_name,
    parse_s3_uri,
    nanomaggies_to_mag,
    get_col,
    compute_lrg_flags,
    compute_region_scores,
    select_top_regions,
    build_brick_to_modes,
)


# ============================================================================
# Test parse_s3_uri
# ============================================================================

class TestParseS3Uri:
    """Tests for S3 URI parsing."""
    
    def test_basic_uri(self):
        """Test parsing a basic S3 URI."""
        bucket, key = parse_s3_uri("s3://mybucket/path/to/file.csv")
        assert bucket == "mybucket"
        assert key == "path/to/file.csv"
    
    def test_root_key(self):
        """Test parsing an S3 URI with a simple key."""
        bucket, key = parse_s3_uri("s3://bucket/key")
        assert bucket == "bucket"
        assert key == "key"
    
    def test_nested_path(self):
        """Test parsing an S3 URI with deeply nested path."""
        bucket, key = parse_s3_uri("s3://darkhaloscope/phase2_analysis/v3_color_relaxed/summary.csv")
        assert bucket == "darkhaloscope"
        assert key == "phase2_analysis/v3_color_relaxed/summary.csv"
    
    def test_trailing_slash(self):
        """Test parsing an S3 URI with trailing slash."""
        bucket, key = parse_s3_uri("s3://bucket/prefix/")
        assert bucket == "bucket"
        assert key == "prefix/"
    
    def test_invalid_uri_raises(self):
        """Test that non-S3 URIs raise ValueError."""
        with pytest.raises(ValueError, match="Not an s3:// URI"):
            parse_s3_uri("https://bucket/key")
    
    def test_http_uri_raises(self):
        """Test that HTTP URIs raise ValueError."""
        with pytest.raises(ValueError, match="Not an s3:// URI"):
            parse_s3_uri("http://example.com/file")


# ============================================================================
# Test nanomaggies_to_mag
# ============================================================================

class TestNanomaggiesToMag:
    """Tests for flux to magnitude conversion."""
    
    def test_unity_flux(self):
        """Test that 1 nanomaggie = 22.5 mag."""
        result = nanomaggies_to_mag(np.array([1.0]))
        np.testing.assert_allclose(result, [22.5])
    
    def test_ten_nanomaggies(self):
        """Test 10 nanomaggies = 20 mag."""
        result = nanomaggies_to_mag(np.array([10.0]))
        np.testing.assert_allclose(result, [20.0])
    
    def test_hundred_nanomaggies(self):
        """Test 100 nanomaggies = 17.5 mag."""
        result = nanomaggies_to_mag(np.array([100.0]))
        np.testing.assert_allclose(result, [17.5])
    
    def test_zero_flux_is_nan(self):
        """Test that zero flux returns NaN."""
        result = nanomaggies_to_mag(np.array([0.0]))
        assert np.isnan(result[0])
    
    def test_negative_flux_is_nan(self):
        """Test that negative flux returns NaN."""
        result = nanomaggies_to_mag(np.array([-1.0]))
        assert np.isnan(result[0])
    
    def test_nan_flux_is_nan(self):
        """Test that NaN flux returns NaN."""
        result = nanomaggies_to_mag(np.array([np.nan]))
        assert np.isnan(result[0])
    
    def test_mixed_array(self):
        """Test array with valid and invalid values."""
        result = nanomaggies_to_mag(np.array([1.0, 0.0, -1.0, 10.0, np.nan]))
        assert result[0] == pytest.approx(22.5)
        assert np.isnan(result[1])
        assert np.isnan(result[2])
        assert result[3] == pytest.approx(20.0)
        assert np.isnan(result[4])
    
    def test_custom_zeropoint(self):
        """Test with a custom zero point."""
        result = nanomaggies_to_mag(np.array([1.0]), zero_point=25.0)
        np.testing.assert_allclose(result, [25.0])


# ============================================================================
# Test get_col (case-insensitive FITS column lookup)
# ============================================================================

class TestGetCol:
    """Tests for case-insensitive FITS column extraction."""
    
    @pytest.fixture
    def mock_fits_data(self):
        """Create mock FITS-like data."""
        dtype = [
            ('RA', 'f8'),
            ('DEC', 'f8'),
            ('BRICKNAME', 'U20'),
            ('TYPE', 'U5'),
            ('FLUX_R', 'f8'),
        ]
        data = np.array([
            (180.0, -30.0, "0001m300", "DEV", 100.0),
            (181.0, -31.0, "0002m310", "PSF", 50.0),
        ], dtype=dtype)
        return data
    
    def test_exact_match(self, mock_fits_data):
        """Test finding column with exact case match."""
        names = list(mock_fits_data.dtype.names)
        result = get_col(mock_fits_data, names, ["RA"])
        np.testing.assert_array_equal(result, [180.0, 181.0])
    
    def test_lowercase_desired(self, mock_fits_data):
        """Test finding uppercase column with lowercase desired."""
        names = list(mock_fits_data.dtype.names)
        result = get_col(mock_fits_data, names, ["brickname"])
        np.testing.assert_array_equal(result, ["0001m300", "0002m310"])
    
    def test_mixed_case(self, mock_fits_data):
        """Test case-insensitive matching."""
        names = list(mock_fits_data.dtype.names)
        result = get_col(mock_fits_data, names, ["BrIcKnAmE"])
        np.testing.assert_array_equal(result, ["0001m300", "0002m310"])
    
    def test_first_match_priority(self, mock_fits_data):
        """Test that first matching desired name is used."""
        names = list(mock_fits_data.dtype.names)
        # Should find 'brickname' first
        result = get_col(mock_fits_data, names, ["brickname", "RA"])
        assert result[0] == "0001m300"  # Not 180.0
    
    def test_missing_column_raises(self, mock_fits_data):
        """Test that missing column raises KeyError."""
        names = list(mock_fits_data.dtype.names)
        with pytest.raises(KeyError, match="Missing FITS column"):
            get_col(mock_fits_data, names, ["NONEXISTENT", "ALSOMISSING"])


# ============================================================================
# Test LRG Variant Definitions
# ============================================================================

class TestLRGVariants:
    """Tests for LRG variant definitions."""
    
    def test_all_variants_defined(self):
        """Test that all 5 variants are defined."""
        assert len(LRG_VARIANTS) == 5
    
    def test_variant_names(self):
        """Test variant names match expected."""
        names = [v.name for v in LRG_VARIANTS]
        assert "v1_pure_massive" in names
        assert "v2_baseline_dr10" in names
        assert "v3_color_relaxed" in names
        assert "v4_mag_relaxed" in names
        assert "v5_very_relaxed" in names
    
    def test_baseline_variant(self):
        """Test baseline variant is v3_color_relaxed."""
        assert BASELINE_VARIANT == "v3_color_relaxed"
    
    def test_v3_thresholds(self):
        """Test v3 (color_relaxed) has correct thresholds."""
        v3 = get_variant_by_name("v3_color_relaxed")
        assert v3 is not None
        assert v3.z_mag_max == 20.4
        assert v3.rz_min == 0.4
        assert v3.zw1_min == 0.8
    
    def test_v3_thresholds_match_phase2(self):
        """
        CRITICAL: Verify v3 thresholds match what Phase 2 uses.
        
        Phase 2 uses: z < 20.4, r-z > 0.4, z-W1 > 0.8
        """
        v3 = get_variant_by_name("v3_color_relaxed")
        # These MUST match Phase 2 exactly
        assert v3.z_mag_max == 20.4, "z_mag_max mismatch with Phase 2!"
        assert v3.rz_min == 0.4, "rz_min mismatch with Phase 2!"
        assert v3.zw1_min == 0.8, "zw1_min mismatch with Phase 2!"


# ============================================================================
# Test compute_lrg_flags
# ============================================================================

class TestComputeLRGFlags:
    """Tests for LRG selection flag computation."""
    
    def test_typical_lrg(self):
        """Test that a typical LRG passes v3 cut."""
        # Typical LRG: z~19.5, r-z~0.6, z-W1~1.0
        mag_z = np.array([19.5])
        r_minus_z = np.array([0.6])
        z_minus_w1 = np.array([1.0])
        
        flags = compute_lrg_flags(mag_z, r_minus_z, z_minus_w1)
        
        assert flags["v3_color_relaxed"][0] == True
    
    def test_faint_object_fails_v3(self):
        """Test that object fainter than z=20.4 fails v3 cut."""
        mag_z = np.array([20.5])  # Too faint
        r_minus_z = np.array([0.6])
        z_minus_w1 = np.array([1.0])
        
        flags = compute_lrg_flags(mag_z, r_minus_z, z_minus_w1)
        
        assert flags["v3_color_relaxed"][0] == False
    
    def test_blue_object_fails_v3(self):
        """Test that blue object (low r-z) fails v3 cut."""
        mag_z = np.array([19.5])
        r_minus_z = np.array([0.3])  # Too blue (< 0.4)
        z_minus_w1 = np.array([1.0])
        
        flags = compute_lrg_flags(mag_z, r_minus_z, z_minus_w1)
        
        assert flags["v3_color_relaxed"][0] == False
    
    def test_low_zw1_fails_v3(self):
        """Test that object with low z-W1 fails v3 cut."""
        mag_z = np.array([19.5])
        r_minus_z = np.array([0.6])
        z_minus_w1 = np.array([0.7])  # Too low (< 0.8)
        
        flags = compute_lrg_flags(mag_z, r_minus_z, z_minus_w1)
        
        assert flags["v3_color_relaxed"][0] == False
    
    def test_edge_case_exact_thresholds(self):
        """Test behavior at exact threshold values."""
        # Exactly at z=20.4, r-z=0.4, z-W1=0.8
        # Note: cuts are z < 20.4, r-z > 0.4, z-W1 > 0.8 (strict inequalities)
        mag_z = np.array([20.4])
        r_minus_z = np.array([0.4])
        z_minus_w1 = np.array([0.8])
        
        flags = compute_lrg_flags(mag_z, r_minus_z, z_minus_w1)
        
        # All are strict inequalities, so this should FAIL
        assert flags["v3_color_relaxed"][0] == False
    
    def test_just_passing_v3(self):
        """Test object just barely passing v3 cut."""
        mag_z = np.array([20.39])  # Just under 20.4
        r_minus_z = np.array([0.41])  # Just over 0.4
        z_minus_w1 = np.array([0.81])  # Just over 0.8
        
        flags = compute_lrg_flags(mag_z, r_minus_z, z_minus_w1)
        
        assert flags["v3_color_relaxed"][0] == True
    
    def test_all_variants_computed(self):
        """Test that all 5 variants are computed."""
        mag_z = np.array([19.0])
        r_minus_z = np.array([0.6])
        z_minus_w1 = np.array([2.0])  # Very red
        
        flags = compute_lrg_flags(mag_z, r_minus_z, z_minus_w1)
        
        assert len(flags) == 5
        # This very red object should pass all variants
        for name, mask in flags.items():
            assert mask[0] == True, f"{name} should pass"


# ============================================================================
# Test Brick Matching (String Comparison)
# ============================================================================

class TestBrickMatching:
    """Tests for brickname string matching between FITS and Phase 2 CSVs."""
    
    def test_exact_match(self):
        """Test that identical bricknames match."""
        fits_bricknames = np.array(["0001m442", "0003p010", "0004m335"])
        phase2_bricks = {"0001m442", "0003p010"}
        
        in_sel = np.isin(fits_bricknames, list(phase2_bricks))
        
        assert in_sel[0] == True
        assert in_sel[1] == True
        assert in_sel[2] == False
    
    def test_case_sensitivity(self):
        """
        Test case sensitivity in brickname matching.
        
        FITS files typically have lowercase bricknames.
        Phase 2 CSVs should also have lowercase.
        """
        # Simulate FITS bricknames (typically lowercase)
        fits_bricknames = np.array(["0001m442", "0003p010"])
        
        # If Phase 2 had uppercase (BUG scenario)
        phase2_uppercase = {"0001M442", "0003P010"}
        
        # This should NOT match (case sensitive)
        in_sel = np.isin(fits_bricknames, list(phase2_uppercase))
        
        # Both should be False because of case mismatch
        assert in_sel[0] == False
        assert in_sel[1] == False
    
    def test_numpy_string_conversion(self):
        """
        Test numpy string conversion behavior.
        
        This tests a potential bug where astype(str) might add whitespace.
        """
        # Simulate FITS data with bytes
        fits_bytes = np.array([b"0001m442", b"0003p010"])
        fits_str = fits_bytes.astype(str)
        
        # Should match exactly
        assert fits_str[0] == "0001m442"
        assert fits_str[1] == "0003p010"
        
        # No trailing whitespace
        assert fits_str[0] == fits_str[0].strip()


# ============================================================================
# Test Type Filter
# ============================================================================

class TestTypeFilter:
    """Tests for TYPE != 'PSF' filtering."""
    
    def test_psf_filtered_out(self):
        """Test that PSF type is filtered out."""
        types = np.array(["DEV", "PSF", "EXP", "PSF", "REX"])
        
        non_psf = types != "PSF"
        
        assert non_psf[0] == True   # DEV passes
        assert non_psf[1] == False  # PSF filtered
        assert non_psf[2] == True   # EXP passes
        assert non_psf[3] == False  # PSF filtered
        assert non_psf[4] == True   # REX passes
    
    def test_lowercase_psf_not_filtered(self):
        """Test that lowercase 'psf' is NOT filtered (case sensitive)."""
        types = np.array(["psf", "PSF"])
        
        non_psf = types != "PSF"
        
        assert non_psf[0] == True   # lowercase 'psf' passes
        assert non_psf[1] == False  # uppercase 'PSF' filtered


# ============================================================================
# Test Region Quality Cuts
# ============================================================================

class TestRegionQualityCuts:
    """Tests for region quality cut filtering."""
    
    @pytest.fixture
    def sample_regions_df(self):
        """Create a sample regions DataFrame."""
        return pd.DataFrame({
            "region_id": [1, 2, 3, 4, 5],
            "median_ebv": [0.05, 0.15, 0.10, 0.11, 0.13],
            "median_psf_r_arcsec": [1.3, 1.5, 1.8, 1.55, 1.4],
            "median_psfdepth_r": [24.0, 24.5, 23.0, 23.8, 24.2],
            "total_area_deg2": [1.0, 2.0, 1.5, 0.5, 3.0],
            "mean_lrg_density_v3_color_relaxed": [100, 200, 150, 80, 250],
            "total_n_lrg_v3_color_relaxed": [100, 400, 225, 40, 750],
        })
    
    def test_default_quality_cuts(self, sample_regions_df):
        """Test default quality cut parameters."""
        max_ebv = 0.12
        max_psf = 1.60
        min_depth = 23.6
        
        df = sample_regions_df
        filtered = df[
            (df["median_ebv"] <= max_ebv) &
            (df["median_psf_r_arcsec"] <= max_psf) &
            (df["median_psfdepth_r"] >= min_depth)
        ]
        
        # Region 1: ebv=0.05 ✓, psf=1.3 ✓, depth=24.0 ✓ -> PASS
        # Region 2: ebv=0.15 ✗ -> FAIL
        # Region 3: ebv=0.10 ✓, psf=1.8 ✗ -> FAIL
        # Region 4: ebv=0.11 ✓, psf=1.55 ✓, depth=23.8 ✓ -> PASS
        # Region 5: ebv=0.13 ✗ -> FAIL
        
        assert len(filtered) == 2
        assert 1 in filtered["region_id"].values
        assert 4 in filtered["region_id"].values
    
    def test_all_regions_filtered_out(self, sample_regions_df):
        """Test scenario where all regions are filtered out (BUG scenario)."""
        # Very strict cuts that filter everything
        max_ebv = 0.01  # Very strict
        max_psf = 1.0   # Very strict
        min_depth = 25.0  # Very strict
        
        df = sample_regions_df
        filtered = df[
            (df["median_ebv"] <= max_ebv) &
            (df["median_psf_r_arcsec"] <= max_psf) &
            (df["median_psfdepth_r"] >= min_depth)
        ]
        
        # All should be filtered out
        assert len(filtered) == 0


# ============================================================================
# Test build_brick_to_modes
# ============================================================================

class TestBuildBrickToModes:
    """Tests for building the brick-to-modes mapping."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample Phase 2 data."""
        regions_summary = pd.DataFrame({
            "region_id": [1, 2, 3],
            "median_ebv": [0.05, 0.06, 0.07],
            "median_psf_r_arcsec": [1.3, 1.4, 1.5],
            "median_psfdepth_r": [24.0, 24.1, 24.2],
            "total_area_deg2": [1.0, 2.0, 1.5],
            "mean_lrg_density_v3_color_relaxed": [100, 200, 150],
            "total_n_lrg_v3_color_relaxed": [100, 400, 225],
        })
        
        regions_bricks = pd.DataFrame({
            "region_id": [1, 1, 2, 2, 2, 3],
            "brickname": ["brick_a", "brick_b", "brick_c", "brick_d", "brick_e", "brick_f"],
        })
        
        return regions_summary, regions_bricks
    
    def test_basic_mapping(self, sample_data):
        """Test basic brick-to-modes mapping."""
        regions_summary, regions_bricks = sample_data
        
        brick_to_modes = build_brick_to_modes(
            regions_summary,
            regions_bricks,
            variant="v3_color_relaxed",
            ranking_modes=["density"],
            num_regions=2,
            max_ebv=0.12,
            max_psf_r_arcsec=1.60,
            min_psfdepth_r=23.6,
        )
        
        # Top 2 by density: region 2 (200), region 3 (150)
        # Bricks for region 2: brick_c, brick_d, brick_e
        # Bricks for region 3: brick_f
        
        assert "brick_c" in brick_to_modes
        assert "brick_d" in brick_to_modes
        assert "brick_e" in brick_to_modes
        assert "brick_f" in brick_to_modes
        
        # Region 1's bricks should NOT be included
        assert "brick_a" not in brick_to_modes
        assert "brick_b" not in brick_to_modes
    
    def test_empty_after_quality_cuts(self, sample_data):
        """Test that empty dict is returned when all regions filtered."""
        regions_summary, regions_bricks = sample_data
        
        brick_to_modes = build_brick_to_modes(
            regions_summary,
            regions_bricks,
            variant="v3_color_relaxed",
            ranking_modes=["density"],
            num_regions=2,
            max_ebv=0.01,  # Very strict - filters all
            max_psf_r_arcsec=1.60,
            min_psfdepth_r=23.6,
        )
        
        # Should be empty since all regions are filtered
        assert len(brick_to_modes) == 0
    
    def test_multiple_modes(self, sample_data):
        """Test that multiple modes create correct mappings."""
        regions_summary, regions_bricks = sample_data
        
        brick_to_modes = build_brick_to_modes(
            regions_summary,
            regions_bricks,
            variant="v3_color_relaxed",
            ranking_modes=["density", "n_lrg"],
            num_regions=2,
            max_ebv=0.12,
            max_psf_r_arcsec=1.60,
            min_psfdepth_r=23.6,
        )
        
        # Each brick can have multiple mode entries
        # brick_c (from region 2) should appear in both density and n_lrg
        assert len(brick_to_modes.get("brick_c", [])) == 2


# ============================================================================
# Integration with Real Phase 2 Data
# ============================================================================

class TestWithRealPhase2Data:
    """Integration tests using real Phase 2 output files."""
    
    def test_load_regions_summary(self, phase2_regions_summary_csv):
        """Test loading real Phase 2 regions summary."""
        if not phase2_regions_summary_csv.exists():
            pytest.skip("Phase 2 regions summary not found")
        
        df = pd.read_csv(phase2_regions_summary_csv)
        
        # Check expected columns exist
        assert "region_id" in df.columns
        assert "median_ebv" in df.columns
        assert "median_psf_r_arcsec" in df.columns
        assert "median_psfdepth_r" in df.columns
        assert "mean_lrg_density_v3_color_relaxed" in df.columns
        assert "total_n_lrg_v3_color_relaxed" in df.columns
        
        # Check we have data
        assert len(df) > 0
    
    def test_load_regions_bricks(self, phase2_regions_bricks_csv):
        """Test loading real Phase 2 regions bricks mapping."""
        if not phase2_regions_bricks_csv.exists():
            pytest.skip("Phase 2 regions bricks not found")
        
        df = pd.read_csv(phase2_regions_bricks_csv)
        
        # Check expected columns exist
        assert "region_id" in df.columns
        assert "brickname" in df.columns
        
        # Check we have data
        assert len(df) > 0
    
    def test_quality_cuts_on_real_data(self, phase2_regions_summary_csv):
        """Test that quality cuts don't filter out ALL regions."""
        if not phase2_regions_summary_csv.exists():
            pytest.skip("Phase 2 regions summary not found")
        
        df = pd.read_csv(phase2_regions_summary_csv)
        
        # Apply default quality cuts
        filtered = df[
            (df["median_ebv"] <= 0.12) &
            (df["median_psf_r_arcsec"] <= 1.60) &
            (df["median_psfdepth_r"] >= 23.6)
        ]
        
        n_before = len(df)
        n_after = len(filtered)
        
        print(f"\nQuality cuts: {n_before} -> {n_after} regions")
        print(f"Percentage passing: {100 * n_after / n_before:.1f}%")
        
        # CRITICAL: At least some regions should pass
        assert n_after > 0, "All regions filtered out by quality cuts!"
        
        # At least 10 regions should pass for a reasonable sample
        assert n_after >= 10, f"Only {n_after} regions passed quality cuts"
    
    def test_build_brick_to_modes_real_data(
        self, 
        phase2_regions_summary_csv, 
        phase2_regions_bricks_csv
    ):
        """Test building brick_to_modes with real Phase 2 data."""
        if not phase2_regions_summary_csv.exists():
            pytest.skip("Phase 2 regions summary not found")
        if not phase2_regions_bricks_csv.exists():
            pytest.skip("Phase 2 regions bricks not found")
        
        regions_summary = pd.read_csv(phase2_regions_summary_csv)
        regions_bricks = pd.read_csv(phase2_regions_bricks_csv)
        
        brick_to_modes = build_brick_to_modes(
            regions_summary,
            regions_bricks,
            variant="v3_color_relaxed",
            ranking_modes=["n_lrg", "area_weighted", "psf_weighted"],
            num_regions=30,
            max_ebv=0.12,
            max_psf_r_arcsec=1.60,
            min_psfdepth_r=23.6,
        )
        
        print(f"\nbrick_to_modes has {len(brick_to_modes)} unique bricks")
        
        # CRITICAL: brick_to_modes should NOT be empty
        assert len(brick_to_modes) > 0, "brick_to_modes is empty!"
        
        # Print some sample bricks for debugging
        sample_bricks = list(brick_to_modes.keys())[:5]
        print(f"Sample bricks: {sample_bricks}")
        
        # Each brick should have at least one mode entry
        for brick, modes in list(brick_to_modes.items())[:5]:
            print(f"  {brick}: {modes}")
            assert len(modes) > 0

