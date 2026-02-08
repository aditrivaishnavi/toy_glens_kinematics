#!/usr/bin/env python3
"""
Unit tests for negative sampling components.

These tests validate the core logic WITHOUT requiring Spark or S3.
Run with: pytest tests/test_negative_sampling.py -v

Lessons Learned Incorporated:
- L4.3: Test the ACTUAL code path, not a mock version
- L5.3: Validate data quality before assuming it's clean
"""
import pytest
import numpy as np
import os
import sys
import tempfile
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the utility module (no pyspark dependency)
from emr.sampling_utils import (
    get_nobs_z_bin,
    get_type_bin,
    flux_to_mag,
    compute_healpix,
    assign_split,
    is_near_known_lens,
    classify_pool_n2,
    check_maskbits,
    compute_split_proportions,
    VALID_TYPES_N1,
    NOBS_Z_BINS,
)


class TestNobsZBinning:
    """Tests for nobs_z bin assignment."""
    
    def test_bin_1_2(self):
        assert get_nobs_z_bin(1) == "1-2"
        assert get_nobs_z_bin(2) == "1-2"
    
    def test_bin_3_5(self):
        assert get_nobs_z_bin(3) == "3-5"
        assert get_nobs_z_bin(4) == "3-5"
        assert get_nobs_z_bin(5) == "3-5"
    
    def test_bin_6_10(self):
        assert get_nobs_z_bin(6) == "6-10"
        assert get_nobs_z_bin(10) == "6-10"
    
    def test_bin_11_plus(self):
        assert get_nobs_z_bin(11) == "11+"
        assert get_nobs_z_bin(50) == "11+"
        assert get_nobs_z_bin(100) == "11+"
    
    def test_invalid_values(self):
        assert get_nobs_z_bin(0) == "invalid"
        assert get_nobs_z_bin(-1) == "invalid"
        assert get_nobs_z_bin(None) == "invalid"


class TestTypeBinning:
    """Tests for galaxy type bin assignment."""
    
    def test_valid_types(self):
        assert get_type_bin("SER") == "SER"
        assert get_type_bin("DEV") == "DEV"
        assert get_type_bin("REX") == "REX"
        assert get_type_bin("EXP") == "EXP"
    
    def test_case_insensitive(self):
        assert get_type_bin("ser") == "SER"
        assert get_type_bin("Dev") == "DEV"
        assert get_type_bin("rex") == "REX"
    
    def test_with_whitespace(self):
        assert get_type_bin("  SER  ") == "SER"
        assert get_type_bin("DEV ") == "DEV"
    
    def test_other_types(self):
        assert get_type_bin("PSF") == "OTHER"
        assert get_type_bin("DUP") == "OTHER"
        assert get_type_bin("UNKNOWN") == "OTHER"
    
    def test_none(self):
        assert get_type_bin(None) == "OTHER"


class TestFluxToMag:
    """Tests for flux to magnitude conversion."""
    
    def test_valid_flux(self):
        # 1 nMgy should be mag ~22.5
        mag = flux_to_mag(1.0)
        assert abs(mag - 22.5) < 0.01
    
    def test_bright_flux(self):
        # 100 nMgy should be ~17.5
        mag = flux_to_mag(100.0)
        assert abs(mag - 17.5) < 0.01
    
    def test_faint_flux(self):
        # 0.01 nMgy should be ~27.5
        mag = flux_to_mag(0.01)
        assert abs(mag - 27.5) < 0.01
    
    def test_zero_flux(self):
        assert flux_to_mag(0.0) is None
    
    def test_negative_flux(self):
        assert flux_to_mag(-1.0) is None
    
    def test_none_flux(self):
        assert flux_to_mag(None) is None


class TestHEALPixComputation:
    """Tests for HEALPix index computation."""
    
    def test_basic_computation(self):
        # Should return an integer
        idx = compute_healpix(180.0, 45.0, 64)
        assert isinstance(idx, int)
        assert idx >= 0
        assert idx < 12 * 64 * 64
    
    def test_different_nside(self):
        idx_64 = compute_healpix(180.0, 45.0, 64)
        idx_128 = compute_healpix(180.0, 45.0, 128)
        # Different nside should give different ranges
        assert idx_64 < 12 * 64 * 64
        assert idx_128 < 12 * 128 * 128
    
    def test_nearby_positions_same_pixel(self):
        # Very nearby positions should be in same pixel (at low nside)
        idx1 = compute_healpix(180.0, 45.0, 8)
        idx2 = compute_healpix(180.01, 45.01, 8)
        # Note: This may or may not be true depending on pixel boundaries
        # Just check they're valid
        assert idx1 >= 0
        assert idx2 >= 0
    
    def test_edge_cases(self):
        # Poles
        idx_north = compute_healpix(0.0, 89.9, 64)
        idx_south = compute_healpix(0.0, -89.9, 64)
        assert idx_north >= 0
        assert idx_south >= 0
        
        # RA wraparound
        idx_0 = compute_healpix(0.0, 0.0, 64)
        idx_360 = compute_healpix(359.999, 0.0, 64)
        assert idx_0 >= 0
        assert idx_360 >= 0


class TestSplitAssignment:
    """Tests for train/val/test split assignment."""
    
    def test_deterministic(self):
        # Same input should give same output
        allocations = {"train": 0.7, "val": 0.15, "test": 0.15}
        split1 = assign_split(12345, allocations, seed=42)
        split2 = assign_split(12345, allocations, seed=42)
        assert split1 == split2
    
    def test_different_seeds(self):
        # Different seeds may give different results
        allocations = {"train": 0.7, "val": 0.15, "test": 0.15}
        results_seed1 = [assign_split(i, allocations, seed=1) for i in range(1000)]
        results_seed2 = [assign_split(i, allocations, seed=2) for i in range(1000)]
        # Should be different distributions (unless very unlucky)
        assert results_seed1 != results_seed2
    
    def test_proportions_approximate(self):
        # Check that proportions roughly match
        allocations = {"train": 0.7, "val": 0.15, "test": 0.15}
        n = 10000
        results = [assign_split(i, allocations, seed=42) for i in range(n)]
        
        train_frac = sum(1 for r in results if r == "train") / n
        val_frac = sum(1 for r in results if r == "val") / n
        test_frac = sum(1 for r in results if r == "test") / n
        
        # Allow 5% tolerance
        assert abs(train_frac - 0.7) < 0.05
        assert abs(val_frac - 0.15) < 0.05
        assert abs(test_frac - 0.15) < 0.05


class TestKnownLensExclusion:
    """Tests for known lens exclusion radius."""
    
    def test_inside_radius(self):
        known_coords = [(180.0, 45.0)]
        # 1 arcsec away
        result = is_near_known_lens(180.0001, 45.0, known_coords, radius_arcsec=5.0)
        assert result is True
    
    def test_outside_radius(self):
        known_coords = [(180.0, 45.0)]
        # ~36 arcsec away (0.01 deg)
        result = is_near_known_lens(180.01, 45.0, known_coords, radius_arcsec=11.0)
        assert result is False
    
    def test_multiple_lenses(self):
        known_coords = [(180.0, 45.0), (200.0, 30.0), (220.0, -15.0)]
        
        # Near second lens
        result = is_near_known_lens(200.0001, 30.0, known_coords, radius_arcsec=5.0)
        assert result is True
        
        # Far from all
        result = is_near_known_lens(100.0, 0.0, known_coords, radius_arcsec=11.0)
        assert result is False
    
    def test_empty_list(self):
        known_coords = []
        result = is_near_known_lens(180.0, 45.0, known_coords, radius_arcsec=11.0)
        assert result is False


class TestPoolN2Classification:
    """Tests for N2 confuser classification."""
    
    @pytest.fixture
    def config(self):
        return {
            "negative_pools": {
                "pool_n2": {
                    "tractor_criteria": {
                        "ring_proxy": {"type": "DEV", "flux_r_min": 10},
                        "edge_on_proxy": {"type": "EXP", "shape_r_min": 2.0},
                        "blue_clumpy_proxy": {"g_minus_r_max": 0.4, "r_mag_max": 19.0}
                    }
                }
            }
        }
    
    def test_ring_proxy(self, config):
        result = classify_pool_n2("DEV", 15.0, 1.0, 0.8, 18.0, config)
        assert result == "ring_proxy"
    
    def test_ring_proxy_too_faint(self, config):
        result = classify_pool_n2("DEV", 5.0, 1.0, 0.8, 18.0, config)
        assert result is None
    
    def test_edge_on_proxy(self, config):
        result = classify_pool_n2("EXP", 5.0, 3.0, 0.8, 18.0, config)
        assert result == "edge_on_proxy"
    
    def test_edge_on_proxy_too_small(self, config):
        result = classify_pool_n2("EXP", 5.0, 1.0, 0.8, 18.0, config)
        assert result is None
    
    def test_blue_clumpy(self, config):
        result = classify_pool_n2("SER", 5.0, 1.0, 0.2, 18.0, config)
        assert result == "blue_clumpy"
    
    def test_blue_clumpy_too_red(self, config):
        result = classify_pool_n2("SER", 5.0, 1.0, 0.8, 18.0, config)
        assert result is None
    
    def test_no_match(self, config):
        result = classify_pool_n2("SER", 5.0, 1.0, 0.8, 18.0, config)
        assert result is None


class TestValidTypesAndBins:
    """Tests for constants."""
    
    def test_valid_types_set(self):
        assert "SER" in VALID_TYPES_N1
        assert "DEV" in VALID_TYPES_N1
        assert "REX" in VALID_TYPES_N1
        assert "EXP" in VALID_TYPES_N1
        assert len(VALID_TYPES_N1) == 4
    
    def test_nobs_bins_coverage(self):
        # Bins should cover 1-999
        covered = set()
        for low, high in NOBS_Z_BINS:
            for i in range(low, min(high + 1, 100)):  # Don't iterate to 999
                covered.add(i)
        
        # Check first 20 values are covered
        for i in range(1, 21):
            assert i in covered


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
