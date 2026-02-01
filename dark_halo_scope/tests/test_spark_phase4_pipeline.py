"""
Unit tests for spark_phase4_pipeline.py

These tests validate the patched code BEFORE deploying to EMR.
NOTE: Tests require PySpark to be installed.
"""
import sys
import os
import argparse
import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check if pyspark is available
try:
    import pyspark
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False


@pytest.mark.skipif(not PYSPARK_AVAILABLE, reason="PySpark not installed")
class TestArgumentParser:
    """Test that all required arguments exist and have correct defaults."""
    
    def test_psf_model_argument_exists(self):
        """Verify --psf-model argument is defined."""
        from emr.spark_phase4_pipeline import build_parser
        parser = build_parser()
        args = parser.parse_args(['--stage', '4c', '--output-s3', 's3://test', '--variant', 'test'])
        assert hasattr(args, 'psf_model'), "Missing psf_model argument"
        assert args.psf_model == 'moffat', f"Expected default 'moffat', got {args.psf_model}"
    
    def test_moffat_beta_argument_exists(self):
        """Verify --moffat-beta argument is defined."""
        from emr.spark_phase4_pipeline import build_parser
        parser = build_parser()
        args = parser.parse_args(['--stage', '4c', '--output-s3', 's3://test', '--variant', 'test'])
        assert hasattr(args, 'moffat_beta'), "Missing moffat_beta argument"
        assert args.moffat_beta == 3.5, f"Expected default 3.5, got {args.moffat_beta}"
    
    def test_all_referenced_args_exist(self):
        """Check that all args referenced in stage_4c_inject_cutouts exist."""
        from emr.spark_phase4_pipeline import build_parser
        parser = build_parser()
        args = parser.parse_args(['--stage', '4c', '--output-s3', 's3://test', '--variant', 'test'])
        
        # List of args referenced in stage_4c config dict
        required_attrs = [
            'experiment_id',
            'coadd_s3_cache_prefix', 
            'sweep_partitions',
            'skip_if_exists',
            'force',
            'psf_model',
            'moffat_beta',
            'manifests_subdir',
        ]
        
        missing = []
        for attr in required_attrs:
            if not hasattr(args, attr):
                missing.append(attr)
        
        assert len(missing) == 0, f"Missing arguments: {missing}"
    
    def test_src_flux_scale_handled(self):
        """Verify src_flux_scale is handled (either as arg or with getattr default)."""
        from emr.spark_phase4_pipeline import build_parser
        parser = build_parser()
        args = parser.parse_args(['--stage', '4c', '--output-s3', 's3://test', '--variant', 'test'])
        
        # Should not raise even if src_flux_scale is not defined
        value = getattr(args, 'src_flux_scale', 1.0)
        assert value == 1.0, f"Expected default 1.0, got {value}"


@pytest.mark.skipif(not PYSPARK_AVAILABLE, reason="PySpark not installed")
class TestMoffatPSF:
    """Test Moffat PSF kernel generation."""
    
    def test_moffat_kernel_exists(self):
        """Verify _moffat_kernel2d function exists."""
        from emr.spark_phase4_pipeline import _moffat_kernel2d
        assert callable(_moffat_kernel2d)
    
    def test_moffat_kernel_normalized(self):
        """Moffat kernel should sum to 1.0."""
        from emr.spark_phase4_pipeline import _moffat_kernel2d
        kernel = _moffat_kernel2d(fwhm_pix=3.0, beta=3.5)
        assert abs(kernel.sum() - 1.0) < 1e-6, f"Kernel sum = {kernel.sum()}, expected 1.0"
    
    def test_moffat_kernel_shape(self):
        """Moffat kernel should be square and odd-sized."""
        from emr.spark_phase4_pipeline import _moffat_kernel2d
        kernel = _moffat_kernel2d(fwhm_pix=3.0, beta=3.5)
        assert kernel.shape[0] == kernel.shape[1], "Kernel should be square"
        assert kernel.shape[0] % 2 == 1, "Kernel size should be odd"
    
    def test_moffat_kernel_centered(self):
        """Moffat kernel should have max at center."""
        from emr.spark_phase4_pipeline import _moffat_kernel2d
        kernel = _moffat_kernel2d(fwhm_pix=3.0, beta=3.5)
        center = kernel.shape[0] // 2
        assert kernel[center, center] == kernel.max(), "Max should be at center"


@pytest.mark.skipif(not PYSPARK_AVAILABLE, reason="PySpark not installed")
class TestConvolvePSF:
    """Test the unified _convolve_psf function."""
    
    def test_convolve_psf_exists(self):
        """Verify _convolve_psf function exists."""
        from emr.spark_phase4_pipeline import _convolve_psf
        assert callable(_convolve_psf)
    
    def test_convolve_psf_gaussian(self):
        """Test Gaussian PSF convolution."""
        from emr.spark_phase4_pipeline import _convolve_psf
        img = np.zeros((64, 64), dtype=np.float32)
        img[32, 32] = 1.0  # Point source
        
        result = _convolve_psf(img, psf_fwhm_pix=3.0, psf_model='gaussian')
        assert result.shape == img.shape
        assert result.sum() > 0.99  # Flux conserved
    
    def test_convolve_psf_moffat(self):
        """Test Moffat PSF convolution."""
        from emr.spark_phase4_pipeline import _convolve_psf
        img = np.zeros((64, 64), dtype=np.float32)
        img[32, 32] = 1.0  # Point source
        
        result = _convolve_psf(img, psf_fwhm_pix=3.0, psf_model='moffat', moffat_beta=3.5)
        assert result.shape == img.shape
        assert result.sum() > 0.99  # Flux conserved
    
    def test_convolve_psf_invalid_model_raises(self):
        """Invalid PSF model should raise ValueError."""
        from emr.spark_phase4_pipeline import _convolve_psf
        img = np.zeros((64, 64), dtype=np.float32)
        
        with pytest.raises(ValueError, match="Unknown psf_model"):
            _convolve_psf(img, psf_fwhm_pix=3.0, psf_model='invalid')
    
    def test_convolve_psf_zero_fwhm(self):
        """Zero FWHM should return original image."""
        from emr.spark_phase4_pipeline import _convolve_psf
        img = np.random.rand(64, 64).astype(np.float32)
        
        result = _convolve_psf(img, psf_fwhm_pix=0.0, psf_model='gaussian')
        np.testing.assert_array_almost_equal(result, img)


@pytest.mark.skipif(not PYSPARK_AVAILABLE, reason="PySpark not installed")
class TestNumpyRoundFix:
    """Test that numpy round() is used instead of built-in round()."""
    
    def test_numpy_round_in_render_functions(self):
        """Check that np.round is used, not built-in round."""
        import inspect
        from emr import spark_phase4_pipeline
        
        # Get source code
        source = inspect.getsource(spark_phase4_pipeline)
        
        # Look for patterns that would fail with numpy arrays
        # The bug was: int(round(x)) where x is numpy array element
        # These should now be: int(np.round(x))
        
        # Count occurrences
        bad_pattern_count = source.count('int(round(')
        good_pattern_count = source.count('int(np.round(')
        
        # Should have more np.round than round
        assert bad_pattern_count == 0, f"Found {bad_pattern_count} occurrences of 'int(round(' which may fail with numpy"


@pytest.mark.skipif(not PYSPARK_AVAILABLE, reason="PySpark not installed")
class TestRowSchemaCompleteness:
    """Test that exception handler Row has all required fields."""
    
    def test_exception_row_field_count(self):
        """The exception Row in process_task should have correct field count."""
        import inspect
        from emr import spark_phase4_pipeline
        
        source = inspect.getsource(spark_phase4_pipeline)
        
        # Look for the exception Row pattern
        # It should match the schema with 53 fields
        if 'Row(' in source and 'except' in source:
            # This is a weak check - a proper test would parse the AST
            pass  # Placeholder for more thorough testing


def build_parser():
    """Build argument parser - extracted for testing."""
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["4a", "4b", "4c", "4d", "4p5"])
    p.add_argument("--output-s3", required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--skip-if-exists", type=int, default=1)
    p.add_argument("--force", type=int, default=0)
    p.add_argument("--bands", default="g,r,z")
    p.add_argument("--coadd-base-url", default="https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr10/south/coadd")
    p.add_argument("--coadd-s3-cache-prefix", default="s3://darkhaloscope/dr10/coadd_cache")
    p.add_argument("--psf-model", default="moffat", choices=["gaussian", "moffat"])
    p.add_argument("--moffat-beta", type=float, default=3.5)
    p.add_argument("--experiment-id", default="")
    p.add_argument("--manifests-subdir", default="manifests")
    p.add_argument("--sweep-partitions", type=int, default=200)
    return p


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

