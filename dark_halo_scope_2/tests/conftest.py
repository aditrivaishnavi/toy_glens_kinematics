"""
Pytest fixtures for Phase 3 tests.
"""

import os
from pathlib import Path

import pytest


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def phase2_analysis_dir(project_root: Path) -> Path:
    """Return the Phase 2 analysis directory for v3_color_relaxed."""
    return project_root / "results" / "phase2_analysis" / "v3_color_relaxed"


@pytest.fixture
def phase2_regions_summary_csv(phase2_analysis_dir: Path) -> Path:
    """Return path to the Phase 2 regions summary CSV."""
    return phase2_analysis_dir / "phase2_regions_summary.csv"


@pytest.fixture
def phase2_regions_bricks_csv(phase2_analysis_dir: Path) -> Path:
    """Return path to the Phase 2 regions bricks CSV."""
    return phase2_analysis_dir / "phase2_regions_bricks.csv"


@pytest.fixture
def sweep_cache_dir() -> Path:
    """Return the local sweep cache directory."""
    return Path.home() / "data" / "sweep_cache"


@pytest.fixture
def sample_sweep_path(sweep_cache_dir: Path) -> Path:
    """Return path to a sample sweep file (smallest one for faster tests)."""
    # sweep-200m090-205m085.fits is ~96MB, smallest in cache
    path = sweep_cache_dir / "sweep-200m090-205m085.fits"
    if not path.exists():
        pytest.skip(f"Sample sweep file not found: {path}")
    return path


@pytest.fixture
def full_sweep_path(sweep_cache_dir: Path) -> Path:
    """Return path to a full sweep file used in integration tests."""
    # sweep-000m035-005m030.fits is ~1.7GB, matches deprecated script output
    path = sweep_cache_dir / "sweep-000m035-005m030.fits"
    if not path.exists():
        pytest.skip(f"Full sweep file not found: {path}")
    return path

