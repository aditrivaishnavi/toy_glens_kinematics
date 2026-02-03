"""
External catalog source definitions.

This module defines the sources and formats for external catalogs used in
anchor baseline validation. All catalog metadata is defined here for reproducibility.

Catalog Sources:
- SLACS: Strong Lensing Legacy Survey (confirmed galaxy-galaxy lenses)
- BELLS: BOSS Emission-Line Lens Survey (spectroscopically confirmed)
- Galaxy Zoo: Ring galaxies and mergers (hard negatives)
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum


class CatalogType(Enum):
    """Type of external catalog."""
    KNOWN_LENS = "known_lens"
    HARD_NEGATIVE = "hard_negative"
    CANDIDATE = "candidate"


@dataclass
class CatalogSource:
    """
    Definition of an external catalog source.
    
    This contains all metadata needed to download and parse a catalog.
    """
    name: str
    catalog_type: CatalogType
    description: str
    
    # Source URL or reference
    source_url: Optional[str] = None
    source_reference: Optional[str] = None  # ADS bibcode
    
    # Data format
    format: str = "csv"  # csv, fits, votable
    
    # Column mappings (source column -> standard column)
    ra_column: str = "ra"
    dec_column: str = "dec"
    
    # Optional columns
    theta_e_column: Optional[str] = None
    z_lens_column: Optional[str] = None
    z_source_column: Optional[str] = None
    grade_column: Optional[str] = None
    
    # Selection criteria
    min_grade: Optional[str] = None  # e.g., "A" or "B"
    
    # Expected count (for validation)
    expected_count_min: int = 0
    expected_count_max: int = 10000


# =============================================================================
# Known Lens Catalogs
# =============================================================================

SLACS_CATALOG = CatalogSource(
    name="SLACS",
    catalog_type=CatalogType.KNOWN_LENS,
    description="Sloan Lens ACS Survey - confirmed galaxy-galaxy lenses from SDSS",
    source_reference="2009ApJ...705.1099A",  # Auger et al. 2009
    format="csv",
    ra_column="RAJ2000",
    dec_column="DEJ2000",
    theta_e_column="theta_E",
    z_lens_column="z_l",
    z_source_column="z_s",
    grade_column="Grade",
    min_grade="A",
    expected_count_min=50,
    expected_count_max=150
)

BELLS_CATALOG = CatalogSource(
    name="BELLS",
    catalog_type=CatalogType.KNOWN_LENS,
    description="BOSS Emission-Line Lens Survey - spectroscopically confirmed lenses",
    source_reference="2012ApJ...744...41B",  # Brownstein et al. 2012
    format="csv",
    ra_column="RA",
    dec_column="DEC",
    theta_e_column="REIN",  # Einstein radius
    z_lens_column="ZLENS",
    z_source_column="ZSRC",
    grade_column="GRADE",
    min_grade="A",
    expected_count_min=20,
    expected_count_max=100
)

SL2S_CATALOG = CatalogSource(
    name="SL2S",
    catalog_type=CatalogType.KNOWN_LENS,
    description="Strong Lensing Legacy Survey - CFHTLS lenses",
    source_reference="2012A&A...544A..62S",  # Sonnenfeld et al. 2012
    format="fits",
    ra_column="RA",
    dec_column="DEC",
    theta_e_column="theta_E",
    expected_count_min=30,
    expected_count_max=100
)


# =============================================================================
# Hard Negative Catalogs
# =============================================================================

GALAXY_ZOO_RINGS_CATALOG = CatalogSource(
    name="GalaxyZoo_Rings",
    catalog_type=CatalogType.HARD_NEGATIVE,
    description="Galaxy Zoo ring galaxy classifications - lens mimics",
    source_url="https://data.galaxyzoo.org/",
    source_reference="2013MNRAS.435.2835W",  # Willett et al. 2013
    format="csv",
    ra_column="ra",
    dec_column="dec",
    expected_count_min=500,
    expected_count_max=5000
)

GALAXY_ZOO_MERGERS_CATALOG = CatalogSource(
    name="GalaxyZoo_Mergers",
    catalog_type=CatalogType.HARD_NEGATIVE,
    description="Galaxy Zoo merger classifications - potential false positives",
    source_url="https://data.galaxyzoo.org/",
    source_reference="2010MNRAS.401.1043D",  # Darg et al. 2010
    format="csv",
    ra_column="ra",
    dec_column="dec",
    expected_count_min=1000,
    expected_count_max=10000
)

SIMBAD_RING_GALAXIES = CatalogSource(
    name="SIMBAD_Rings",
    catalog_type=CatalogType.HARD_NEGATIVE,
    description="SIMBAD ring galaxy catalog",
    source_url="https://simbad.u-strasbg.fr/",
    format="votable",
    ra_column="RA_d",
    dec_column="DEC_d",
    expected_count_min=100,
    expected_count_max=2000
)


# =============================================================================
# Catalog Registry
# =============================================================================

CATALOG_REGISTRY: Dict[str, CatalogSource] = {
    "slacs": SLACS_CATALOG,
    "bells": BELLS_CATALOG,
    "sl2s": SL2S_CATALOG,
    "gz_rings": GALAXY_ZOO_RINGS_CATALOG,
    "gz_mergers": GALAXY_ZOO_MERGERS_CATALOG,
    "simbad_rings": SIMBAD_RING_GALAXIES,
}


def get_known_lens_catalogs() -> List[CatalogSource]:
    """Get all known lens catalogs."""
    return [c for c in CATALOG_REGISTRY.values() if c.catalog_type == CatalogType.KNOWN_LENS]


def get_hard_negative_catalogs() -> List[CatalogSource]:
    """Get all hard negative catalogs."""
    return [c for c in CATALOG_REGISTRY.values() if c.catalog_type == CatalogType.HARD_NEGATIVE]


def get_catalog(name: str) -> CatalogSource:
    """Get a catalog by name."""
    if name.lower() not in CATALOG_REGISTRY:
        raise ValueError(f"Unknown catalog: {name}. Available: {list(CATALOG_REGISTRY.keys())}")
    return CATALOG_REGISTRY[name.lower()]

