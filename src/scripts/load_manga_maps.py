"""
Run script: Load and visualize MaNGA MAPS data.

This script demonstrates loading a MaNGA DAP MAPS file,
extracting emission-line flux and stellar velocity maps,
and displaying them side-by-side.

Usage:
    cd toy_glens_kinematics
    python -m src.scripts.load_manga_maps
"""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from data_io import MaNGAMapsExtractor  # noqa: E402


# Path to the MaNGA MAPS file (relative to project root)
MAPS_PATH = "data/manga-8138-12704-MAPS-HYB10-MILESHC-MASTARHC2.fits"


def main():
    """Main function to demonstrate loading and visualizing MaNGA maps."""
    
    print("=" * 60)
    print("MaNGA MAPS Loader - Demo")
    print("=" * 60)
    
    # Create extractor (using H-alpha by default, index 18)
    # To use a different emission line, e.g., OIII-5007:
    #   extractor = MaNGAMapsExtractor(MAPS_PATH, emline_index=12)
    extractor = MaNGAMapsExtractor(MAPS_PATH)
    
    # Print FITS file info
    print("\n1. Inspecting FITS file structure...")
    extractor.print_info()
    
    # Extract flux and velocity maps
    print("2. Extracting maps...")
    flux_map, velocity_map = extractor.extract_maps()
    
    print(f"   Flux map shape: {flux_map.shape}")
    print(f"   Flux map range: [{flux_map.min():.2f}, {flux_map.max():.2f}]")
    print(f"   Velocity map shape: {velocity_map.shape}")
    print(f"   Velocity map range: [{velocity_map.min():.2f}, {velocity_map.max():.2f}] km/s")
    
    # Plot the maps
    print("\n3. Plotting maps...")
    extractor.plot_maps()
    
    print("\nDone!")


if __name__ == "__main__":
    main()

