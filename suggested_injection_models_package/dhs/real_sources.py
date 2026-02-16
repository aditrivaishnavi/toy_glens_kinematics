
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

@dataclass
class RealSourceStamp:
    # source-plane image in arbitrary units, with known pixel scale (arcsec/pix)
    img: np.ndarray  # (H,W)
    pixel_scale_arcsec: float

class CosmosSourceLibrary:
    """
    Optional Model 3 component.
    Uses GalSim COSMOS galaxy library (requires external files).

    Download:
    - Install galsim
    - Download COSMOS catalog files as per GalSim docs (COSMOS_25.2_training_sample and associated data).
    - Set COSMOS_DIR environment variable to the directory.

    If you prefer HUDF+MUSE sources like HOLISMOKES, you will need a different source catalog.
    """
    def __init__(self, cosmos_dir: str, rng: Optional[np.random.Generator] = None):
        try:
            import galsim
        except Exception as e:
            raise ImportError("Model 3 requires galsim. Install galsim and COSMOS files.") from e
        self.galsim = galsim
        self.cosmos_dir = cosmos_dir
        self.rng = rng or np.random.default_rng()
        self.catalog = galsim.COSMOSCatalog(dir=cosmos_dir, file_name='real_galaxy_catalog_25.2.fits')

    def sample(self) -> RealSourceStamp:
        gal = self.catalog.makeGalaxy(self.rng.integers(0, self.catalog.nobjects))
        # draw at native COSMOS pixel scale (0.03 arcsec/pix typically)
        im = gal.drawImage(scale=0.03).array.astype(np.float32)
        return RealSourceStamp(img=im, pixel_scale_arcsec=0.03)
