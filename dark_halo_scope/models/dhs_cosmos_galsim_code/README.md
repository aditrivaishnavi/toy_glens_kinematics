# COSMOS + GalSim source morphology modules

Standalone modules to:
- Build a COSMOS RealGalaxy source bank (HDF5).
- Debug-inject a COSMOS template through an SIE+shear lens model (lenstronomy).
- Validate Parquet stamp datasets (stamp_npz) using lightweight diagnostics.
- Show a Spark-friendly integration pattern.

## Dependencies
cosmos_source_loader.py:
- galsim
- numpy
- h5py

cosmos_lens_injector.py:
- lenstronomy
- astropy
- numpy
- h5py

validate_cosmos_injection.py:
- pyarrow
- numpy

## Notes
- COSMOS data must be downloaded separately. Point --cosmos-dir to the catalog directory.
- Whitening does not denoise. Whitening generally adds noise to decorrelate drizzle noise.
- RealGalaxy rendering can amplify noise if you render at higher resolution than supported by the original PSF.
  This loader can optionally convolve with a small intrinsic PSF before rendering.

