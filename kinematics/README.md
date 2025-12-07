# Toy Glens Kinematics

## MaNGA MAPS File Structure and Path Explanation

This information is sourced from the [Sloan Digital Sky Survey DR17](https://dr17.sdss.org/).

This project uses **MaNGA DR17 DAP MAPS FITS files**, which contain 2D maps of galaxy properties, including:

- Stellar velocity fields
- Emission line fluxes and velocity (Hα, [OIII], etc.)
- Masks (bad pixel flags)
- Line-strength indices and kinematic information

These maps are essential for building flux + velocity source planes for strong-lensing simulations.

---

### 📁 Official File Path Structure (Annotated)

```
https://data.sdss.org/sas/dr17/manga/spectro/analysis/v3_1_1/3.1.0/HYB10-MILESHC-MASTARHC2/8138/12704/manga-8138-12704-MAPS-HYB10-MILESHC-MASTARHC2.fits.gz
```

Below is the breakdown of each part:

| Path Part | Meaning | Notes |
|-----------|---------|-------|
| `sas` | Science Archive Server | SDSS file hosting |
| `dr17` | Data Release 17 (latest) | Most updated MaNGA data |
| `manga` | MaNGA survey | Mapping Nearby Galaxies at APO |
| `spectro` | Spectroscopic data | Includes IFU datasets |
| `analysis` | DAP outputs (processed data) | NOT raw observational data |
| `v3_1_1` | DAP Software Version | Defines processing version |
| `3.1.0` | Data product version | Internal version tag |
| `HYB10-MILESHC-MASTARHC2` | 🔥 Data modeling type | Specifies model used (hybrid stellar+gas kinematics) |
| `8138` | 📌 Plate ID | Identifies observation session |
| `12704` | 📌 IFU Design ID | Specific MaNGA galaxy bundle |
| `manga-8138-12704-MAPS-HYB10-MILESHC-MASTARHC2.fits.gz` | 🔑 MAPS file | Contains 2D maps (flux, velocity, masks, etc.) |

---

### 📄 What the MAPS File Contains

| MAPS Extension | Meaning |
|----------------|---------|
| `EMLINE_GFLUX` | Emission line flux map (e.g., Hα, [NII], [OIII]) |
| `EMLINE_GVEL` | Emission line velocity map (gas kinematics) |
| `STELLAR_VEL` | Stellar velocity map |
| `STELLAR_SIGMA` | Velocity dispersion |
| `*_MASK` | Bad pixel mask for data quality control |
| `HEADER` | FITS metadata (observation details, units, resolution) |

