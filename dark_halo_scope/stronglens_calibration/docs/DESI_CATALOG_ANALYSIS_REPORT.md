# DESI DR1 Strong Lensing Catalog Analysis Report

**Generated**: 2026-02-07T14:10:41.590250
**Purpose**: Assess catalog for independent validation of our lens finder

## 1. Catalog Summary

- **Total rows**: 2176
- **Total columns**: 45

### Redshift Distribution
- Valid redshifts: 2166 (99.5%)
- Invalid/missing: 10
- Range: 0.0000 - 3.1651
- Median: 0.5789
- Mean: 0.6952

### Redshift Warning Flags
- ZWARN=0 (reliable): 1637 (75.2%)
- ZWARN≠0 (caution): 539

### Spectral Type Distribution
- GALAXY: 2134 (98.1%)
- QSO: 25 (1.1%)
- STAR: 17 (0.8%)

### Morphological Type Distribution (Tractor)
- nan: 1730 (79.5%)
- SER: 284 (13.1%)
- DEV: 126 (5.8%)
- REX: 29 (1.3%)
- EXP: 6 (0.3%)
- PSF: 1 (0.0%)

### r-band Magnitude
- Valid flux measurements: 512
- Range: 16.33 - 22.16 mag
- Median: 19.23 mag

### Sky Coverage
- RA: 0.02° - 359.97°
- Dec: -18.99° - 75.03°

### Spatial Distribution
- Unique HEALPix cells (nside=64): 1201

## 2. Samples for Independent Validation

### Quality Cuts Applied
- valid_z: 2166 pass (99.5%)
- zwarn_0: 1637 pass (75.2%)
- is_galaxy: 2134 pass (98.1%)

**Samples passing ALL cuts**: 1603 (73.7%)

### Sample of Passing Entries (first 10)
| RA | Dec | z | Brickname |
|-----|-----|-----|-----------|
| 221.4063 | 4.9046 | 0.4220 | 2214p050 |
| 218.1147 | 28.7213 | 0.3675 | 2180p287 |
| 220.3159 | 36.3880 | 0.5336 | 2201p365 |
| 253.5000 | 27.5225 | 0.4027 | 2535p275 |
| 249.3569 | 31.6652 | 0.4509 | 2493p317 |
| 260.9006 | 34.1995 | 0.4423 | 2609p342 |
| 236.5553 | 35.0802 | 0.3820 | 2364p350 |
| 207.4187 | 38.5021 | 0.5784 | 2075p385 |
| 180.7746 | 69.8060 | 0.4316 | 1810p697 |
| 337.5729 | -0.3126 | 0.4064 | 3376m002 |

## 3. Cross-match with Our Imaging Candidates

- Imaging candidates file: `/Users/balaji/code/oss/toy_glens_kinematics/dark_halo_scope/data/positives/desi_candidates.csv`
- Number of imaging candidates: 5104
- Match radius: 5.0 arcsec
- **Matches found**: 822 (16.1% of imaging candidates)
- Unique DESI entries matched: 822

## 4. Recommendations for Use

### As Independent Validation Set
- **1603 high-quality spectroscopic observations** available
- These are spectroscopically-selected (different method from imaging-CNN)
- Breaks circularity with Paper IV ML candidates

### Quality Criteria for Selection
- Use only `ZWARN=0` for reliable redshifts
- Use only `ZCAT_PRIMARY=True` to avoid duplicates
- Filter `SPECTYPE='GALAXY'` for galaxy-galaxy lenses

### Integration Steps
1. Download catalog: `desi-sl-vac-v1.fits`
2. Apply quality cuts (ZWARN=0, ZCAT_PRIMARY=True)
3. Cross-match with our imaging candidates
4. Exclude overlapping systems from training
5. Use remaining DESI systems as independent validation
