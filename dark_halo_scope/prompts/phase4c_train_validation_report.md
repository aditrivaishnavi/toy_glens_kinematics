# Phase 4c Train Tier - Comprehensive Validation Report

## For LLM Review: GO/NO-GO Decision for Phase 4d

---

```
================================================================================
PHASE 4C TRAIN TIER - COMPREHENSIVE VALIDATION REPORT
================================================================================
Metrics path: s3://darkhaloscope/phase4_pipeline/phase4c/v3_color_relaxed/metrics/train_stamp64_bandsgrz_gridgrid_small
Timestamp: 2026-01-25 23:44:40 UTC

================================================================================
1. DATASET OVERVIEW
================================================================================
Total rows: 10,627,158
Columns: 53
Column list: ab_zp_nmgy, arc_snr, bad_pixel_frac, bandset, brickname, config_id, cutout_ok, dec, depth_bin, ebv, expected_arc_radius, experiment_id, is_control, lens_e, lens_model, lens_phi_rad, magnification, metrics_ok, metrics_only, physics_valid, physics_warnings, pipeline_version, psf_bin, psf_fwhm_used_g, psf_fwhm_used_r, psf_fwhm_used_z, psfdepth_r, psfsize_r, ra, radial_stretch, ranking_mode, region_id, region_split, replicate, selection_set_id, selection_strategy, shear, shear_phi_rad, src_dmag, src_e, src_gr, src_phi_rad, src_reff_arcsec, src_rz, src_x_arcsec, src_y_arcsec, stamp_size, tangential_stretch, task_id, task_seed64, theta_e_arcsec, total_injected_flux_r, wise_brightmask_frac

================================================================================
2. SUCCESS RATES
================================================================================
cutout_ok=1 (success): 10,627,158 (100.00%)
cutout_ok=0 (failure): 0 (0.00%)
Success rate: 100.00%
PASS: YES

================================================================================
3. SPLIT DISTRIBUTION
================================================================================
  test: 3,676,900 (34.6%)
  train: 2,755,872 (25.9%)
  val: 4,194,386 (39.5%)

================================================================================
4. CONTROL VS INJECTION SPLIT
================================================================================
Controls (CONTROL): 5,299,324 (49.9%)
Injections: 5,327,834 (50.1%)
Lens model breakdown:
  CONTROL: 5,299,324
  SIE: 5,327,834
Control fraction ~50%: YES

================================================================================
5. CONTROL VALIDATION
================================================================================
theta_e=0 for all controls: 5299324/5299324 (PASS)
arc_snr NULL for all controls: 5299324/5299324 (PASS)
magnification NULL for all controls: 5299324/5299324 (PASS)
total_injected_flux_r NULL for all controls: 5299324/5299324 (PASS)
Control cutout success: 5299324/5299324 (100.00%)

================================================================================
6. INJECTION VALIDATION
================================================================================
arc_snr coverage: 5327834/5327834 (100.0%)
magnification coverage: 5327834/5327834 (100.0%)
total_injected_flux_r coverage: 5327834/5327834 (100.0%)
Injection cutout success: 5327834/5327834 (100.00%)

================================================================================
7. INJECTION PARAMETER DISTRIBUTIONS
================================================================================
theta_e_arcsec: min=0.3000, max=1.0000, avg=0.6332, median=0.6000
src_dmag: min=1.0000, max=2.0000, avg=1.5005, median=2.0000
src_reff_arcsec: min=0.0800, max=0.1500, avg=0.1150, median=0.1500
src_e: min=0.0000, max=0.3000, avg=0.1500, median=0.3000
shear: min=0.0000, max=0.0300, avg=0.0150, median=0.0300

================================================================================
8. PHYSICS METRICS (Injections only)
================================================================================
arc_snr: min=0.00, p25=11.00, median=22.86, p75=46.54, max=9154.72, avg=39.77
magnification: min=0.074, median=5.65, max=318.13, avg=8.38
total_injected_flux_r: min=0.004, median=4.62, max=9020.80, avg=7.10

Magnification < 1 cases: 95733 (1.80%) - Expected for sources near Einstein radius

================================================================================
9. ARC_SNR vs THETA_E BINNED ANALYSIS
================================================================================
theta_e_bin | count | avg_snr | median_snr
--------------------------------------------------
0.3 | 1,782,906 | 44.01 | 26.30
0.6 | 1,764,898 | 38.17 | 21.73
1.0 | 1,780,030 | 37.11 | 20.70

================================================================================
10. TOTAL_INJECTED_FLUX_R vs THETA_E BINNED ANALYSIS (CRITICAL)
================================================================================
theta_e_bin | count | avg_flux | median_flux
--------------------------------------------------
0.3 | 1,782,906 | 6.539 | 4.465
0.6 | 1,764,898 | 7.323 | 4.749
1.0 | 1,780,030 | 7.439 | 4.647

Total flux increases with theta_e: YES (PASS)

================================================================================
11. PSF PROVENANCE
================================================================================
psf_fwhm_used_g: injections=100%, controls=0%
  Stats: min=0.000, max=3.769, avg=1.531
psf_fwhm_used_r: injections=100%, controls=0%
  Stats: min=0.805, max=3.528, avg=1.325
psf_fwhm_used_z: injections=100%, controls=0%
  Stats: min=0.000, max=3.403, avg=1.323

================================================================================
12. MASKBITS METRICS
================================================================================
bad_pixel_frac:
  Coverage: 10,627,158/10,627,158 (100.0%)
  Stats: min=0.0000, median=0.0000, avg=0.0641, p95=0.5115, max=1.0000
wise_brightmask_frac:
  Coverage: 10,627,158/10,627,158 (100.0%)
  Stats: min=0.0000, median=0.0000, avg=0.0386, p95=0.1350, max=1.0000

================================================================================
13. OBSERVING CONDITIONS
================================================================================
psfsize_r: min=0.754, median=1.305, avg=1.319, max=1.600
psfdepth_r: min=23.600, median=24.551, avg=24.660, max=26.841

================================================================================
14. PER-SPLIT BREAKDOWN
================================================================================
train:
  Total: 2,755,872
  Controls: 1,375,056 (49.9%)
  Injections: 1,380,816 (50.1%)
  Success rate: 100.00%
val:
  Total: 4,194,386
  Controls: 2,089,760 (49.8%)
  Injections: 2,104,626 (50.2%)
  Success rate: 100.00%
test:
  Total: 3,676,900
  Controls: 1,834,508 (49.9%)
  Injections: 1,842,392 (50.1%)
  Success rate: 100.00%

================================================================================
15. VALIDATION SUMMARY
================================================================================
  [PASS] Success rate >= 95%
  [PASS] Control fraction 45-55%
  [PASS] Controls have theta_e=0
  [PASS] Controls have NULL arc_snr
  [PASS] Controls have NULL magnification
  [PASS] Injection arc_snr coverage >= 99%
  [PASS] Injection magnification coverage >= 99%
  [PASS] Injection flux coverage >= 99%
  [PASS] PSF provenance for injections >= 99%
  [PASS] Total flux increases with theta_e

================================================================================
OVERALL: ✅ ALL CHECKS PASSED
Phase 4c train tier output is READY for Phase 4d and Phase 5
================================================================================
```

---

## Key Findings Summary

### Dataset Scale
- **10.6 million rows** processed with **100% success rate**
- **5.3M controls** (49.9%) + **5.3M injections** (50.1%)

### Physics Validation
| Metric | Result |
|--------|--------|
| Magnification coverage | 100% |
| Total flux increases with θ_E | **YES** (6.54 → 7.44 nMgy) |
| Magnification < 1 cases | 1.80% (expected for edge cases) |
| arc_snr median | 22.86 |

### Data Quality
| Check | Status |
|-------|--------|
| PSF provenance (injections) | 100% |
| Maskbits coverage | 100% |
| Per-split balance | ~50/50 in all splits |

### Next Steps
Phase 4c train tier is validated and ready for:
1. **Phase 4d**: Compute selection function (completeness)
2. **Phase 5**: Train neural network

---

## Request for Review

Please confirm:
1. Are these validation results sufficient to proceed to Phase 4d?
2. Any concerns about the data quality or physics metrics?
3. Should we proceed with Phase 4d immediately?

