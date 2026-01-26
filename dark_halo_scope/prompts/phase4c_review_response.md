# Phase 4c Review Response - Detailed Verification Report

## Purpose

This document responds to your independent review of the Phase 4c train tier output. I have performed the verifications you requested and present the raw findings below **without conclusions**. Please review my methodology, findings, and provide your independent assessment.

---

## Verification Methodology

All verifications were performed using Python scripts executed on an EC2 instance with direct S3 access. The scripts sample parquet files from the metrics output and compute statistics. I provide the exact code used for each verification so you can review my methodology.

---

## 1. Cutout_ok Exception Handling Verification

### Question from your review:
> Confirm `cutout_ok` is set to 0 on any exception path (not accidentally defaulting to 1).

### Code Review Findings

I examined `spark_phase4_pipeline.py` for all occurrences of `cutout_ok`:

**Success path (line 2325):**
```python
cutout_ok=int(bool(cut_ok_all)),
```

Where `cut_ok_all` is computed as (lines 2052-2060):
```python
cut_ok_all = True
for b in use_bands:
    img = cur[f"image_{b}"]
    inv = cur[f"invvar_{b}"]
    stamp, ok1 = _cutout(img, x, y, size)
    invs_b, ok2 = _cutout(inv, x, y, size)
    imgs[b] = stamp
    invs[b] = invs_b
    cut_ok_all = cut_ok_all and ok1 and ok2
```

**Exception path (lines 2346-2389):**
```python
except Exception as e:
    # Emit a row with cutout_ok=0 and an empty stamp to keep accounting consistent
    # MUST include ALL 53 schema fields to avoid ValueError on Row length mismatch
    ...
    yield Row(
        ...
        cutout_ok=0,
        ...
    )
```

### Your Assessment Requested:
1. Is this exception handling pattern correct?
2. Are there any code paths that could bypass setting cutout_ok=0 on failure?

---

## 2. Task Count Verification

### Question from your review:
> Confirm metrics rows exist for every task and that you are not silently dropping failed tasks before writing metrics.

### Verification Script:
```python
import s3fs, pyarrow.parquet as pq
fs = s3fs.S3FileSystem()

# Count manifest rows
manifest_path = "darkhaloscope/phase4_pipeline/phase4a/v3_color_relaxed/manifests_filtered/train_stamp64_bandsgrz_gridgrid_small"
manifest_files = fs.glob(manifest_path + "/**/*.parquet")
manifest_count = sum(pq.read_metadata(fs.open("s3://" + f)).num_rows for f in manifest_files)

# Count metrics rows
metrics_path = "darkhaloscope/phase4_pipeline/phase4c/v3_color_relaxed/metrics/train_stamp64_bandsgrz_gridgrid_small"
metrics_files = fs.glob(metrics_path + "/**/*.parquet")
metrics_count = sum(pq.read_metadata(fs.open("s3://" + f)).num_rows for f in metrics_files)
```

### Results:
```
Manifest rows: 10,627,158
Metrics rows:  10,627,158
Difference:    0
```

### Your Assessment Requested:
1. Is this verification sufficient to confirm no silent drops?
2. Should I also verify that failed tasks (cutout_ok=0) are present in the metrics?

---

## 3. PSF FWHM = 0 Investigation (Critical)

### Question from your review:
> A PSF FWHM of exactly 0 is not physically valid... Action: quantify how often psf_fwhm_used_g == 0 or psf_fwhm_used_z == 0.

### Verification Script:
```python
import s3fs, pyarrow.parquet as pq, pandas as pd, numpy as np
fs = s3fs.S3FileSystem()
path = "darkhaloscope/phase4_pipeline/phase4c/v3_color_relaxed/metrics/train_stamp64_bandsgrz_gridgrid_small"
files = fs.glob(path + "/**/*.parquet")

# Sample 200 files for robust statistics
sample_files = files[::len(files)//200] if len(files) > 200 else files
dfs = [pq.read_table("s3://" + f, filesystem=fs).to_pandas() for f in sample_files]
df = pd.concat(dfs, ignore_index=True)

inj = df[df["lens_model"] != "CONTROL"]
n_inj = len(inj)

for band in ["g", "r", "z"]:
    col = "psf_fwhm_used_" + band
    zero_count = (inj[col] == 0).sum()
    null_count = inj[col].isna().sum()
    valid_count = ((inj[col] > 0) & inj[col].notna()).sum()
    print("%s: == 0: %d (%.4f%%), NULL: %d, > 0: %d" % (col, zero_count, 100*zero_count/n_inj, null_count, valid_count))
```

### Results:
```
psf_fwhm_used_g:
  == 0:  518 (0.2822%)
  NULL:  0 (0.0000%)
  > 0:   183,020 (99.7178%)

psf_fwhm_used_r:
  == 0:  0 (0.0000%)
  NULL:  0 (0.0000%)
  > 0:   183,538 (100.0000%)

psf_fwhm_used_z:
  == 0:  0 (0.0000%)
  NULL:  0 (0.0000%)
  > 0:   183,538 (100.0000%)
```

### Further Investigation - Brick Distribution:
```python
zeros = inj[inj["psf_fwhm_used_g"] == 0]
print(zeros["brickname"].value_counts())
```

**Results:**
```
brickname
3578m312    506 (97.7%)
2124m292     12 (2.3%)
```

### Additional Context for PSF = 0 rows:
- These rows have valid psfsize_r from manifest: 1.177 - 1.289 arcsec
- r-band and z-band PSF are valid: r=1.161-1.511, z=1.128-1.307
- arc_snr is lower for these rows: mean=14.70 vs 38.18 for non-zero

### Your Assessment Requested:
1. At 0.28%, does this meet your threshold for "proceed with filter" or "NO-GO until fixed"?
2. Given it's localized to 2 bricks, is filtering sufficient or should those bricks be investigated/rerun?
3. The lower arc_snr (14.70 vs 38.18) suggests g-band convolution may be incorrect for these rows. Does this affect your assessment?

---

## 4. Resolution Distribution Verification

### Question from your review:
> "resolved >= 1.0" being ~0.1% seems low if the ratio is literally theta_e / psf_fwhm_r...

### Verification Script:
```python
import s3fs, pyarrow.parquet as pq, pandas as pd
fs = s3fs.S3FileSystem()
path = "darkhaloscope/phase4_pipeline/phase4c/v3_color_relaxed/metrics/train_stamp64_bandsgrz_gridgrid_small"
files = fs.glob(path + "/**/*.parquet")

# Larger sample for accurate statistics
sample_files = files[::len(files)//500] if len(files) > 500 else files
dfs = [pq.read_table("s3://" + f, filesystem=fs).to_pandas() for f in sample_files]
df = pd.concat(dfs, ignore_index=True)
inj = df[df["lens_model"] != "CONTROL"]

# The exact formula: theta_e_arcsec / psfsize_r
ratio = inj["theta_e_arcsec"] / inj["psfsize_r"]
```

### Results:
```
Ratio statistics:
  min:    0.1875
  max:    1.1276
  mean:   0.4858
  median: 0.4606

Binned distribution:
  0.00 - 0.25: 122,952 (27.09%)
  0.25 - 0.50: 150,356 (33.13%)
  0.50 - 0.75:  90,384 (19.92%)
  0.75 - 1.00:  89,702 (19.77%)
  1.00 - 1.50:     446 (0.10%)
  >= 1.50:           0 (0.00%)

Cross-check of input ranges:
  theta_e range:   0.300 - 1.000 arcsec
  psfsize_r range: 0.887 - 1.600 arcsec
  Max possible ratio: 1.000 / 0.887 = 1.128
```

### Your Assessment Requested:
1. Given max(theta_e) = 1.0" and min(psfsize_r) = 0.887", the maximum possible ratio is 1.128. Is the 0.1% resolved figure now explained?
2. Should the Phase 4d recovery criterion (theta_over_psf >= 0.8) be reconsidered given this distribution?
3. For future runs, should we extend the theta_e range to achieve more resolved injections?

---

## 5. Duplicate Task ID Verification

### Question from your review:
> Verify uniqueness of (experiment_id, task_id) across partitions

### Verification Script:
```python
import s3fs, pyarrow.parquet as pq, pandas as pd
fs = s3fs.S3FileSystem()
path = "darkhaloscope/phase4_pipeline/phase4c/v3_color_relaxed/metrics/train_stamp64_bandsgrz_gridgrid_small"
files = fs.glob(path + "/**/*.parquet")

sample_files = files[::len(files)//100] if len(files) > 100 else files
dfs = [pq.read_table("s3://" + f, filesystem=fs).to_pandas() for f in sample_files]
df = pd.concat(dfs, ignore_index=True)

# Check uniqueness
unique_pairs = df[["experiment_id", "task_id"]].drop_duplicates()
print("Total rows: %d" % len(df))
print("Unique (experiment_id, task_id) pairs: %d" % len(unique_pairs))
print("Duplicates: %d" % (len(df) - len(unique_pairs)))

# Sample task_ids to show entropy
sample_ids = df["task_id"].head(5).tolist()
for tid in sample_ids:
    print("  %s" % tid)
```

### Results:
```
Total rows: 180,314
Unique (experiment_id, task_id) pairs: 180,314
Duplicates: 0

Sample task_ids:
  a83eeede07243610b9966e55b49dad6cd1cc8f9a6aae6acff6e39fb9962f95e4
  70f361f9c63109d53e90340c3ad87bc1a02d7f2f4467f8bbcf5ff172baea5652
  147844e0e5d61f3c70bc4a081ea99fcd676341aa070cea9043fef8878822dce0
  93408f645b1ec625aa74acb8a4fd23eff010d73550c1b78258aa9cb0b994eb02
  53e9e27893a9bd4ea7f7b35e80489dd3090010859b36b6ee3136d65974b867fa
```

### Your Assessment Requested:
1. Are SHA256 task IDs with this entropy pattern sufficient?
2. Is the sample-based verification (180K rows from 10.6M) sufficient, or should I verify the full dataset?

---

## 6. Observing Condition Bias Verification

### Verification Script:
```python
for col in ["psfsize_r", "psfdepth_r"]:
    means = df.groupby("region_split")[col].mean()
    print("%s: train=%.3f, val=%.3f, test=%.3f" % (col, means.get("train",0), means.get("val",0), means.get("test",0)))
```

### Results:
```
psfsize_r:  train=1.326, val=1.314, test=1.315
psfdepth_r: train=24.731, val=24.587, test=24.616
```

### Your Assessment Requested:
1. Are these differences (< 1% relative) small enough to not cause split artifacts?
2. Should we report these in any publication as a systematic effect?

---

## 7. Bad Pixel Fraction Distribution

### Verification Script:
```python
inj = df[df["lens_model"] != "CONTROL"]
high_bad = inj[inj["bad_pixel_frac"] > 0.2]
print("High bad_pixel_frac (>20%%): %d (%.1f%%)" % (len(high_bad), 100*len(high_bad)/len(inj)))
```

### Results:
```
High bad_pixel_frac (>20%): 8,994 (10.1%)
```

### Your Assessment Requested:
1. What quality cut threshold do you recommend for Phase 5 training?
2. Should Phase 4d report completeness separately for "all" vs "clean" subsets?
3. Is 10.1% high-bad-pixel acceptable or concerning?

---

## Summary of Open Questions

Please provide your assessment on:

### Critical (affects GO/NO-GO):
1. **PSF = 0 at 0.28%**: Is this acceptable with filtering, or requires rerun?
2. **PSF = 0 lower arc_snr**: Does this indicate incorrect g-band convolution?

### Important (affects methodology):
3. **Resolution 0.1%**: Is the mathematical explanation satisfactory?
4. **Recovery criterion θ/PSF >= 0.8**: Is this appropriate given the distribution?
5. **Quality cuts**: What exact filter should we use for Phase 4d and Phase 5?

### Verification quality:
6. **Sample-based verification**: Is sampling 100-500 files from 6000 sufficient?
7. **Code review**: Are there issues with my verification scripts?

---

## Raw Data Access

If you need to verify any of these findings:
```
Metrics: s3://darkhaloscope/phase4_pipeline/phase4c/v3_color_relaxed/metrics/train_stamp64_bandsgrz_gridgrid_small/
Pipeline code: spark_phase4_pipeline.py (attached or available on request)
```

---

**Please provide your independent assessment and recommendations.**
