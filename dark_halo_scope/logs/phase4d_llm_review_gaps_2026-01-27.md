# Phase 4d LLM Review Gap Analysis - 2026-01-27

## Executive Summary

The independent LLM reviewed Phase 4d output and provided detailed scientific and code feedback. This document analyzes every claim, verifies against our current implementation, identifies gaps, and documents action items.

**Key Finding:** The LLM's claim that our code lacks Wilson CI is **INCORRECT**. We implemented Wilson CI in our Phase 4d upgrade (commit 40c66b5). The LLM likely reviewed an older version of our code.

---

## LLM Verdict Summary

**GO for Phase 5 as engineering step, but NOT publication-ready yet.**

### LLM's Three Must-Fix Items

| # | LLM Claim | Our Status | Gap? |
|---|-----------|------------|------|
| 1 | Define completeness relative to actual detector (Phase 5 model score) | Pending Phase 5/6 | ⚠️ Expected - requires Phase 5 |
| 2 | Rework region-variance summary (dominated by tiny bins) | Partially addressed | ⚠️ Action needed |
| 3 | Wilson CI not computed in pipeline | **INCORRECT** - We have it | ✅ No gap |

---

## Detailed Code Comparison

### File Comparison

| Aspect | Our Code (`spark_phase4_pipeline.py`) | LLM's Code (`spark_phase4_pipeline_phase4d_ci.py`) |
|--------|---------------------------------------|---------------------------------------------------|
| Wilson CI | ✅ Python UDF-based | ✅ Pure Spark SQL expressions |
| Integration | Part of main pipeline | Standalone script |
| Lines | ~300 lines (4d section) | 339 lines total |

### 1. Wilson CI Implementation

**Our Implementation (lines 2708-2734):**
```python
@F.udf(T.StructType([
    T.StructField("ci_low", T.DoubleType(), True),
    T.StructField("ci_high", T.DoubleType(), True),
]))
def wilson_ci_udf(k, n):
    import math
    if n is None or n == 0 or k is None:
        return (None, None)
    k, n = int(k), int(n)
    z = ci_z
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))
```

**LLM's Implementation (lines 67-86):**
```python
def wilson_ci_cols(k_col, n_col, z: float = 1.96) -> Tuple[F.Column, F.Column]:
    z2 = z * z
    n = n_col.cast("double")
    k = k_col.cast("double")
    p = F.when(n > 0, k / n).otherwise(F.lit(None).cast("double"))
    denom = (F.lit(1.0) + F.lit(z2) / n)
    center = (p + F.lit(z2) / (F.lit(2.0) * n)) / denom
    half = (F.lit(z) / denom) * F.sqrt((p * (F.lit(1.0) - p) + F.lit(z2) / (F.lit(4.0) * n)) / n)
    low = F.when(n > 0, center - half).otherwise(F.lit(None).cast("double"))
    high = F.when(n > 0, center + half).otherwise(F.lit(None).cast("double"))
    return low, high
```

**Verdict:** Both are mathematically correct. LLM's is more efficient (no UDF overhead). Consider adopting.

---

### 2. Validity Definition

**Our Implementation:**
```python
df = df.withColumn("valid_all", 
    ((F.col("cutout_ok") == 1) & F.col("arc_snr").isNotNull()).cast("int")
)
```

**LLM's Implementation:**
```python
metrics_ok = (
    (F.col("cutout_ok") == F.lit(1)) &
    F.col("arc_snr").isNotNull() &
    (F.col("psfsize_r").isNotNull()) & (F.col("psfsize_r") > F.lit(0.0)) &
    (F.col("psfdepth_r").isNotNull()) & (F.col("psfdepth_r") > F.lit(0.0))
)
```

**Gap:** ⚠️ LLM's version is stricter - requires non-null AND positive psfsize_r and psfdepth_r.

**Impact:** Our validation showed 100% valid (n_valid_all = n_attempt = 5,327,834). This suggests all our injections already have valid psfsize_r/psfdepth_r, so no practical impact. However, for publication defensibility, we should adopt the stricter definition.

---

### 3. Binning Definition

**Our Implementation:**
```python
df = df.withColumn("psf_bin", F.floor(F.col("psfsize_r") / F.lit(psf_bin_width)).cast("int"))
df = df.withColumn("depth_bin", F.floor(F.col("psfdepth_r") / F.lit(depth_bin_width)).cast("int"))
```

**LLM's Implementation:**
```python
inj = inj.withColumn("psf_bin", F.round(F.floor(F.col("psfsize_r") / F.lit(psf_w)) * F.lit(psf_w), 2))
inj = inj.withColumn("depth_bin", F.round(F.floor(F.col("psfdepth_r") / F.lit(depth_w)) * F.lit(depth_w), 2))
```

**Difference:** 
- Ours: Integer bin index (e.g., 12 means 1.2")
- LLM's: Actual value (e.g., 1.2 arcsec)

**Gap:** Minor - both work, LLM's is more self-documenting in output tables.

---

### 4. Resolution Bin Edges

**Our Implementation:** Hardcoded
```python
.when(F.col("theta_over_psf") < 0.4, F.lit("<0.4"))
.when(F.col("theta_over_psf") < 0.6, F.lit("0.4-0.6"))
...
```

**LLM's Implementation:** Configurable via CLI
```python
p.add_argument("--resolution-edges", default="0.0,0.4,0.6,0.8,1.0,inf")
```

**Gap:** ⚠️ For threshold sensitivity analysis, we need configurable edges.

---

### 5. Group By Columns

**Our Implementation:**
```python
grp_cols = [
    "region_split", "region_id", "selection_set_id", "ranking_mode",
    "theta_e_arcsec", "src_dmag", "src_reff_arcsec",
    "psf_bin", "depth_bin", "resolution_bin",
]
```

**LLM's Implementation:**
```python
group_cols = [
    "region_id", "region_split", "selection_set_id", "ranking_mode",
    "theta_e_arcsec", "src_dmag", "src_reff_arcsec",
    "psf_bin", "depth_bin", "resolution_bin",
]
```

**Verdict:** ✅ Identical (order doesn't matter).

---

### 6. Performance Optimization

**Our Implementation:**
```python
df.persist(StorageLevel.MEMORY_AND_DISK)
# ... aggregations ...
df.unpersist()
```

**LLM's Implementation:** No explicit persist/unpersist.

**Verdict:** ✅ We are more optimized.

---

## Scientific Review - Gap Analysis

### LLM's Key Scientific Points

| Point | LLM Claim | Our Status | Action |
|-------|-----------|------------|--------|
| Selection = seeing wall | θ/PSF ≥ 0.8 creates binary selection | Confirmed | Document in paper |
| SNR threshold not contributing | SNR >> 5 for most | Confirmed (median 31.7) | Report both dimensions |
| Only θ_E=1" recoverable | 0.3" and 0.6" all fail | Confirmed | Consider better-seeing bricks |
| Region variance method flawed | Dominated by tiny bins | ⚠️ Needs fix | Add min sample size |
| Need threshold sensitivity | Sweep 0.6/0.7/0.8/0.9 | Not done | Add to Phase 6 |
| Need smooth completeness curve | Fit logistic vs θ/PSF | Not done | Add to Phase 6 |

---

## Action Items (Prioritized)

### Must-Do Before Publication (Blocking)

| # | Action | Owner | Effort |
|---|--------|-------|--------|
| 1 | Add minimum sample size filter (n_valid ≥ 50, n_regions ≥ 5) to region variance summary | Code | 30 min |
| 2 | Adopt stricter validity definition (require psfsize_r > 0, psfdepth_r > 0) | Code | 15 min |
| 3 | Document that completeness is "resolvability proxy", not detection completeness | Doc | 1 hr |
| 4 | Verify end-to-end: run Phase 5 model, define selection on model score | Phase 5/6 | Days |

### Should-Do (Strengthens Paper)

| # | Action | Owner | Effort |
|---|--------|-------|--------|
| 5 | Make resolution edges configurable via CLI | Code | 30 min |
| 6 | Add threshold sensitivity plots (0.6/0.7/0.8/0.9) | Analysis | 2 hr |
| 7 | Fit smooth logistic completeness vs θ/PSF | Analysis | 2 hr |
| 8 | Adopt pure Spark SQL Wilson CI (no UDF) | Code | 30 min |
| 9 | Investigate topk_density_k100 outlier (3.6% vs 10%) | Analysis | 1 hr |
| 10 | Report depth trends conditional on PSF bin | Analysis | 1 hr |

### Nice-to-Have

| # | Action |
|---|--------|
| 11 | Expand injection grid beyond {0.3, 0.6, 1.0}" |
| 12 | Validate injected morphologies against known lenses |
| 13 | Consider hierarchical beta-binomial model |

---

## Code Verification Summary

### What LLM Got Wrong

1. **"Wilson CI not computed"** - FALSE. Our code has Wilson CI (lines 2705-2734, committed 40c66b5).
2. **"Reproducibility gap"** - FALSE. CI is computed in the pipeline stage, not in analysis.

### What LLM Got Right

1. Validity definition could be stricter
2. Resolution edges should be configurable
3. Region variance summary is problematic (no min sample size)
4. Selection function is dominated by resolvability (θ/PSF), not SNR
5. Need Phase 5/6 for true detection completeness

---

## Outputs Verification

### Schema Comparison

**Our output schema (from analysis report):**
```
region_id, selection_set_id, ranking_mode, theta_e_arcsec, src_dmag, 
src_reff_arcsec, psf_bin, depth_bin, resolution_bin, n_attempt, 
n_valid_all, n_valid_clean, n_recovered_all, n_recovered_clean, 
arc_snr_mean, arc_snr_p50, theta_over_psf_mean, 
completeness_valid_all, completeness_valid_clean, 
completeness_overall_all, completeness_overall_clean, 
valid_frac_all, valid_frac_clean, 
ci_low_valid_all, ci_high_valid_all,      <-- Wilson CI present
ci_low_valid_clean, ci_high_valid_clean,  <-- Wilson CI present
region_split
```

**LLM's expected schema:** Same columns - ✅ Match.

---

## Conclusion

**The LLM reviewed an outdated version of our code.** Our current implementation already has:
- ✅ Wilson confidence intervals
- ✅ Injection-only filtering
- ✅ All/Clean subset stratification
- ✅ Region-level variance output
- ✅ PSF provenance summary
- ✅ DataFrame persistence optimization

**Remaining gaps are minor code improvements and additional analyses for publication-grade completeness**, not fundamental issues with the pipeline.

**Verdict: GO for Phase 5.** Address the action items in parallel with Phase 5 development.

