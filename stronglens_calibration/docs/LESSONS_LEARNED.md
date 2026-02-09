# Lessons Learned: StrongLens Calibration Project

**Purpose:** This document captures every mistake, bug, misconfiguration, and incorrect assumption encountered during development. It serves as a reference to prevent repeating these errors.

**Origin:** This document was created during the Gen5 COSMOS integration phase (Plan B) and updated during the transition to real-image training (Plan C / stronglens_calibration). Many lessons remain relevant because they concern EMR, AWS, Spark, and data pipeline issues that apply to any DR10 processing.

**Date Range:** 2026-02-01 to 2026-02-09  
**Author:** Aditrivaishnavi Balaji

---

## Table of Contents

1. [Code Bugs](#1-code-bugs)
2. [Configuration Errors](#2-configuration-errors)
3. [Command Line Mistakes](#3-command-line-mistakes)
4. [Incorrect Assumptions](#4-incorrect-assumptions)
5. [AI Assistant Mistakes](#5-ai-assistant-mistakes)
6. [Process Failures](#6-process-failures)
7. [Data Quality Issues](#7-data-quality-issues)
8. [Prevention Checklist](#8-prevention-checklist)
9. [Code Review Patterns](#9-code-review-patterns-llm-generated-code)
10. [Sim-to-Real Gap Analysis](#10-sim-to-real-gap-analysis) (Why we switched to real-image training)
11. [Shortcut Detection](#21-critical-core-brightness-shortcut-discovery)

---

## 1. Code Bugs

### 1.1 boto3 Import at Module Level (CRITICAL)

**What happened:**
```python
# At module level in spark_phase4_pipeline_gen5.py
try:
    import boto3
except ImportError:
    boto3 = None  # This becomes None on EMR 6.x driver!
```

**Why it failed:**
- EMR 6.x does NOT have boto3 pre-installed (unlike EMR 7.x)
- The driver evaluates this at module load time → `boto3 = None`
- Spark serializes UDFs capturing this `None` reference
- Executors receive `None` even if bootstrap installed boto3

**Fix:**
```python
# Import INSIDE functions that use it
def _s3_client():
    import boto3  # Fresh import on executor
    return boto3.client("s3")
```

**Lesson:** Never use `try/except` for optional imports at module level in Spark code. Import inside functions.

---

### 1.2 Duplicate Function Definitions (CRITICAL)

**What happened:**
Large Python files had MULTIPLE duplicate function definitions. Python uses the LAST definition, so earlier correct implementations were overwritten by later placeholder versions.

**How we missed it:**
- File was 3000+ lines
- Didn't do systematic duplicate check
- Grep for function name only showed count, not locations

**Fix:**
```bash
# Always check for duplicates before production
grep -n "^def " script.py | sort -t: -k2 | uniq -d -f1
```

**Lesson:** Large files accumulate cruft. Always search for duplicate function definitions before deployment.

---

### 1.3 Variable Referenced Before Assignment

**What happened:**
```python
if theta_e > 0:
    source_mode = getattr(args, 'source_mode', 'sersic')
    # ... injection code ...

# Later, OUTSIDE the if block:
cosmos_idx = cosmos_idx if source_mode == "cosmos" else None  # ERROR!
```

**Why it failed:**
Control samples have `theta_e = 0`, so `source_mode` was never defined.

**Fix:**
Move variable initialization BEFORE the conditional:
```python
source_mode = getattr(args, 'source_mode', 'sersic')  # BEFORE if block
```

**Lesson:** Initialize all variables that might be used outside conditional blocks.

---

### 1.4 _parse_s3 Not Handling s3a:// URIs

**What happened:**
```python
# Original regex only matched s3://
match = re.match(r"^s3://([^/]+)/?(.*)$", uri)
# Failed on s3a://darkhaloscope/...
```

**Fix:**
```python
match = re.match(r"^s3[a]?://([^/]+)/?(.*)$", uri)  # Handle both s3:// and s3a://
```

**Lesson:** Spark uses `s3a://` for Hadoop compatibility. Handle both URI schemes.

---

## 2. Configuration Errors

### 2.1 Wrong AWS Region (CRITICAL)

**What happened:** Launched EMR in `us-east-1` but all data/quotas were in `us-east-2`.

**Wasted time:** ~2 hours, multiple cluster launches

**Fix:** Always specify `--region us-east-2` explicitly.

**Lesson:** Never rely on default region. Always be explicit.

---

### 2.2 Missing/Wrong Subnet

**What happened:** Used subnet from `us-east-1` when launching in `us-east-2`.

**Error:** `The subnet subnet-01e8e1839aabcdb77 does not exist`

**Fix:** List subnets in correct region first:
```bash
aws ec2 describe-subnets --region us-east-2 --query 'Subnets[*].SubnetId'
```

**Lesson:** Subnets are region-specific. Verify before use.

---

### 2.3 Spark Memory Settings Too High

**What happened:** Requested 8GB executor memory on m5.xlarge (only 16GB total, ~7GB available after YARN overhead).

**Result:** No executors could spawn. Job hung indefinitely.

**Fix:** Match memory to instance size:
- m5.xlarge (16GB): executor-memory ≤ 4g
- m5.2xlarge (32GB): executor-memory ≤ 10g

**Lesson:** Calculate available memory = (instance_memory - 4GB_for_OS) / executors_per_node

---

### 2.4 Missing NUMBA_CACHE_DIR

**What happened:**
```
RuntimeError: cannot cache function 'rotate': no locator available for 
file '/usr/local/lib/python3.9/site-packages/lenstronomy/Util/util.py'
```

**Why it failed:** Numba tried to write cache to read-only system directory.

**Fix:** In bootstrap AND Spark config:
```bash
# Bootstrap
mkdir -p /tmp/numba_cache && chmod 777 /tmp/numba_cache
export NUMBA_CACHE_DIR=/tmp/numba_cache

# Spark submit
--conf spark.executorEnv.NUMBA_CACHE_DIR=/tmp/numba_cache
--conf spark.yarn.appMasterEnv.NUMBA_CACHE_DIR=/tmp/numba_cache
```

**Lesson:** JIT compilers (Numba, etc.) need writable cache directories.

---

## 3. Command Line Mistakes

### 3.1 --bands Argument Parsed Incorrectly

**What happened:**
```bash
# WRONG - commas become separate arguments
--bands g,r,z  # Parsed as --bands g, then r and z as positional args

# ERROR: unrecognized arguments: r z
```

**Fix:**
```bash
# Quote the value
--bands "g,r,z"
# OR use in EMR Args array:
"--bands","g,r,z"  # As separate array elements
```

**Lesson:** In EMR step Args, each comma-separated value needs careful quoting.

---

### 3.2 spark-submit Incorrectly in Args

**What happened:**
```bash
Args=[spark-submit, --deploy-mode, cluster, ...]
```

**Error:** `File spark-submit does not exist. Please specify --class.`

**Why:** EMR `Type=Spark` steps handle spark-submit internally.

**Fix:** Remove `spark-submit` from Args when using `Type=Spark`:
```bash
Type=Spark,Args=[--deploy-mode,cluster,s3://bucket/script.py,...]
```

**Lesson:** EMR step types (`Spark`, `Hive`, etc.) have implicit launchers.

---

## 4. Incorrect Assumptions

### 4.1 "EMR 6.x Has boto3 Pre-installed"

**Reality:** EMR 7.x has boto3, EMR 6.x does NOT.

**Impact:** Spent hours debugging "boto3 not available" errors.

**Lesson:** Don't assume package availability. Bootstrap should install ALL dependencies explicitly.

---

### 4.2 "test-limit Will Be Fast"

**Reality:** `--test-limit 50` still read the ENTIRE 12M-row manifest before applying limit.

**Impact:** "Smoke tests" took 30+ minutes just to start processing.

**Fix:** Modified code to read single partition file for smoke tests.

**Lesson:** `.limit(N)` in Spark doesn't reduce data scanned, only output. Use partition pruning.

---

### 4.3 "Smoke Test Validates the Pipeline"

**Reality:** The "smoke test" was a standalone Python script, not the actual Spark pipeline.

**Impact:** Bugs like `task_id` undefined passed smoke test but failed in production.

**Lesson:** Smoke tests MUST run the actual code path with `--test-limit`.

---

### 4.4 "Data Has No NaN Values"

**Reality:** 0.08% of stamps had NaN in raw pixel data, but `cutout_ok=1`.

**Impact:** Training produced `loss=nan` starting at batch 300.

**Fix:** Added NaN filter in training data loader:
```python
if np.isnan(x).any() or np.isinf(x).any():
    continue
```

**Lesson:** Always validate data quality before training. Never assume clean data.

---

### 4.5 "Train/Val/Test Split Is 70/15/15"

**Reality:** Split was actually 26/39/35 (inverted!).

**Impact:** Training had less data than evaluation sets.

**Fix:** Implemented hash-based split relabeling in compaction step.

**Lesson:** Always verify actual split proportions, not assumed ones.

---

## 5. AI Assistant Mistakes

### 5.1 Declaring Victory Prematurely

**What happened:** Multiple times said "job launched successfully" or "fix applied" without verifying:
- Job actually ran to completion
- Output was correct
- No errors in logs

**Examples:**
- "EMR cluster launched!" → but it failed in bootstrap
- "Training started!" → but loss went to NaN
- "Bug fixed!" → but code wasn't uploaded to S3

**Lesson:** Never declare success until verified by:
1. Job COMPLETED (not just RUNNING)
2. Output exists and is valid
3. Logs show no errors

---

### 5.2 Not Checking Logs Immediately

**What happened:** EMR jobs failed but didn't check logs until user asked.

**Impact:** Wasted time waiting for jobs we could have known were failing.

**Lesson:** After launching any job, immediately:
1. Check cluster state
2. Check step state
3. If FAILED, immediately fetch logs

---

### 5.3 Not Validating Code Before Upload

**What happened:** Uploaded code to S3 without verifying:
- No syntax errors
- No duplicate functions
- All imports work

**Lesson:** Before uploading to S3:
```bash
python -m py_compile script.py
grep -c "^def func_name" script.py  # Check for duplicates
```

---

## 6. Process Failures

### 6.1 No Local Testing Before EMR

**What happened:** Deployed directly to EMR without local Spark testing.

**Impact:** Each iteration took 20-30 minutes (cluster startup + failure).

**Fix:** Set up local Spark with S3 support for rapid iteration:
```bash
# Local spark with S3
spark-submit --packages org.apache.hadoop:hadoop-aws:3.3.4 \
  --conf spark.hadoop.fs.s3a.impl=org.apache.hadoop.fs.s3a.S3AFileSystem \
  script.py --test-limit 50
```

**Lesson:** Local testing saves hours of EMR iteration time.

---

### 6.2 Not Verifying S3 Code Matches Local

**What happened:** Fixed code locally but forgot to upload to S3.

**Impact:** EMR ran old, buggy code.

**Fix:** Always verify after upload:
```bash
aws s3 cp s3://bucket/code/script.py - | md5sum
md5sum local/script.py
```

**Lesson:** S3 upload must be verified, not assumed.

---

## 7. Data Quality Issues

### 7.1 NaN Values in Raw Stamps

**Cause:** Unknown - possibly failed coadd reads or processing errors.

**Detection:** Training loss went to NaN at batch 300.

**Prevalence:** 0.08% of samples (180 out of 230k checked).

**Fix:** Filter in data loader.

**Prevention:** Add data validation in cutout generation:
```python
if np.isnan(stamp).any():
    return Row(cutout_ok=0, physics_warnings="NaN in stamp")
```

---

### 7.2 Missing Bands in Some Stamps

**Cause:** Some stamps saved with only r-band (bandset="r").

**Detection:** `KeyError: 'image_g is not a file in the archive'`

**Fix:** Training script handles missing bands gracefully:
```python
def decode_stamp_npz(npz_bytes):
    with np.load(bio) as npz:
        g = npz.get("image_g", np.zeros((64,64)))
        # ...
```

---

## 8. Prevention Checklist

### Before Writing Code
- [ ] Understand the library's expected units and formats
- [ ] Check existing code for similar functions (avoid duplicates)
- [ ] Plan variable scoping for conditionals

### Before Deploying to EMR
- [ ] Run `python -m py_compile script.py`
- [ ] Search for duplicate function definitions
- [ ] Test locally with `--test-limit`
- [ ] Verify all required arguments in `--help`
- [ ] Confirm code uploaded to S3 matches local

### When Launching EMR
- [ ] Specify `--region us-east-2` explicitly
- [ ] Verify subnet exists in target region
- [ ] Match memory settings to instance type
- [ ] Set NUMBA_CACHE_DIR for JIT libraries
- [ ] Include ALL dependencies in bootstrap

### After Launching
- [ ] Wait for RUNNING state (not just STARTING)
- [ ] Check step status within 5 minutes
- [ ] If FAILED, immediately fetch logs
- [ ] Don't declare success until output verified

### Before Training
- [ ] Check data for NaN/Inf values
- [ ] Verify split proportions
- [ ] Sample and visualize a few examples
- [ ] Start with small epoch count to verify convergence

### When Fixing Bugs
- [ ] Read the EXACT error message
- [ ] Fix root cause, not symptoms
- [ ] Verify fix locally before redeploying
- [ ] Upload fixed code AND verify upload
- [ ] Document the fix for future reference

---

## Summary Statistics

| Category | Count |
|----------|-------|
| Code bugs | 7 major |
| Config errors | 4 |
| Command line mistakes | 3 |
| Wrong assumptions | 6 |
| AI mistakes | 5 |
| Process failures | 3 |
| Data quality issues | 3 |

**Estimated time wasted:** 15-20 hours across the project

**Key insight:** Most issues could have been caught with:
1. Local testing before EMR
2. Reading error messages carefully
3. Verifying uploads and outputs
4. Not assuming—always checking

---

## 9. Code Review Patterns (LLM-Generated Code)

When reviewing LLM-generated code, apply these lessons:

### 9.1 Data Format Incompatibility

**Pattern:** LLM generates code with assumed data formats that don't match actual data.

**Example:**
```python
# LLM assumed format
arr = z["img"]  # Single key

# Our actual format
g = z["image_g"]
r = z["image_r"]
z_band = z["image_z"]
```

**Lesson:** Always verify input/output formats against actual data before integrating.

### 9.2 Precision Loss (float16 vs float32)

**Pattern:** Using float16 to save space causes precision issues.

**Lesson:** Use float32 for scientific data. Only use float16 after explicit validation.

### 9.3 Missing Unit Documentation

**Pattern:** Generated code doesn't document units (flux/pixel vs flux/arcsec²).

**Lesson:** Always document:
- Input units expected
- Output units produced
- Conversion factors used

### 9.4 Relative Import Issues in Distributed Systems

**Pattern:** `from ..module import X` works locally but fails on distributed workers.

**Lesson:** Test imports on actual execution environment (EMR executors).

---

## 10. Sim-to-Real Gap Analysis

**This section documents why we switched from simulation-based training (Plan B) to real-image training (Plan C / stronglens_calibration).**

### 10.1 Signal Strength Mismatch (CRITICAL)

**What happened:**
Gen5 model achieved AUC=0.9945 on synthetic test data but only **4.4% recall** on real SLACS/BELLS anchor lenses.

**Root Cause:**
| Metric | Training Injections | Real SLACS/BELLS |
|--------|---------------------|------------------|
| Mean central r-band max | 0.0878 | 0.0099 |
| 50th percentile | 0.0609 | 0.0085 |

**Our synthetic injections were ~9x brighter than real SLACS/BELLS lenses in DR10 imaging.**

### 10.2 SLACS/BELLS: Wrong Anchor Set

**Key insight:** SLACS/BELLS lenses were discovered via **spectroscopy + HST follow-up**, not ground-based imaging. Their arcs are often invisible in DR10.

**Conclusion:** We needed lenses discovered in ground-based imaging. This led to using the lenscat catalog of DESI imaging candidates.

### 10.3 Model Learned Shortcuts, Not Physics

**Proof:** Adding a simple synthetic ring pattern to ANY image made the model score it 1.0.

**Root cause:** Model learned "bright center = lens" because training data had this pattern.

**Solution:** Switch to real-image training where positives and negatives come from the same distribution (real DR10 cutouts).

---

## 21. CRITICAL: Core Brightness Shortcut Discovery

### What We Found

Following the external LLM's recommendation to add "shortcut gates", we discovered catastrophic failures in the simulation-based approach:

| Gate | Result | Key Metric |
|------|--------|------------|
| Gate 1.6 (Core-Only Baseline) | **FAIL** | AUC=0.98 with only r<10px! |
| Gate 1.8 (Core Brightness) | **FAIL** | Positives 64% brighter in core |

### Root Cause

Positives have 64% brighter cores than controls due to arc flux overlapping with the PSF-blurred galaxy center.

### Lesson Learned

**Passing quality gates ≠ Learning correct features**

You must add shortcut-detection gates:
- Core-only baseline (can a simple classifier use only the center?)
- Arc-suppressed test (does prediction drop when arcs removed?)
- Core brightness matching (are classes matched in core brightness?)

### Prevention

Before ANY training:

```python
# Gate 1.6: Core-only baseline - MUST be near random (AUC < 0.55)
clf = LogisticRegression()
clf.fit(X_core_only, y)
assert roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1]) < 0.55

# Gate 1.8: Core brightness - MUST match (ratio 0.95-1.05)
core_pos = compute_core_brightness(positives)
core_ctrl = compute_core_brightness(controls)
assert 0.95 < core_pos.mean() / core_ctrl.mean() < 1.05
```

---

## Updated Prevention Checklist (After Shortcut Discovery)

Before declaring any model generation successful:

### Data Quality Gates
- [ ] Compare raw pixel statistics: training vs anchor
- [ ] Verify quality columns matched between classes
- [ ] Confirm bandset consistency
- [ ] Check arc_snr distribution covers near-threshold cases

### Shortcut Detection Gates (CRITICAL)
- [ ] **Gate 1.5:** Normalization stats matched (Cohen's d < 0.1 for all)
- [ ] **Gate 1.6:** Core-only baseline AUC < 0.55 (near random)
- [ ] **Gate 1.7:** Arc-suppressed mean p < 0.3 
- [ ] **Gate 1.8:** Core brightness ratio 0.95-1.05

### Evaluation
- [ ] Verify anchor set was discovered using same detection method as training target
- [ ] Use central aperture brightness, not full-stamp mean
- [ ] Report Tier-A (primary) and Tier-B (stress) separately
- [ ] Use region-disjoint splits (HEALPix) for publication

---

*Last updated: 2026-02-09*
