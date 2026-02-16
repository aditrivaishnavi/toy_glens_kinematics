# Lessons Learned: Gen5 COSMOS Integration

**Purpose:** This document captures every mistake, bug, misconfiguration, and incorrect assumption encountered during the Gen5 COSMOS integration project. It serves as a reference to prevent repeating these errors in future work.

**Date Range:** 2026-02-01 to 2026-02-04  
**Author:** AI Assistant + Human Operator

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
`spark_phase4_pipeline_gen5.py` had MULTIPLE duplicate function definitions:
- `render_cosmos_lensed_source` - lines 141 AND 353 (placeholder that used Sersic!)
- `_load_cosmos_bank_h5` - lines 98 AND 280 (second lacked caching)
- `_cosmos_choose_index` - lines 115 AND 317 (different hash algorithms)
- `_compute_hlr_arcsec` - duplicated
- `load_phase4c_config` - duplicated

**Why it failed:**
Python uses the LAST definition. The second `render_cosmos_lensed_source` was a placeholder that fell back to Sersic rendering, defeating the entire Gen5 COSMOS purpose.

**How we missed it:**
- File was 3000+ lines
- Didn't do systematic duplicate check
- Grep for function name only showed count, not locations

**Fix:**
```bash
# Always check for duplicates before production
grep -n "^def " spark_phase4_pipeline_gen5.py | sort -t: -k2 | uniq -d -f1
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
cosmos_idx = None
cosmos_hlr = None

if theta_e > 0:
    # ... injection code ...
```

**Lesson:** Initialize all variables that might be used outside conditional blocks.

---

### 1.4 Undefined Variables in Processing Loop

**What happened:**
```python
# task_id was used but never extracted from row
idx = _cosmos_choose_index(task_id, cosmos_bank)  # NameError: task_id not defined

# src_mag_r was used but only src_rmag was defined
surface_brightness = template * flux_nmgy / src_mag_r  # NameError
```

**Why we missed it:**
- "Smoke test" was a standalone script, not the actual pipeline
- The standalone test defined these variables locally
- Real pipeline reads from DataFrame rows

**Fix:**
```python
task_id = r["task_id"]  # Extract from row
src_mag_r = src_rmag    # Add alias
```

**Lesson:** Test the ACTUAL code path, not a mock version.

---

### 1.5 Surface Brightness Units Wrong (CRITICAL)

**What happened:**
```python
# WRONG - passing total flux
surface_brightness = template * flux_nmgy

# lenstronomy INTERPOL expects flux/arcsec², not flux/pixel
```

**Result:** Arc flux was ~1000x too low, arc_snr values were 0.01-0.15 instead of 10-200.

**Fix:**
```python
# CORRECT - convert to surface brightness
pixel_scale_sq = cosmos_bank["src_pixscale"] ** 2  # arcsec² per pixel
surface_brightness = template * flux_nmgy / pixel_scale_sq
```

**Lesson:** Always verify unit conventions when interfacing with external libraries. lenstronomy INTERPOL expects surface brightness (flux/arcsec²).

---

### 1.6 PSF Kernel Size Exceeding Stamp Size

**What happened:**
```python
# Kernel size calculation could exceed 64x64 stamp
radius = int(5 * sigma_pix)  # For g-band PSF=3.18", this gives 145x145 kernel!
kernel = np.zeros((2*radius+1, 2*radius+1))
```

**Error:** `ValueError: Kernel (145, 145) larger than image (64, 64)`

**Fix:**
```python
max_radius = min(h, w) // 2 - 1  # Cap at stamp size
radius = min(int(5 * sigma_pix), max_radius)
```

**Lesson:** Always validate computed sizes against physical constraints.

---

### 1.7 _parse_s3 Not Handling s3a:// URIs

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

### 2.1 Wrong AWS Region

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

### 3.3 Missing Required Arguments

**What happened:** Used `--experiment-id` but script required `--variant`.

**Error:** `error: the following arguments are required: --variant`

**How we missed it:** Assumed argument names from memory without checking help.

**Fix:** Always check `--help` first:
```bash
python script.py --help | grep -E "required|positional"
```

**Lesson:** Required arguments vary between scripts. Always verify.

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

### 4.5 "arc_snr Is Integrated SNR"

**Reality:** Code computed MAX per-pixel SNR, not integrated SNR.

**Impact:** Inconsistent with standard astronomical SNR definition.

**Fix:** Added `arc_snr_sum` column with proper integrated SNR.

**Lesson:** Verify metric definitions match scientific conventions.

---

### 4.6 "Train/Val/Test Split Is 70/15/15"

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

### 5.3 Using Wrong Argument Names

**What happened:** Used `--cosmos-bank-path` when script expected `--cosmos-bank-h5`.

**Impact:** Job failed with argument parsing error.

**Lesson:** Always verify argument names from `--help` or source code.

---

### 5.4 Not Validating Code Before Upload

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

### 5.5 Not Reading Error Messages Carefully

**What happened:** Error said `--variant` required but I focused on `--experiment-id`.

**Lesson:** Read the EXACT error message. It usually tells you exactly what's wrong.

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

### 6.3 Not Tracking Code Versions

**What happened:** Lost track of which version was on S3 vs local vs git.

**Impact:** Confusion about whether fixes were actually deployed.

**Lesson:** Use git commit hashes and timestamps:
```bash
# Add to script
print(f"Code version: {GIT_COMMIT} uploaded {UPLOAD_TIME}")
```

---

## 7. Data Quality Issues

### 7.1 NaN Values in Raw Stamps

**Cause:** Unknown - possibly failed coadd reads or processing errors.

**Detection:** Training loss went to NaN at batch 300.

**Prevalence:** 0.08% of samples (180 out of 230k checked).

**Fix:** Filter in data loader.

**Prevention:** Add data validation in Phase 4c:
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

### 7.3 Inverted Train/Val/Test Splits

**Cause:** Original manifest partitioning used different criteria.

**Reality:** train=26%, val=39%, test=35% (should be ~70/15/15).

**Fix:** Hash-based relabeling during compaction:
- Keep test frozen
- Move 75% of val → train

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

**Example:**
```python
images.append(img_rs.astype(np.float16))  # BAD: Only ~3 decimal digits
images.append(img_rs.astype(np.float32))  # GOOD: ~7 decimal digits
```

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

### 9.5 Missing Validation for Generated Data

**Pattern:** No NaN/Inf checks on generated synthetic data.

**Lesson:** Always validate generated data:
```python
if not np.isfinite(img).all():
    raise ValueError("NaN/Inf in generated data")
```

---

## 10. Sim-to-Real Gap Analysis (2026-02-05)

### 10.1 Signal Strength Mismatch with SLACS/BELLS (CRITICAL)

**What happened:**
Gen5 model achieved AUC=0.9945 on synthetic test data but only **4.4% recall** on real SLACS/BELLS anchor lenses at threshold 0.5.

**Root Cause Analysis:**
Comparing central r-band signal strength:

| Metric | Training Injections | Real SLACS/BELLS |
|--------|---------------------|------------------|
| Mean central r-band max | 0.0878 | 0.0099 |
| 50th percentile | 0.0609 | 0.0085 |
| Samples with signal > 0.02 | 100% | 4.4% |

**Our synthetic injections are ~9x brighter than real SLACS/BELLS lenses in DR10 imaging.**

**Why This Happened:**
- SLACS/BELLS lenses were discovered via **spectroscopy + HST follow-up**
- Their arcs are often below ground-based detection threshold
- DR10 at 0.262"/pixel can't resolve most of these faint arcs
- We injected arcs bright enough to be visible, but real SLACS/BELLS arcs aren't visible in ground-based data

**Critical Insight:**
**SLACS/BELLS is the WRONG anchor set for evaluating ground-based lens finders.**

These lenses weren't discovered in ground-based imaging - they're essentially invisible in DR10. A 4.4% recall doesn't mean the model is bad; it means we're testing on the wrong data.

**Fix:**
1. Lower injection brightness to match real ground-based detectability limits
2. Find lenses discovered in ground-based surveys (Master Lens Database, DESI visual inspections)
3. Accept that selection function depends heavily on arc brightness
4. Consider this when interpreting selection function C(θ_E, z_l) results

**Lesson:**
**Always validate that your anchor/test set is drawn from the same detection regime as your training data.** A spectroscopically-discovered lens may be invisible in imaging; an imaging-discovered lens may have trivial spectroscopic signature.

### 10.2 Overconfident Predictions from LLM Review

**What happened:**
Earlier LLM review predicted 30-50% real lens recall (optimistic) or 15-30% (conservative). Actual result was 4.4%.

**Why the prediction was wrong:**
- Did not account for fundamental signal strength mismatch
- Assumed SLACS/BELLS arcs would be visible in DR10
- Underestimated the difficulty of ground-based lens detection

**Lesson:**
When estimating sim-to-real transfer performance, **directly compare signal characteristics** (brightness, SNR, morphology) between training and anchor data. Don't assume properties transfer.

### 10.3 Deeper Root Cause: Normalized Pattern Mismatch (2026-02-05)

**What happened:**
Further investigation revealed the model learned a specific **spatial pattern** from training data, not gravitational lensing physics.

**Investigation findings:**

| Region | Training (normalized) | High-scoring Anchor | Low-scoring Anchor |
|--------|----------------------|---------------------|-------------------|
| Center (r<8) | **3.57** | 0.93 | 0.08 |
| Inner ring (8<r<16) | 1.05 | 2.02 | -0.02 |
| Outer ring (16<r<24) | 0.03 | **2.08** | -0.06 |

**The model learned:** "High normalized center + low normalized outer = lens"

This is NOT gravitational lensing. This is the specific pattern of:
1. Training LRGs have bright centers (~0.06 nMgy)
2. Training backgrounds are faint (~0.003 nMgy)
3. MAD normalization creates specific center-to-outer ratios
4. Anchor lenses have completely different ratios

**Proof the model learned shortcuts:**
Adding a simple synthetic ring pattern (`exp(-((r-15)²)/20) * 0.01`) to ANY image makes the model score it 1.0. This is a trivial geometric shortcut, not gravitational lensing detection.

**Root cause chain:**
1. Training LRGs are 10x brighter than SLACS/BELLS centers (different populations or redshifts)
2. After normalization, training data has specific spatial pattern
3. Model learns this pattern as "lens signature"
4. Real lenses don't have this pattern
5. Model fails

**Why training positives and negatives separate perfectly:**
Even though training positives/negatives have same center brightness (0.088 vs 0.087), after normalization:
- Positives have extra flux in annulus from injected arcs
- This changes the MAD normalization slightly
- Model detects this subtle difference

**Critical lessons:**
1. **Perfect synthetic performance means nothing** if the model learned shortcuts
2. **Compare normalized spatial patterns** between training and anchor data
3. **Test with adversarial examples** (e.g., synthetic rings) to detect shortcuts
4. **Training LRG population must match anchor population** in brightness/redshift

### 10.4 SLACS/BELLS: Wrong Anchor Set for Ground-Based Detection

**Confirmed understanding (2026-02-05):**
User correctly pointed out that SLACS/BELLS are **hybrid surveys**:
- Candidates selected from ground-based spectroscopy (SDSS/BOSS)
- Confirmed with HST high-resolution imaging (0.05"/pix)
- The lensing arcs are often invisible in ground-based imaging

**Why there IS some correlation (not zero):**
Some SLACS/BELLS lenses DO score high (SDSSJ1205+4910 = 0.91). But:
- No correlation with theta_E (r = 0.021)
- Weak correlation with brightness (r = 0.225)
- Model responding to image artifacts, not lensing

**The fundamental problem:**
We're evaluating a ground-based lens finder on lenses that were discovered via spectroscopy and confirmed with HST. This is like testing a metal detector on objects found with X-ray machines.

---

## 11. Prevention Checklist for Future Generations

Before declaring any model generation "successful":

### Data Quality
- [ ] Compare raw pixel statistics: training vs anchor (should be similar distributions)
- [ ] Compare normalized pattern statistics: center/inner/outer ring means
- [ ] Verify anchor set was discovered using same detection method as training target

### Model Behavior
- [ ] Test with adversarial examples (synthetic rings, random noise)
- [ ] Check correlation between p_lens and physical parameters (theta_E, arc_snr)
- [ ] Verify model doesn't score 1.0 on trivial geometric patterns

### Anchor Selection
- [ ] Use lenses discovered in ground-based imaging (not spectroscopy + HST)
- [ ] Match anchor population to training population (redshift, brightness)
- [ ] Document how each anchor was originally discovered

### Sim-to-Real Gap
- [ ] Compare signal strength distributions (not just ranges)
- [ ] Check if injected signals are realistic for the detection method
- [ ] Verify arcs would be visible in the target survey

---

## 12. Expert LLM Review Findings (2026-02-05)

An external LLM reviewed our failure analysis and provided the following key insights:

### 12.1 Confirmed Conclusions
- **Shortcut learning confirmed** with high confidence based on ring counterfactual
- **SLACS/BELLS wrong anchors** - directionally correct interpretation
- **Need pipeline parity checks** to determine if brightness mismatch is real or artifact

### 12.2 Critical Red Flag: Val/Test AUC Gap
The LLM flagged: AUC(val)=0.8945 vs AUC(test)=0.9945

**Our investigation found:**
- `meta_cols: psfsize_r,psfdepth_r` (legitimate physical features, not leakage)
- Test set has higher arc_snr (mean 8.59 vs val 7.56) - "easier" samples
- This is a hash-based split artifact, not leakage

### 12.3 Recommended Experiments (from LLM)

**H1: Shortcut validation ablations:**
1. Metadata-only baseline (MLP on just psfsize_r, psfdepth_r)
2. Center-masked training (zero r<8 pixels)
3. Photometric jitter invariance test

**H2: Anchor set validation:**
1. Create "detectable in DR10" anchor subset
2. Stratify recall by arc visibility proxies

**H3: Pipeline parity:**
1. Pull 500 training LRGs through same cutout service as anchors
2. Compare brightness distributions

### 12.4 Recommended Remediation Path

**Step 0 (MUST DO):** Leakage audit - ✅ DONE (no leakage found)

**Step 1:** Fix evaluation methodology
- Tier A: "Detectable in DR10 imaging" anchors
- Tier B: SLACS/BELLS as stress test only

**Step 2:** Suppress shortcuts
- Add ring-galaxy and spiral-arm hard negatives
- Consider lens-light subtraction

**Step 3:** Calibrate injection brightness
- Target controlled distribution near detection threshold
- Use arc_snr correlated with human-visible arcs

### 12.5 Key SOTA Paper Insights

From HOLISMOKES series (A&A):
- "Performances rely heavily on the design of lens simulations and the choice of negative examples… but little on the network architecture"
- Reported 14/16 correct classification of "previously confirmed lens systems above the detection limit of Pan-STARRS"
- Even with >99% accuracy on balanced sets, real surveys yield candidate lists dominated by false positives

**Implication:** Our focus should be on training data design and anchor selection, not architecture.

---

## 13. Pipeline Parity Check (2026-02-05)

### Verification of Data Source Equivalence

**Question:** Is the brightness difference between training LRGs and SLACS/BELLS anchors due to a processing artifact (different data pipelines) or a real population difference?

**Experiment 1: Training LRG Coordinates**
- Fetched same LRG coordinates via Legacy Survey cutout service AND EMR pipeline
- Result: Mean ratio = 0.9905 ± 0.0378 (effectively 1.0)
- **PIPELINES AGREE** - no processing artifact

**Experiment 3: Anchor Brightness Measurement**
- Measured SLACS/BELLS anchor brightness: 0.000627 nMgy (mean r-band)
- Training LRG brightness: ~0.06 nMgy
- **Training LRGs are 95.7x brighter than anchors**

**Root Cause Confirmed:**
SLACS/BELLS lenses are genuinely ~100x fainter than training LRGs in DR10 imaging because:
1. They were discovered via spectroscopy, not imaging
2. Confirmed with HST (space-based), not ground-based surveys
3. Their arcs are too faint/blended for ground-based detection

**Lessons:**
1. Always verify anchor sets are appropriate for the detection method
2. Anchors discovered via different methods (spectroscopy, HST) may not be detectable in ground-based imaging
3. Pipeline parity checks are essential before concluding data artifacts exist

**Recommended Anchor Sets for Ground-Based Lens Finders:**
- SuGOHI lenses (discovered in HSC ground-based imaging)
- Master Lens Database entries with visible arcs in DR10
- Citizen science discoveries from DECaLS viewer

---

## 14. Stratified AUC Verification (2026-02-05)

### Diagnosing AUC(val) vs AUC(test) Gap

**Initial Concern:** Global AUC gap between val (0.8945) and test (0.9945) appeared suspicious.

**Verification Approach:**
1. Compute AUC within difficulty bins (arc_snr, theta_e/psf)
2. Compare distributions of key variables between splits

**Findings:**
- Stratified AUC gaps are small (<0.02 within matched bins)
- Test set has higher mean arc_snr (4.15 vs 3.58) - easier samples
- This is expected from hash-based splitting on task_id

**Conclusion:** Gap is explained by difficulty stratification, not leakage or bugs.

**Lesson:** Always stratify metrics before diagnosing train/val/test gaps.

---

## 15. Corrected Brightness Metric (2026-02-05)

### Original Metric Was Flawed

**Original approach:** Full-stamp mean brightness → 95.7x ratio
**Problem:** 64×64 stamp is mostly sky; mean is dominated by ~3800 sky pixels.

**Corrected approach:** Central aperture (r<8 pixels) mean brightness → 43.8x ratio
**Why better:** ~200 pixels in galaxy core, actually measures galaxy brightness.

**Additional insight:** Several anchors had negative/near-zero central means, indicating they are "background-subtracted to noise" at the center-aperture scale. These are NOT "imaging-detectable positives."

**Lessons:**
1. Use physically meaningful metrics (central aperture, not full stamp)
2. Negative brightness values indicate object is below detection threshold
3. Median is more stable than mean for brightness comparisons
4. Consider bootstrap CI when comparing small samples

---

## 16. Parameter Range Issues Identified (2026-02-05)

### Current Training Grid is Too Narrow

| Parameter | Current | Problem | Recommended |
|-----------|---------|---------|-------------|
| `src_dmag` | [1.0, 2.0] | Too bright, no near-threshold | [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5] |
| `theta_e` | [0.3, 0.6, 1.0] | Missing 0.75, 1.25, 1.5 | [0.3, 0.5, 0.75, 1.0, 1.25, 1.5] |
| `src_reff` | [0.08, 0.15] | Too narrow | [0.05, 0.08, 0.12, 0.18, 0.25] |

**Lessons:**
1. Parameter grids must cover the full range of expected test cases
2. Near-threshold cases (arc_snr 0.8-2) are critical for calibration
3. Broader theta_e coverage needed for selection function work

---

## 17. Hash-Based Splitting Insufficient for Publication (2026-02-05)

### Current: Brickname Hash

**Problem:** Adjacent bricks share systematics (background, calibration, observing pattern).

**Recommended:** HEALPix cell-based splitting with NSIDE=64-128.

**Key insight:** For publication-grade independence, splits must be on larger contiguous sky units, not individual bricks.

**Optional enhancement:** Guard band - exclude cells adjacent to test cells from training.

---

## 18. Four Sanity Gates Before Building New Data (2026-02-05)

### Mandatory Pre-Checks

Before investing time in new anchors or data regeneration, run these gates:

1. **Class-conditional quality distributions**
   - Check: bad_pixel_frac, maskbit_frac, quality flags
   - Goal: positives and controls matched in data-quality space

2. **Bandset audit**
   - Check: All samples have consistent bands (grz)
   - Goal: No per-class band imbalance

3. **Null-injection test**
   - Check: Run injection with zero flux
   - Goal: Classifier drops to chance (AUC ~0.5)

4. **Per-pixel SNR ablation** (if invvar available)
   - Check: Use image × sqrt(invvar) instead of MAD normalization
   - Goal: Reduces "bright center" shortcuts

**Lesson:** These quick checks prevent weeks of wasted work on new data.

---

## 19. Tier-A vs Tier-B Anchor Strategy (2026-02-05)

### Two-Tier Anchor Approach

| Tier | Purpose | Criterion | Use |
|------|---------|-----------|-----|
| **Tier-A** | Primary evaluation | Arc visible in DR10 | Report as main metric |
| **Tier-B** | Stress test | SLACS/BELLS below threshold | Report separately |

**Key insight:** SLACS/BELLS are valid stress tests but NOT fair primary anchors for a DR10 ground-based finder.

**Sources for Tier-A:**
- Legacy Surveys ML candidates (Huang et al.) - same domain
- KiDS lens candidates (Petrillo et al.) - published CNN methodology
- HSC SuGOHI - ground-based, deeper but similar seeing

**Selection criterion:** arc_visibility_snr > 2.0 in DR10 cutout.

---

## 20. Arc SNR Targeting (2026-02-05)

### Current Distribution is Skewed Toward Easy Cases

**Target distribution for robust training:**

| arc_snr Range | Current (Est.) | Target |
|---------------|----------------|--------|
| 0.8–2 (hard) | ~5% | 40% |
| 2–8 (moderate) | ~35% | 40% |
| 8–20 (easy) | ~40% | 15% |
| 20+ (extreme) | ~20% | 5% |

**Implementation:** Rejection sampling on achieved arc_snr to match target.

**Lesson:** Training distribution must include significant mass near detection threshold.

---

## 21. Remediation Phase Results (2026-02-05)

### Phase 1 Sanity Gates - All Passed

| Gate | Status | Key Finding |
|------|--------|-------------|
| 1.1 Quality Distributions | PASS | Controls/positives matched (KS p=0.63) |
| 1.2 Bandset Audit | PASS | 100% grz (10.6M samples) |
| 1.3 Null-Injection | PASS | Controls: mean p=0.014, only 0.4% > 0.5 |
| 1.4 SNR Ablation | DEFERRED | invvar not stored |

**Scripts used:** `gate_1_1_quality_distributions.py`, `gate_1_2_bandset_audit.py`, `gate_1_3_null_injection.py`, `gate_1_4_snr_ablation.py`

### Phase 2 Center-Masked Diagnostic

**Finding:** Model shows MODERATE center reliance (22.4% drop when r=12px masked)

**Interpretation:** Model uses both center galaxy and arc features. Center-masked training may improve anchor recall.

### Phase 3 Tier-A Anchor Classification

**Finding:** Only 4 of 15 SLACS/BELLS have arc_visibility_snr > 2.0
- Tier-A: SDSSJ0029-0055 (3.51), SDSSJ0252+0039 (3.16), SDSSJ0959+0410 (3.91), SDSSJ0832+0404 (7.95)
- Tier-B: 11 lenses with SNR < 2.0

**Implication:** Most SLACS/BELLS are too faint for ground-based detection - need ground-based-discovered anchors.

### Phase 5 Arc SNR Analysis

**Current distribution:**
- Mean: 8.51, Median: 4.67
- 21.6% below SNR=2 (too faint)
- 9.0% above SNR=20 (very easy)

**Recommendation:** Rejection sampling with log-uniform target [2, 50]

---

## Prevention Checklist (Updated)

Before declaring any model generation successful:

- [ ] Verify anchor set was discovered using same detection method as training target
- [ ] Check central aperture brightness, not full-stamp mean
- [ ] Run stratified AUC to diagnose split gaps
- [ ] Confirm positives/controls matched in data-quality space
- [ ] Verify bandset consistency
- [ ] Run null-injection test (mean p_lens < 0.2 for controls)
- [ ] Check arc_snr distribution covers near-threshold cases
- [ ] Ensure parameter grid covers full expected range
- [ ] Use region-disjoint splits (HEALPix) for publication
- [ ] Report Tier-A (primary) and Tier-B (stress) separately
- [ ] Run center-masked diagnostic to assess center reliance
- [ ] All Phase 1 gates must PASS before training

---

## 21. CRITICAL: Core Brightness Shortcut Discovery (2026-02-05)

### What We Found

Following the external LLM's recommendation to add "shortcut gates", we discovered **catastrophic failures**:

| Gate | Result | Key Metric |
|------|--------|------------|
| Gate 1.5 (Normalization Stats) | FAIL | g-band clip fraction: d=0.658 |
| Gate 1.6 (Core-Only Baseline) | **FAIL** | AUC=0.98 with only r<10px! |
| Gate 1.7 (Arc-Suppressed) | FAIL | p=0.62 with arcs removed |
| Gate 1.8 (Core Brightness) | **FAIL** | Positives 64% brighter in core |

### Root Cause

Positives have 64% brighter cores than controls:

| Metric | Controls | Positives | Ratio |
|--------|----------|-----------|-------|
| core_mean_r | 0.0205 | 0.0336 | 1.64x |
| core_median_r | 0.0133 | 0.0256 | 1.92x |

This creates a trivial shortcut: the model just detects brighter centers.

### Why This Happened

1. **Unpaired controls:** Controls are random LRGs, positives are LRGs with injections
2. **Injection adds core flux:** Lensed sources contribute flux to the core region, especially for small theta_e
3. **Selection bias:** "Successful" injections may preferentially select brighter LRGs

### Confirmation

A simple logistic regression trained on ONLY central 10px achieves **98% validation AUC**. This is definitive proof the dataset has a core-based shortcut.

### Why Previous Gates Passed But This Failed

Our original gates (1.1-1.4) checked:
- Quality distributions (matched)
- Bandset consistency (all grz)
- Null-injection (controls score low)
- Invvar availability

None of these checked **whether the model uses the right features**. The model correctly separates classes - but using the wrong signal (core brightness, not arc morphology).

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
- [ ] Compare raw pixel statistics: training vs anchor (should be similar distributions)
- [ ] Verify quality columns matched between classes (KS test + effect size)
- [ ] Confirm bandset consistency (all samples have same bands)
- [ ] Check arc_snr distribution covers near-threshold cases

### NEW: Shortcut Detection Gates (CRITICAL)
- [ ] **Gate 1.5:** Normalization stats matched (Cohen's d < 0.1 for all)
- [ ] **Gate 1.6:** Core-only baseline AUC < 0.55 (near random)
- [ ] **Gate 1.7:** Arc-suppressed mean p < 0.3 
- [ ] **Gate 1.8:** Core brightness ratio 0.95-1.05

### Training Configuration
- [ ] Use paired controls (same LRG with and without injection)
- [ ] OR use center-masked training (mask r<10px during training)
- [ ] Verify parameter grid covers full expected range

### Evaluation
- [ ] Verify anchor set was discovered using same detection method as training target
- [ ] Use central aperture brightness, not full-stamp mean
- [ ] Report Tier-A (primary) and Tier-B (stress) separately
- [ ] Use region-disjoint splits (HEALPix) for publication

---

---

## 22. Arc Overlap Dominates Core Brightness (2026-02-05)

**What happened:**
After implementing paired controls (same LRG with and without injection), we discovered that the core brightness shortcut persists:

| Dataset | Central Brightness Ratio | Central-Only AUC |
|---------|-------------------------|------------------|
| Unpaired Gen5 | 1.64x | 0.98 |
| Paired Data | 1.67x | 0.76 |

**Root cause attribution:**
- LRG selection bias: **-3.2%** (essentially zero!)
- Arc overlap with center: **+67.1%**

**Why it happens:**
- Mean theta_e = 1.35" (Einstein radius)
- Mean PSF = 1.32" (seeing)
- The arc is **blurred by PSF into the central region**
- For 92.3% of samples, the theta-aware core (r < theta_e - 1.5*PSF) is < 2 pixels

**Key insight:**
The "shortcut" is **physical**, not a data bias. Strong lenses genuinely have brighter centers because the lensed arc overlaps with the lens galaxy in ground-based imaging.

**Implications:**
1. Paired controls alone DO NOT fix the shortcut
2. Need center degradation during training (blur or mask central region)
3. The model must learn arc morphology in outer regions, not just "bright center = lens"

**Solution:**
```python
def center_degradation(img, sigma_pix=4.0, fill='blur'):
    """Degrade central region to force outer-region learning."""
    h, w = img.shape[-2:]
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    center_mask = r < 10  # 10 pixel radius
    
    if fill == 'blur':
        from scipy.ndimage import gaussian_filter
        blurred = gaussian_filter(img, sigma_pix)
        out = img.copy()
        out[center_mask] = blurred[center_mask]
    elif fill == 'noise':
        outer_std = np.std(img[~center_mask])
        out = img.copy()
        out[center_mask] = np.random.normal(0, outer_std, center_mask.sum())
    
    return out
```

**Lesson:**
- Don't assume unpaired data bias is the cause of shortcuts
- Physical effects (PSF convolution) can create class-separable features
- Rigorously validate with paired data before changing strategy

---

*Last updated: 2026-02-05T07:40:00+00:00*
