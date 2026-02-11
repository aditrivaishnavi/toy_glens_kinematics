# LLM Review: Training & Evaluation for Strong Lens Calibration (DESI DR10)

**Purpose:** We are preparing a paper on a detector audit / selection function for strong lens finding in DESI Legacy Survey DR10. Our model training and evaluation section must be comparable in rigor and correctness to the strong lens papers we reference (e.g. Huang et al., DESI Strong Lens Foundry, single-fiber/pairwise spectroscopic search papers). We need your careful review of the **code**, **metrics**, and **conclusions**, and your **Go/No-go** for next steps.

**What we are sharing:** (1) Full training **input** setup and **training** metrics; (2) Full **evaluation** metrics and methodology; (3) Our **conclusions** and an **honest audit** we already performed; (4) A **zip of relevant code** (`llm_review_training_eval.zip`) with a manifest (`MANIFEST.md`) explaining each file. This prompt is also included inside the zip at `llm_review_package/LLM_PROMPT_TRAINING_AND_EVAL_REVIEW.md`. We ask you to review the code for issues that could derail honest conclusions, answer direct clarity questions, and—if you can—provide **fully generated code** (e.g. scripts for split verification, bootstrap intervals) as **download links** or inline.

---

## PART A: Training input (data and config)

### A.1 Data sources and counts

| Item | Value |
|------|--------|
| **Manifest** | Single parquet: `training_v1.parquet` (path on training host: `/lambda/nfs/darkhaloscope-training-dc/stronglens_calibration/manifests/training_v1.parquet`). |
| **Total samples** | 416,449 |
| **Positives** | 4,788 |
| **Negatives** | 411,661 |
| **Neg:pos ratio** | ~86:1 |
| **Tier-A (confident) count** | 389 (sample_weight = 1.0 in loss) |
| **Tier-B (probable) count** | 4,399 (sample_weight = 0.5) |
| **Positive cutout source** | S3: `s3://darkhaloscope/stronglens_calibration/cutouts/positives/20260208_205758/` (5,101 .npz), synced to Lambda NFS for training. |
| **Negative cutout source** | S3: `s3://darkhaloscope/stronglens_calibration/cutouts/negatives/20260209_040454/` (~416K .npz), synced to Lambda NFS. |
| **Split assignment** | From NPZ metadata or manifest generator; exact method (HEALPix vs stratified, 70/15/15 vs 80/10/10) to be verified. Train/val/test disjointness **assumed** from upstream process, **not** verified by a script in this package. |
| **Seed** | 1337 (dataset/config). |

### A.2 Preprocessing and augmentation

| Item | Value |
|------|--------|
| **Preprocessing** | `raw_robust`: outer-annulus (r=20–32 px) median/MAD normalization, then clip ±10. |
| **Input size** | Cutouts 101×101 (g,r,z); center-cropped to **64×64** for training. |
| **Augmentation** | HFlip, VFlip, Rot90 (no brightness/contrast). |

### A.3 Training config (YAML)

- **Config file:** `configs/resnet18_baseline_v1.yaml` (included in zip).
- **Architecture:** ResNet-18 (torchvision, weights=None), conv1 → 3 channels, fc → 1 output.
- **Optimizer:** AdamW, lr = 0.0003, weight_decay = 0.0001.
- **Batch size:** 256.
- **Max epochs:** 50; **early stopping patience:** 10 (on val AUC).
- **LR schedule:** CosineAnnealingLR, T_max=50.
- **Mixed precision (AMP):** true.
- **Loss:** BCEWithLogitsLoss with reduction='none', then per-sample weighted mean using `sample_weight` from manifest (Tier-A=1.0, Tier-B=0.5; negatives=1.0).

---

## PART B: Training metrics (from run)

| Metric | Value |
|--------|--------|
| **Host** | Lambda Labs GPU, NFS mount `darkhaloscope-training-dc`. |
| **Epochs completed** | 16 (early stopped). |
| **Best epoch** | 6 (by val AUC). |
| **Best val AUC** | 0.9592 |
| **Final train loss (last epoch)** | 0.0013 |
| **Stop reason** | Early stopping: 10 consecutive epochs without improvement in val AUC. |
| **Command** | `python code/dhs/scripts/run_experiment.py --config code/configs/resnet18_baseline_v1.yaml` (cwd: project root on Lambda). |

---

## PART C: Evaluation metrics (test set, single run)

| Metric | Value |
|--------|--------|
| **Test samples** | 62,760 |
| **Test AUC (overall)** | 0.9579 |
| **Recall @ threshold 0.5 (overall)** | 0.4360 |
| **Precision @ 0.5 (overall)** | 0.7653 |
| **ECE (overall, 15 equal-frequency bins)** | 0.0027 |
| **MCE (overall)** | 0.0156 |
| **Tier-A in test (held-out)** | n = 48 (all positives). Recall @ 0.5 = 0.6042. We do **not** report ECE/MCE for this stratum (single-class; see honest audit). |
| **Tier-B in test** | n = 640 positives. Recall @ 0.5 = 0.4234. |

**Note on “Tier-A”:** These are rows in the **test** set with `tier == A` (confident lenses with DR10 match). They are **held out from training** but come from the **same** imaging-candidate catalog (lenscat), not from a separate spectroscopic-search catalog. We use the term “held-out Tier-A evaluation,” not “independent spectroscopic validation” in the strong sense.

---

## PART D: Our conclusions (and honest audit)

1. **Metrics are not faked:** The evaluation code computes AUC, recall, precision, ECE, MCE as stated; inference order aligns with manifest rows; we evaluate on test only and select the model by val AUC.
2. **Overall ECE** is low (0.0027) but **dominated by the majority class** (negatives). It does **not** certify calibration on the **positive** class; for that we would need a separate analysis (e.g. reliability diagram in the high-score region or positive-only metrics with caveats).
3. **Tier-A / Tier-B ECE/MCE** on positive-only strata were misleading and are no longer reported in the eval script for single-class strata.
4. **“Independent spectroscopic validation”** was overstated; we now say “held-out Tier-A evaluation” and clarify that we do not have a separate spectroscopic-channel catalog.
5. **Split disjointness** (train ∩ test = ∅) is **assumed**, not verified by script. We recommend adding a verification step and documenting it.
6. **Uncertainty:** We report point estimates only. For publication we need at least bootstrap (or similar) intervals for AUC and recall @ 0.5 on test and on Tier-A.

We have documented these points in `EVALUATION_HONEST_AUDIT.md` (included in zip).

---

## PART E: Direct questions (we need your clarity)

1. **Split assignment:** Our manifest has a `split` column (train/val/test). The exact upstream method (HEALPix-based vs label-stratified, and ratio 70/15/15 vs 80/10/10) is not fully confirmed. For the paper, should we (a) trace and document the exact assignment (which script, which logic), or (b) run an explicit check that train and test IDs (e.g. `galaxy_id` or `cutout_path`) are disjoint, or both?

2. **Tier-A wording:** Is it scientifically defensible to call our Tier-A test subset “evaluation on held-out Tier-A anchors” and to **avoid** claiming “independent spectroscopic validation” unless we add a truly separate catalog (e.g. DESI single-fiber or pairwise spectroscopic lens candidates)? Do you see any remaining overclaim in our current wording?

3. **Calibration on positives:** For the paper, is it sufficient to state that “overall ECE summarizes calibration over the full (imbalanced) set and does not certify calibration on the positive class,” and to defer positive-class calibration to future work? Or do you recommend we add a specific analysis (e.g. reliability diagram restricted to high-score region or to positives only) before submission?

4. **Uncertainty quantification:** What minimal uncertainty reporting do you recommend for the methods section? (e.g. bootstrap 95% or 68% for test AUC and recall @ 0.5; and for Tier-A recall given n=48?)

5. **Rigor vs referenced papers:** In your view, what are the 2–3 most important gaps between our current training/evaluation description and the level of rigor in typical DESI strong lens / Huang et al. / Foundry papers (e.g. split description, uncertainty, calibration, or independence of validation)?

6. **Shortcut gates:** We have not yet run “annulus-only” and “core-only” classifier checks (checklist 4.7, 4.8). How critical are these for a first submission, versus leaving them as “future work”?

7. **Reproducibility:** We do not currently log git commit or run timestamp in the checkpoint directory. Should we add a small `run_info.json` (or similar) next to `best.pt` with commit hash, config path, and command for every future run?

---

## PART E2: Meticulous operations (anything else?)

We want to be thorough before and during the paper. Please consider and advise on any of the following we may have missed:

8. **Environment and versions:** We have not pinned Python, PyTorch, torchvision, scikit-learn, pandas, numpy, pyarrow versions in the package. Should we add a `requirements.txt` or conda `environment.yml` (with versions) and cite it in the paper for reproducibility? If so, what minimal set do you recommend?

9. **Evaluation output:** The full evaluation JSON (reliability curve bins, exact counts) from the Lambda run is not in the zip; only the summary numbers are in this prompt. Should we include a sanitized (path-stripped) copy of the eval JSON in the supplement or repo for reviewers?

10. **FPR by negative pool (N1/N2):** We have not run FPR stratified by N2 confuser category. Is that important to run before or alongside the selection function, or can it follow as failure-mode analysis (Phase 6)?

11. **Preprocessing consistency check:** We assert “same preprocessing at train and eval” but do not have a unit test that compares preprocess_stack output for a fixed input across calls. Should we add a small regression test or assert (e.g. checksum on a reference cutout) to lock preprocessing?

12. **Path and portability:** Training/eval use absolute paths in the config (Lambda NFS). For reproducibility, should we support a path override (e.g. env var or `--data_root`) so the same config can be run on another machine with different mount points?

13. **Any other gaps:** From your experience with method papers, what other operational details (logging, random seeds, checkpoint naming, artifact versioning) would you add so our pipeline is “meticulous” and defensible under review?

---

## PART F: Code review request

We are providing a zip of relevant code for **training** and **evaluation** (and upstream manifest generation). Please:

1. **Review the code carefully** and point out **major issues** that could derail our honest conclusions (e.g. data leakage, wrong metric formula, order mismatch between predictions and labels, misuse of splits, or any subtle bug that would invalidate reported numbers).

2. **Focus on:**  
   - Correctness of train/val/test usage (no test in training or model selection).  
   - Correctness of AUC, recall, precision, ECE/MCE implementation.  
   - Alignment of inference output order with manifest row order (for tier/stratum reporting).  
   - Any place where “independent” or “spectroscopic” validation is overstated in comments or docstrings.

3. **Manifest:** The zip includes a `MANIFEST.md` that explains each file and its intended behavior. Use it to navigate; if any file’s stated behavior does not match the code, please flag it.

---

## PART G: Conclusion and next steps

- **Our conclusion:** The evaluation code does not fudge numbers; the main issues were **wording** (“independent spectroscopic validation”) and **interpretation** (overall ECE as proof of calibration on positives; reporting ECE/MCE for positive-only strata). We have tightened terminology and stopped reporting misleading stratum ECE/MCE. For publication we still need: split disjointness verification, uncertainty quantification (bootstrap intervals), and possibly positive-class calibration analysis and shortcut gates.

- **Next steps we envision:**  
  1. Add a script to verify train/test ID disjointness and document result.  
  2. Add bootstrap (or similar) intervals for test AUC, test recall @ 0.5, and Tier-A recall @ 0.5; append to evaluation JSON/summary.  
  3. Add `run_info.json` (or equivalent) to checkpoint directory for reproducibility.  
  4. Optionally: reliability diagram or calibration analysis restricted to positives/high-score region.  
  5. Optionally: run annulus-only and core-only shortcut gates (4.7, 4.8).  
  6. Proceed to selection function (injection-recovery grid with frozen model) once the above are done.

**We need from you:**

1. **Assessment:** Based on the data and code we shared, do our **conclusions** look **correct**, **accurate**, **scientifically rigorous**, and **defensible** for a paper whose model-training section will be compared to strong lens papers we reference?

2. **Go/No-go:** Do you recommend a **Go** for the next steps above (with or without modifications), or a **No-go** until specific issues are fixed? If No-go, please list the blocking issues.

3. **Anything else:** Is there anything else we should do before writing the training/evaluation section of the paper?

4. **Fully generated code:** If possible, please provide **fully generated code** (e.g. Python script to verify train/test disjointness, or to compute bootstrap intervals for AUC and recall) as **download links** or paste the code inline so we can drop it into the repo and run it. We will attribute and use it for reproducibility.

---

## PART H: Selection function (injection-recovery) – complete code and watchouts

Our **next major step** after the above is the **selection function**: injection-recovery grid with the **frozen** ResNet-18 model to estimate completeness C(θE, PSF, depth) with uncertainty. We need **complete, runnable code** (or a detailed blueprint we can implement) and a clear list of **things to watch out for** so we do not undermine our conclusions.

### H.1 Specification (from our checklist)

- **Grid axes:**  
  - **θE (Einstein radius):** 0.5" to 3.0" in **0.25"** steps → **11 bins**.  
  - **PSF FWHM:** 0.9" to 1.8" in **0.15"** steps → **7 bins**.  
  - **Depth (e.g. psfdepth_r or galdepth_r):** 22.5 to 24.5 mag in **0.5** mag steps → **5 bins**.  
  - Total cells: **11 × 7 × 5 = 385**.

- **Minimum injections per cell:** **200** (so ≥ 77,000 injections total). Stratify by DR10 conditions so each cell has enough real cutouts with matching conditions (or document how we sample cutouts per cell).

- **Frozen detector:** Load `best.pt` (ResNet-18), **no training**. Run inference with **same preprocessing** as training (`raw_robust`, 64×64 center crop). Record score (sigmoid output) or binary detection at a chosen threshold.

- **Completeness:** Per cell, completeness = (number of injections with score ≥ threshold) / (total injections in cell). Report **Bayesian binomial 68%** intervals (optionally 95% in appendix). **Low-N bins:** do not smooth unless justified; **mark** cells with N < 200 as “insufficient” or merge adjacent cells with a stated rule.

- **Output:** Lookup table or artifact (e.g. C(θE, PSF, depth) with lower/upper bounds) in a public-release format.

### H.2 Injection realism (from our prior LLM guidance)

Injections must be **photometrically calibrated** to DR10:

- **Zeropoint:** AB zeropoint **22.5** (nanomaggies); we have `AB_ZEROPOINT_MAG = 22.5` in constants.
- **Source magnitude prior:** e.g. r-band 22–26 (unlensed), magnification proxy μ ~ 5–30; target **arc annulus SNR** in a range comparable to real Tier-A anchors (e.g. median within 0.5×–2×).
- **PSF:** Prefer **per-cutout PSF** when available; otherwise document use of brick/cell-average and impact.
- **Noise:** Add noise using **measured background** (e.g. MAD-based) from the same cutout, not a global constant.
- **Acceptance (GO/NO-GO):** Before trusting the selection function, we should pass: (1) arc annulus SNR distribution (injections vs real) within 0.5×–2× at median and percentiles; (2) color (g-r, r-z) median within ±0.2 mag; (3) noise histogram KS test p > 0.05; (4) visual sanity (injections not systematically “cleaner” than real).

**Base for injections:** Use **negative cutouts only** (no positive cutouts) so we are measuring “recovery of synthetic lenses on real non-lens backgrounds.” Do not use any cutout that appears in the training set for injection hosts (or document and justify if a subset is shared).

### H.3 What we are asking from you

1. **Complete code (or blueprint):** Please provide **full, runnable code** (or a step-by-step blueprint with pseudocode and file layout) for:  
   - (a) Building or loading the injection grid (θE, PSF, depth) and sampling/assigning negative cutouts to cells (with stratification if needed).  
   - (b) Generating synthetic lens injections on each cutout (flux in DR10 units, PSF, noise) and saving or streaming “injected” images.  
   - (c) Running the **frozen** model on injected images with the **same** preprocessing as training, recording scores (and optionally binary at threshold).  
   - (d) Aggregating by cell: completeness, counts, and **Bayesian binomial 68%** intervals.  
   - (e) Outputting a lookup table or structured artifact (e.g. CSV/Parquet with cell axes and C_lo, C, C_hi).  
   Prefer Python; we use PyTorch for the model and NumPy/Pandas for data. If you cannot provide full code, please give a **detailed blueprint** (modules, inputs/outputs, formulas) so we can implement it without guessing.

2. **Things to watch out for:** Please list **concrete pitfalls** that could derail an honest selection function, for example:  
   - Using wrong zeropoint or flux units so injections are not DR10-like.  
   - Using training or positive cutouts as injection hosts (leakage).  
   - Preprocessing mismatch (e.g. different normalization or crop) between training and injection scoring.  
   - Ignoring per-cutout PSF/depth so cell assignment is inconsistent with real conditions.  
   - Reporting completeness without uncertainty or with inappropriate smoothing.  
   - Any other subtle bug or methodological slip that would make the selection function invalid or overstated.

3. **Deliverable:** If possible, provide the code (or blueprint) and watchouts as **download links** or paste them inline. We will integrate and attribute them.

Thank you for a rigorous and honest review.
