# Prompt 25: S3 Code Integrity and Completeness Verification

## Context

We have shut down our GPU compute instance (lambda3) and deleted the NFS filesystem that held all project data. All data has been synced to S3. Additionally, the entire local git repository (including `.git` history) has been rcloned to S3 as a full backup.

**Your task:** Verify that S3 contains all the code, configs, and artifacts needed to fully reproduce the MNRAS paper "The morphological barrier" from scratch, and that the code is the **latest version** (matching the git repo).

## Access details

### S3 bucket

- **Bucket:** `s3://darkhaloscope/` (region: `us-east-2`)
- **Access from:** SSH into `emr-launcher` — this machine has full S3 access via `aws` CLI and `rclone`
  ```bash
  ssh emr-launcher
  aws s3 ls s3://darkhaloscope/
  ```

### Key S3 locations

| S3 path | Contents |
|---------|----------|
| `s3://darkhaloscope/toy_glens_kinematics_backup/` | **Full git repo backup** — rsync'd from local Mac including `.git/`, all paper `.tex` files, all code, all LLM prompts. This is the source of truth. |
| `s3://darkhaloscope/stronglens_calibration/code/` | Mirror of the Python codebase (synced from NFS before shutdown — may be stale) |
| `s3://darkhaloscope/stronglens_calibration/dhs/` | Core library (`data.py`, `train.py`, `injection_engine.py`, `s3io.py`, etc.) |
| `s3://darkhaloscope/stronglens_calibration/scripts/` | Standalone scripts |
| `s3://darkhaloscope/stronglens_calibration/configs/` | YAML training and injection configs |
| `s3://darkhaloscope/stronglens_calibration/checkpoints/` | All trained model weights (`.pt` files) |
| `s3://darkhaloscope/stronglens_calibration/results/D06_20260216_corrected_priors/` | Final production results (229,926 files) |
| `s3://darkhaloscope/stronglens_calibration/manifests/` | Training/validation split manifests |
| `s3://darkhaloscope/stronglens_calibration/cutouts/` | Training cutouts (positives + negatives) |

### Git repo details

- **GitHub remote:** `git@github.com:aditrivaishnavi/toy_glens_kinematics.git`
- **Branch:** `master`
- **Latest commit:** `a3ef5ea` — "Fix 6 issues from blind code review"
- **NOTE:** The repo is **8 commits ahead of origin** — those 8 commits have NOT been pushed to GitHub yet. The S3 backup at `toy_glens_kinematics_backup/` contains the full `.git` directory with all unpushed commits.

### Recent git history (latest first)

```
a3ef5ea Fix 6 issues from blind code review
ec02817 Sync configs/docs/paper with corrected injection priors, 2-phase bright arc test
b79512c Fix injection engine realism: K-corrected colours, narrowed priors, sky noise
1e475cb MNRAS paper v10: fix citations, soften overclaims, strengthen PSF caveat
78e8600 D01 pre-retrain diagnostics: results and GO decision for annulus retrain
2c33eaa LLM-reviewed injection-recovery pipeline: code fixes, diagnostics, and MNRAS documentation
2a3d26d Add injection engine validation tests (28 tests) and lenstronomy cross-validation
598e1c6 Add LLM prompt and code zip for injection-recovery pipeline review
```

## What to check

### Q1: Is the full git repo backup intact?

On `emr-launcher`, download the backup and verify git integrity:

```bash
aws s3 sync s3://darkhaloscope/toy_glens_kinematics_backup/ /tmp/repo_check/ 
cd /tmp/repo_check
git fsck
git log --oneline -10
```

Confirm:
- `git fsck` reports no errors
- Latest commit is `a3ef5ea`
- All 8 unpushed commits are present

### Q2: Does the backup contain all critical files?

Check that these files/directories exist in the backup:

**Paper files (these are ONLY in the git repo, not elsewhere in S3):**
- `stronglens_calibration/paper/mnras_merged_draft_v14.tex` (latest paper version)
- `stronglens_calibration/paper/mnras_merged_draft_v1.tex` through `v13.tex` (all prior versions)
- `stronglens_calibration/paper/generate_all_figures.py`
- `stronglens_calibration/paper/fig1_completeness.pdf` through `fig5_comparison.pdf`

**LLM review prompts (also only in git repo):**
- `stronglens_calibration/llm_review_package/PROMPT_20_D06_RESULTS.md` through `PROMPT_24_V14_FIGURE5_REDESIGN.md`

**Critical code files:**
- `stronglens_calibration/dhs/injection_engine.py` — the injection engine with K-corrected colours fix
- `stronglens_calibration/dhs/data.py` — data loader
- `stronglens_calibration/dhs/train.py` — training loop
- `stronglens_calibration/configs/injection_priors.yaml` — corrected priors used in D06
- `stronglens_calibration/configs/paperIV_bottlenecked_resnet.yaml` — primary model config
- `stronglens_calibration/scripts/generate_comparison_figure.py` — Figure 5 generator
- `stronglens_calibration/scripts/selection_function_grid.py` — grid evaluation script

**Docs:**
- `stronglens_calibration/docs/S3_BUCKET_INVENTORY.md`
- `stronglens_calibration/docs/REPRODUCIBILITY_GUIDE.md`
- `stronglens_calibration/docs/MNRAS_INJECTION_VALIDATION.md`

### Q3: Is the NFS-synced code in S3 consistent with the git backup?

Compare key files between the two S3 locations:

```bash
# Example: compare injection_engine.py
aws s3 cp s3://darkhaloscope/stronglens_calibration/dhs/injection_engine.py /tmp/from_nfs.py
aws s3 cp s3://darkhaloscope/toy_glens_kinematics_backup/stronglens_calibration/dhs/injection_engine.py /tmp/from_git.py
diff /tmp/from_nfs.py /tmp/from_git.py
```

Do this for at least:
- `dhs/injection_engine.py`
- `dhs/data.py`
- `dhs/train.py`
- `configs/injection_priors.yaml`
- `configs/paperIV_bottlenecked_resnet.yaml`

If differences exist, document which version is newer and what changed.

### Q4: Are model checkpoints present and intact?

Verify the primary model checkpoint exists:
```bash
aws s3 ls s3://darkhaloscope/stronglens_calibration/checkpoints/paperIV_bottlenecked_resnet/best.pt
```

List all checkpoint directories and confirm file counts match expectations (136 total files across 15 model directories).

### Q5: Are D06 results complete?

```bash
aws s3 ls s3://darkhaloscope/stronglens_calibration/results/D06_20260216_corrected_priors/ --recursive | wc -l
```

Expected: **229,926 files** (verified before NFS deletion). Spot-check:
- `grid_no_poisson/`: 110,003 files
- `grid_poisson/`: 110,003 files
- `ba_baseline/`: 1,605 files
- `provenance.json`: exists

### Q6: Are there any files in the NFS-synced S3 locations that are NOT in the git backup?

This would identify data files (cutouts, results, checkpoints) that only exist in S3 and not in the git repo — which is expected and fine, but should be documented.

```bash
# List files in stronglens_calibration/ S3 prefix
aws s3 ls s3://darkhaloscope/stronglens_calibration/ --recursive | awk '{print $4}' | sort > /tmp/s3_sl.txt

# List files in the backup
aws s3 ls s3://darkhaloscope/toy_glens_kinematics_backup/stronglens_calibration/ --recursive | awk '{print $4}' | sort > /tmp/s3_backup.txt

# Files in S3 not in backup (expected: large data files)
comm -23 /tmp/s3_sl.txt /tmp/s3_backup.txt | head -30
```

## Expected output

Please provide a clear YES/NO for each question, with evidence. Flag anything concerning. If any critical files are missing or differ unexpectedly, explain exactly what is missing and what the impact would be on reproducibility.
