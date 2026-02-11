# No-Go Remediation Runbook

This runbook describes how to clear the LLM's **No-Go** (split disjointness + bootstrap intervals) and where to record results. Run these steps on a machine that has access to the training manifest and (for bootstrap) test predictions.

**Region:** All AWS commands must use `--region us-east-2` (see [LESSONS_LEARNED.md](LESSONS_LEARNED.md)).

---

## 1. Split disjointness verification (blocking for paper claims)

**Goal:** Run `verify_splits.py` on the manifest used for the reported training run and commit the report so it can be cited in the methods section.

### 1.1 Obtain the manifest

- **On Lambda (training host):**  
  Path: `/lambda/nfs/darkhaloscope-training-dc/stronglens_calibration/manifests/training_v1.parquet`
- **From elsewhere:** Download from S3 (if synced) or copy from Lambda, e.g.:
  ```bash
  aws s3 cp s3://darkhaloscope/stronglens_calibration/manifests/training_v1.parquet ./manifests/ --region us-east-2
  ```
  Or use rclone / scp from the Lambda NFS path.

### 1.2 Run verify_splits

From the **stronglens_calibration** directory (so `common` and scripts resolve):

```bash
cd /path/to/stronglens_calibration
PYTHONPATH=. python scripts/verify_splits.py \
  --manifest /path/to/training_v1.parquet \
  --out_json docs/split_verification_report.json \
  --fail_on_overlap
```

- Use the actual path to `training_v1.parquet` (Lambda path or local copy).
- `--fail_on_overlap` makes the script exit non-zero if any train/val/test overlap exists (safe for CI).
- Output is written to `docs/split_verification_report.json`.

### 1.3 Record the result

- **Commit** `docs/split_verification_report.json` to the repo (or place it in a known artifact location).
- In the paper methods / supplement: *"We verified train/val/test disjointness by `galaxy_id` and `cutout_path`; report in `split_verification_report.json`."*
- If overlaps are non-zero: fix upstream split assignment (e.g. HEALPix/hash in `emr/sampling_utils.py` or manifest generation), regenerate the manifest, then re-run until overlaps are 0.

---

## 2. Bootstrap intervals (required for Go)

**Goal:** Add bootstrap 68% (and optionally 95%) intervals for test AUC and test recall at the operating threshold, and for Tier-A recall.

### 2.1 Get test-set predictions

For the run you reported (e.g. `best.pt`, epoch 6):

- Re-run evaluation on Lambda and export predictions with columns `cutout_path` and `score` (or `logit`), or use an existing export.
- Save as CSV or Parquet (e.g. `test_predictions.csv`).

### 2.2 Run bootstrap_eval

From **stronglens_calibration**:

```bash
cd /path/to/stronglens_calibration
PYTHONPATH=. python scripts/bootstrap_eval.py \
  --manifest /path/to/training_v1.parquet \
  --preds /path/to/test_predictions.csv \
  --split test \
  --threshold 0.5 \
  --out_json docs/bootstrap_eval_test.json
```

- Output includes `overall` (AUC, recall, precision), `overall_bootstrap` (68% and 95% intervals), and `by_tier` if the merged data has a `tier` column (e.g. Tier-A recall with n=48).

### 2.3 Record and use in paper

- Add `docs/bootstrap_eval_test.json` to the repo or supplement.
- In the methods section: report *"test AUC and recall at threshold 0.5 with bootstrap 68% (and 95%) intervals"* and cite the file; for Tier-A, report *"Tier-A recall with 68% interval"* and note n=48.

---

## 3. Training reproducibility (optional)

- **run_info.json:** Each training run writes `run_info.json` in the checkpoint directory (next to `best.pt`) with git commit, timestamp, config path/hash, command, and dataset seed.
- **Path portability:** Use `--data_root /path/to/root` when running on a host other than Lambda so that manifest and checkpoint paths in the config are rewritten from the default Lambda NFS root to your path.

---

## 4. Order of operations

1. Run split verification; commit `docs/split_verification_report.json` (fix splits until overlaps = 0).
2. Export test predictions; run `bootstrap_eval`; commit `docs/bootstrap_eval_test.json`.
3. Proceed to write the training/evaluation section with both artifacts cited.
