# Dark Halo Scope: Reproducible Runbook (Phase 4 and Phase 5)

This runbook assumes:
- AWS EMR (Spark 3.5, Python 3)
- S3 bucket: s3://darkhaloscope/
- Training on a single GH200/H100-class GPU (PyTorch 2.x, Python 3.12)

## 0) One-time setup

### EMR bootstrap (Python deps)
Install: numpy, pyarrow, fitsio, astropy.
If you use lenstronomy in Spark executors, also install lenstronomy.

### Lambda (training host) environment
Create a venv and install:
- torch, torchvision (CUDA build)
- pyarrow, numpy, pandas, tqdm

## 1) Phase 4a: Build SOTA manifest (unpaired controls, resolvable grid)

Use the dedicated builder:

spark-submit \
  --deploy-mode cluster \
  dhs_package/spark_phase4a_build_manifest_sota.py \
  --input s3://darkhaloscope/phase3/parent_sample/targets.parquet \
  --out s3://darkhaloscope/phase4_pipeline/phase4a/v4_sota/ \
  --grid_name grid_sota_v1 \
  --control_frac 0.50 \
  --unpaired_control_frac 0.50 \
  --seed 1337

Notes:
- unpaired_control_frac=0.50 forces half of controls to come from different galaxies but PSF/depth matched.
- control_frac controls total negative fraction.

## 2) Phase 4b: Cache DR10 coadds (if not already cached)

spark-submit \
  --deploy-mode cluster \
  dhs_package/spark_phase4_pipeline.py \
  --stage 4b \
  --bricklist s3://darkhaloscope/phase4_pipeline/bricklists/dr10_south_bricks.txt \
  --out_coadd_cache s3://darkhaloscope/dr10/coadd_cache/

## 3) Phase 4c: Inject stamps (PSF model patched)

Recommended PSF setting:
- --psf-model moffat
- --moffat-beta 3.5

spark-submit \
  --deploy-mode cluster \
  dhs_package/spark_phase4_pipeline.py \
  --stage 4c \
  --manifests s3://darkhaloscope/phase4_pipeline/phase4a/v4_sota/ \
  --coadd_cache s3://darkhaloscope/dr10/coadd_cache/ \
  --out_stamps s3://darkhaloscope/phase4_pipeline/phase4c/v4_sota/stamps/ \
  --stamp_size 64 \
  --bandsets grz \
  --psf-model moffat \
  --moffat-beta 3.5 \
  --max_bad_pixel_frac 0.02 \
  --min_psf_fwhm_arcsec 0.9 \
  --max_psf_fwhm_arcsec 1.8

## 4) Transfer stamps to training host (one-time copy)

Example with rclone:

rclone copy \
  s3:darkhaloscope/phase4_pipeline/phase4c/v4_sota/stamps/ \
  /mnt/data/darkhaloscope/phase4c/v4_sota/stamps/ \
  --transfers 32 --checkers 64 --fast-list

## 5) Phase 5 training: recommended hyperparameters (publication-oriented)

### Path A (resolved-only curriculum)

Stage A1: high-confidence morphology learning
python -u dhs_package/phase5_train_fullscale_gh200_v2.py \
  --data /mnt/data/darkhaloscope/phase4c/v4_sota/stamps/ \
  --out_dir /mnt/data/models/pathA_stageA1 \
  --arch convnext_tiny \
  --epochs 6 \
  --batch_size 256 \
  --lr 3e-4 \
  --weight_decay 1e-2 \
  --dropout 0.10 \
  --use_bf16 \
  --augment \
  --loss focal \
  --focal_alpha 0.25 --focal_gamma 2.0 \
  --min_theta_over_psf 0.80 \
  --min_arc_snr 7.0 \
  --meta_cols psfsize_r,psfdepth_r \
  --contract_json dhs_package/phase5_required_columns_contract.json

Stage A2: expand to your target resolved regime
python -u dhs_package/phase5_train_fullscale_gh200_v2.py \
  --data /mnt/data/darkhaloscope/phase4c/v4_sota/stamps/ \
  --out_dir /mnt/data/models/pathA_stageA2 \
  --arch convnext_tiny \
  --epochs 6 \
  --batch_size 256 \
  --lr 1e-4 \
  --weight_decay 1e-2 \
  --dropout 0.10 \
  --use_bf16 \
  --augment \
  --loss focal \
  --focal_alpha 0.25 --focal_gamma 2.0 \
  --min_theta_over_psf 0.50 \
  --min_arc_snr 3.0 \
  --meta_cols psfsize_r,psfdepth_r \
  --contract_json dhs_package/phase5_required_columns_contract.json \
  --resume /mnt/data/models/pathA_stageA1/best.pt

### Path B (unpaired controls + hard negatives)

After mining hard negatives (step 7), retrain with:
- min_theta_over_psf 0.50
- min_arc_snr 0.0 to avoid biasing selection function
- loss focal, same optimizer, same augmentation

## 6) Inference on heldout test split

python -u dhs_package/phase5_infer_scores_v2.py \
  --input /mnt/data/darkhaloscope/phase4c/v4_sota/stamps/ \
  --split test \
  --ckpt /mnt/data/models/pathA_stageA2/best.pt \
  --out /mnt/data/scores/pathA_stageA2_test_scores.parquet \
  --batch_size 1024 \
  --use_bf16

## 7) Stratified evaluation (FPR curves)

python -u dhs_package/phase5_eval_stratified_fpr.py \
  --scores /mnt/data/scores/pathA_stageA2_test_scores.parquet \
  --out_csv /mnt/data/reports/stratified_fpr.csv \
  --theta_bins 0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.5 \
  --res_bins 0.5,0.7,0.9,1.1,1.4,2.0 \
  --tpr_targets 0.80,0.85,0.90,0.95 \
  --fpr_targets 1e-5,1e-4,1e-3,1e-2

## 8) Mine hard negatives (top scoring false positives)

python -u dhs_package/phase5_mine_hard_negatives.py \
  --scores /mnt/data/scores/pathA_stageA2_test_scores.parquet \
  --out_parquet /mnt/data/hard_negs/hard_negs_top20000.parquet \
  --topk 20000 \
  --min_score 0.90

Append these to training negatives and retrain (Path B).

## 9) Reproducibility checks
- Use fixed seeds (1337).
- Keep region-disjoint splits (train/val/test by region_id).
- Report stratified FPR at fixed TPR, not only AUROC.
