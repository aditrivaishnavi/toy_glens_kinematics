#!/bin/bash
# Gen5 Phase 4c Production Run - COSMOS Source Integration (CORRECTED)
# Created: 2026-02-04
# 
# THIS IS THE CORRECTED VERSION - USE THIS, NOT v5_cosmos_production
#
# CORRECTIONS APPLIED:
# - arc_snr_sum: Added integrated SNR calculation
# - lensed_hlr_arcsec: Renamed from cosmos_hlr_arcsec for clarity
# - Surface brightness: Fixed units for lenstronomy INTERPOL
#
# PREFLIGHT CHECKS PASSED:
# - Pipeline code: Fixed surface brightness conversion, added arc_snr_sum
# - Bootstrap: boto3, h5py, lenstronomy, scipy, astropy, numba cache
# - Manifest: 12M+ tasks, 133k bricks
# - Coadd cache: 99.85% coverage (132,970 / 133,163 bricks)
# - COSMOS bank: 20k galaxies, 474MB H5 file
# - Output path: v5_cosmos_corrected (new)

set -e

CLUSTER_NAME="Gen5-Phase4c-COSMOS-Production-$(date +%Y%m%d-%H%M)"
RELEASE_LABEL="emr-7.6.0"

echo "========================================="
echo "Gen5 Production Run: $CLUSTER_NAME"
echo "========================================="

# Create cluster with step
aws emr create-cluster \
  --name "$CLUSTER_NAME" \
  --release-label "$RELEASE_LABEL" \
  --applications Name=Spark \
  --ec2-attributes KeyName=root,SubnetId=subnet-01e8e1839aabcdb77 \
  --instance-groups '[
    {"InstanceGroupType":"MASTER","InstanceCount":1,"InstanceType":"m5.xlarge","Name":"Master"},
    {"InstanceGroupType":"CORE","InstanceCount":30,"InstanceType":"m5.2xlarge","Name":"Core"}
  ]' \
  --bootstrap-actions '[
    {"Path":"s3://darkhaloscope/code/gen5/emr_bootstrap_gen5.sh","Name":"Gen5-Bootstrap"}
  ]' \
  --steps '[
    {
      "Name": "Gen5-Phase4c-COSMOS",
      "ActionOnFailure": "CONTINUE",
      "HadoopJarStep": {
        "Jar": "command-runner.jar",
        "Args": [
          "spark-submit",
          "--deploy-mode", "cluster",
          "--driver-memory", "8g",
          "--executor-memory", "10g",
          "--executor-cores", "4",
          "--conf", "spark.sql.parquet.compression.codec=gzip",
          "--conf", "spark.dynamicAllocation.enabled=true",
          "--conf", "spark.shuffle.service.enabled=true",
          "s3://darkhaloscope/code/gen5/spark_phase4_pipeline_gen5.py",
          "--stage", "4c",
          "--output-s3", "s3://darkhaloscope/phase4_pipeline",
          "--variant", "cosmos_corrected",
          "--experiment-id", "train_stamp64_bandsgrz_cosmos_corrected",
          "--parent-s3", "s3://darkhaloscope/phase4_pipeline/phase4a/v4_sota_moffat/manifests/train_stamp64_bandsgrz_gridgrid_sota",
          "--coadd-s3-cache-prefix", "s3://darkhaloscope/dr10/coadd_cache",
          "--bands", "g,r,z",
          "--source-mode", "cosmos",
          "--cosmos-bank-h5", "s3://darkhaloscope/data/cosmos_banks/cosmos_bank_20k_gen5.h5",
          "--cosmos-salt", "production_v1",
          "--psf-model", "moffat",
          "--moffat-beta", "3.5",
          "--sweep-partitions", "600",
          "--skip-if-exists", "1"
        ]
      }
    }
  ]' \
  --service-role EMR_DefaultRole \
  --enable-debugging \
  --log-uri "s3://darkhaloscope/emr_logs/$CLUSTER_NAME/" \
  --auto-terminate \
  --region us-east-2

echo ""
echo "Cluster submitted! Check AWS Console for status."
echo "Log location: s3://darkhaloscope/emr_logs/$CLUSTER_NAME/"
