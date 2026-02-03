#!/bin/bash
# Gen5 Phase 4c Smoke Test EMR Launch
# 
# Purpose: Validate COSMOS integration with minimal cost (~$2-5, ~10 min runtime)
# Output: ~50 stamps in debug tier
#
# Usage: Run this from emr-launcher
#   bash launch_gen5_smoke_test.sh

set -e

echo "========================================="
echo "Gen5 Phase 4c COSMOS Smoke Test"
echo "========================================="
echo "Purpose: Validate pipeline before full run"
echo "Cost: ~\$2-5 (2 nodes, ~10 min)"
echo "Output: ~50 stamps (debug tier)"
echo "========================================="
echo ""

# Launch EMR cluster
aws emr create-cluster \
  --name "Gen5-Phase4c-COSMOS-SmokeTest-$(date +%Y%m%d-%H%M)" \
  --region us-east-1 \
  --release-label emr-6.10.0 \
  --applications Name=Spark \
  --instance-type m5.2xlarge \
  --instance-count 2 \
  --service-role EMR_DefaultRole \
  --ec2-attributes InstanceProfile=EMR_EC2_DefaultRole \
  --bootstrap-actions Path=s3://darkhaloscope/scripts/gen5/emr_bootstrap_gen5.sh \
  --log-uri s3://darkhaloscope/emr-logs/ \
  --steps Type=Spark,Name="Phase4c-COSMOS-SmokeTest",ActionOnFailure=CONTINUE,Args=[\
--deploy-mode,cluster,\
--driver-memory,4g,\
--executor-memory,12g,\
--executor-cores,4,\
--conf,spark.sql.parquet.compression.codec=gzip,\
--conf,spark.dynamicAllocation.enabled=false,\
s3://darkhaloscope/scripts/gen5/spark_phase4_pipeline_gen5.py,\
--stage,4c,\
--output-s3,s3://darkhaloscope/phase4_pipeline,\
--variant,v5_cosmos_source_test,\
--experiment-id,test_stamp64_bandsgrz_cosmos_smoke,\
--source-mode,cosmos,\
--cosmos-bank-h5,s3://darkhaloscope/data/cosmos_banks/cosmos_bank_20k_gen5.h5,\
--cosmos-salt,gen5_test_v1,\
--seed-base,1337,\
--psf-model,moffat,\
--moffat-beta,3.5,\
--split-seed,42,\
--tiers,debug,\
--grid-debug,grid_small,\
--n-per-config-debug,5,\
--manifests-subdir,manifests,\
--parent-s3,s3://darkhaloscope/phase4_pipeline/phase4a/v4_sota_moffat/manifests/train_stamp64_bandsgrz_gridgrid_sota\
] \
  --auto-terminate

echo ""
echo "========================================="
echo "✅ Smoke test launched successfully"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Wait for cluster to complete (~10 min)"
echo "2. Check output: aws s3 ls s3://darkhaloscope/phase4_pipeline/phase4c/v5_cosmos_source_test/"
echo "3. Download sample: aws s3 cp s3://darkhaloscope/phase4_pipeline/phase4c/v5_cosmos_source_test/stamps/test_stamp64_bandsgrz_cosmos_smoke/part-00000*.parquet /tmp/"
echo "4. Validate: python3 validate_gen5_smoke_test.py /tmp/part-00000*.parquet"
echo ""
echo "If validation passes, proceed to full production run (270 vcores)"

