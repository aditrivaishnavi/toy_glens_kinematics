#!/bin/bash
set -e

echo "=========================================="
echo "Launching Gen5 COSMOS Bank Builder EMR"
echo "=========================================="

# Configuration
S3_BUCKET="s3://darkhaloscope"
S3_BOOTSTRAP="${S3_BUCKET}/emr_bootstrap/cosmos_bank_builder_bootstrap.sh"
S3_CODE="${S3_BUCKET}/code/gen5/"

# Upload files
echo "📤 Uploading files to S3..."
aws s3 cp emr/gen5/cosmos_bank_builder_bootstrap.sh ${S3_BOOTSTRAP}
aws s3 cp emr/gen5/build_cosmos_bank_emr.py ${S3_CODE}build_cosmos_bank_emr.py
aws s3 cp --recursive models/dhs_cosmos_galsim_code/dhs_cosmos/ ${S3_CODE}dhs_cosmos/

echo ""
echo "🚀 Launching EMR cluster..."

# Launch with explicit roles and simpler configuration
CLUSTER_ID=$(aws emr create-cluster \
  --name "gen5-cosmos-bank-builder" \
  --release-label emr-6.15.0 \
  --applications Name=Hadoop Name=Spark \
  --ec2-attributes KeyName=phase5-key \
  --instance-type r6i.8xlarge \
  --instance-count 1 \
  --service-role EMR_DefaultRole \
  --ec2-attributes InstanceProfile=EMR_EC2_DefaultRole \
  --bootstrap-actions Path=${S3_BOOTSTRAP} \
  --steps Type=CUSTOM_JAR,Name="Download-Code",ActionOnFailure=TERMINATE_CLUSTER,Jar=command-runner.jar,Args=[bash,-c,"sudo mkdir -p /mnt/code && sudo chmod 777 /mnt/code && aws s3 cp --recursive ${S3_CODE}dhs_cosmos/ /mnt/code/dhs_cosmos/ && aws s3 cp ${S3_CODE}build_cosmos_bank_emr.py /mnt/code/"] \
          Type=CUSTOM_JAR,Name="Build-COSMOS-Bank",ActionOnFailure=CONTINUE,Jar=command-runner.jar,Args=[python3,/mnt/code/build_cosmos_bank_emr.py] \
  --auto-terminate \
  --log-uri ${S3_BUCKET}/emr_logs/ \
  --region us-east-1 \
  --query 'ClusterId' \
  --output text)

if [ -z "$CLUSTER_ID" ]; then
  echo "❌ Failed to launch cluster"
  exit 1
fi

echo ${CLUSTER_ID} > /tmp/cosmos_bank_cluster_id.txt

echo ""
echo "✅ Cluster launched: ${CLUSTER_ID}"
echo "   Region: us-east-1"
echo "   Runtime: 30-60 minutes"
echo ""
echo "Monitor: aws emr describe-cluster --cluster-id ${CLUSTER_ID} --region us-east-1"
echo ""
