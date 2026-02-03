#!/bin/bash
# ============================================================================
# Gen5 COSMOS Bank Builder - Launch Instructions for emr-launcher
# ============================================================================
# 
# This job MUST be run from emr-launcher (not laptop) per user requirements.
# 
# INSTRUCTIONS:
# 1. SSH to emr-launcher
# 2. Clone/update the repo
# 3. Run this script
# 
# ============================================================================

cat << 'EOF'

╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║  Gen5 COSMOS Bank Builder - Launch on emr-launcher            ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝

⚠️  This job MUST run on emr-launcher (not laptop)

STEP 1: SSH to emr-launcher
────────────────────────────────────────────────────────────────
ssh emr-launcher

STEP 2: Setup repository
────────────────────────────────────────────────────────────────
cd /home/ec2-user/
if [ ! -d "toy_glens_kinematics" ]; then
  git clone <your-repo-url> toy_glens_kinematics
fi
cd toy_glens_kinematics/dark_halo_scope
git pull

STEP 3: Launch EMR job for COSMOS bank creation
────────────────────────────────────────────────────────────────
bash emr/gen5/launch_cosmos_bank_builder.sh

STEP 4: Monitor progress
────────────────────────────────────────────────────────────────
# The script will output monitoring commands, or use:
CLUSTER_ID=$(cat /tmp/cosmos_bank_cluster_id.txt)

# Check cluster status
aws emr describe-cluster --cluster-id $CLUSTER_ID \
  --query 'Cluster.Status.State' --output text

# Check step progress
aws emr list-steps --cluster-id $CLUSTER_ID \
  --query 'Steps[*].[Name,Status.State]' --output table

# View logs
aws emr ssh --cluster-id $CLUSTER_ID --command "tail -f /mnt/var/log/hadoop/steps/*/stdout"

EXPECTED OUTPUT:
────────────────────────────────────────────────────────────────
After 30-60 minutes:
✅ s3://darkhaloscope/cosmos/cosmos_bank_20k_parametric_v1.h5 (~150 MB)
✅ s3://darkhaloscope/cosmos/cosmos_bank_config_20k_v1.json

CLUSTER SPECIFICATIONS:
────────────────────────────────────────────────────────────────
• Instance: 1x r6i.8xlarge (32 vCPUs, 256 GB RAM)
• Storage: 500 GB GP3 EBS (3000 IOPS)
• Auto-terminate: Yes (saves cost after completion)
• Estimated cost: ~$3-5 USD for the full run

WHAT IT DOES:
────────────────────────────────────────────────────────────────
1. Downloads GalSim COSMOS catalog (~2.3 GB)
2. Renders 20,000 galaxy templates at 0.03"/pix
3. Filters by HLR (0.1-1.5 arcsec)
4. Computes clumpiness metrics
5. Saves to HDF5 (float32, compressed)
6. Uploads to S3

TROUBLESHOOTING:
────────────────────────────────────────────────────────────────
If cluster fails:
1. Check logs: aws s3 ls s3://darkhaloscope/emr_logs/$CLUSTER_ID/ --recursive
2. SSH to cluster: aws emr ssh --cluster-id $CLUSTER_ID
3. Check /mnt/cosmos_workspace/ for partial output

╔════════════════════════════════════════════════════════════════╗
║  Ready to launch? Run this script on emr-launcher!            ║
╚════════════════════════════════════════════════════════════════╝

EOF

