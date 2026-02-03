#!/bin/bash
# ============================================================================
# COMPLETE EMR LAUNCH SCRIPT - Run this on emr-launcher
# ============================================================================
# 
# This script:
# 1. Updates the repository
# 2. Launches the EMR cluster with 32 vCPUs
# 3. Runs in background with nohup
# 4. Shows monitoring commands
# 
# ============================================================================

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Gen5 COSMOS Bank Builder - EMR Launch                        ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Navigate to repo
cd /home/ec2-user/toy_glens_kinematics/dark_halo_scope || {
  echo "❌ Repository not found at /home/ec2-user/toy_glens_kinematics/dark_halo_scope"
  echo "Please clone the repository first"
  exit 1
}

echo "✅ Working directory: $(pwd)"
echo ""

# Update repo
echo "📥 Updating repository..."
git pull 2>/dev/null || echo "Could not pull, using current version"
echo ""

# Launch with nohup
echo "🚀 Launching EMR job with nohup..."
echo "   Output: ~/cosmos_bank_emr_launch.log"
echo ""

nohup bash emr/gen5/launch_cosmos_bank_builder.sh > ~/cosmos_bank_emr_launch.log 2>&1 &
LAUNCH_PID=$!

echo "✅ EMR launch started in background"
echo "   PID: $LAUNCH_PID"
echo ""

# Wait for initial startup
echo "⏳ Waiting 20 seconds for cluster to launch..."
sleep 20

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Initial Launch Output                                         ║"
echo "╚════════════════════════════════════════════════════════════════╝"
head -60 ~/cosmos_bank_emr_launch.log 2>/dev/null || echo "Log file not yet created"

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Cluster Information                                           ║"
echo "╚════════════════════════════════════════════════════════════════╝"

if [ -f /tmp/cosmos_bank_cluster_id.txt ]; then
  CLUSTER_ID=$(cat /tmp/cosmos_bank_cluster_id.txt)
  echo "✅ Cluster ID: $CLUSTER_ID"
  echo ""
  echo "📊 Checking cluster status..."
  aws emr describe-cluster --cluster-id $CLUSTER_ID \
    --query 'Cluster.{State:Status.State,Name:Name,Created:Status.Timeline.CreationDateTime}' \
    --output table 2>/dev/null || echo "Status check will be available shortly"
else
  echo "⏳ Cluster ID not yet available"
  echo "   Check in 30 seconds: cat /tmp/cosmos_bank_cluster_id.txt"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Monitoring Commands                                           ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "📋 View live logs:"
echo "   tail -f ~/cosmos_bank_emr_launch.log"
echo ""
echo "🔍 Get cluster ID:"
echo "   cat /tmp/cosmos_bank_cluster_id.txt"
echo ""
echo "📊 Check cluster status:"
echo "   CLUSTER_ID=\$(cat /tmp/cosmos_bank_cluster_id.txt)"
echo "   aws emr describe-cluster --cluster-id \$CLUSTER_ID \\"
echo "     --query 'Cluster.Status.State' --output text"
echo ""
echo "📝 Check step progress:"
echo "   aws emr list-steps --cluster-id \$CLUSTER_ID \\"
echo "     --query 'Steps[*].[Name,Status.State]' --output table"
echo ""
echo "☁️  Check S3 output (after completion):"
echo "   aws s3 ls s3://darkhaloscope/cosmos/"
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  What Happens Next                                             ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "⏱️  Runtime: 30-60 minutes"
echo "💰 Cost: ~\$3-5 USD"
echo "🔄 Auto-terminates: Yes"
echo ""
echo "Steps:"
echo "  1. ✅ EMR cluster launching (32 vCPUs, r6i.8xlarge)"
echo "  2. ⏳ Downloads COSMOS catalog (~2.3 GB)"
echo "  3. ⏳ Renders 20,000 galaxy templates"
echo "  4. ⏳ Computes metrics and validates"
echo "  5. ⏳ Uploads to S3"
echo "  6. ⏳ Cluster terminates"
echo ""
echo "Output:"
echo "  s3://darkhaloscope/cosmos/cosmos_bank_20k_parametric_v1.h5"
echo "  s3://darkhaloscope/cosmos/cosmos_bank_config_20k_v1.json"
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  You can now disconnect - the job will continue running!       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

