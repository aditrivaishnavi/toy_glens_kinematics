#!/bin/bash
# ============================================================================
# Run this script on emr-launcher to start the COSMOS bank EMR job
# ============================================================================
set -e

echo "=========================================="
echo "Gen5 COSMOS Bank Builder - EMR Launch"
echo "=========================================="

# Navigate to repo
cd /home/ec2-user/toy_glens_kinematics/dark_halo_scope

echo "✅ Working directory: $(pwd)"
echo ""

# Launch with nohup
echo "🚀 Launching EMR job with nohup..."
nohup bash emr/gen5/launch_cosmos_bank_builder.sh > ~/cosmos_bank_emr_launch.log 2>&1 &
LAUNCH_PID=$!

echo ""
echo "✅ EMR launch started in background"
echo "   PID: $LAUNCH_PID"
echo "   Log: ~/cosmos_bank_emr_launch.log"
echo ""
echo "📊 Monitor with:"
echo "   tail -f ~/cosmos_bank_emr_launch.log"
echo ""
echo "🔍 Check cluster ID (after ~30 seconds):"
echo "   cat /tmp/cosmos_bank_cluster_id.txt"
echo ""
echo "⏱️  The cluster will take 30-60 minutes to complete."
echo "   It will auto-terminate when done."
echo ""
echo "=========================================="

# Wait a bit and show initial output
sleep 10

echo "📋 First 30 lines of output:"
echo "=========================================="
head -30 ~/cosmos_bank_emr_launch.log
echo "=========================================="
echo ""
echo "✅ Launch initiated successfully!"
echo "   Continue monitoring: tail -f ~/cosmos_bank_emr_launch.log"
echo ""

