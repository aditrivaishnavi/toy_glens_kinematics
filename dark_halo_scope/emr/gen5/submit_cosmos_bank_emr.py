#!/usr/bin/env python3
"""
Launch EMR cluster to build COSMOS source bank.
Based on submit_phase4_pipeline_emr_cluster.py pattern.
"""
import boto3
import sys

def main():
    print("=" * 70)
    print("Gen5 COSMOS Bank Builder - EMR Launch")
    print("=" * 70)
    
    # Configuration
    region = "us-east-2"
    subnet_id = "subnet-01ca3ae3325cec025"
    key_name = "root"
    
    # S3 paths
    bootstrap_s3 = "s3://darkhaloscope/emr_bootstrap/cosmos_bank_builder_bootstrap.sh"
    code_s3_base = "s3://darkhaloscope/code/gen5/"
    
    # Upload files
    print("\n📤 Uploading bootstrap and code to S3...")
    s3 = boto3.client("s3", region_name=region)
    
    # Upload bootstrap
    with open("emr/gen5/cosmos_bank_builder_bootstrap.sh", "rb") as f:
        s3.put_object(Bucket="darkhaloscope", Key="emr_bootstrap/cosmos_bank_builder_bootstrap.sh", Body=f)
    
    # Upload main script
    with open("emr/gen5/build_cosmos_bank_emr.py", "rb") as f:
        s3.put_object(Bucket="darkhaloscope", Key="code/gen5/build_cosmos_bank_emr.py", Body=f)
    
    # Upload COSMOS loader
    import os
    for root, dirs, files in os.walk("models/dhs_cosmos_galsim_code/dhs_cosmos"):
        for file in files:
            if file.endswith('.py'):
                local_path = os.path.join(root, file)
                s3_key = local_path.replace("models/dhs_cosmos_galsim_code/", "code/gen5/")
                with open(local_path, "rb") as f:
                    s3.put_object(Bucket="darkhaloscope", Key=s3_key, Body=f)
    
    print("✅ Files uploaded")
    
    # Create EMR cluster
    print("\n🚀 Launching EMR cluster...")
    emr = boto3.client("emr", region_name=region)
    
    response = emr.run_job_flow(
        Name="gen5-cosmos-bank-builder",
        ReleaseLabel="emr-6.15.0",
        LogUri="s3://darkhaloscope/emr_logs/",
        VisibleToAllUsers=True,
        
        Applications=[
            {"Name": "Hadoop"},
            {"Name": "Spark"},
        ],
        
        Instances={
            "InstanceGroups": [
                {
                    "Name": "Master",
                    "InstanceRole": "MASTER",
                    "InstanceType": "r6i.8xlarge",
                    "InstanceCount": 1,
                },
            ],
            "KeepJobFlowAliveWhenNoSteps": False,
            "TerminationProtected": False,
            "Ec2KeyName": key_name,
            "Ec2SubnetId": subnet_id,
        },
        
        Steps=[
            {
                "Name": "Download-Code",
                "ActionOnFailure": "TERMINATE_CLUSTER",
                "HadoopJarStep": {
                    "Jar": "command-runner.jar",
                    "Args": [
                        "bash", "-c",
                        f"sudo mkdir -p /mnt/code && sudo chmod 777 /mnt/code && "
                        f"aws s3 cp --recursive {code_s3_base}dhs_cosmos/ /mnt/code/dhs_cosmos/ && "
                        f"aws s3 cp {code_s3_base}build_cosmos_bank_emr.py /mnt/code/"
                    ],
                },
            },
            {
                "Name": "Build-COSMOS-Bank",
                "ActionOnFailure": "CONTINUE",
                "HadoopJarStep": {
                    "Jar": "command-runner.jar",
                    "Args": ["python3", "/mnt/code/build_cosmos_bank_emr.py"],
                },
            },
        ],
        
        BootstrapActions=[
            {
                "Name": "Install-Dependencies",
                "ScriptBootstrapAction": {
                    "Path": bootstrap_s3,
                },
            },
        ],
        
        ServiceRole="EMR_DefaultRole",
        JobFlowRole="EMR_EC2_DefaultRole",
    )
    
    cluster_id = response["JobFlowId"]
    
    # Save cluster ID
    with open("/tmp/cosmos_bank_cluster_id.txt", "w") as f:
        f.write(cluster_id)
    
    print(f"\n✅ Cluster launched: {cluster_id}")
    print(f"   Region: {region}")
    print(f"   Instance: r6i.8xlarge (32 vCPUs, 256 GB RAM)")
    print(f"\n📊 Monitor:")
    print(f"   aws emr describe-cluster --cluster-id {cluster_id} --region {region}")
    print(f"\n⏱️  Runtime: 30-60 minutes")
    print(f"💰 Cost: ~$3-5 USD")
    print(f"\n📦 Output:")
    print(f"   s3://darkhaloscope/cosmos/cosmos_bank_20k_parametric_v1.h5")
    print("=" * 70)
    
    return cluster_id

if __name__ == "__main__":
    try:
        cluster_id = main()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

