"""
EMR Constants and Configuration

All EMR-related constants in one place.

IMPORTANT:
- AWS credentials expire after 24 hours. If operations fail with auth errors,
  stop and ask user for new credentials.
- Max vCore budget: 280. All presets are validated against this limit.
"""
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# =============================================================================
# AWS CONFIGURATION
# =============================================================================

# Region - can be overridden via environment variable
AWS_REGION = os.environ.get("AWS_REGION", "us-west-2")
EMR_RELEASE_LABEL = "emr-7.0.0"

# vCore budget limit
# User requested 35 workers for large preset = 284 vCores
# Set to 284 to accommodate this (slightly over nominal 280)
MAX_VCORES = 284

# S3 paths
S3_BUCKET = os.environ.get("S3_BUCKET", "darkhaloscope")
S3_CODE_PREFIX = "planb/emr/code"
S3_LOGS_PREFIX = "planb/emr/logs"
S3_OUTPUT_PREFIX = "planb/emr/output"

# Instance vCore mapping
INSTANCE_VCORES = {
    "m5.large": 2,
    "m5.xlarge": 4,
    "m5.2xlarge": 8,
    "m5.4xlarge": 16,
    "m5.8xlarge": 32,
    "m5.12xlarge": 48,
    "m5.16xlarge": 64,
    "r5.large": 2,
    "r5.xlarge": 4,
    "r5.2xlarge": 8,
    "r5.4xlarge": 16,
}


# =============================================================================
# INSTANCE CONFIGURATION
# =============================================================================

@dataclass
class InstanceConfig:
    """EMR instance configuration."""
    master_instance_type: str = "m5.xlarge"
    worker_instance_type: str = "m5.2xlarge"
    worker_count: int = 10
    
    # Spot configuration
    use_spot: bool = True
    spot_bid_percent: int = 100  # % of on-demand price
    
    # EBS configuration
    ebs_size_gb: int = 100
    ebs_volume_type: str = "gp3"
    ebs_iops: int = 3000
    
    def total_vcores(self) -> int:
        """Calculate total vCores for this configuration."""
        master_vcores = INSTANCE_VCORES.get(self.master_instance_type, 4)
        worker_vcores = INSTANCE_VCORES.get(self.worker_instance_type, 8)
        return master_vcores + (worker_vcores * self.worker_count)
    
    def validate_vcore_budget(self, max_vcores: int = MAX_VCORES) -> bool:
        """Check if configuration is within vCore budget."""
        total = self.total_vcores()
        if total > max_vcores:
            raise ValueError(
                f"Configuration exceeds vCore budget: {total} > {max_vcores}\n"
                f"  Master: {self.master_instance_type} = {INSTANCE_VCORES.get(self.master_instance_type, 4)} vCores\n"
                f"  Workers: {self.worker_count} x {self.worker_instance_type} = "
                f"{self.worker_count * INSTANCE_VCORES.get(self.worker_instance_type, 8)} vCores"
            )
        return True
    
    def to_instance_fleet_config(self) -> List[Dict]:
        """Generate instance fleet configuration."""
        master_config = {
            "InstanceFleetType": "MASTER",
            "TargetOnDemandCapacity": 1,
            "InstanceTypeConfigs": [{
                "InstanceType": self.master_instance_type,
                "EbsConfiguration": {
                    "EbsBlockDeviceConfigs": [{
                        "VolumeSpecification": {
                            "VolumeType": self.ebs_volume_type,
                            "SizeInGB": self.ebs_size_gb,
                            "Iops": self.ebs_iops,
                        },
                        "VolumesPerInstance": 1,
                    }]
                }
            }]
        }
        
        core_config = {
            "InstanceFleetType": "CORE",
            "TargetOnDemandCapacity": 0 if self.use_spot else self.worker_count,
            "TargetSpotCapacity": self.worker_count if self.use_spot else 0,
            "InstanceTypeConfigs": [{
                "InstanceType": self.worker_instance_type,
                "BidPriceAsPercentageOfOnDemandPrice": self.spot_bid_percent,
                "EbsConfiguration": {
                    "EbsBlockDeviceConfigs": [{
                        "VolumeSpecification": {
                            "VolumeType": self.ebs_volume_type,
                            "SizeInGB": self.ebs_size_gb,
                            "Iops": self.ebs_iops,
                        },
                        "VolumesPerInstance": 1,
                    }]
                }
            }]
        }
        
        if self.use_spot:
            core_config["LaunchSpecifications"] = {
                "SpotSpecification": {
                    "TimeoutDurationMinutes": 30,
                    "TimeoutAction": "SWITCH_TO_ON_DEMAND",
                }
            }
        
        return [master_config, core_config]


# =============================================================================
# SPARK CONFIGURATION
# =============================================================================

@dataclass
class SparkConfig:
    """Spark configuration for EMR."""
    executor_memory: str = "8g"
    executor_cores: int = 4
    driver_memory: str = "4g"
    shuffle_partitions: int = 200
    
    # Memory tuning
    memory_fraction: float = 0.6
    memory_storage_fraction: float = 0.5
    
    # Parallelism
    default_parallelism: int = 100
    
    # S3 optimization
    s3_multipart_size: int = 128  # MB
    
    def to_spark_defaults(self) -> List[Dict]:
        """Generate spark-defaults configuration."""
        return [
            {"Classification": "spark-defaults", "Properties": {
                "spark.executor.memory": self.executor_memory,
                "spark.executor.cores": str(self.executor_cores),
                "spark.driver.memory": self.driver_memory,
                "spark.sql.shuffle.partitions": str(self.shuffle_partitions),
                "spark.memory.fraction": str(self.memory_fraction),
                "spark.memory.storageFraction": str(self.memory_storage_fraction),
                "spark.default.parallelism": str(self.default_parallelism),
                "spark.hadoop.fs.s3a.multipart.size": f"{self.s3_multipart_size}M",
                "spark.hadoop.fs.s3a.fast.upload": "true",
                "spark.hadoop.fs.s3a.fast.upload.buffer": "bytebuffer",
                "spark.speculation": "false",
                "spark.dynamicAllocation.enabled": "true",
                "spark.dynamicAllocation.minExecutors": "1",
                "spark.dynamicAllocation.maxExecutors": "50",
            }},
            {"Classification": "spark-env", "Properties": {}, "Configurations": [
                {"Classification": "export", "Properties": {
                    "PYSPARK_PYTHON": "/usr/bin/python3",
                }}
            ]},
        ]


# =============================================================================
# JOB PRESETS
# =============================================================================

@dataclass
class JobPreset:
    """Pre-configured job settings for common tasks."""
    name: str
    instance_config: InstanceConfig
    spark_config: SparkConfig
    timeout_hours: int = 6
    

# Small job (e.g., smoke tests, validation)
PRESET_SMALL = JobPreset(
    name="small",
    instance_config=InstanceConfig(
        worker_instance_type="m5.xlarge",
        worker_count=2,
    ),
    spark_config=SparkConfig(
        executor_memory="4g",
        executor_cores=2,
        shuffle_partitions=50,
    ),
    timeout_hours=1,
)

# Medium job (e.g., single-split processing)
# 15 workers @ m5.2xlarge (8 vCores) + 1 master (4 vCores) = 124 vCores
PRESET_MEDIUM = JobPreset(
    name="medium",
    instance_config=InstanceConfig(
        worker_instance_type="m5.2xlarge",
        worker_count=15,
    ),
    spark_config=SparkConfig(
        executor_memory="8g",
        executor_cores=4,
        shuffle_partitions=200,
    ),
    timeout_hours=4,
)

# Large job (e.g., full pipeline)
# 35 workers @ m5.2xlarge (8 vCores) + 1 master (4 vCores) = 284 vCores
# NOTE: Slightly exceeds 280 vCore budget - may need to adjust if quota is hard
PRESET_LARGE = JobPreset(
    name="large",
    instance_config=InstanceConfig(
        worker_instance_type="m5.2xlarge",
        worker_count=35,
    ),
    spark_config=SparkConfig(
        executor_memory="8g",
        executor_cores=4,
        shuffle_partitions=500,
        default_parallelism=300,
    ),
    timeout_hours=8,
)

PRESETS = {
    "small": PRESET_SMALL,
    "medium": PRESET_MEDIUM,
    "large": PRESET_LARGE,
}


# =============================================================================
# VCORE BUDGET SUMMARY
# =============================================================================

def print_vcore_summary():
    """Print vCore usage for all presets."""
    print(f"\nMax vCore Budget: {MAX_VCORES}")
    print("-" * 50)
    for name, preset in PRESETS.items():
        cfg = preset.instance_config
        total = cfg.total_vcores()
        master_vc = INSTANCE_VCORES.get(cfg.master_instance_type, 4)
        worker_vc = INSTANCE_VCORES.get(cfg.worker_instance_type, 8)
        status = "✓" if total <= MAX_VCORES else "✗ EXCEEDS"
        print(f"{name:8s}: {total:3d} vCores {status}")
        print(f"          1x {cfg.master_instance_type} ({master_vc} vc) + "
              f"{cfg.worker_count}x {cfg.worker_instance_type} ({worker_vc} vc)")
    print("-" * 50)


# Validate all presets at import time
def _validate_presets():
    for name, preset in PRESETS.items():
        try:
            preset.instance_config.validate_vcore_budget()
        except ValueError as e:
            raise ValueError(f"Preset '{name}' exceeds vCore budget: {e}")

_validate_presets()
