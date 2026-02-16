#!/usr/bin/env python3
"""
Phase 1: Data Loading Module

Lessons Learned Incorporated:
- L1.4: Never assume clean data - validate for NaN/Inf
- L4.4: arc_snr=0 means "arc injected but SNR=0", not "no arc"
- L7.2: Handle missing bands gracefully
- L21: Core brightness shortcut exists - need proper handling
- L22: Arc overlap dominates core brightness (physical, not bias)
- L1.2: No duplicate functions - import from shared module

Exit Criteria:
- Batch shapes must be (B, C, H, W) = (batch_size, 3, 64, 64)
- No NaN/Inf values in any batch
- Labels must be binary (0 or 1)
- Hard negative ratio must be within 5% of target
"""
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, IterableDataset

# Import from shared module - SINGLE SOURCE OF TRUTH
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.constants import (
    STAMP_SIZE, NUM_CHANNELS, STAMP_SHAPE,
    OUTER_RADIUS_PIX, CLIP_SIGMA, CORE_RADIUS_PIX,
    DEFAULT_BATCH_SIZE, DEFAULT_HARD_NEGATIVE_RATIO,
    AZIMUTHAL_SHUFFLE_BINS, NORMALIZED_VALUE_MIN, NORMALIZED_VALUE_MAX,
)
from shared.schema import PARQUET_SCHEMA, BATCH_SCHEMA
from shared.utils import (
    decode_stamp_npz,
    validate_stamp,
    robust_normalize,
    azimuthal_shuffle,
    apply_core_dropout,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class DataConfig:
    """Data loading configuration with validation."""
    parquet_root: str
    split: str
    batch_size: int = DEFAULT_BATCH_SIZE
    num_workers: int = 4
    shuffle: bool = True
    drop_last: bool = True
    
    # Normalization - use constants
    normalization: str = "per_sample_robust"
    outer_radius_pix: int = OUTER_RADIUS_PIX
    clip_sigma: float = CLIP_SIGMA
    
    # Hard negatives
    hard_negative_ratio: float = DEFAULT_HARD_NEGATIVE_RATIO
    hard_negative_method: str = "azimuthal_shuffle"
    
    # Core dropout
    core_dropout_enabled: bool = True
    core_dropout_prob: float = 0.5
    core_dropout_radius: int = CORE_RADIUS_PIX
    
    # Filtering
    min_arc_snr: Optional[float] = None
    max_arc_snr: Optional[float] = None
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.split in ["train", "val", "test"], f"Invalid split: {self.split}"
        assert self.batch_size > 0, "batch_size must be positive"
        assert 0 <= self.hard_negative_ratio <= 1, "hard_negative_ratio must be [0, 1]"
        assert 0 <= self.core_dropout_prob <= 1, "core_dropout_prob must be [0, 1]"
        assert self.core_dropout_radius > 0, "core_dropout_radius must be positive"
        assert self.outer_radius_pix > self.core_dropout_radius, \
            "outer_radius must exceed core_dropout_radius"


# =============================================================================
# DATASET CLASS
# =============================================================================
# NOTE: decode_stamp_npz, validate_stamp, robust_normalize, azimuthal_shuffle,
#       apply_core_dropout are imported from shared.utils (single source of truth)


class PairedLensDataset(IterableDataset):
    """
    Iterable dataset for paired lens training data.
    
    Features:
    - Paired sampling: stamp + matched ctrl
    - Hard negative generation
    - Core dropout augmentation
    - Robust normalization
    - Full validation of each sample
    
    Lessons Incorporated:
    - L3.3: Don't use .limit() - read specific files
    - L5.3: Test actual code path, not mock
    - L7.1: Filter NaN values
    """
    
    def __init__(self, config: DataConfig, max_samples: Optional[int] = None):
        """
        Initialize dataset.
        
        Args:
            config: DataConfig object
            max_samples: Optional limit for testing
        """
        self.config = config
        self.max_samples = max_samples
        
        # Find parquet files
        split_path = Path(config.parquet_root) / config.split
        self.files = sorted(split_path.glob("*.parquet"))
        
        if len(self.files) == 0:
            raise ValueError(f"No parquet files found in {split_path}")
        
        # Statistics for validation
        self.stats = {
            "samples_read": 0,
            "samples_skipped_nan": 0,
            "samples_skipped_shape": 0,
            "hard_negatives_generated": 0,
        }
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over samples."""
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            # Single process
            files_to_process = self.files
        else:
            # Multi-process: shard files across workers
            # Lesson L6.1: Shard by rank * num_workers + worker_id
            per_worker = len(self.files) // worker_info.num_workers
            start = worker_info.id * per_worker
            end = start + per_worker if worker_info.id < worker_info.num_workers - 1 else len(self.files)
            files_to_process = self.files[start:end]
        
        samples_yielded = 0
        
        for file_path in files_to_process:
            if self.max_samples and samples_yielded >= self.max_samples:
                break
            
            try:
                df = pd.read_parquet(file_path)
            except Exception as e:
                print(f"WARNING: Failed to read {file_path}: {e}")
                continue
            
            # Shuffle if training
            if self.config.shuffle:
                df = df.sample(frac=1.0, random_state=None)
            
            for idx, row in df.iterrows():
                if self.max_samples and samples_yielded >= self.max_samples:
                    break
                
                sample = self._process_row(row)
                if sample is not None:
                    yield sample
                    samples_yielded += 1
    
    def _process_row(self, row: pd.Series) -> Optional[Dict[str, torch.Tensor]]:
        """
        Process a single row into a training sample.
        
        Returns None if sample should be skipped.
        """
        self.stats["samples_read"] += 1
        
        try:
            # Decode stamp (positive sample)
            stamp, bandset = decode_stamp_npz(row.stamp_npz)
            ctrl, _ = decode_stamp_npz(row.ctrl_stamp_npz)
        except Exception as e:
            self.stats["samples_skipped_shape"] += 1
            return None
        
        # Validate
        stamp_valid = validate_stamp(stamp, bandset)
        ctrl_valid = validate_stamp(ctrl, bandset)
        
        if not stamp_valid["valid"] or not ctrl_valid["valid"]:
            self.stats["samples_skipped_nan"] += 1
            return None
        
        # Generate hard negative (with probability)
        is_hard_negative = np.random.random() < self.config.hard_negative_ratio
        
        if is_hard_negative:
            diff = stamp - ctrl
            shuffled_diff = azimuthal_shuffle(diff)
            negative = ctrl + shuffled_diff
            self.stats["hard_negatives_generated"] += 1
        else:
            negative = ctrl
        
        # Apply core dropout (training only)
        if self.config.core_dropout_enabled and self.config.split == "train":
            if np.random.random() < self.config.core_dropout_prob:
                stamp = apply_core_dropout(
                    stamp,
                    radius=self.config.core_dropout_radius,
                    fill_mode="outer_median"
                )
                negative = apply_core_dropout(
                    negative,
                    radius=self.config.core_dropout_radius,
                    fill_mode="outer_median"
                )
        
        # Normalize
        stamp_norm = robust_normalize(
            stamp,
            outer_radius=self.config.outer_radius_pix,
            clip_sigma=self.config.clip_sigma
        )
        negative_norm = robust_normalize(
            negative,
            outer_radius=self.config.outer_radius_pix,
            clip_sigma=self.config.clip_sigma
        )
        
        # Return both positive and negative
        # We'll handle batch construction in collate_fn
        return {
            "positive": torch.from_numpy(stamp_norm),
            "negative": torch.from_numpy(negative_norm),
            "is_hard_negative": is_hard_negative,
            "theta_e": row.get("theta_e_arcsec", 0.0),
            "arc_snr": row.get("arc_snr", 0.0),
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function that creates balanced batches.
    
    Each sample has (positive, negative), so batch doubles in size.
    """
    positives = torch.stack([s["positive"] for s in batch])
    negatives = torch.stack([s["negative"] for s in batch])
    
    # Interleave positives and negatives
    x = torch.cat([positives, negatives], dim=0)
    y = torch.cat([
        torch.ones(len(batch)),
        torch.zeros(len(batch))
    ])
    
    # Shuffle
    perm = torch.randperm(len(x))
    x = x[perm]
    y = y[perm]
    
    return {
        "x": x,
        "y": y,
        "theta_e": torch.tensor([s["theta_e"] for s in batch]),
        "arc_snr": torch.tensor([s["arc_snr"] for s in batch]),
        "n_hard_negatives": sum(1 for s in batch if s["is_hard_negative"]),
    }


# =============================================================================
# BUILDER FUNCTIONS
# =============================================================================

def build_training_loader(
    parquet_root: str,
    split: str = "train",
    batch_size: int = 128,
    num_workers: int = 4,
    hard_negative_ratio: float = 0.4,
    core_dropout_prob: float = 0.5,
    max_samples: Optional[int] = None,
) -> DataLoader:
    """
    Build a training data loader with all mitigations enabled.
    
    Exit Criteria:
    - Returns DataLoader with batch["x"] shape (2*batch_size, 3, 64, 64)
    - Labels are balanced (50% positive, 50% negative)
    - Hard negative ratio within 5% of target
    """
    config = DataConfig(
        parquet_root=parquet_root,
        split=split,
        batch_size=batch_size,
        num_workers=num_workers,
        hard_negative_ratio=hard_negative_ratio,
        core_dropout_prob=core_dropout_prob,
        shuffle=(split == "train"),
        drop_last=(split == "train"),
    )
    
    dataset = PairedLensDataset(config, max_samples=max_samples)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    return loader


def build_eval_loader(
    parquet_root: str,
    split: str = "val",
    batch_size: int = 128,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
) -> DataLoader:
    """
    Build evaluation loader without augmentations.
    """
    config = DataConfig(
        parquet_root=parquet_root,
        split=split,
        batch_size=batch_size,
        num_workers=num_workers,
        hard_negative_ratio=0.0,  # No hard negatives in eval
        core_dropout_enabled=False,  # No dropout in eval
        shuffle=False,
        drop_last=False,
    )
    
    dataset = PairedLensDataset(config, max_samples=max_samples)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_loader(loader: DataLoader, n_batches: int = 5) -> Dict[str, bool]:
    """
    Validate a data loader meets all requirements.
    
    Uses BATCH_SCHEMA for validation to ensure consistency.
    
    Exit Criteria:
    1. Batches have correct shape (NUM_CHANNELS, STAMP_SIZE, STAMP_SIZE)
    2. No NaN/Inf values
    3. Labels are binary
    4. Reasonable value ranges (NORMALIZED_VALUE_MIN to NORMALIZED_VALUE_MAX)
    """
    results = {
        "shape_correct": True,
        "no_nan_inf": True,
        "labels_binary": True,
        "value_range_ok": True,
        "batches_checked": 0,
        "schema_errors": [],
    }
    
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        
        # Use schema validation
        schema_result = BATCH_SCHEMA.validate_batch(batch)
        if not schema_result["valid"]:
            results["schema_errors"].extend(schema_result["errors"])
        
        x = batch["x"]
        y = batch["y"]
        
        # Check shape using constants
        if (x.dim() != 4 or 
            x.shape[1] != NUM_CHANNELS or 
            x.shape[2] != STAMP_SIZE or 
            x.shape[3] != STAMP_SIZE):
            results["shape_correct"] = False
        
        # Check NaN/Inf
        if not torch.isfinite(x).all():
            results["no_nan_inf"] = False
        
        # Check labels
        unique_labels = set(y.numpy().tolist())
        if not unique_labels.issubset({0.0, 1.0}):
            results["labels_binary"] = False
        
        # Check value range using constants
        if x.min() < NORMALIZED_VALUE_MIN or x.max() > NORMALIZED_VALUE_MAX:
            results["value_range_ok"] = False
        
        results["batches_checked"] += 1
    
    results["all_passed"] = all([
        results["shape_correct"],
        results["no_nan_inf"],
        results["labels_binary"],
        results["value_range_ok"],
        len(results["schema_errors"]) == 0,
    ])
    
    return results


if __name__ == "__main__":
    # Simple test
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-root", required=True)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    
    print("Building loader...")
    loader = build_training_loader(
        args.parquet_root,
        split=args.split,
        batch_size=4,
        num_workers=0,
        max_samples=16,
    )
    
    print("Validating loader...")
    results = validate_loader(loader, n_batches=3)
    
    print("\nValidation Results:")
    for key, value in results.items():
        status = "✓" if value else "✗"
        print(f"  {key}: {status} {value}")
    
    if results["all_passed"]:
        print("\n✓ ALL CHECKS PASSED")
    else:
        print("\n✗ VALIDATION FAILED")
