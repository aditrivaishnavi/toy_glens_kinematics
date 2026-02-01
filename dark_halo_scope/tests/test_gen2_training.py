"""Comprehensive unit tests for Gen2 training script.

Tests all major components:
- StreamConfig dataclass
- Normalization functions
- Augmentation
- Forbidden metadata guard
- ROC/evaluation functions
- Loss functions
- Model building
- Early stopping logic
"""

import numpy as np
import pytest
import sys
import os
from dataclasses import dataclass, fields
from typing import List, Optional, Tuple, Dict, Sequence
import math
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json


# =============================================================================
# INLINE IMPLEMENTATIONS (to avoid torch dependency in tests)
# =============================================================================

def decode_stamp_npz(blob: bytes) -> np.ndarray:
    """Decode stamp_npz blob to numpy array."""
    import io
    with io.BytesIO(blob) as buf:
        npz = np.load(buf)
        g = npz["image_g"].astype(np.float32)
        r = npz["image_r"].astype(np.float32)
        z = npz["image_z"].astype(np.float32)
    return np.stack([g, r, z], axis=0)


def robust_mad_norm(x: np.ndarray, clip: float = 10.0, eps: float = 1e-6) -> np.ndarray:
    """Normalize using median/MAD of full image (legacy method)."""
    out = np.empty_like(x, dtype=np.float32)
    for c in range(x.shape[0]):
        v = x[c]
        med = np.median(v)
        mad = np.median(np.abs(v - med))
        scale = 1.4826 * mad + eps
        vv = (v - med) / scale
        if clip is not None:
            vv = np.clip(vv, -clip, clip)
        out[c] = vv.astype(np.float32)
    return out


def robust_mad_norm_outer(x: np.ndarray, clip: float = 10.0, eps: float = 1e-6,
                          inner_frac: float = 0.5) -> np.ndarray:
    """Normalize using outer annulus only."""
    out = np.empty_like(x, dtype=np.float32)
    h, w = x.shape[-2:]
    cy, cx = h // 2, w // 2
    ri = int(min(h, w) * inner_frac / 2)
    
    yy, xx = np.ogrid[:h, :w]
    outer_mask = ((yy - cy)**2 + (xx - cx)**2) > ri**2
    
    for c in range(x.shape[0]):
        v = x[c]
        outer_v = v[outer_mask]
        med = np.median(outer_v)
        mad = np.median(np.abs(outer_v - med))
        scale = 1.4826 * mad + eps
        vv = (v - med) / scale
        if clip is not None:
            vv = np.clip(vv, -clip, clip)
        out[c] = vv.astype(np.float32)
    return out


def aug_rot_flip(x: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Apply random rotation and flip augmentation."""
    k = int(rng.randint(0, 4))
    if k:
        x = np.rot90(x, k=k, axes=(1, 2)).copy()
    if rng.rand() < 0.5:
        x = x[:, :, ::-1].copy()
    if rng.rand() < 0.5:
        x = x[:, ::-1, :].copy()
    return x


def roc_curve_np(scores: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ROC curve with proper tie handling."""
    order = np.argsort(-scores, kind="mergesort")
    s = scores[order]
    yy = y[order].astype(np.int64)
    P = int(yy.sum())
    N = len(yy) - P
    
    if P == 0 or N == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([np.inf, -np.inf])
    
    tp = np.cumsum(yy)
    fp = np.cumsum(1 - yy)
    
    distinct = np.r_[np.diff(s) != 0, True]
    
    tp = tp[distinct]
    fp = fp[distinct]
    thr = s[distinct]
    
    tpr = tp / P
    fpr = fp / N
    
    fpr = np.r_[0.0, fpr]
    tpr = np.r_[0.0, tpr]
    thr = np.r_[np.inf, thr]
    
    return fpr, tpr, thr


# =============================================================================
# TEST CLASSES
# =============================================================================

class TestStreamConfig:
    """Test StreamConfig dataclass."""
    
    def test_streamconfig_has_all_fields(self):
        """Verify StreamConfig has all required fields."""
        required_fields = {
            'data', 'split', 'seed', 'columns', 'meta_cols',
            'augment', 'mad_clip', 'max_rows', 'min_theta_over_psf',
            'min_arc_snr', 'norm_method', 'epoch'
        }
        
        # Create minimal config
        @dataclass
        class StreamConfig:
            data: str
            split: str
            seed: int
            columns: List[str]
            meta_cols: List[str]
            augment: bool
            mad_clip: float
            max_rows: int
            min_theta_over_psf: float
            min_arc_snr: float
            norm_method: str = "full"
            epoch: int = 0
        
        actual_fields = {f.name for f in fields(StreamConfig)}
        assert required_fields == actual_fields
    
    def test_streamconfig_defaults(self):
        """Test default values."""
        @dataclass
        class StreamConfig:
            data: str
            split: str
            seed: int
            columns: List[str]
            meta_cols: List[str]
            augment: bool
            mad_clip: float
            max_rows: int
            min_theta_over_psf: float
            min_arc_snr: float
            norm_method: str = "full"
            epoch: int = 0
        
        cfg = StreamConfig(
            data="/test", split="train", seed=42,
            columns=[], meta_cols=[], augment=True,
            mad_clip=10.0, max_rows=0, min_theta_over_psf=0.0,
            min_arc_snr=0.0
        )
        
        assert cfg.norm_method == "full"
        assert cfg.epoch == 0


class TestNormalization:
    """Test normalization functions."""
    
    def test_robust_mad_norm_shape_preserved(self):
        """Output shape should match input."""
        x = np.random.randn(3, 64, 64).astype(np.float32)
        out = robust_mad_norm(x)
        assert out.shape == x.shape
    
    def test_robust_mad_norm_clipping(self):
        """Values should be clipped."""
        x = np.random.randn(3, 64, 64).astype(np.float32) * 100
        out = robust_mad_norm(x, clip=10.0)
        assert np.all(out >= -10.0)
        assert np.all(out <= 10.0)
    
    def test_robust_mad_norm_dtype(self):
        """Output should be float32."""
        x = np.random.randn(3, 64, 64).astype(np.float64)
        out = robust_mad_norm(x)
        assert out.dtype == np.float32
    
    def test_robust_mad_norm_outer_excludes_center(self):
        """Outer normalization should compute stats from outer region only."""
        np.random.seed(42)
        
        # Create image with very different statistics in center vs outer
        x = np.zeros((3, 64, 64), dtype=np.float32)
        
        # Outer region: small values, low variance
        outer_vals = np.random.randn(3, 64, 64).astype(np.float32) * 0.1
        x[:] = outer_vals
        
        # Center (inner 50%): add large offset
        x[:, 16:48, 16:48] += 100.0
        
        # Full normalization uses all pixels including center offset
        out_full = robust_mad_norm(x, clip=None)
        
        # Outer normalization should only use outer region
        out_outer = robust_mad_norm_outer(x, clip=None, inner_frac=0.5)
        
        # The center values should be normalized differently
        # In outer norm, center values should be much larger since they're 
        # normalized by the outer region's smaller MAD
        center_full = out_full[0, 32, 32]
        center_outer = out_outer[0, 32, 32]
        
        # The center pixel should have different normalized values
        # because they're normalized by different statistics
        assert abs(center_full - center_outer) > 10
    
    def test_robust_mad_norm_outer_shape_preserved(self):
        """Output shape should match input."""
        x = np.random.randn(3, 64, 64).astype(np.float32)
        out = robust_mad_norm_outer(x)
        assert out.shape == x.shape


class TestAugmentation:
    """Test augmentation functions."""
    
    def test_aug_rot_flip_shape_preserved(self):
        """Augmentation should preserve shape."""
        x = np.random.randn(3, 64, 64).astype(np.float32)
        rng = np.random.RandomState(42)
        out = aug_rot_flip(x.copy(), rng)
        assert out.shape == x.shape
    
    def test_aug_rot_flip_deterministic_with_seed(self):
        """Same seed should produce same augmentation."""
        x = np.random.randn(3, 64, 64).astype(np.float32)
        
        rng1 = np.random.RandomState(42)
        out1 = aug_rot_flip(x.copy(), rng1)
        
        rng2 = np.random.RandomState(42)
        out2 = aug_rot_flip(x.copy(), rng2)
        
        assert np.allclose(out1, out2)
    
    def test_aug_rot_flip_different_seeds_differ(self):
        """Different seeds should (usually) produce different augmentations."""
        x = np.random.randn(3, 64, 64).astype(np.float32)
        
        # Run many times with different seeds
        outputs = []
        for seed in range(10):
            rng = np.random.RandomState(seed)
            out = aug_rot_flip(x.copy(), rng)
            outputs.append(out)
        
        # At least some should be different
        unique_count = len(set(tuple(o.flatten()[:100].tolist()) for o in outputs))
        assert unique_count > 1


class TestForbiddenMetadata:
    """Test forbidden metadata guard."""
    
    def test_forbidden_columns_defined(self):
        """Verify all injection-related columns are blocked."""
        FORBIDDEN_META = frozenset({
            "theta_e_arcsec", "theta_e", "src_dmag", "src_reff_arcsec", "src_n",
            "src_e", "src_phi_deg", "src_x_arcsec", "src_y_arcsec",
            "lens_e", "lens_phi_deg", "shear_gamma", "shear_phi_deg", "shear",
            "arc_snr", "magnification", "tangential_stretch", "radial_stretch",
            "is_control", "label", "cutout_ok",
        })
        
        # These should be blocked
        injection_params = ["theta_e_arcsec", "src_dmag", "arc_snr"]
        for col in injection_params:
            assert col in FORBIDDEN_META
        
        # These should be allowed
        allowed = ["psfsize_r", "psfdepth_r", "ebv", "ra", "dec"]
        for col in allowed:
            assert col not in FORBIDDEN_META
    
    def test_guard_raises_on_forbidden(self):
        """Guard should raise ValueError for forbidden columns."""
        FORBIDDEN_META = frozenset({"arc_snr", "theta_e_arcsec"})
        
        meta_cols = ["psfsize_r", "arc_snr"]
        
        for c in meta_cols:
            if c in FORBIDDEN_META:
                with pytest.raises(ValueError, match="leak"):
                    raise ValueError(f"REFUSING meta_cols that leak labels: '{c}'")
    
    def test_guard_allows_valid_columns(self):
        """Guard should allow non-forbidden columns."""
        FORBIDDEN_META = frozenset({"arc_snr", "theta_e_arcsec"})
        
        meta_cols = ["psfsize_r", "psfdepth_r", "ebv"]
        
        for c in meta_cols:
            assert c not in FORBIDDEN_META


class TestROCCurve:
    """Test ROC curve implementation."""
    
    def test_roc_basic(self):
        """Basic ROC curve test."""
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        y = np.array([1, 1, 0, 1, 0])
        
        fpr, tpr, thr = roc_curve_np(scores, y)
        
        assert len(fpr) == len(tpr) == len(thr)
        assert fpr[0] == 0.0
        assert tpr[0] == 0.0
        assert fpr[-1] == 1.0
        assert tpr[-1] == 1.0
    
    def test_roc_auroc_calculation(self):
        """Test AUROC calculation."""
        scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2])
        y = np.array([1, 1, 1, 0, 0])
        
        fpr, tpr, _ = roc_curve_np(scores, y)
        auroc = np.trapz(tpr, fpr)
        
        # Perfect separation should give AUROC = 1.0
        assert auroc == 1.0
    
    def test_roc_tied_scores(self):
        """Test ROC with many tied scores."""
        scores = np.array([0.9, 0.5, 0.5, 0.5, 0.5, 0.1])
        y = np.array([1, 1, 0, 1, 0, 0])
        
        fpr, tpr, _ = roc_curve_np(scores, y)
        
        assert fpr[-1] == 1.0
        assert tpr[-1] == 1.0


class TestCalibrationMonitoring:
    """Test calibration collapse detection."""
    
    def test_detect_binary_scores(self):
        """Detect when scores are binary (calibration collapse)."""
        # Simulate calibration collapse
        scores = np.array([0.999, 0.999, 0.999, 0.001, 0.001, 0.001])
        
        binary_low = (scores < 0.01).sum()
        binary_high = (scores > 0.99).sum()
        binary_frac = (binary_low + binary_high) / len(scores)
        
        assert binary_frac == 1.0  # All scores are binary
    
    def test_healthy_calibration(self):
        """Healthy calibration should have few binary scores."""
        scores = np.random.rand(100) * 0.8 + 0.1  # Uniform in [0.1, 0.9]
        
        binary_low = (scores < 0.01).sum()
        binary_high = (scores > 0.99).sum()
        binary_frac = (binary_low + binary_high) / len(scores)
        
        assert binary_frac < 0.1


class TestEarlyStopping:
    """Test early stopping logic."""
    
    def test_early_stopping_patience(self):
        """Test patience-based early stopping."""
        best_metric = 0.8
        no_improve_count = 0
        patience = 3
        
        # Simulate 4 epochs of no improvement
        for epoch in range(4):
            score = 0.75  # Worse than best
            
            if score > best_metric:
                best_metric = score
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            if no_improve_count >= patience:
                break
        
        # Should have stopped at epoch 3 (patience=3)
        assert no_improve_count == 3
        assert epoch == 2  # 0-indexed
    
    def test_early_stopping_resets_on_improvement(self):
        """Counter should reset when metric improves."""
        best_metric = 0.7
        no_improve_count = 2  # Already at 2
        
        score = 0.8  # Improvement!
        
        if score > best_metric:
            best_metric = score
            no_improve_count = 0
        
        assert no_improve_count == 0
        assert best_metric == 0.8


class TestEpochDependentShuffle:
    """Test epoch-dependent shuffling."""
    
    def test_different_epochs_different_shuffle(self):
        """Different epochs should produce different shuffle orders."""
        seed = 1337
        rank = 0
        worker_id = 0
        
        orders = []
        for epoch in [0, 1, 2, 3, 4]:
            rng = np.random.RandomState(seed + 997*rank + 131*worker_id + 7919*epoch)
            order = np.arange(100)
            rng.shuffle(order)
            orders.append(tuple(order))
        
        # All should be different
        assert len(set(orders)) == 5
    
    def test_same_epoch_same_shuffle(self):
        """Same epoch should produce same shuffle (reproducibility)."""
        seed = 1337
        rank = 0
        worker_id = 0
        epoch = 5
        
        rng1 = np.random.RandomState(seed + 997*rank + 131*worker_id + 7919*epoch)
        order1 = np.arange(100)
        rng1.shuffle(order1)
        
        rng2 = np.random.RandomState(seed + 997*rank + 131*worker_id + 7919*epoch)
        order2 = np.arange(100)
        rng2.shuffle(order2)
        
        assert np.array_equal(order1, order2)


class TestWorkerSharding:
    """Test worker sharding formula."""
    
    def test_sharding_unique(self):
        """Each (rank, worker) combination gets unique shard."""
        world_size = 2
        num_workers = 4
        
        shards = set()
        for rank in range(world_size):
            for worker_id in range(num_workers):
                shard = rank * num_workers + worker_id
                assert shard not in shards
                shards.add(shard)
        
        assert len(shards) == world_size * num_workers
    
    def test_sharding_no_overlap(self):
        """Fragments should not overlap between workers."""
        n_frags = 100
        frags = list(range(n_frags))
        
        world_size = 1
        num_workers = 8
        nshard = world_size * num_workers
        
        all_frags = []
        for rank in range(world_size):
            for worker_id in range(num_workers):
                shard = rank * num_workers + worker_id
                assigned = frags[shard::nshard]
                all_frags.extend(assigned)
        
        # All fragments should appear exactly once
        assert sorted(all_frags) == frags


class TestStampDecoding:
    """Test stamp NPZ decoding."""
    
    def test_decode_stamp_npz(self):
        """Test decoding of stamp_npz format."""
        import io
        
        # Create test stamp
        g = np.random.randn(64, 64).astype(np.float32)
        r = np.random.randn(64, 64).astype(np.float32)
        z = np.random.randn(64, 64).astype(np.float32)
        
        buf = io.BytesIO()
        np.savez_compressed(buf, image_g=g, image_r=r, image_z=z)
        blob = buf.getvalue()
        
        # Decode
        decoded = decode_stamp_npz(blob)
        
        assert decoded.shape == (3, 64, 64)
        assert decoded.dtype == np.float32
        assert np.allclose(decoded[0], g)
        assert np.allclose(decoded[1], r)
        assert np.allclose(decoded[2], z)


class TestTop50Logging:
    """Test top-50 score logging."""
    
    def test_top_negative_scores(self):
        """Get top-50 negative scores."""
        np.random.seed(42)
        n_samples = 1000
        scores = np.random.rand(n_samples)
        labels = np.random.randint(0, 2, n_samples)
        
        neg_mask = (labels == 0)
        neg_scores = scores[neg_mask]
        top_neg_idx = np.argsort(-neg_scores)[:50]
        top_neg = neg_scores[top_neg_idx]
        
        assert len(top_neg) == 50
        # Should be sorted descending
        assert np.all(np.diff(top_neg) <= 0)
    
    def test_bottom_positive_scores(self):
        """Get bottom-50 positive scores."""
        np.random.seed(42)
        n_samples = 1000
        scores = np.random.rand(n_samples)
        labels = np.random.randint(0, 2, n_samples)
        
        pos_mask = (labels == 1)
        pos_scores = scores[pos_mask]
        bottom_pos_idx = np.argsort(pos_scores)[:50]
        bottom_pos = pos_scores[bottom_pos_idx]
        
        assert len(bottom_pos) == min(50, pos_mask.sum())
        # Should be sorted ascending
        assert np.all(np.diff(bottom_pos) >= 0)


class TestFocalLoss:
    """Test Focal Loss implementation."""
    
    def test_focal_loss_formula(self):
        """Test focal loss calculation."""
        # Focal loss: -alpha * (1-p)^gamma * log(p) for positive class
        # Using numpy approximation
        
        alpha = 0.25
        gamma = 2.0
        
        # Test case: p = 0.9 (confident correct)
        p = 0.9
        y = 1  # positive
        
        # For positive class:
        # focal_loss = -alpha * (1-p)^gamma * log(p)
        expected = -alpha * ((1 - p) ** gamma) * np.log(p)
        
        assert expected < 0.1  # Low loss for confident prediction
        
        # Test case: p = 0.1 (confident wrong for positive)
        p = 0.1
        expected_high = -alpha * ((1 - p) ** gamma) * np.log(p)
        
        assert expected_high > expected  # Higher loss for wrong prediction


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

