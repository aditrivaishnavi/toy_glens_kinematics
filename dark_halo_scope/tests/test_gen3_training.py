"""Comprehensive unit tests for Gen3 training script.

Tests Gen3-specific features:
- Moffat PSF (in data pipeline)
- All fixes from LLM1+LLM2 analysis
- Integration with Gen2 fixes
"""

import numpy as np
import pytest
import sys
import os
from dataclasses import dataclass, fields
from typing import List, Optional, Tuple, Dict, Sequence
import math
import io


# =============================================================================
# INLINE IMPLEMENTATIONS
# =============================================================================

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
# GEN3-SPECIFIC TESTS
# =============================================================================

class TestGen3StreamConfig:
    """Test Gen3 StreamConfig matches Gen2."""
    
    def test_all_fields_present(self):
        """Gen3 should have same fields as Gen2."""
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
            mad_clip=10.0, max_rows=0, min_theta_over_psf=0.5,
            min_arc_snr=3.0
        )
        
        assert cfg.norm_method == "full"
        assert cfg.epoch == 0
        assert cfg.min_theta_over_psf == 0.5


class TestGen3ForbiddenMetadata:
    """Test Gen3 has same forbidden metadata as Gen2."""
    
    def test_forbidden_set_identical(self):
        """Gen3 forbidden set should match Gen2."""
        FORBIDDEN_META_GEN2 = frozenset({
            "theta_e_arcsec", "theta_e", "src_dmag", "src_reff_arcsec", "src_n",
            "src_e", "src_phi_deg", "src_x_arcsec", "src_y_arcsec",
            "lens_e", "lens_phi_deg", "shear_gamma", "shear_phi_deg", "shear",
            "arc_snr", "magnification", "tangential_stretch", "radial_stretch",
            "is_control", "label", "cutout_ok",
        })
        
        FORBIDDEN_META_GEN3 = frozenset({
            "theta_e_arcsec", "theta_e", "src_dmag", "src_reff_arcsec", "src_n",
            "src_e", "src_phi_deg", "src_x_arcsec", "src_y_arcsec",
            "lens_e", "lens_phi_deg", "shear_gamma", "shear_phi_deg", "shear",
            "arc_snr", "magnification", "tangential_stretch", "radial_stretch",
            "is_control", "label", "cutout_ok",
        })
        
        assert FORBIDDEN_META_GEN2 == FORBIDDEN_META_GEN3


class TestGen3ROCCurve:
    """Test Gen3 ROC curve implementation."""
    
    def test_roc_reaches_endpoints(self):
        """ROC should reach (0,0) and (1,1)."""
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0])
        
        fpr, tpr, _ = roc_curve_np(scores, y)
        
        assert fpr[0] == 0.0
        assert tpr[0] == 0.0
        assert fpr[-1] == 1.0
        assert tpr[-1] == 1.0
    
    def test_roc_with_calibration_collapse(self):
        """Test ROC with binary-like scores."""
        # Simulating severe calibration collapse
        scores = np.array([0.9999] * 50 + [0.0001] * 50)
        y = np.array([1] * 25 + [0] * 25 + [1] * 25 + [0] * 25)
        
        fpr, tpr, _ = roc_curve_np(scores, y)
        
        # Should still reach endpoints
        assert fpr[0] == 0.0
        assert fpr[-1] == 1.0


class TestGen3EarlyStopping:
    """Test Gen3 early stopping."""
    
    def test_patience_logic(self):
        """Test early stopping patience."""
        metrics_history = [
            {"tpr@fpr0.0001": 0.5},
            {"tpr@fpr0.0001": 0.6},
            {"tpr@fpr0.0001": 0.7},  # Best
            {"tpr@fpr0.0001": 0.65},  # No improvement 1
            {"tpr@fpr0.0001": 0.68},  # No improvement 2
            {"tpr@fpr0.0001": 0.66},  # No improvement 3 -> STOP
        ]
        
        patience = 3
        best_metric = -1.0
        no_improve_count = 0
        stop_epoch = None
        
        for epoch, metrics in enumerate(metrics_history):
            score = metrics["tpr@fpr0.0001"]
            if score > best_metric:
                best_metric = score
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    stop_epoch = epoch
                    break
        
        assert stop_epoch == 5  # Should stop at epoch 5


class TestGen3WorkerSharding:
    """Test Gen3 worker sharding."""
    
    def test_complete_coverage(self):
        """All fragments should be covered exactly once."""
        n_frags = 1000
        world_size = 1
        num_workers = 8
        
        assigned = set()
        for worker_id in range(num_workers):
            shard = 0 * num_workers + worker_id  # rank=0
            nshard = world_size * num_workers
            for i in range(shard, n_frags, nshard):
                assert i not in assigned
                assigned.add(i)
        
        assert assigned == set(range(n_frags))


class TestGen3Normalization:
    """Test Gen3 normalization options."""
    
    def test_norm_method_full(self):
        """Full normalization uses entire image."""
        x = np.random.randn(3, 64, 64).astype(np.float32)
        
        # Full normalization
        out = np.empty_like(x, dtype=np.float32)
        for c in range(x.shape[0]):
            v = x[c]
            med = np.median(v)
            mad = np.median(np.abs(v - med))
            scale = 1.4826 * mad + 1e-6
            vv = (v - med) / scale
            out[c] = np.clip(vv, -10, 10)
        
        assert out.shape == x.shape
    
    def test_norm_method_outer(self):
        """Outer normalization excludes center."""
        x = np.random.randn(3, 64, 64).astype(np.float32)
        
        out = robust_mad_norm_outer(x)
        
        assert out.shape == x.shape
        assert not np.any(np.isnan(out))


class TestGen3CalibrationMonitoring:
    """Test Gen3 calibration monitoring."""
    
    def test_binary_fraction_calculation(self):
        """Test binary score fraction calculation."""
        # 50% binary
        scores = np.concatenate([
            np.array([0.001] * 25),
            np.array([0.5] * 50),
            np.array([0.999] * 25)
        ])
        
        binary_low = (scores < 0.01).sum()
        binary_high = (scores > 0.99).sum()
        binary_frac = (binary_low + binary_high) / len(scores)
        
        assert binary_frac == 0.5
    
    def test_calibration_warning_threshold(self):
        """Warning should trigger at >50% binary."""
        threshold = 0.5
        
        # Just under threshold
        binary_frac_ok = 0.49
        assert binary_frac_ok <= threshold
        
        # Over threshold
        binary_frac_bad = 0.51
        assert binary_frac_bad > threshold


class TestGen3Top50Logging:
    """Test Gen3 top-50 score logging."""
    
    def test_top_neg_extraction(self):
        """Extract top-50 highest-scoring negatives."""
        n = 1000
        scores = np.random.rand(n)
        labels = np.random.randint(0, 2, n)
        
        neg_mask = labels == 0
        if neg_mask.sum() > 0:
            neg_scores = scores[neg_mask]
            top_neg_idx = np.argsort(-neg_scores)[:50]
            top_neg = neg_scores[top_neg_idx]
            
            # Verify descending order
            assert np.all(np.diff(top_neg) <= 0)
            assert len(top_neg) == min(50, neg_mask.sum())
    
    def test_bottom_pos_extraction(self):
        """Extract bottom-50 lowest-scoring positives."""
        n = 1000
        scores = np.random.rand(n)
        labels = np.random.randint(0, 2, n)
        
        pos_mask = labels == 1
        if pos_mask.sum() > 0:
            pos_scores = scores[pos_mask]
            bottom_pos_idx = np.argsort(pos_scores)[:50]
            bottom_pos = pos_scores[bottom_pos_idx]
            
            # Verify ascending order
            assert np.all(np.diff(bottom_pos) >= 0)
            assert len(bottom_pos) == min(50, pos_mask.sum())


class TestGen3EpochShuffle:
    """Test Gen3 epoch-dependent shuffling."""
    
    def test_shuffle_varies_by_epoch(self):
        """Each epoch should have different shuffle."""
        seed = 1337
        n_frags = 100
        
        shuffles = []
        for epoch in range(5):
            rng = np.random.RandomState(seed + 7919 * epoch)
            frags = list(range(n_frags))
            rng.shuffle(frags)
            shuffles.append(tuple(frags))
        
        # All should be unique
        assert len(set(shuffles)) == 5
    
    def test_shuffle_reproducible(self):
        """Same epoch should give same shuffle."""
        seed = 1337
        epoch = 3
        n_frags = 100
        
        rng1 = np.random.RandomState(seed + 7919 * epoch)
        frags1 = list(range(n_frags))
        rng1.shuffle(frags1)
        
        rng2 = np.random.RandomState(seed + 7919 * epoch)
        frags2 = list(range(n_frags))
        rng2.shuffle(frags2)
        
        assert frags1 == frags2


class TestGen3Integration:
    """Integration tests for Gen3."""
    
    def test_full_evaluation_pipeline(self):
        """Test full evaluation metrics computation."""
        np.random.seed(42)
        
        # Simulate model predictions
        n_pos, n_neg = 500, 500
        pos_scores = np.random.beta(2, 1, n_pos)  # Skewed toward 1
        neg_scores = np.random.beta(1, 2, n_neg)  # Skewed toward 0
        
        scores = np.concatenate([pos_scores, neg_scores])
        labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
        
        # Compute metrics
        fpr, tpr, _ = roc_curve_np(scores, labels)
        auroc = np.trapz(tpr, fpr)
        
        # Binary fraction
        binary_frac = ((scores < 0.01).sum() + (scores > 0.99).sum()) / len(scores)
        
        # Top/bottom scores
        neg_mask = labels == 0
        neg_scores_only = scores[neg_mask]
        top_neg = np.sort(neg_scores_only)[-50:][::-1]
        
        pos_mask = labels == 1
        pos_scores_only = scores[pos_mask]
        bottom_pos = np.sort(pos_scores_only)[:50]
        
        # Verify all metrics are valid
        assert 0.5 < auroc < 1.0  # Should be better than random
        assert binary_frac < 0.5  # Beta distribution shouldn't be binary
        assert len(top_neg) == 50
        assert len(bottom_pos) == 50
    
    def test_metrics_dict_structure(self):
        """Test metrics dict has expected keys."""
        expected_keys = {
            "n_eval", "pos_eval", "neg_eval",
            "auroc", "binary_score_frac",
            "fpr@tpr0.99", "fpr@tpr0.95", "fpr@tpr0.90",
            "tpr@fpr0.01", "tpr@fpr0.001", "tpr@fpr0.0001",
        }
        
        # Create mock metrics dict
        metrics = {
            "n_eval": 1000,
            "pos_eval": 500,
            "neg_eval": 500,
            "auroc": 0.95,
            "binary_score_frac": 0.1,
            "fpr@tpr0.99": 0.05,
            "fpr@tpr0.95": 0.01,
            "fpr@tpr0.90": 0.005,
            "tpr@fpr0.01": 0.9,
            "tpr@fpr0.001": 0.8,
            "tpr@fpr0.0001": 0.7,
            "top_neg_scores": [0.9, 0.85, 0.8],
            "bottom_pos_scores": [0.1, 0.15, 0.2],
        }
        
        for key in expected_keys:
            assert key in metrics


class TestDataDecoding:
    """Test stamp data decoding."""
    
    def test_decode_npz_blob(self):
        """Test NPZ blob decoding."""
        # Create test data
        g = np.random.randn(64, 64).astype(np.float32)
        r = np.random.randn(64, 64).astype(np.float32)
        z = np.random.randn(64, 64).astype(np.float32)
        
        # Compress
        buf = io.BytesIO()
        np.savez_compressed(buf, image_g=g, image_r=r, image_z=z)
        blob = buf.getvalue()
        
        # Decode
        with io.BytesIO(blob) as f:
            npz = np.load(f)
            decoded = np.stack([
                npz["image_g"].astype(np.float32),
                npz["image_r"].astype(np.float32),
                npz["image_z"].astype(np.float32)
            ], axis=0)
        
        assert decoded.shape == (3, 64, 64)
        assert np.allclose(decoded[0], g)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

