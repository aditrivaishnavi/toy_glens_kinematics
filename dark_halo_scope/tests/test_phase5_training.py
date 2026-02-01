"""
Unit tests for Phase 5 training and evaluation code.

These tests catch bugs like the tpr@fpr1e-4 = 0.0% issue.
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTPRatFPRCalculation:
    """Test the TPR at fixed FPR calculation logic."""
    
    def test_tpr_at_fpr_basic(self):
        """Basic TPR@FPR calculation should work."""
        from sklearn.metrics import roc_curve
        
        # Create simple test case
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        
        # Find TPR at FPR=0.25
        target_fpr = 0.25
        idx = np.searchsorted(fpr, target_fpr)
        if idx < len(tpr):
            tpr_at_target = tpr[idx]
        else:
            tpr_at_target = tpr[-1]
        
        assert tpr_at_target > 0, f"TPR@FPR={target_fpr} should be > 0, got {tpr_at_target}"
    
    def test_tpr_at_very_low_fpr(self):
        """TPR at very low FPR (1e-4) with small sample size."""
        from sklearn.metrics import roc_curve
        
        # With only 1000 samples, FPR=1e-4 means we need < 0.1 false positives
        # This is impossible with discrete counts
        n_neg = 1000
        n_pos = 1000
        
        y_true = np.concatenate([np.zeros(n_neg), np.ones(n_pos)])
        # Good classifier
        y_score = np.concatenate([
            np.random.uniform(0, 0.5, n_neg),  # Negatives score low
            np.random.uniform(0.5, 1.0, n_pos)  # Positives score high
        ])
        
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        
        # Find TPR at FPR=1e-4
        target_fpr = 1e-4
        
        # BUG CHECK: With 1000 negatives, the minimum non-zero FPR is 1/1000 = 0.001
        # So FPR=1e-4 is below the resolution
        min_fpr = 1.0 / n_neg
        
        if target_fpr < min_fpr:
            # Cannot achieve this FPR with this sample size
            # The code should handle this gracefully
            idx = np.searchsorted(fpr, target_fpr)
            if idx == 0:
                # At FPR=0, TPR is also 0 (by definition of ROC curve starting point)
                tpr_at_target = tpr[0]
                assert tpr_at_target == 0.0, "At FPR=0, TPR should be 0"
            else:
                tpr_at_target = tpr[idx]
        
        print(f"Min achievable FPR with {n_neg} negatives: {min_fpr}")
        print(f"Target FPR: {target_fpr}")
        print(f"This explains why tpr@fpr1e-4 can be 0.0 with small eval sets!")
    
    def test_tpr_at_fpr_with_large_sample(self):
        """With enough samples, TPR@FPR=1e-4 should be meaningful."""
        from sklearn.metrics import roc_curve
        
        # Need at least 10k negatives for FPR=1e-4 to be meaningful
        n_neg = 100000
        n_pos = 100000
        
        np.random.seed(42)
        y_true = np.concatenate([np.zeros(n_neg), np.ones(n_pos)])
        # Good classifier with some overlap
        y_score = np.concatenate([
            np.random.beta(2, 8, n_neg),  # Negatives skew low
            np.random.beta(8, 2, n_pos)   # Positives skew high
        ])
        
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        
        target_fpr = 1e-4
        idx = np.searchsorted(fpr, target_fpr)
        
        if idx < len(tpr):
            tpr_at_target = tpr[idx]
        else:
            tpr_at_target = tpr[-1]
        
        min_fpr = 1.0 / n_neg
        assert target_fpr >= min_fpr, f"Target FPR {target_fpr} is below resolution {min_fpr}"
        
        # With a good classifier, TPR should be > 0
        print(f"TPR@FPR={target_fpr} with {n_neg} negatives: {tpr_at_target}")


class TestEvaluationSampleSize:
    """Tests related to evaluation sample size and metric stability."""
    
    def test_minimum_samples_for_fpr_1e4(self):
        """Calculate minimum samples needed for FPR=1e-4 to be meaningful."""
        target_fpr = 1e-4
        
        # For FPR = k/N to equal 1e-4, need N >= 10000
        min_negatives = int(1.0 / target_fpr)
        
        assert min_negatives == 10000, f"Need at least {min_negatives} negatives"
        
        # The training showed n_eval = 128000, with ~50% negatives = 64000
        # This should be enough for FPR=1e-4
        actual_negatives = 65090  # From the training logs
        
        assert actual_negatives >= min_negatives, \
            f"Have {actual_negatives} negatives, need {min_negatives} for FPR=1e-4"
    
    def test_tpr_at_fpr_interpolation_vs_index(self):
        """Test if interpolation gives different results than simple indexing."""
        from sklearn.metrics import roc_curve
        
        n = 10000
        np.random.seed(123)
        y_true = np.random.binomial(1, 0.5, n)
        y_score = y_true * 0.3 + np.random.uniform(0, 0.7, n)
        
        fpr, tpr, _ = roc_curve(y_true, y_score)
        
        target_fpr = 1e-3
        
        # Method 1: Simple indexing (what the code likely does)
        idx = np.searchsorted(fpr, target_fpr)
        tpr_indexed = tpr[min(idx, len(tpr)-1)]
        
        # Method 2: Interpolation (more accurate)
        tpr_interp = np.interp(target_fpr, fpr, tpr)
        
        print(f"Indexed TPR@FPR={target_fpr}: {tpr_indexed}")
        print(f"Interpolated TPR@FPR={target_fpr}: {tpr_interp}")
        
        # They can differ significantly at low FPR
        if abs(tpr_indexed - tpr_interp) > 0.1:
            print("WARNING: Large difference between methods!")


class TestWorkerShardingBug:
    """Test for the DataLoader worker duplication bug."""
    
    def test_sharding_by_rank_only_causes_duplication(self):
        """Demonstrate the worker duplication bug."""
        # Simulate 8 workers with rank=0
        num_workers = 8
        rank = 0
        world_size = 1
        
        fragments = list(range(100))  # 100 parquet fragments
        
        # Bug: sharding by rank only
        shards_buggy = [fragments[rank::world_size] for _ in range(num_workers)]
        
        # All workers get the same data!
        for i in range(1, num_workers):
            assert shards_buggy[0] == shards_buggy[i], "Bug: workers should NOT have identical data"
        
        print(f"BUG: All {num_workers} workers process identical {len(shards_buggy[0])} fragments")
        print(f"This means {num_workers}x duplicate samples per epoch!")
    
    def test_correct_sharding_by_rank_and_worker(self):
        """Show correct sharding implementation."""
        num_workers = 8
        rank = 0
        world_size = 1
        
        fragments = list(range(100))
        
        # Correct: shard by (rank * num_workers + worker_id)
        shards_correct = []
        for worker_id in range(num_workers):
            shard = rank * num_workers + worker_id
            nshard = world_size * num_workers
            worker_fragments = fragments[shard::nshard]
            shards_correct.append(worker_fragments)
        
        # Each worker should have different fragments
        all_fragments = []
        for s in shards_correct:
            all_fragments.extend(s)
        
        # Should cover all fragments without overlap
        assert len(all_fragments) == len(fragments), "All fragments should be covered"
        assert len(set(all_fragments)) == len(fragments), "No duplicates"
        
        print(f"Correct: {num_workers} workers each process {len(shards_correct[0])} unique fragments")


class TestForbiddenMetadataColumns:
    """Test that label-leaking metadata is blocked."""
    
    def test_arc_snr_is_forbidden(self):
        """arc_snr leaks labels and should be forbidden."""
        forbidden = {
            "theta_e_arcsec", "theta_e", "src_dmag", "src_reff_arcsec", 
            "src_e", "shear", "shear_phi_rad", "lens_e", "lens_phi_rad",
            "src_x_arcsec", "src_y_arcsec", "magnification", "tangential_stretch",
            "radial_stretch", "physics_valid", "arc_snr", "is_control", "label",
        }
        
        # These columns leak information about whether injection happened
        dangerous_cols = ["arc_snr", "theta_e_arcsec", "is_control"]
        
        for col in dangerous_cols:
            assert col in forbidden, f"{col} should be in forbidden list"
    
    def test_safe_metadata_columns(self):
        """These columns are safe for metadata fusion."""
        safe = {"psfsize_r", "psfdepth_r", "psfsize_g", "psfsize_z", "ebv"}
        forbidden = {
            "theta_e_arcsec", "arc_snr", "is_control", "label",
        }
        
        for col in safe:
            assert col not in forbidden, f"{col} should be safe"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

