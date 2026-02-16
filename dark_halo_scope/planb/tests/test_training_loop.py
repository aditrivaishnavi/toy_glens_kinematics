#!/usr/bin/env python3
"""
Unit Tests for Training Loop Components

Tests training loop functions with mock data to verify:
1. Loss computation works
2. Gradient flow is correct
3. Gate evaluation produces valid results
4. No NaN during training

Usage:
    pytest test_training_loop.py -v
    python test_training_loop.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent / "phase1_baseline"))


class TestLossComputation:
    """Tests for loss computation."""
    
    def test_bce_loss_basic(self):
        """Test BCE loss computes without error."""
        criterion = nn.BCEWithLogitsLoss()
        
        logits = torch.randn(16, 1)
        targets = torch.randint(0, 2, (16, 1)).float()
        
        loss = criterion(logits, targets)
        
        assert torch.isfinite(loss), "Loss is NaN/Inf"
        assert loss.item() > 0, "Loss should be positive"
    
    def test_bce_loss_gradient_flow(self):
        """Test that gradients flow through BCE loss."""
        criterion = nn.BCEWithLogitsLoss()
        
        logits = torch.randn(16, 1, requires_grad=True)
        targets = torch.randint(0, 2, (16, 1)).float()
        
        loss = criterion(logits, targets)
        loss.backward()
        
        assert logits.grad is not None, "No gradients"
        assert torch.isfinite(logits.grad).all(), "NaN in gradients"


class TestMetricComputation:
    """Tests for metric computation."""
    
    def test_auroc_basic(self):
        """Test AUROC computation."""
        # Perfect separation
        probs = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])
        labels = np.array([0, 0, 0, 1, 1, 1])
        
        auroc = roc_auc_score(labels, probs)
        
        assert auroc == 1.0, f"Perfect separation should give AUROC=1.0, got {auroc}"
    
    def test_auroc_random(self):
        """Test AUROC on random predictions."""
        np.random.seed(42)
        probs = np.random.rand(1000)
        labels = np.random.randint(0, 2, 1000)
        
        auroc = roc_auc_score(labels, probs)
        
        # Random should be close to 0.5
        assert 0.4 < auroc < 0.6, f"Random should give AUROC~0.5, got {auroc}"
    
    def test_auroc_handles_edge_cases(self):
        """Test AUROC handles edge cases gracefully."""
        # All same class - should raise or handle
        probs = np.array([0.1, 0.2, 0.3])
        labels = np.array([1, 1, 1])
        
        try:
            auroc = roc_auc_score(labels, probs)
            # sklearn raises ValueError for single class
            assert False, "Should have raised error"
        except ValueError:
            pass  # Expected


class TestGateEvaluation:
    """Tests for gate evaluation logic."""
    
    def test_core_lr_auc_mock(self):
        """Test Core LR AUC computation with mock data."""
        from sklearn.linear_model import LogisticRegression
        
        np.random.seed(42)
        
        # Create mock core features with NO shortcut
        # Core should NOT distinguish classes
        n_samples = 100
        X_core = np.random.randn(n_samples, 300)  # 10x10x3 = 300
        y = np.random.randint(0, 2, n_samples)
        
        lr = LogisticRegression(max_iter=100, random_state=42)
        lr.fit(X_core, y)
        probs = lr.predict_proba(X_core)[:, 1]
        
        auc = roc_auc_score(y, probs)
        
        # Should be near random (0.5) since no shortcut
        assert 0.4 < auc < 0.7, f"Random core should give ~0.5 AUC, got {auc}"
    
    def test_core_lr_auc_with_shortcut(self):
        """Test that Core LR AUC detects a shortcut."""
        from sklearn.linear_model import LogisticRegression
        
        np.random.seed(42)
        
        # Create mock core features WITH shortcut
        # Class 1 has brighter cores
        n_samples = 100
        y = np.random.randint(0, 2, n_samples)
        
        X_core = np.random.randn(n_samples, 300)
        X_core[y == 1, :] += 2.0  # Add bias for class 1
        
        lr = LogisticRegression(max_iter=100, random_state=42)
        lr.fit(X_core, y)
        probs = lr.predict_proba(X_core)[:, 1]
        
        auc = roc_auc_score(y, probs)
        
        # Should detect the shortcut (high AUC)
        assert auc > 0.8, f"Shortcut should give high AUC, got {auc}"


class TestGradientClipping:
    """Tests for gradient clipping."""
    
    def test_gradient_clipping_basic(self):
        """Test that gradient clipping works."""
        from model import build_model, ModelConfig
        
        config = ModelConfig(arch="resnet18", pretrained=False)
        model = build_model(config=config, device="cpu")
        model.train()
        
        # Forward pass with large input
        x = torch.randn(4, 3, 64, 64) * 100  # Large values
        out = model(x)
        loss = out.sum()
        loss.backward()
        
        # Clip gradients
        max_norm = 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
        # Check all gradients are clipped
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        
        # After clipping, norm should be <= max_norm (approximately)
        assert total_norm <= max_norm * 1.1, f"Grad norm {total_norm} exceeds {max_norm}"


class TestEarlyStopping:
    """Tests for early stopping logic."""
    
    def test_patience_counting(self):
        """Test patience counter logic."""
        best_auroc = 0.0
        patience_counter = 0
        patience = 5
        
        aurocs = [0.5, 0.6, 0.7, 0.65, 0.68, 0.69, 0.69, 0.69, 0.69, 0.69]
        
        for epoch, auroc in enumerate(aurocs):
            if auroc > best_auroc:
                best_auroc = auroc
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break
        
        # Should stop at epoch 7 (index)
        assert epoch == 7, f"Should stop at epoch 7, stopped at {epoch}"
        assert best_auroc == 0.7, f"Best should be 0.7, got {best_auroc}"


class TestDataPipeline:
    """Tests for data pipeline integration."""
    
    def test_collate_fn_output_shape(self):
        """Test that collate function produces correct shapes."""
        # Mock batch
        batch = [
            {
                "positive": torch.randn(3, 64, 64),
                "negative": torch.randn(3, 64, 64),
                "is_hard_negative": False,
                "theta_e": 1.5,
                "arc_snr": 10.0,
            }
            for _ in range(4)
        ]
        
        from data_loader import collate_fn
        
        result = collate_fn(batch)
        
        assert result["x"].shape == (8, 3, 64, 64), f"Wrong x shape: {result['x'].shape}"
        assert result["y"].shape == (8,), f"Wrong y shape: {result['y'].shape}"
        assert result["y"].sum().item() == 4, "Should have 4 positives"
    
    def test_collate_fn_shuffles(self):
        """Test that collate function shuffles pos/neg."""
        batch = [
            {
                "positive": torch.ones(3, 64, 64),  # All ones
                "negative": torch.zeros(3, 64, 64),  # All zeros
                "is_hard_negative": False,
                "theta_e": 1.5,
                "arc_snr": 10.0,
            }
            for _ in range(4)
        ]
        
        from data_loader import collate_fn
        
        torch.manual_seed(42)
        result = collate_fn(batch)
        
        # Labels should be shuffled, not [1,1,1,1,0,0,0,0]
        y = result["y"].tolist()
        assert y != [1, 1, 1, 1, 0, 0, 0, 0], "Collate should shuffle"


def run_all_tests():
    """Run all tests and report results."""
    import traceback
    
    test_classes = [
        TestLossComputation,
        TestMetricComputation,
        TestGateEvaluation,
        TestGradientClipping,
        TestEarlyStopping,
        TestDataPipeline,
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        instance = test_class()
        
        test_methods = [m for m in dir(instance) if m.startswith("test_")]
        
        print(f"\n{test_class.__name__}")
        print("-" * 40)
        
        for method_name in test_methods:
            total_tests += 1
            method = getattr(instance, method_name)
            
            try:
                method()
                print(f"  ✓ {method_name}")
                passed_tests += 1
            except Exception as e:
                print(f"  ✗ {method_name}")
                print(f"    Error: {e}")
                failed_tests.append({
                    "class": test_class.__name__,
                    "method": method_name,
                    "error": str(e),
                })
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed_tests}/{total_tests}")
    print(f"Failed: {len(failed_tests)}")
    
    if failed_tests:
        print("\nFailed tests:")
        for failure in failed_tests:
            print(f"  - {failure['class']}.{failure['method']}: {failure['error']}")
        return False
    else:
        print("\n✓ ALL TESTS PASSED")
        return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
