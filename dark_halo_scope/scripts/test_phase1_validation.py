#!/usr/bin/env python3
"""
Phase 1 Validation Tests for Gen5-Prime Pipeline

Run this on Lambda to verify all infrastructure is working before training.

Tests:
    1. Imports - verify all modules load
    2. Data pipeline - load pilot batch, verify 6-channel output
    3. Model forward - verify LensFinder6CH works
    4. Gates (no model) - run GateRunner on samples
    5. Coadd cache - verify paired control cutouts work

Usage:
    python test_phase1_validation.py

Author: DarkHaloScope Team
Date: 2026-02-05
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path

# Configuration
PARQUET_ROOT = "/lambda/nfs/darkhaloscope-training-dc/phase4c_v5_cosmos"
COADD_CACHE = "/lambda/nfs/darkhaloscope-training-dc/dr10/coadd_cache"
MAX_TEST_PAIRS = 100
BATCH_SIZE = 16
USE_API_FALLBACK = True  # Use Legacy Survey cutout service when local cache unavailable

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_result(name: str, passed: bool, details: str = ""):
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  [{status}] {name}")
    if details:
        for line in details.split("\n"):
            print(f"           {line}")


def test_imports():
    """Test 1: Verify all imports work."""
    print_header("TEST 1: Imports")
    
    errors = []
    
    try:
        import numpy as np
        print_result("numpy", True, f"version {np.__version__}")
    except Exception as e:
        errors.append(f"numpy: {e}")
        print_result("numpy", False, str(e))
    
    try:
        import torch
        print_result("torch", True, f"version {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    except Exception as e:
        errors.append(f"torch: {e}")
        print_result("torch", False, str(e))
    
    try:
        from training.paired_training_v2 import (
            decode_stamp_npz,
            CoaddCutoutProvider,
            PairedParquetDataset,
            Preprocess6CH,
            build_training_loader,
            run_gates_quick,
        )
        print_result("paired_training_v2", True)
    except Exception as e:
        errors.append(f"paired_training_v2: {e}")
        print_result("paired_training_v2", False, str(e))
    
    try:
        from training.convnext_6ch import LensFinder6CH, patch_convnext_stem_to_6ch
        print_result("convnext_6ch", True)
    except Exception as e:
        errors.append(f"convnext_6ch: {e}")
        print_result("convnext_6ch", False, str(e))
    
    try:
        import pyarrow.dataset as ds
        print_result("pyarrow.dataset", True)
    except Exception as e:
        errors.append(f"pyarrow: {e}")
        print_result("pyarrow.dataset", False, str(e))
    
    try:
        from astropy.io import fits
        from astropy.wcs import WCS
        print_result("astropy", True)
    except Exception as e:
        errors.append(f"astropy: {e}")
        print_result("astropy", False, str(e))
    
    try:
        from sklearn.metrics import roc_auc_score
        from sklearn.linear_model import LogisticRegression
        print_result("sklearn", True)
    except Exception as e:
        errors.append(f"sklearn: {e}")
        print_result("sklearn", False, str(e))
    
    return len(errors) == 0, errors


def test_coadd_cache():
    """Test 2: Verify coadd cache exists and has data."""
    print_header("TEST 2: Coadd Cache")
    
    errors = []
    
    # Check directory exists
    if os.path.exists(COADD_CACHE):
        print_result("Cache directory exists", True, COADD_CACHE)
    else:
        errors.append(f"Cache directory not found: {COADD_CACHE}")
        print_result("Cache directory exists", False, f"Not found: {COADD_CACHE}")
        return False, errors
    
    # Count bricks
    try:
        bricks = [d for d in os.listdir(COADD_CACHE) if os.path.isdir(os.path.join(COADD_CACHE, d))]
        if len(bricks) > 0:
            print_result("Cached bricks", True, f"{len(bricks)} bricks found")
            
            # Sample a brick
            sample_brick = bricks[0]
            brick_path = os.path.join(COADD_CACHE, sample_brick)
            files = os.listdir(brick_path)
            fits_files = [f for f in files if f.endswith('.fits.fz') or f.endswith('.fits')]
            print_result(f"Sample brick ({sample_brick})", True, f"{len(fits_files)} FITS files")
        else:
            errors.append("No cached bricks found")
            print_result("Cached bricks", False, "No bricks found")
    except Exception as e:
        errors.append(f"Error listing cache: {e}")
        print_result("Cached bricks", False, str(e))
    
    # Check disk usage
    try:
        import subprocess
        result = subprocess.run(["du", "-sh", COADD_CACHE], capture_output=True, text=True)
        if result.returncode == 0:
            size = result.stdout.split()[0]
            print_result("Cache size", True, size)
    except Exception:
        pass
    
    return len(errors) == 0, errors


def test_data_pipeline():
    """Test 3: Verify data pipeline loads correctly."""
    print_header("TEST 3: Data Pipeline")
    
    errors = []
    
    # Check parquet root exists
    if not os.path.exists(PARQUET_ROOT):
        errors.append(f"Parquet root not found: {PARQUET_ROOT}")
        print_result("Parquet root exists", False, PARQUET_ROOT)
        return False, errors
    
    print_result("Parquet root exists", True, PARQUET_ROOT)
    
    try:
        from training.paired_training_v2 import build_training_loader
        
        start = time.time()
        loader = build_training_loader(
            parquet_root=PARQUET_ROOT,
            coadd_cache_root=COADD_CACHE,
            split="train",
            batch_pairs=BATCH_SIZE,
            num_workers=0,  # Single worker for testing
            max_pairs_index=MAX_TEST_PAIRS,
            use_api_fallback=USE_API_FALLBACK,
        )
        print_result("Build loader", True, f"{time.time()-start:.2f}s")
        
        # Get a batch
        start = time.time()
        batch = next(iter(loader))
        print_result("Load batch", True, f"{time.time()-start:.2f}s")
        
        # Verify shapes
        x6 = batch.x6
        y = batch.y
        
        if x6.shape[1] == 6 and x6.shape[2] == 64 and x6.shape[3] == 64:
            print_result("x6 shape", True, str(tuple(x6.shape)))
        else:
            errors.append(f"Unexpected x6 shape: {x6.shape}")
            print_result("x6 shape", False, f"Expected (B,6,64,64), got {x6.shape}")
        
        if y.ndim == 1:
            print_result("y shape", True, str(tuple(y.shape)))
        else:
            errors.append(f"Unexpected y shape: {y.shape}")
            print_result("y shape", False, f"Expected (B,), got {y.shape}")
        
        # Check for NaN/Inf
        import torch
        if torch.isfinite(x6).all():
            print_result("No NaN/Inf in x6", True)
        else:
            nan_count = (~torch.isfinite(x6)).sum().item()
            errors.append(f"NaN/Inf in x6: {nan_count}")
            print_result("No NaN/Inf in x6", False, f"{nan_count} bad values")
        
        # Check mixing
        pos_count = y.sum().item()
        neg_count = (1 - y).sum().item()
        hardneg_count = batch.meta["is_hardneg"].sum().item()
        print_result("Sample mix", True, f"pos={int(pos_count)}, neg={int(neg_count)}, hardneg={int(hardneg_count)}")
        
        # Check x_ratio
        x_ratio = batch.meta["x_ratio"]
        print_result("x_ratio range", True, f"[{x_ratio.min():.2f}, {x_ratio.max():.2f}]")
        
    except Exception as e:
        errors.append(f"Pipeline error: {e}")
        print_result("Data pipeline", False, str(e))
        traceback.print_exc()
    
    return len(errors) == 0, errors


def test_model_forward():
    """Test 4: Verify model forward pass works."""
    print_header("TEST 4: Model Forward Pass")
    
    errors = []
    
    try:
        import torch
        from training.convnext_6ch import LensFinder6CH
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print_result("Device", True, str(device))
        
        # Create model
        start = time.time()
        model = LensFinder6CH(arch="tiny", pretrained=True, init="copy_or_zero")
        model = model.to(device)
        model.eval()
        print_result("Create model", True, f"{time.time()-start:.2f}s, {sum(p.numel() for p in model.parameters()):,} params")
        
        # Forward pass
        x = torch.randn(4, 6, 64, 64).to(device)
        meta = torch.randn(4, 2).to(device)
        
        with torch.no_grad():
            start = time.time()
            logits = model(x, meta)
            print_result("Forward pass", True, f"{time.time()-start:.4f}s")
        
        # Check output
        if logits.shape == (4,):
            print_result("Output shape", True, str(tuple(logits.shape)))
        else:
            errors.append(f"Unexpected output shape: {logits.shape}")
            print_result("Output shape", False, f"Expected (4,), got {logits.shape}")
        
        # Check output values
        probs = torch.sigmoid(logits)
        print_result("Output probs", True, f"range [{probs.min():.3f}, {probs.max():.3f}]")
        
        # Check stem weight initialization
        stem_weight = model.backbone.features[0][0].weight.data
        rgb_norm = stem_weight[:, :3].norm().item()
        resid_norm = stem_weight[:, 3:].norm().item()
        print_result("Stem init", True, f"RGB norm={rgb_norm:.2f}, Resid norm={resid_norm:.4f}")
        
        if resid_norm > 1e-4:
            errors.append(f"Residual channels not zero-initialized: norm={resid_norm}")
            print_result("Zero-init check", False, f"Resid norm should be ~0, got {resid_norm}")
        else:
            print_result("Zero-init check", True)
        
    except Exception as e:
        errors.append(f"Model error: {e}")
        print_result("Model forward", False, str(e))
        traceback.print_exc()
    
    return len(errors) == 0, errors


def test_gates_no_model():
    """Test 5: Verify gate runner works without model."""
    print_header("TEST 5: Gate Runner (No Model)")
    
    errors = []
    
    try:
        from training.paired_training_v2 import run_gates_quick
        
        start = time.time()
        results = run_gates_quick(
            parquet_root=PARQUET_ROOT,
            coadd_cache_root=COADD_CACHE,
            split="train",
            model=None,
            device="cpu",
            max_pairs=200,
            use_api_fallback=USE_API_FALLBACK,
        )
        elapsed = time.time() - start
        print_result("Run gates", True, f"{elapsed:.2f}s")
        
        # Check results structure
        if "n_pairs" in results:
            print_result("n_pairs", True, str(results["n_pairs"]))
        else:
            errors.append("Missing n_pairs in results")
            print_result("n_pairs", False, "Missing")
        
        if "strata" in results:
            strata = results["strata"]
            for name, data in strata.items():
                if isinstance(data, dict) and "core_auc_lr" in data:
                    print_result(f"Stratum {name}", True, 
                                f"core={data['core_auc_lr']:.3f}, ann={data['annulus_auc_lr']:.3f}")
                elif isinstance(data, dict) and "status" in data:
                    print_result(f"Stratum {name}", True, data["status"])
        else:
            errors.append("Missing strata in results")
            print_result("strata", False, "Missing")
        
        if "distributions" in results:
            dist = results["distributions"]
            if "x_ratio" in dist:
                print_result("x_ratio distribution", True, 
                            f"mean={dist['x_ratio']['mean']:.2f}, frac_ge_1={dist['x_ratio']['frac_ge_1']:.2f}")
        
    except Exception as e:
        errors.append(f"Gate error: {e}")
        print_result("Gate runner", False, str(e))
        traceback.print_exc()
    
    return len(errors) == 0, errors


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("  PHASE 1 VALIDATION TESTS - Gen5-Prime Pipeline")
    print("=" * 70)
    print(f"\nParquet root: {PARQUET_ROOT}")
    print(f"Coadd cache: {COADD_CACHE}")
    print(f"Test samples: {MAX_TEST_PAIRS}")
    
    all_passed = True
    all_errors = []
    
    # Run tests
    tests = [
        ("Imports", test_imports),
        ("Coadd Cache", test_coadd_cache),
        ("Data Pipeline", test_data_pipeline),
        ("Model Forward", test_model_forward),
        ("Gates (No Model)", test_gates_no_model),
    ]
    
    results = {}
    for name, test_fn in tests:
        try:
            passed, errors = test_fn()
            results[name] = {"passed": passed, "errors": errors}
            if not passed:
                all_passed = False
                all_errors.extend(errors)
        except Exception as e:
            results[name] = {"passed": False, "errors": [str(e)]}
            all_passed = False
            all_errors.append(str(e))
    
    # Summary
    print_header("SUMMARY")
    
    for name, result in results.items():
        status = "✓ PASS" if result["passed"] else "✗ FAIL"
        print(f"  {name}: {status}")
    
    print("\n" + "=" * 70)
    if all_passed:
        print("  ALL TESTS PASSED - Ready for training!")
        print("=" * 70)
        print("\nNext step: Run training with:")
        print(f"  python scripts/train_gen5_prime.py")
        return 0
    else:
        print("  SOME TESTS FAILED - Fix issues before training")
        print("=" * 70)
        print("\nErrors:")
        for err in all_errors:
            print(f"  - {err}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
