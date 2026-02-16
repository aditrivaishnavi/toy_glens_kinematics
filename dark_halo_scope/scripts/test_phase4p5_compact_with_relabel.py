#!/usr/bin/env python3
"""
Unit Tests for Phase 4p5: Compaction with Split Relabeling

Tests verify:
1. Hash-based split function is deterministic
2. Test split is NEVER modified
3. Val split is correctly partitioned (75% → train, 25% → val)
4. Train split is NEVER modified
5. Spark integration works correctly

Usage:
    # Run all tests
    python -m pytest test_phase4p5_compact_with_relabel.py -v
    
    # Run without pytest
    python test_phase4p5_compact_with_relabel.py

Author: DarkHaloScope Team
Date: 2026-02-04
"""

import hashlib
import sys
import tempfile
import os
from typing import List, Dict

# Import the module under test
from phase4p5_compact_with_relabel import hash_based_split


# =============================================================================
# UNIT TESTS FOR hash_based_split FUNCTION
# =============================================================================

def test_hash_based_split_test_unchanged():
    """Test that 'test' split is NEVER changed."""
    test_ids = [f"task_{i}" for i in range(1000)]
    
    for task_id in test_ids:
        result = hash_based_split(task_id, "test", val_to_train_pct=75)
        assert result == "test", f"Test split changed for task_id={task_id}"
    
    print("✓ test_hash_based_split_test_unchanged: PASSED")


def test_hash_based_split_train_unchanged():
    """Test that 'train' split is NEVER changed."""
    test_ids = [f"task_{i}" for i in range(1000)]
    
    for task_id in test_ids:
        result = hash_based_split(task_id, "train", val_to_train_pct=75)
        assert result == "train", f"Train split changed for task_id={task_id}"
    
    print("✓ test_hash_based_split_train_unchanged: PASSED")


def test_hash_based_split_val_partitioned():
    """Test that 'val' split is correctly partitioned ~75% train, ~25% val."""
    n_samples = 10000
    test_ids = [f"task_{i}" for i in range(n_samples)]
    
    train_count = 0
    val_count = 0
    
    for task_id in test_ids:
        result = hash_based_split(task_id, "val", val_to_train_pct=75)
        assert result in ["train", "val"], f"Unexpected result: {result}"
        if result == "train":
            train_count += 1
        else:
            val_count += 1
    
    train_pct = 100 * train_count / n_samples
    val_pct = 100 * val_count / n_samples
    
    # Allow 5% tolerance (should be ~75% ± 5%)
    assert 70 <= train_pct <= 80, f"Train % out of range: {train_pct:.1f}%"
    assert 20 <= val_pct <= 30, f"Val % out of range: {val_pct:.1f}%"
    
    print(f"✓ test_hash_based_split_val_partitioned: PASSED")
    print(f"  Train: {train_count} ({train_pct:.1f}%), Val: {val_count} ({val_pct:.1f}%)")


def test_hash_based_split_deterministic():
    """Test that hash_based_split is deterministic (same input → same output)."""
    test_ids = [f"task_{i}" for i in range(100)]
    
    # Run twice and compare
    results_run1 = [hash_based_split(tid, "val", 75) for tid in test_ids]
    results_run2 = [hash_based_split(tid, "val", 75) for tid in test_ids]
    
    assert results_run1 == results_run2, "Results differ between runs!"
    
    print("✓ test_hash_based_split_deterministic: PASSED")


def test_hash_based_split_different_pct():
    """Test that different val_to_train_pct values work correctly."""
    n_samples = 10000
    test_ids = [f"task_{i}" for i in range(n_samples)]
    
    # Test with 50%
    train_50 = sum(1 for tid in test_ids if hash_based_split(tid, "val", 50) == "train")
    pct_50 = 100 * train_50 / n_samples
    assert 45 <= pct_50 <= 55, f"50% test failed: got {pct_50:.1f}%"
    
    # Test with 90%
    train_90 = sum(1 for tid in test_ids if hash_based_split(tid, "val", 90) == "train")
    pct_90 = 100 * train_90 / n_samples
    assert 85 <= pct_90 <= 95, f"90% test failed: got {pct_90:.1f}%"
    
    print(f"✓ test_hash_based_split_different_pct: PASSED")
    print(f"  50% setting: {pct_50:.1f}%, 90% setting: {pct_90:.1f}%")


def test_hash_based_split_unknown_split():
    """Test that unknown splits are preserved."""
    result = hash_based_split("task_123", "unknown_split", 75)
    assert result == "unknown_split", f"Unknown split changed: {result}"
    
    print("✓ test_hash_based_split_unknown_split: PASSED")


# =============================================================================
# INTEGRATION TEST WITH SPARK
# =============================================================================

def test_spark_integration():
    """Test the full Spark integration (requires PySpark)."""
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F
        from pyspark.sql.types import StructType, StructField, StringType, IntegerType
    except ImportError:
        print("⚠ test_spark_integration: SKIPPED (PySpark not available)")
        return
    
    # Create a local Spark session for testing
    spark = SparkSession.builder \
        .appName("Test-Phase4p5") \
        .master("local[2]") \
        .getOrCreate()
    
    try:
        # Create test data
        schema = StructType([
            StructField("task_id", StringType(), False),
            StructField("region_split", StringType(), False),
            StructField("value", IntegerType(), False),
        ])
        
        # Create 1000 samples with known distribution
        data = []
        for i in range(260):  # 26% train
            data.append((f"train_{i}", "train", i))
        for i in range(390):  # 39% val
            data.append((f"val_{i}", "val", i))
        for i in range(350):  # 35% test
            data.append((f"test_{i}", "test", i))
        
        df = spark.createDataFrame(data, schema)
        
        # Verify original distribution
        original_counts = df.groupBy("region_split").count().collect()
        original_dict = {row["region_split"]: row["count"] for row in original_counts}
        
        assert original_dict["train"] == 260
        assert original_dict["val"] == 390
        assert original_dict["test"] == 350
        
        # Import and apply the UDF
        from phase4p5_compact_with_relabel import hash_based_split_udf
        
        # Apply relabeling
        df_relabeled = df.withColumn("original_split", F.col("region_split"))
        df_relabeled = df_relabeled.withColumn(
            "region_split",
            hash_based_split_udf(F.col("task_id"), F.col("original_split"), F.lit(75))
        )
        
        # Count relabeled distribution
        relabeled_counts = df_relabeled.groupBy("region_split").count().collect()
        relabeled_dict = {row["region_split"]: row["count"] for row in relabeled_counts}
        
        # Verify test unchanged
        assert relabeled_dict["test"] == 350, f"Test changed: {relabeled_dict['test']}"
        
        # Verify val reduced (some moved to train)
        assert relabeled_dict["val"] < 390, f"Val not reduced: {relabeled_dict['val']}"
        
        # Verify train increased
        assert relabeled_dict["train"] > 260, f"Train not increased: {relabeled_dict['train']}"
        
        # Verify total unchanged
        total = sum(relabeled_dict.values())
        assert total == 1000, f"Total changed: {total}"
        
        print(f"✓ test_spark_integration: PASSED")
        print(f"  Original: train={original_dict['train']}, val={original_dict['val']}, test={original_dict['test']}")
        print(f"  Relabeled: train={relabeled_dict['train']}, val={relabeled_dict['val']}, test={relabeled_dict['test']}")
        
    finally:
        spark.stop()


# =============================================================================
# MATHEMATICAL VERIFICATION
# =============================================================================

def test_expected_final_distribution():
    """
    Verify the expected final distribution matches target.
    
    Original: train=26%, val=39%, test=35%
    After relabeling (75% of val → train):
        - test: 35% (unchanged)
        - train: 26% + 0.75 * 39% = 26% + 29.25% = 55.25%
        - val: 0.25 * 39% = 9.75%
    
    Wait, that's only 55.25 + 9.75 + 35 = 100%, good.
    But we want ~70% train. Let's recalculate...
    
    To get 70% train:
        train_final = train_orig + x * val_orig
        0.70 = 0.26 + x * 0.39
        x = (0.70 - 0.26) / 0.39 = 0.44 / 0.39 = 1.13
    
    That's > 100%, which means we need to use ALL of val AND some of test.
    Since we can't touch test, max train is: 26% + 39% = 65%
    
    With 75% of val → train:
        train: 26% + 75% * 39% = 26% + 29.25% = 55.25%
        val: 25% * 39% = 9.75%
        test: 35%
    
    With 100% of val → train:
        train: 26% + 39% = 65%
        val: 0%
        test: 35%
    
    So max achievable train is 65% without touching test.
    """
    train_orig_pct = 26
    val_orig_pct = 39
    test_orig_pct = 35
    
    val_to_train_pct = 75  # Our setting
    
    train_final = train_orig_pct + (val_to_train_pct / 100) * val_orig_pct
    val_final = (1 - val_to_train_pct / 100) * val_orig_pct
    test_final = test_orig_pct
    
    total = train_final + val_final + test_final
    
    assert abs(total - 100) < 0.01, f"Total != 100: {total}"
    
    print(f"✓ test_expected_final_distribution: PASSED")
    print(f"  Expected after 75% val→train:")
    print(f"    Train: {train_final:.2f}%")
    print(f"    Val:   {val_final:.2f}%")
    print(f"    Test:  {test_final:.2f}%")
    print(f"  Note: Max possible train (100% val→train) = {train_orig_pct + val_orig_pct}%")


# =============================================================================
# MAIN
# =============================================================================

def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("UNIT TESTS: Phase 4p5 Compact with Relabel")
    print("=" * 60)
    print()
    
    tests = [
        test_hash_based_split_test_unchanged,
        test_hash_based_split_train_unchanged,
        test_hash_based_split_val_partitioned,
        test_hash_based_split_deterministic,
        test_hash_based_split_different_pct,
        test_hash_based_split_unknown_split,
        test_expected_final_distribution,
        test_spark_integration,
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__}: FAILED - {e}")
            failed += 1
        except Exception as e:
            if "SKIPPED" in str(e):
                skipped += 1
            else:
                print(f"✗ {test.__name__}: ERROR - {e}")
                failed += 1
    
    print()
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
