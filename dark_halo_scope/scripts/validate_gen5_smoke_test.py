#!/usr/bin/env python3
"""Gen5 Phase 4c Smoke Test Validation Script

Validates output from smoke test run to ensure COSMOS integration is working correctly.
"""

import sys
import pyarrow.parquet as pq
import numpy as np
import io

def validate_smoke_test(parquet_path: str):
    """Validate smoke test output."""
    
    print("="*60)
    print("Gen5 Phase 4c Smoke Test Validation")
    print("="*60)
    
    # Read sample
    print(f"\n📂 Reading: {parquet_path}")
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    
    print(f"✅ Rows: {len(df)}")
    print(f"✅ Columns: {len(df.columns)}")
    
    # Check required columns
    print("\n🔍 Checking schema...")
    required = ['stamp_npz', 'is_control', 'source_mode', 'cosmos_index', 'cosmos_hlr_arcsec']
    missing = [col for col in required if col not in df.columns]
    
    if missing:
        print(f"❌ FAIL: Missing columns: {missing}")
        return False
    print(f"✅ All required columns present")
    
    # Check source_mode
    print("\n🔍 Checking source_mode...")
    if not (df['source_mode'] == 'cosmos').all():
        print(f"❌ FAIL: source_mode should be 'cosmos' for all rows")
        print(f"   Found: {df['source_mode'].unique()}")
        return False
    print(f"✅ source_mode = 'cosmos' for all {len(df)} rows")
    
    # Check cosmos_index range
    print("\n🔍 Checking cosmos_index...")
    if not (df['cosmos_index'] >= 0).all() or not (df['cosmos_index'] < 20000).all():
        print(f"❌ FAIL: cosmos_index out of range [0, 20000)")
        print(f"   Min: {df['cosmos_index'].min()}, Max: {df['cosmos_index'].max()}")
        return False
    print(f"✅ cosmos_index in [0, 20000)")
    print(f"   Range: {df['cosmos_index'].min()} - {df['cosmos_index'].max()}")
    print(f"   Unique sources used: {df['cosmos_index'].nunique()}")
    
    # Check COSMOS HLR distribution
    print("\n🔍 Checking COSMOS HLR distribution...")
    hlr_valid = df['cosmos_hlr_arcsec'][df['cosmos_hlr_arcsec'].notna()]
    if len(hlr_valid) == 0:
        print(f"❌ FAIL: No valid HLR values")
        return False
    
    hlr_min, hlr_max = hlr_valid.min(), hlr_valid.max()
    hlr_median = hlr_valid.median()
    
    print(f"✅ HLR statistics:")
    print(f"   Min:    {hlr_min:.3f} arcsec")
    print(f"   Median: {hlr_median:.3f} arcsec")
    print(f"   Max:    {hlr_max:.3f} arcsec")
    print(f"   Expected range: 0.1-1.25 arcsec (from COSMOS bank)")
    
    if hlr_min < 0.05 or hlr_max > 1.5:
        print(f"⚠️  WARNING: HLR range unusual (but not necessarily wrong)")
    
    # Decode one stamp and check shape
    print("\n🔍 Checking stamp format...")
    try:
        npz_bytes = df['stamp_npz'].iloc[0]
        with io.BytesIO(npz_bytes) as f:
            npz = np.load(f)
            img = npz['image']
            print(f"✅ Stamp shape: {img.shape}")
            
            if img.shape != (3, 64, 64):
                print(f"❌ FAIL: Expected shape (3, 64, 64) for grz, got {img.shape}")
                return False
            print(f"   Expected: (3, 64, 64) for grz ✅")
            
            # Check for NaN/Inf
            if not np.isfinite(img).all():
                print(f"❌ FAIL: Stamp contains NaN or Inf values")
                return False
            print(f"✅ No NaN/Inf values in stamp")
            
    except Exception as e:
        print(f"❌ FAIL: Could not decode stamp: {e}")
        return False
    
    # Check positive/negative split
    print("\n🔍 Checking positive/negative split...")
    n_pos = (df['is_control'] == 0).sum()
    n_neg = (df['is_control'] == 1).sum()
    print(f"   Positives: {n_pos}")
    print(f"   Negatives: {n_neg}")
    print(f"   Ratio: {n_neg/n_pos:.2f}:1" if n_pos > 0 else "   (no positives)")
    
    if n_pos == 0:
        print(f"⚠️  WARNING: No positive samples (may be expected for debug tier)")
    
    # Success
    print("\n" + "="*60)
    print("🎉 SMOKE TEST VALIDATION PASSED!")
    print("="*60)
    print("\nKey Metrics:")
    print(f"  - Total stamps: {len(df)}")
    print(f"  - Source mode: COSMOS")
    print(f"  - Unique sources: {df['cosmos_index'].nunique()}")
    print(f"  - HLR range: {hlr_min:.3f} - {hlr_max:.3f} arcsec")
    print(f"  - Stamp shape: {img.shape}")
    print("\n✅ Ready for full production run")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_smoke_test.py <path_to_parquet>")
        print("Example: python validate_smoke_test.py /tmp/sample.parquet")
        sys.exit(1)
    
    success = validate_smoke_test(sys.argv[1])
    sys.exit(0 if success else 1)

