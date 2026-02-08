#!/usr/bin/env python3
"""
Integration test for N2 classification fix.

Verifies that classify_pool_n2 correctly identifies confuser galaxies.
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emr.sampling_utils import classify_pool_n2


def test_n2_classification():
    """Test N2 classification with synthetic data matching real criteria."""
    
    # Config matching negative_sampling_v1.yaml
    config = {
        "negative_pools": {
            "pool_n2": {
                "tractor_criteria": {
                    "ring_proxy": {
                        "type": "DEV",
                        "flux_r_min": 10,  # nMgy
                    },
                    "edge_on_proxy": {
                        "type": "EXP",
                        "shape_r_min": 2.0,  # arcsec
                    },
                    "blue_clumpy_proxy": {
                        "g_minus_r_max": 0.4,
                        "r_mag_max": 19.0,
                    },
                }
            }
        }
    }
    
    # Test cases - each should return a specific category or None
    test_cases = [
        # (galaxy_type, flux_r, shape_r, g_minus_r, mag_r, expected_category)
        
        # Ring proxy: DEV with bright flux
        ("DEV", 15.0, 1.0, 0.8, 18.0, "ring_proxy"),
        ("DEV", 5.0, 1.0, 0.8, 18.0, None),  # flux_r too low
        ("SER", 15.0, 1.0, 0.8, 18.0, None),  # wrong type
        
        # Edge-on proxy: EXP with large shape_r
        ("EXP", 5.0, 3.0, 0.8, 18.0, "edge_on_proxy"),
        ("EXP", 5.0, 1.5, 0.8, 18.0, None),  # shape_r too small
        ("DEV", 5.0, 3.0, 0.8, 18.0, None),  # wrong type (but would match ring if flux high)
        
        # Blue clumpy: any type with blue color and bright mag
        ("SER", 5.0, 1.0, 0.3, 18.0, "blue_clumpy"),
        ("REX", 5.0, 1.0, 0.2, 17.0, "blue_clumpy"),
        ("SER", 5.0, 1.0, 0.5, 18.0, None),  # g-r too red
        ("SER", 5.0, 1.0, 0.3, 20.0, None),  # mag_r too faint
        
        # N1 cases - should return None
        ("SER", 5.0, 1.0, 0.8, 18.0, None),
        ("REX", 3.0, 0.5, 0.6, 19.5, None),
        
        # Edge cases with None values
        ("DEV", None, 1.0, 0.8, 18.0, None),  # flux_r is None
        ("EXP", 5.0, None, 0.8, 18.0, None),  # shape_r is None
        ("SER", 5.0, 1.0, None, 18.0, None),  # g_minus_r is None
    ]
    
    print("=" * 60)
    print("N2 Classification Integration Test")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for i, (galaxy_type, flux_r, shape_r, g_minus_r, mag_r, expected) in enumerate(test_cases):
        result = classify_pool_n2(galaxy_type, flux_r, shape_r, g_minus_r, mag_r, config)
        
        if result == expected:
            status = "✓ PASS"
            passed += 1
        else:
            status = "✗ FAIL"
            failed += 1
        
        print(f"  [{i+1:2d}] {status}: type={galaxy_type}, flux_r={flux_r}, shape_r={shape_r}, "
              f"g-r={g_minus_r}, mag_r={mag_r} -> {result} (expected: {expected})")
    
    print()
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    # Summary statistics for realistic data
    print("\nSimulating realistic manifest data (like actual sweep file):")
    
    # Simulate distribution similar to actual manifest
    import random
    random.seed(42)
    
    n1_count = 0
    n2_count = 0
    n2_categories = {"ring_proxy": 0, "edge_on_proxy": 0, "blue_clumpy": 0}
    
    n_samples = 10000
    
    for _ in range(n_samples):
        # Realistic type distribution
        galaxy_type = random.choices(
            ["SER", "DEV", "REX", "EXP"],
            weights=[0.4, 0.12, 0.35, 0.13]
        )[0]
        
        # Realistic property distributions
        flux_r = random.lognormvariate(1.5, 1.0)  # Log-normal, most are dim
        shape_r = random.lognormvariate(0.0, 0.5)  # Half-light radius
        g_minus_r = random.gauss(0.6, 0.25)  # Color distribution
        mag_r = random.gauss(19.5, 1.5)  # Magnitude distribution
        
        result = classify_pool_n2(galaxy_type, flux_r, shape_r, g_minus_r, mag_r, config)
        
        if result is not None:
            n2_count += 1
            n2_categories[result] += 1
        else:
            n1_count += 1
    
    n2_pct = n2_count / n_samples * 100
    
    print(f"  Total samples: {n_samples}")
    print(f"  N1: {n1_count} ({n1_count/n_samples*100:.1f}%)")
    print(f"  N2: {n2_count} ({n2_pct:.1f}%)")
    print(f"    - ring_proxy: {n2_categories['ring_proxy']}")
    print(f"    - edge_on_proxy: {n2_categories['edge_on_proxy']}")
    print(f"    - blue_clumpy: {n2_categories['blue_clumpy']}")
    
    # Validation gate
    if n2_count > 0:
        print(f"\n✓ N2 POOL IS NON-EMPTY ({n2_pct:.1f}%)")
    else:
        print(f"\n✗ N2 POOL IS EMPTY - BUG NOT FIXED")
        return False
    
    return failed == 0 and n2_count > 0


if __name__ == "__main__":
    success = test_n2_classification()
    sys.exit(0 if success else 1)
