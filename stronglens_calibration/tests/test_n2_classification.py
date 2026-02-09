#!/usr/bin/env python3
"""
Integration test for N2 classification.

Verifies that classify_pool_n2 correctly identifies confuser galaxies
and achieves the target ~15% N2 rate.

Updated 2026-02-09: New thresholds calibrated for DR10 distributions.
"""
import sys
import os
import math

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emr.sampling_utils import classify_pool_n2, classify_pool_n2_simple


def test_n2_classification():
    """Test N2 classification with synthetic data matching real criteria."""
    
    # Config matching negative_sampling_v1.yaml (tightened thresholds 2026-02-09)
    config = {
        "negative_pools": {
            "pool_n2": {
                "tractor_criteria": {
                    "ring_proxy": {
                        "types": ["DEV", "SER"],  # DEV always, SER if high sersic
                        "flux_r_min": 5.0,        # nMgy (tightened from 3.0)
                        "sersic_min": 4.0,        # Very high for SER
                    },
                    "edge_on_proxy": {
                        "types": ["EXP", "SER", "DEV"],
                        "ellipticity_min": 0.50,  # LLM: loosened for 2-4% edge-on
                        "shape_r_min": 0.6,
                        "shape_r_min_legacy": 1.8,  # loosened from 2.0
                    },
                    "blue_clumpy_proxy": {
                        "g_minus_r_max": 0.4,   # tightened from 0.5
                        "r_mag_max": 20.5,      # tightened from 21.0
                    },
                    "large_galaxy_proxy": {
                        "shape_r_min": 2.0,     # tightened from 1.5
                        "flux_r_min": 3.0,      # tightened from 2.0
                    },
                    # bright_core_proxy removed - redundant with ring_proxy
                }
            }
        }
    }
    
    # Test cases - each should return a specific category or None
    # Format: (galaxy_type, flux_r, shape_r, g_minus_r, mag_r, ellipticity, sersic, expected)
    test_cases = [
        # Ring proxy: DEV with flux >= 5 nMgy
        ("DEV", 6.0, 1.0, 0.8, 18.0, None, None, "ring_proxy"),
        ("DEV", 4.0, 1.0, 0.8, 18.0, None, None, None),  # flux_r too low (need >= 5)
        ("SER", 6.0, 1.0, 0.8, 18.0, None, 4.5, "ring_proxy"),  # SER with very high sersic
        ("SER", 6.0, 1.0, 0.8, 18.0, None, 3.5, None),  # SER with sersic < 4.0
        ("SER", 6.0, 1.0, 0.8, 18.0, None, None, None),  # SER without sersic info
        
        # Edge-on proxy: high ellipticity (>= 0.50)
        ("EXP", 5.0, 1.0, 0.8, 18.0, 0.55, None, "edge_on_proxy"),
        ("EXP", 5.0, 1.0, 0.8, 18.0, 0.45, None, None),  # ellipticity too low (need >= 0.50)
        ("EXP", 5.0, 2.0, 0.8, 18.0, None, None, "edge_on_proxy"),  # legacy: shape_r >= 1.8
        ("EXP", 5.0, 1.5, 0.8, 18.0, None, None, None),  # legacy: shape_r < 1.8
        
        # Blue clumpy: g-r <= 0.4 and mag_r <= 20.5
        ("REX", 5.0, 1.0, 0.35, 20.0, None, None, "blue_clumpy"),
        ("REX", 5.0, 1.0, 0.3, 18.0, None, None, "blue_clumpy"),
        ("SER", 5.0, 1.0, 0.5, 18.0, None, None, None),  # g-r > 0.4, too red
        ("SER", 5.0, 1.0, 0.35, 21.0, None, None, None),  # mag_r > 20.5, too faint
        
        # Large galaxy: shape_r >= 2.0 and flux_r >= 3.0
        ("REX", 4.0, 2.5, 0.7, 19.0, None, None, "large_galaxy"),
        ("REX", 2.0, 2.5, 0.7, 19.0, None, None, None),  # flux_r < 3.0
        ("REX", 4.0, 1.5, 0.7, 19.0, None, None, None),  # shape_r < 2.0
        
        # Bright DEV still matches ring_proxy (no separate bright_core)
        ("DEV", 10.0, 0.8, 0.8, 17.0, None, None, "ring_proxy"),
        
        # N1 cases - should return None
        ("REX", 2.0, 0.5, 0.7, 20.0, None, None, None),  # typical faint galaxy
        ("REX", 1.0, 0.3, 0.8, 21.0, None, None, None),  # dim, small, red
        ("SER", 4.0, 1.0, 0.6, 19.0, None, None, None),  # moderate everything
        
        # Edge cases with None values
        ("DEV", None, 1.0, 0.8, 18.0, None, None, None),  # flux_r is None
        ("EXP", 5.0, None, 0.8, 18.0, None, None, None),  # shape_r is None
        ("REX", 5.0, 1.0, None, 18.0, None, None, None),  # g_minus_r is None
    ]
    
    print("=" * 70)
    print("N2 Classification Integration Test (Updated 2026-02-09)")
    print("=" * 70)
    
    passed = 0
    failed = 0
    
    for i, (galaxy_type, flux_r, shape_r, g_minus_r, mag_r, ellipticity, sersic, expected) in enumerate(test_cases):
        result = classify_pool_n2(
            galaxy_type, flux_r, shape_r, g_minus_r, mag_r, config,
            ellipticity=ellipticity, sersic=sersic
        )
        
        if result == expected:
            status = "✓ PASS"
            passed += 1
        else:
            status = "✗ FAIL"
            failed += 1
        
        print(f"  [{i+1:2d}] {status}: type={galaxy_type}, flux_r={flux_r}, shape_r={shape_r}, "
              f"g-r={g_minus_r}, mag_r={mag_r}, e={ellipticity}, n={sersic} -> {result} (expected: {expected})")
    
    print()
    print(f"Unit test results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    # Summary statistics for realistic DR10 data
    print("\nSimulating realistic DR10 galaxy population:")
    
    import random
    import numpy as np
    np.random.seed(42)
    random.seed(42)
    
    n1_count = 0
    n2_count = 0
    n2_categories = {
        "ring_proxy": 0, 
        "edge_on_proxy": 0, 
        "blue_clumpy": 0,
        "large_galaxy": 0,
    }
    
    n_samples = 50000
    
    for _ in range(n_samples):
        # Realistic DR10 type distribution (from actual sweeps)
        galaxy_type = random.choices(
            ["SER", "DEV", "REX", "EXP"],
            weights=[0.35, 0.12, 0.38, 0.15]  # Adjusted to match DR10
        )[0]
        
        # Realistic property distributions based on DR10 Tractor catalogs
        # flux_r: log-normal, median ~5 nMgy, 10th percentile ~1, 90th ~20
        flux_r = np.random.lognormal(mean=1.5, sigma=1.0)
        
        # shape_r: log-normal, median ~0.8 arcsec
        shape_r = np.random.lognormal(mean=-0.2, sigma=0.6)
        
        # g-r color: Gaussian, bimodal but approximated
        g_minus_r = np.random.normal(0.65, 0.28)
        
        # mag_r from flux_r (22.5 - 2.5*log10(flux))
        mag_r = 22.5 - 2.5 * math.log10(max(flux_r, 0.01))
        
        # ellipticity: distribution skewed toward low values
        ellipticity = abs(np.random.beta(2, 5) * 0.8)
        
        # sersic: only for SER type, distribution peaks around 1-2
        sersic = np.random.lognormal(mean=0.5, sigma=0.6) if galaxy_type == "SER" else None
        
        result = classify_pool_n2(
            galaxy_type, flux_r, shape_r, g_minus_r, mag_r, config,
            ellipticity=ellipticity, sersic=sersic
        )
        
        if result is not None:
            n2_count += 1
            if result in n2_categories:
                n2_categories[result] += 1
            else:
                n2_categories[result] = 1
        else:
            n1_count += 1
    
    n2_pct = n2_count / n_samples * 100
    
    print(f"  Total samples: {n_samples:,}")
    print(f"  N1 (deployment-representative): {n1_count:,} ({n1_count/n_samples*100:.1f}%)")
    print(f"  N2 (hard confusers): {n2_count:,} ({n2_pct:.1f}%)")
    print()
    print("  N2 breakdown by category:")
    for cat, count in sorted(n2_categories.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"    - {cat}: {count:,} ({count/n_samples*100:.1f}%)")
    
    # Validation gates
    print()
    print("=" * 70)
    print("VALIDATION GATES")
    print("=" * 70)
    
    gates_passed = 0
    gates_total = 3
    
    # Gate 1: N2 is non-empty
    if n2_count > 0:
        print(f"  ✓ Gate 1: N2 pool is non-empty ({n2_count:,} galaxies)")
        gates_passed += 1
    else:
        print(f"  ✗ Gate 1: N2 pool is EMPTY - classification broken")
    
    # Gate 2: N2 rate is in target range (10-25%)
    if 10 <= n2_pct <= 25:
        print(f"  ✓ Gate 2: N2 rate is in target range ({n2_pct:.1f}% vs target 10-25%)")
        gates_passed += 1
    elif n2_pct < 10:
        print(f"  ✗ Gate 2: N2 rate too LOW ({n2_pct:.1f}% vs target 10-25%)")
    else:
        print(f"  ✗ Gate 2: N2 rate too HIGH ({n2_pct:.1f}% vs target 10-25%)")
    
    # Gate 3: Multiple categories represented
    active_categories = sum(1 for c in n2_categories.values() if c > 0)
    if active_categories >= 3:
        print(f"  ✓ Gate 3: Multiple N2 categories active ({active_categories} categories)")
        gates_passed += 1
    else:
        print(f"  ✗ Gate 3: Too few N2 categories ({active_categories}, need >=3)")
    
    print()
    print(f"Gates passed: {gates_passed}/{gates_total}")
    print("=" * 70)
    
    success = (failed == 0) and (gates_passed == gates_total)
    
    if success:
        print("\n✓ ALL TESTS PASSED")
    else:
        print("\n✗ SOME TESTS FAILED")
    
    return success


def test_backward_compatibility():
    """Test that classify_pool_n2_simple works for legacy code."""
    print("\n" + "=" * 70)
    print("Backward Compatibility Test (classify_pool_n2_simple)")
    print("=" * 70)
    
    config = {"negative_pools": {"pool_n2": {"tractor_criteria": {}}}}
    
    # Should work without ellipticity/sersic
    result = classify_pool_n2_simple("DEV", 5.0, 1.0, 0.8, 18.0, config)
    if result == "ring_proxy":
        print("  ✓ classify_pool_n2_simple works correctly")
        return True
    else:
        print(f"  ✗ classify_pool_n2_simple returned {result}, expected ring_proxy")
        return False


if __name__ == "__main__":
    success1 = test_n2_classification()
    success2 = test_backward_compatibility()
    sys.exit(0 if (success1 and success2) else 1)
