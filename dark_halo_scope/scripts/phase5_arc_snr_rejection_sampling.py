#!/usr/bin/env python3
"""
Phase 5: Implement arc_snr rejection sampling for brightness calibration.

Problem:
- Current training data has arc_snr distribution with mean=8.5, but only 9% > 20
- Real detectable lenses may have different SNR distribution
- Need to calibrate injection brightness to match expected detection threshold

Solution:
- Use rejection sampling to shape the arc_snr distribution
- Target distribution: uniform in log-space from 2 to 50
- Or: match empirical distribution from ground-based lens searches

This script provides:
1. Analysis of current arc_snr distribution
2. Rejection sampling implementation
3. Recommended brightness calibration parameters
"""
import numpy as np
import pyarrow.dataset as ds
import json
from datetime import datetime, timezone


def analyze_current_arc_snr_distribution():
    """Analyze the current arc_snr distribution in training data."""
    print("=" * 70)
    print("PHASE 5: ARC_SNR REJECTION SAMPLING")
    print("=" * 70)
    
    data_path = "/lambda/nfs/darkhaloscope-training-dc/phase4c_v5_cosmos"
    dataset = ds.dataset(data_path, format="parquet", partitioning="hive")
    
    # Read arc_snr for positives
    filt = (ds.field("region_split") == "train") & (ds.field("cutout_ok") == 1) & (ds.field("is_control") == 0)
    table = dataset.to_table(filter=filt, columns=["arc_snr", "src_dmag", "theta_e_arcsec"])
    
    arc_snr = np.array(table["arc_snr"].to_pandas().dropna())
    src_dmag = np.array(table["src_dmag"].to_pandas().dropna())
    theta_e = np.array(table["theta_e_arcsec"].to_pandas().dropna())
    
    print(f"\nCurrent Distribution (n={len(arc_snr)}):")
    print(f"  Mean arc_snr: {arc_snr.mean():.2f}")
    print(f"  Median arc_snr: {np.median(arc_snr):.2f}")
    print(f"  Std arc_snr: {arc_snr.std():.2f}")
    
    # Percentiles
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    print("\n  Percentiles:")
    for p in percentiles:
        val = np.percentile(arc_snr, p)
        print(f"    {p}th: {val:.2f}")
    
    # Bins
    bins = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 50), (50, np.inf)]
    print("\n  Bin distribution:")
    for low, high in bins:
        frac = ((arc_snr >= low) & (arc_snr < high)).mean()
        print(f"    [{low}, {high}): {frac:.1%}")
    
    return {
        "n_samples": len(arc_snr),
        "mean": float(arc_snr.mean()),
        "median": float(np.median(arc_snr)),
        "std": float(arc_snr.std()),
        "percentiles": {str(p): float(np.percentile(arc_snr, p)) for p in percentiles},
        "current_src_dmag_values": list(np.unique(src_dmag)),
        "current_theta_e_values": list(np.unique(theta_e))
    }


def recommended_parameter_changes():
    """
    Recommend changes to injection parameters to achieve better arc_snr distribution.
    """
    print("\n" + "=" * 70)
    print("RECOMMENDED PARAMETER CHANGES")
    print("=" * 70)
    
    recommendations = {
        "src_dmag": {
            "current": [1.0, 2.0],
            "recommended": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            "rationale": "Extend range to include brighter (0.5) and fainter (3.0) sources"
        },
        "theta_e_arcsec": {
            "current": [0.3, 0.6, 1.0],
            "recommended": [0.3, 0.5, 0.7, 1.0, 1.3, 1.6],
            "rationale": "Extend to larger Einstein radii for more obvious arcs"
        },
        "src_reff_arcsec": {
            "current": [0.08, 0.15],
            "recommended": [0.05, 0.10, 0.15, 0.25],
            "rationale": "Include smaller sources (more compact arcs) and larger sources"
        },
        "arc_snr_target_distribution": {
            "type": "log_uniform",
            "range": [2.0, 50.0],
            "rationale": "Uniform in log-space covers both faint (detection limit) and bright (easy) lenses"
        }
    }
    
    for param, rec in recommendations.items():
        print(f"\n{param}:")
        if "current" in rec:
            print(f"  Current: {rec['current']}")
            print(f"  Recommended: {rec['recommended']}")
        print(f"  Rationale: {rec['rationale']}")
    
    return recommendations


def rejection_sampling_implementation():
    """
    Provide code for rejection sampling to shape arc_snr distribution.
    """
    print("\n" + "=" * 70)
    print("REJECTION SAMPLING IMPLEMENTATION")
    print("=" * 70)
    
    code = '''
def rejection_sample_arc_snr(arc_snr_value, target_dist="log_uniform", snr_min=2.0, snr_max=50.0):
    """
    Rejection sampling to shape arc_snr distribution.
    
    For pipeline integration:
    - After computing arc_snr, call this function
    - If returns False, skip this sample (generate new injection)
    - If returns True, include this sample
    
    Args:
        arc_snr_value: Computed arc SNR for this injection
        target_dist: Target distribution type
        snr_min: Minimum SNR to include
        snr_max: Maximum SNR to include
    
    Returns:
        bool: True if sample should be kept
    """
    import numpy as np
    
    # Reject samples outside range
    if arc_snr_value < snr_min or arc_snr_value > snr_max:
        return False
    
    if target_dist == "log_uniform":
        # For log-uniform: acceptance probability ∝ 1/snr
        # This converts uniform sampling to log-uniform
        # Normalize by 1/snr_min (max acceptance prob)
        accept_prob = (snr_min / arc_snr_value)
        return np.random.random() < accept_prob
    
    elif target_dist == "uniform":
        # Keep all samples in range
        return True
    
    elif target_dist == "empirical":
        # Match empirical distribution from ground-based searches
        # Would need to load empirical CDF and sample accordingly
        raise NotImplementedError("Empirical distribution not yet implemented")
    
    return True


# Pipeline integration example:
def generate_injection_with_rejection(manifest_row, max_attempts=10):
    """
    Generate injection with rejection sampling on arc_snr.
    """
    for attempt in range(max_attempts):
        # Run normal injection pipeline
        result = run_injection(manifest_row)
        
        if result["cutout_ok"] and result.get("arc_snr"):
            # Apply rejection sampling
            if rejection_sample_arc_snr(result["arc_snr"]):
                return result
    
    # If all attempts rejected, return last result anyway
    return result
'''
    
    print(code)
    
    return {
        "implementation": "rejection_sampling",
        "function": "rejection_sample_arc_snr",
        "target_dist": "log_uniform",
        "snr_range": [2.0, 50.0]
    }


def main():
    RESULTS = {
        "phase": "5",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    # Analyze current distribution
    current_dist = analyze_current_arc_snr_distribution()
    RESULTS["current_distribution"] = current_dist
    
    # Get recommendations
    recommendations = recommended_parameter_changes()
    RESULTS["recommendations"] = recommendations
    
    # Get implementation
    implementation = rejection_sampling_implementation()
    RESULTS["implementation"] = implementation
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Current arc_snr: mean={current_dist['mean']:.2f}, median={current_dist['median']:.2f}")
    print(f"Current src_dmag: {current_dist['current_src_dmag_values']}")
    print(f"Current theta_e: {current_dist['current_theta_e_values']}")
    print("\nRecommended changes:")
    print(f"  src_dmag: {recommendations['src_dmag']['recommended']}")
    print(f"  theta_e: {recommendations['theta_e_arcsec']['recommended']}")
    print(f"  src_reff: {recommendations['src_reff_arcsec']['recommended']}")
    print(f"  Target arc_snr distribution: log-uniform [2, 50]")
    
    RESULTS["status"] = "FRAMEWORK_READY"
    RESULTS["next_steps"] = [
        "1. Update pipeline config with expanded parameter ranges",
        "2. Integrate rejection_sample_arc_snr into injection loop",
        "3. Regenerate training data with new distribution",
        "4. Verify arc_snr distribution matches target"
    ]
    
    # Save
    with open("/lambda/nfs/darkhaloscope-training-dc/phase5_arc_snr_config.json", "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)
    
    print("\nResults saved to phase5_arc_snr_config.json")
    return RESULTS


if __name__ == "__main__":
    main()
