#!/usr/bin/env python3
"""
Phase 6-7: Update parameter grids (theta_e, src_reff) in pipeline code.

This script documents the recommended parameter grid changes based on:
1. Current parameter ranges (too narrow)
2. LLM recommendations
3. Scientific requirements for ground-based detection

Changes needed in:
- dark_halo_scope/configs/gen5/*.json
- dark_halo_scope/emr/gen5/spark_phase4_pipeline_gen5.py (if hardcoded)
"""
import json
from datetime import datetime, timezone


def get_current_parameter_grids():
    """Document current parameter grids."""
    current = {
        "grid_small (training)": {
            "theta_e_arcsec": [0.3, 0.6, 1.0],
            "src_dmag": [1.0, 2.0],
            "src_reff_arcsec": [0.08, 0.15],
            "src_e": [0.0, 0.2],
            "shear": [0.0, 0.02]
        },
        "grid_grid (full)": {
            "theta_e_arcsec": [0.3, 0.5, 0.7, 1.0, 1.3],
            "src_dmag": [0.5, 1.0, 1.5, 2.0, 2.5],
            "src_reff_arcsec": [0.05, 0.10, 0.15, 0.25],
            "src_e": [0.0, 0.15, 0.3],
            "shear": [0.0, 0.02, 0.05]
        }
    }
    return current


def get_recommended_parameter_grids():
    """Recommended parameter grids for Gen5'."""
    recommended = {
        "grid_small_v2 (training - expanded)": {
            "theta_e_arcsec": {
                "values": [0.3, 0.5, 0.7, 1.0, 1.3, 1.6],
                "rationale": "Extend to 1.6\" to include more obvious arcs"
            },
            "src_dmag": {
                "values": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
                "rationale": "Extend to 3.0 for fainter sources, 0.5 for brighter"
            },
            "src_reff_arcsec": {
                "values": [0.05, 0.10, 0.15, 0.25, 0.35],
                "rationale": "Include compact (0.05\") and extended (0.35\") sources"
            },
            "src_e": {
                "values": [0.0, 0.2, 0.4],
                "rationale": "Include more elliptical sources"
            },
            "shear": {
                "values": [0.0, 0.02, 0.05],
                "rationale": "Include moderate external shear"
            }
        },
        "notes": {
            "psf_considerations": "Ground-based PSF FWHM ~1.0-1.5\", so theta_e should extend to ~1.5× PSF",
            "detectability": "theta_e/PSF > 0.7 is typical detection threshold",
            "magnitude_scaling": "src_dmag relative to LRG magnitude; 3.0 means 3 mag fainter"
        }
    }
    return recommended


def generate_config_diff():
    """Generate a diff showing what to change in config files."""
    diff = '''
# Changes for gen5 config files

## File: dark_halo_scope/configs/gen5/gen5_cosmos_production.json

### OLD:
"grid_small": {
    "theta_e_arcsec": [0.3, 0.6, 1.0],
    "src_dmag": [1.0, 2.0],
    "src_reff_arcsec": [0.08, 0.15]
}

### NEW:
"grid_small_v2": {
    "theta_e_arcsec": [0.3, 0.5, 0.7, 1.0, 1.3, 1.6],
    "src_dmag": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    "src_reff_arcsec": [0.05, 0.10, 0.15, 0.25, 0.35]
}

## Rationale:
- theta_e: Extended range covers PSF-sized to 1.5× PSF radii
- src_dmag: Extended range covers brighter and fainter sources
- src_reff: Extended range covers compact to extended sources

## Impact:
- Grid size increases from 3×2×2=12 to 6×6×5=180 configs
- Recommend rejection sampling to maintain balanced arc_snr distribution
'''
    return diff


def main():
    print("=" * 70)
    print("PHASE 6-7: PARAMETER GRID UPDATES")
    print("=" * 70)
    
    RESULTS = {
        "phase": "6-7",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "description": "Update parameter grids for Gen5' retraining"
    }
    
    # Current grids
    current = get_current_parameter_grids()
    print("\nCurrent Parameter Grids:")
    print(json.dumps(current, indent=2))
    RESULTS["current_grids"] = current
    
    # Recommended grids
    recommended = get_recommended_parameter_grids()
    print("\n\nRecommended Parameter Grids:")
    print(json.dumps(recommended, indent=2))
    RESULTS["recommended_grids"] = recommended
    
    # Config diff
    diff = generate_config_diff()
    print("\n\nConfig File Diff:")
    print(diff)
    RESULTS["config_diff"] = diff
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    old_size = 3 * 2 * 2  # theta × dmag × reff
    new_size = 6 * 6 * 5
    print(f"Old grid size: {old_size} configurations")
    print(f"New grid size: {new_size} configurations")
    print(f"Increase: {new_size / old_size:.1f}×")
    
    RESULTS["grid_size_comparison"] = {
        "old": old_size,
        "new": new_size,
        "increase_factor": new_size / old_size
    }
    
    RESULTS["status"] = "READY_TO_IMPLEMENT"
    RESULTS["next_steps"] = [
        "1. Update gen5_cosmos_production.json with new grid",
        "2. Update spark_phase4_pipeline_gen5.py if grids are hardcoded",
        "3. Verify grid parsing in pipeline code",
        "4. Run with rejection sampling to balance arc_snr"
    ]
    
    print("\nNext Steps:")
    for step in RESULTS["next_steps"]:
        print(f"  {step}")
    
    # Save
    with open("/lambda/nfs/darkhaloscope-training-dc/phase6_7_parameter_grids.json", "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)
    
    print("\nResults saved to phase6_7_parameter_grids.json")
    return RESULTS


if __name__ == "__main__":
    main()
