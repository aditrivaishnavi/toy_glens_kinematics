#!/usr/bin/env python3
"""
Phase 4: Curate hard negatives from Galaxy Zoo DECaLS.

Hard negatives are non-lens galaxies that look lens-like (e.g., rings, spirals,
mergers). These help the model learn to distinguish true lenses from look-alikes.

Sources for hard negatives:
1. Galaxy Zoo DECaLS morphology classifications
2. Ring galaxy catalogs
3. Spiral galaxy catalogs
4. Merger catalogs

This script provides the framework and sample queries for curating hard negatives.
"""
import numpy as np
import json
from datetime import datetime, timezone

# ============================================================
# Configuration
# ============================================================

HARD_NEGATIVE_CONFIG = {
    "sources": [
        {
            "name": "Galaxy Zoo DECaLS",
            "description": "Citizen science morphology classifications",
            "url": "https://data.galaxyzoo.org/",
            "categories_to_use": [
                "ring",
                "spiral_arms_prominent",
                "merger",
                "odd_featured"
            ]
        },
        {
            "name": "Legacy Survey ring candidates",
            "description": "Objects flagged as ring-like in ML catalogs",
            "selection_criteria": "High ring probability, not confirmed lenses"
        }
    ],
    "target_count": 10000,  # Target number of hard negatives
    "mixing_strategy": "5-10% of training batch",
    "selection_criteria": {
        "ring": "p_ring > 0.5 AND not_in_known_lens_catalog",
        "spiral": "p_spiral > 0.7 AND prominent_arms",
        "merger": "p_merger > 0.5 AND close_pairs"
    }
}

# ============================================================
# Galaxy Zoo Query (Pseudocode - requires access to GZ data)
# ============================================================

def get_galaxy_zoo_hard_negatives_sql():
    """
    SQL query for Galaxy Zoo DECaLS hard negatives.
    
    This would be run against the Galaxy Zoo data release.
    """
    query = """
    SELECT 
        gz.ra, gz.dec, gz.iauname,
        gz.smooth_or_featured_featured_or_disk_fraction as p_featured,
        gz.disk_edge_on_no_fraction as p_face_on,
        gz.has_spiral_arms_yes_fraction as p_spiral,
        gz.spiral_arm_count_more_than_4_fraction as p_multi_arm,
        gz.bulge_size_dominant_fraction as p_bulge,
        gz.odd_feature_ring_fraction as p_ring,
        gz.odd_feature_lens_or_arc_fraction as p_lens_like,
        gz.merging_merger_fraction as p_merger
    FROM 
        gz_decals_dr5 gz
    WHERE 
        -- Ring galaxies (most lens-like)
        (gz.odd_feature_ring_fraction > 0.5 
         AND gz.odd_feature_lens_or_arc_fraction < 0.3)  -- Not already lens-like
        
        OR
        
        -- Prominent spirals (can mimic arcs)
        (gz.has_spiral_arms_yes_fraction > 0.7
         AND gz.spiral_arm_count_more_than_4_fraction > 0.3)
        
        OR
        
        -- Mergers (double nuclei can look like lensed pairs)
        (gz.merging_merger_fraction > 0.5)
    
    ORDER BY 
        gz.odd_feature_ring_fraction DESC
    LIMIT 
        20000
    """
    return query


# ============================================================
# Sample Hard Negative Coordinates
# ============================================================

# Example ring galaxies known from catalogs
SAMPLE_RING_GALAXIES = [
    # Hoag's Object (famous ring galaxy)
    {"name": "Hoag's Object", "ra": 226.1833, "dec": 21.5839, "type": "ring"},
    # AM 0644-741 (Cartwheel-like)
    {"name": "AM 0644-741", "ra": 101.3833, "dec": -74.2583, "type": "ring"},
    # Other ring candidates would be added from Galaxy Zoo
]

# Example prominent spirals
SAMPLE_SPIRALS = [
    {"name": "M51", "ra": 202.4696, "dec": 47.1952, "type": "spiral"},
    {"name": "NGC 1232", "ra": 47.4362, "dec": -20.5794, "type": "spiral"},
]

# Example mergers
SAMPLE_MERGERS = [
    {"name": "Antennae", "ra": 180.4710, "dec": -18.8732, "type": "merger"},
    {"name": "NGC 6240", "ra": 253.2454, "dec": 2.4008, "type": "merger"},
]


# ============================================================
# Main Execution
# ============================================================

def main():
    print("=" * 70)
    print("PHASE 4: CURATE HARD NEGATIVES")
    print("=" * 70)
    
    RESULTS = {
        "phase": "4",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": HARD_NEGATIVE_CONFIG,
        "sample_query": get_galaxy_zoo_hard_negatives_sql(),
        "status": "FRAMEWORK_READY",
        "next_steps": [
            "1. Access Galaxy Zoo DECaLS data release",
            "2. Run provided SQL query to get candidates",
            "3. Cross-match with known lens catalogs to exclude real lenses",
            "4. Download cutouts from Legacy Survey",
            "5. Add to training with 5-10% mixing ratio"
        ],
        "sample_coordinates": {
            "rings": SAMPLE_RING_GALAXIES,
            "spirals": SAMPLE_SPIRALS,
            "mergers": SAMPLE_MERGERS
        }
    }
    
    print("\nConfiguration:")
    print(json.dumps(HARD_NEGATIVE_CONFIG, indent=2))
    
    print("\n\nSQL Query for Galaxy Zoo:")
    print("-" * 70)
    print(RESULTS["sample_query"])
    print("-" * 70)
    
    print("\n\nNext Steps:")
    for step in RESULTS["next_steps"]:
        print(f"  {step}")
    
    print("\n\nSample Coordinates (for testing):")
    print(f"  Rings: {len(SAMPLE_RING_GALAXIES)}")
    print(f"  Spirals: {len(SAMPLE_SPIRALS)}")
    print(f"  Mergers: {len(SAMPLE_MERGERS)}")
    
    # Save
    with open("/lambda/nfs/darkhaloscope-training-dc/phase4_hard_negatives_config.json", "w") as f:
        json.dump(RESULTS, f, indent=2)
    
    print("\nConfiguration saved to phase4_hard_negatives_config.json")
    print("\nNote: Full implementation requires Galaxy Zoo data access.")
    
    return RESULTS


if __name__ == "__main__":
    main()
