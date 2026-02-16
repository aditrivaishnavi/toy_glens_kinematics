#!/usr/bin/env python3
"""
Phase 3: Build Tier-A anchor set with arc visibility criterion.

This script evaluates potential anchor lenses and classifies them into:
- Tier-A: Arc visible in DR10 ground-based imaging (arc_visibility_snr > 2.0)
- Tier-B: Arc too faint for ground-based detection (stress test only)

Tier-A anchors are the primary evaluation metric for ground-based lens finders.
Tier-B anchors (e.g., SLACS/BELLS) are used only for stress testing.
"""
import numpy as np
import requests
import tempfile
import os
from astropy.io import fits
import json
from datetime import datetime, timezone
import time

# ============================================================
# Configuration
# ============================================================

STAMP_SIZE = 64
PIXSCALE = 0.262  # arcsec/pixel

# Arc visibility SNR threshold
ARC_VISIBILITY_THRESHOLD = 2.0

# SLACS lenses (from our previous analysis - known to be faint in DR10)
SLACS_LENSES = [
    {"name": "SDSSJ0029-0055", "ra": 7.4543, "dec": -0.9254},
    {"name": "SDSSJ0037-0942", "ra": 9.3004, "dec": -9.7095},
    {"name": "SDSSJ0252+0039", "ra": 43.1313, "dec": 0.6651},
    {"name": "SDSSJ0330-0020", "ra": 52.5019, "dec": -0.3419},
    {"name": "SDSSJ0728+3835", "ra": 112.1879, "dec": 38.5900},
    {"name": "SDSSJ0737+3216", "ra": 114.4121, "dec": 32.2793},
    {"name": "SDSSJ0912+0029", "ra": 138.0200, "dec": 0.4894},
    {"name": "SDSSJ0959+0410", "ra": 149.7954, "dec": 4.1755},
    {"name": "SDSSJ1016+3859", "ra": 154.1092, "dec": 38.9937},
    {"name": "SDSSJ1020+1122", "ra": 155.1029, "dec": 11.3690},
]

# BELLS lenses
BELLS_LENSES = [
    {"name": "SDSSJ0747+5055", "ra": 116.8850, "dec": 50.9261},
    {"name": "SDSSJ0755+3445", "ra": 118.8175, "dec": 34.7544},
    {"name": "SDSSJ0801+4727", "ra": 120.3808, "dec": 47.4569},
    {"name": "SDSSJ0830+5116", "ra": 127.5371, "dec": 51.2753},
    {"name": "SDSSJ0832+0404", "ra": 128.2038, "dec": 4.0725},
]

# Ground-based discovered lenses (higher priority - should be Tier-A)
# These are from Legacy Survey ML lens searches
GROUND_BASED_CANDIDATES = [
    # Add coordinates from Legacy Survey ML lens catalogs
    # These should have visible arcs in ground-based imaging
]

# ============================================================
# Helper Functions
# ============================================================

def arc_visibility_snr(cutout, inner_r=4, outer_r=16):
    """
    Compute arc visibility SNR in annulus region.
    
    The arc lives in an annulus around the lens galaxy. We measure
    the excess flux in this region above the background.
    
    Args:
        cutout: 2D array (r-band preferred)
        inner_r: inner radius of arc annulus (pixels)
        outer_r: outer radius of arc annulus (pixels)
    
    Returns:
        snr: float, visibility SNR
        is_tier_a: bool, True if snr > threshold
    """
    h, w = cutout.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r2 = (yy - cy)**2 + (xx - cx)**2
    
    # Arc annulus
    annulus = (r2 >= inner_r**2) & (r2 < outer_r**2)
    
    # Outer region for background estimation
    outer = r2 >= outer_r**2
    
    if outer.sum() == 0 or annulus.sum() == 0:
        return 0.0, False
    
    # Background from outer region
    bg = np.median(cutout[outer])
    
    # Excess flux in annulus
    annulus_excess = np.sum(cutout[annulus] - bg)
    
    # Noise estimate from outer region
    outer_mad = np.median(np.abs(cutout[outer] - bg))
    noise = 1.4826 * outer_mad * np.sqrt(annulus.sum())
    
    snr = annulus_excess / (noise + 1e-10)
    return float(snr), snr > ARC_VISIBILITY_THRESHOLD


def central_aperture_flux(cutout, r=8):
    """
    Compute central aperture flux.
    
    Args:
        cutout: 2D array
        r: aperture radius in pixels
    
    Returns:
        mean_flux: float
        max_flux: float
    """
    h, w = cutout.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r2 = (yy - cy)**2 + (xx - cx)**2
    mask = r2 < r**2
    
    vals = cutout[mask]
    return float(np.mean(vals)), float(np.max(vals))


def fetch_dr10_cutout(ra, dec, size=STAMP_SIZE, band='r', retries=3):
    """Fetch cutout from Legacy Survey DR10."""
    url = f"https://www.legacysurvey.org/viewer/cutout.fits?ra={ra}&dec={dec}&size={size}&layer=ls-dr10&pixscale={PIXSCALE}&bands={band}"
    
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200:
                time.sleep(1)
                continue
            
            with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
                f.write(resp.content)
                tmp_path = f.name
            
            try:
                with fits.open(tmp_path) as hdul:
                    data = hdul[0].data
                    if data is None:
                        return None
                    return data.astype(np.float32)
            finally:
                os.unlink(tmp_path)
                
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            time.sleep(1)
    
    return None


def evaluate_anchors(candidates, source_name="unknown"):
    """Evaluate anchor candidates and classify into Tier-A/B."""
    tier_a = []
    tier_b = []
    failed = []
    
    for cand in candidates:
        name = cand.get("name", f"{cand['ra']:.4f},{cand['dec']:.4f}")
        print(f"  {name}...", end=" ", flush=True)
        
        cutout = fetch_dr10_cutout(cand["ra"], cand["dec"])
        if cutout is None:
            print("FAILED")
            failed.append({**cand, "error": "fetch failed"})
            continue
        
        snr, is_tier_a = arc_visibility_snr(cutout)
        mean_flux, max_flux = central_aperture_flux(cutout)
        
        result = {
            **cand,
            "source": source_name,
            "arc_visibility_snr": snr,
            "central_mean_flux": mean_flux,
            "central_max_flux": max_flux,
            "is_tier_a": is_tier_a
        }
        
        if is_tier_a:
            tier_a.append(result)
            print(f"TIER-A (snr={snr:.2f})")
        else:
            tier_b.append(result)
            print(f"TIER-B (snr={snr:.2f})")
    
    return tier_a, tier_b, failed


# ============================================================
# Main Execution
# ============================================================

def main():
    print("=" * 70)
    print("PHASE 3: BUILD TIER-A ANCHOR SET")
    print("=" * 70)
    
    RESULTS = {
        "phase": "3",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "threshold": ARC_VISIBILITY_THRESHOLD,
        "tier_a": [],
        "tier_b": [],
        "failed": [],
        "summary": {}
    }
    
    # Evaluate SLACS
    print("\nEvaluating SLACS lenses...")
    slacs_a, slacs_b, slacs_f = evaluate_anchors(SLACS_LENSES, "SLACS")
    RESULTS["tier_a"].extend(slacs_a)
    RESULTS["tier_b"].extend(slacs_b)
    RESULTS["failed"].extend(slacs_f)
    
    # Evaluate BELLS
    print("\nEvaluating BELLS lenses...")
    bells_a, bells_b, bells_f = evaluate_anchors(BELLS_LENSES, "BELLS")
    RESULTS["tier_a"].extend(bells_a)
    RESULTS["tier_b"].extend(bells_b)
    RESULTS["failed"].extend(bells_f)
    
    # Evaluate ground-based candidates (if any)
    if GROUND_BASED_CANDIDATES:
        print("\nEvaluating ground-based candidates...")
        gb_a, gb_b, gb_f = evaluate_anchors(GROUND_BASED_CANDIDATES, "ground-based")
        RESULTS["tier_a"].extend(gb_a)
        RESULTS["tier_b"].extend(gb_b)
        RESULTS["failed"].extend(gb_f)
    
    # Summary
    RESULTS["summary"] = {
        "n_tier_a": len(RESULTS["tier_a"]),
        "n_tier_b": len(RESULTS["tier_b"]),
        "n_failed": len(RESULTS["failed"]),
        "tier_a_sources": list(set(a["source"] for a in RESULTS["tier_a"])),
        "tier_b_sources": list(set(a["source"] for a in RESULTS["tier_b"]))
    }
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Tier-A (arc visible in DR10): {RESULTS['summary']['n_tier_a']}")
    print(f"Tier-B (arc too faint): {RESULTS['summary']['n_tier_b']}")
    print(f"Failed to fetch: {RESULTS['summary']['n_failed']}")
    
    if RESULTS["tier_a"]:
        print(f"\nTier-A candidates:")
        for a in RESULTS["tier_a"]:
            print(f"  {a['name']}: snr={a['arc_visibility_snr']:.2f}")
    
    # Interpretation
    if RESULTS["summary"]["n_tier_a"] == 0:
        RESULTS["interpretation"] = "NO TIER-A ANCHORS FOUND - SLACS/BELLS are not suitable primary anchors"
        RESULTS["recommendation"] = "Need to source anchors from ground-based lens searches (Legacy Survey ML, KiDS, etc.)"
    else:
        RESULTS["interpretation"] = f"Found {RESULTS['summary']['n_tier_a']} Tier-A anchors"
        RESULTS["recommendation"] = "Use Tier-A for primary evaluation, Tier-B for stress testing"
    
    print(f"\nInterpretation: {RESULTS['interpretation']}")
    print(f"Recommendation: {RESULTS['recommendation']}")
    
    # Save
    with open("/lambda/nfs/darkhaloscope-training-dc/phase3_anchor_results.json", "w") as f:
        json.dump(RESULTS, f, indent=2)
    
    print("\nResults saved to phase3_anchor_results.json")
    return RESULTS


if __name__ == "__main__":
    main()
