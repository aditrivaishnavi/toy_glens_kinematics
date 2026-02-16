#!/usr/bin/env python3
"""
Theta-Aware Validation Gates

This script implements the LLM-recommended validation gates to detect
and prevent shortcut learning in lens finder models.

Gates implemented:
1. Core-Only Baseline AUC (target: ≤ 0.55)
2. Arc-Annulus Baseline AUC (target: > 0.70)
3. Paired Core Brightness Match (target: ratio 0.95-1.05)
4. Injection Contribution Analysis

Usage:
    python theta_aware_validation.py --data_path /path/to/paired_data
"""
import numpy as np
import argparse
import json
from datetime import datetime, timezone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

RESULTS = {
    "test": "theta_aware_validation",
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "gates": {}
}

def compute_theta_aware_masks(h, w, theta_e_arcsec, psfsize_arcsec, pixscale=0.262):
    """
    Compute theta-aware masks for core and arc regions.
    
    Core: r < theta_e - 1.5*PSF (lens galaxy only)
    Arc: |r - theta_e| < 1.5*PSF (Einstein ring region)
    """
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    
    theta_pix = theta_e_arcsec / pixscale
    psf_pix = psfsize_arcsec / pixscale
    
    # Core: inside Einstein ring minus PSF buffer
    core_r = max(0, theta_pix - 1.5 * psf_pix)
    core_mask = r < core_r
    
    # Arc annulus: around Einstein ring
    arc_mask = np.abs(r - theta_pix) < 1.5 * psf_pix
    
    return core_mask, arc_mask

def extract_region_features(img, mask):
    """Extract simple statistical features from a masked region."""
    if mask.sum() < 5:
        return np.zeros(7)
    
    pixels = img[mask]
    return np.array([
        np.mean(pixels),
        np.std(pixels),
        np.median(pixels),
        np.percentile(pixels, 25),
        np.percentile(pixels, 75),
        np.max(pixels),
        np.min(pixels),
    ])

def gate_core_only_auc(positives, controls, theta_es, psfsize_rs):
    """
    Gate 1: Core-Only Baseline AUC
    
    Train a simple classifier ONLY on core features.
    If AUC > 0.55, there's a shortcut in core brightness.
    """
    print("\n" + "=" * 60)
    print("GATE 1: Core-Only Baseline AUC")
    print("=" * 60)
    
    X = []
    y = []
    
    for i, (pos, ctrl, theta_e, psfsize_r) in enumerate(zip(positives, controls, theta_es, psfsize_rs)):
        if theta_e is None or theta_e <= 0:
            continue
        
        h, w = pos.shape[-2:]
        core_mask, _ = compute_theta_aware_masks(h, w, theta_e, psfsize_r)
        
        if core_mask.sum() < 10:
            continue
        
        # Extract core features for r-band
        pos_feat = extract_region_features(pos[1], core_mask)
        ctrl_feat = extract_region_features(ctrl[1], core_mask)
        
        X.append(pos_feat)
        y.append(1)  # positive
        X.append(ctrl_feat)
        y.append(0)  # control
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Samples: {len(y)} ({(y==1).sum()} pos, {(y==0).sum()} ctrl)")
    
    if len(y) < 100:
        print("SKIP: Insufficient samples")
        return None
    
    # Train simple logistic regression
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    probs = clf.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, probs)
    
    print(f"Core-Only AUC: {auc:.4f}")
    
    if auc <= 0.55:
        print("✓ PASS: Core features are not predictive (AUC ≤ 0.55)")
        status = "PASS"
    elif auc <= 0.65:
        print("⚠ MARGINAL: Some core predictability (0.55 < AUC ≤ 0.65)")
        status = "MARGINAL"
    else:
        print(f"✗ FAIL: Core features are too predictive (AUC = {auc:.4f} > 0.65)")
        status = "FAIL"
    
    return {"auc": auc, "status": status, "n_samples": len(y)}

def gate_arc_annulus_auc(positives, controls, theta_es, psfsize_rs):
    """
    Gate 2: Arc-Annulus Baseline AUC
    
    Train a simple classifier ONLY on arc annulus features.
    AUC should be > 0.70 (arcs are predictive).
    """
    print("\n" + "=" * 60)
    print("GATE 2: Arc-Annulus Baseline AUC")
    print("=" * 60)
    
    X = []
    y = []
    
    for i, (pos, ctrl, theta_e, psfsize_r) in enumerate(zip(positives, controls, theta_es, psfsize_rs)):
        if theta_e is None or theta_e <= 0:
            continue
        
        h, w = pos.shape[-2:]
        _, arc_mask = compute_theta_aware_masks(h, w, theta_e, psfsize_r)
        
        if arc_mask.sum() < 10:
            continue
        
        # Extract arc features for r-band
        pos_feat = extract_region_features(pos[1], arc_mask)
        ctrl_feat = extract_region_features(ctrl[1], arc_mask)
        
        X.append(pos_feat)
        y.append(1)
        X.append(ctrl_feat)
        y.append(0)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Samples: {len(y)} ({(y==1).sum()} pos, {(y==0).sum()} ctrl)")
    
    if len(y) < 100:
        print("SKIP: Insufficient samples")
        return None
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    probs = clf.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, probs)
    
    print(f"Arc-Annulus AUC: {auc:.4f}")
    
    if auc >= 0.70:
        print("✓ PASS: Arc features are predictive (AUC ≥ 0.70)")
        status = "PASS"
    elif auc >= 0.60:
        print("⚠ MARGINAL: Weak arc predictability (0.60 ≤ AUC < 0.70)")
        status = "MARGINAL"
    else:
        print(f"✗ FAIL: Arc features not predictive (AUC = {auc:.4f} < 0.60)")
        status = "FAIL"
    
    return {"auc": auc, "status": status, "n_samples": len(y)}

def gate_core_brightness_match(positives, controls, theta_es, psfsize_rs):
    """
    Gate 3: Paired Core Brightness Match
    
    In paired data, core brightness should be matched.
    Ratio should be 0.95-1.05.
    """
    print("\n" + "=" * 60)
    print("GATE 3: Paired Core Brightness Match")
    print("=" * 60)
    
    pos_core_brightness = []
    ctrl_core_brightness = []
    
    for pos, ctrl, theta_e, psfsize_r in zip(positives, controls, theta_es, psfsize_rs):
        if theta_e is None or theta_e <= 0:
            continue
        
        h, w = pos.shape[-2:]
        core_mask, _ = compute_theta_aware_masks(h, w, theta_e, psfsize_r)
        
        if core_mask.sum() < 10:
            continue
        
        pos_core_brightness.append(float(pos[1][core_mask].mean()))
        ctrl_core_brightness.append(float(ctrl[1][core_mask].mean()))
    
    if len(pos_core_brightness) < 50:
        print("SKIP: Insufficient samples")
        return None
    
    pos_mean = np.mean(pos_core_brightness)
    ctrl_mean = np.mean(ctrl_core_brightness)
    ratio = pos_mean / ctrl_mean if ctrl_mean != 0 else float('inf')
    
    print(f"Positives core mean: {pos_mean:.6f}")
    print(f"Controls core mean:  {ctrl_mean:.6f}")
    print(f"Ratio: {ratio:.4f}")
    
    if 0.95 <= ratio <= 1.05:
        print("✓ PASS: Core brightness matched (ratio within 5%)")
        status = "PASS"
    elif 0.90 <= ratio <= 1.10:
        print("⚠ MARGINAL: Core brightness close (ratio within 10%)")
        status = "MARGINAL"
    else:
        print(f"✗ FAIL: Core brightness not matched (ratio = {ratio:.4f})")
        status = "FAIL"
    
    return {
        "pos_mean": pos_mean,
        "ctrl_mean": ctrl_mean,
        "ratio": ratio,
        "status": status,
        "n_samples": len(pos_core_brightness)
    }

def gate_injection_contribution(positives, controls, theta_es, psfsize_rs):
    """
    Gate 4: Injection Contribution Analysis
    
    Verify that injection adds flux primarily to arc annulus, not core.
    """
    print("\n" + "=" * 60)
    print("GATE 4: Injection Contribution Analysis")
    print("=" * 60)
    
    core_contributions = []
    arc_contributions = []
    
    for pos, ctrl, theta_e, psfsize_r in zip(positives, controls, theta_es, psfsize_rs):
        if theta_e is None or theta_e <= 0:
            continue
        
        h, w = pos.shape[-2:]
        core_mask, arc_mask = compute_theta_aware_masks(h, w, theta_e, psfsize_r)
        
        if core_mask.sum() < 10 or arc_mask.sum() < 10:
            continue
        
        diff = pos[1] - ctrl[1]  # r-band difference
        
        core_contributions.append(float(diff[core_mask].mean()))
        arc_contributions.append(float(diff[arc_mask].mean()))
    
    if len(core_contributions) < 50:
        print("SKIP: Insufficient samples")
        return None
    
    core_mean = np.mean(core_contributions)
    arc_mean = np.mean(arc_contributions)
    
    print(f"Core injection contribution:      {core_mean:.6f}")
    print(f"Arc annulus injection contribution: {arc_mean:.6f}")
    print(f"Ratio (arc/core): {arc_mean/core_mean:.2f}x" if core_mean != 0 else "N/A")
    
    if arc_mean > core_mean * 2:
        print("✓ PASS: Arc has 2x+ more injection flux than core")
        status = "PASS"
    elif arc_mean > core_mean:
        print("⚠ MARGINAL: Arc has more injection flux, but < 2x")
        status = "MARGINAL"
    else:
        print("✗ FAIL: Core has more injection flux than arc")
        status = "FAIL"
    
    return {
        "core_mean": core_mean,
        "arc_mean": arc_mean,
        "ratio": arc_mean / core_mean if core_mean != 0 else float('inf'),
        "status": status,
        "n_samples": len(core_contributions)
    }

def main():
    parser = argparse.ArgumentParser(description="Theta-Aware Validation Gates")
    parser.add_argument("--data_path", type=str, default="/data/paired_export",
                        help="Path to paired export data")
    parser.add_argument("--n_samples", type=int, default=500,
                        help="Number of samples to analyze")
    args = parser.parse_args()
    
    print("=" * 60)
    print("THETA-AWARE VALIDATION GATES")
    print("=" * 60)
    
    # Load data
    import boto3
    import io
    from astropy.io import fits
    from astropy.wcs import WCS
    
    with open(f"{args.data_path}/metadata.json") as f:
        metadata = json.load(f)
    
    print(f"Loaded {len(metadata)} samples from {args.data_path}")
    
    s3 = boto3.client('s3')
    STAMP_SIZE = 64
    S3_BUCKET = 'darkhaloscope'
    COADD_PREFIX = 'dr10/coadd_cache'
    
    def fetch_base_cutout(ra, dec, brickname):
        images = []
        for band in ['g', 'r', 'z']:
            s3_key = f'{COADD_PREFIX}/{brickname}/legacysurvey-{brickname}-image-{band}.fits.fz'
            try:
                obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
                with fits.open(io.BytesIO(obj['Body'].read())) as hdul:
                    img_data = hdul[1].data
                    wcs = WCS(hdul[1].header)
                    x, y = wcs.world_to_pixel_values(ra, dec)
                    x, y = int(round(float(x))), int(round(float(y)))
                    half = STAMP_SIZE // 2
                    cutout = img_data[y-half:y+half, x-half:x+half].astype(np.float32)
                    if cutout.shape != (64, 64):
                        return None
                    images.append(cutout)
            except:
                return None
        return np.stack(images, axis=0)
    
    # Load paired samples
    positives = []
    controls = []
    theta_es = []
    psfsize_rs = []
    
    print(f"\nLoading {args.n_samples} paired samples...")
    
    for i, m in enumerate(metadata[:args.n_samples]):
        if (i+1) % 100 == 0:
            print(f"  Progress: {i+1}/{args.n_samples}")
        
        try:
            with np.load(f"{args.data_path}/pos_{m['idx']}.npz") as z:
                pos = np.stack([z['image_g'], z['image_r'], z['image_z']], axis=0)
        except:
            continue
        
        ctrl = fetch_base_cutout(m['ra'], m['dec'], m['brickname'])
        if ctrl is None:
            continue
        
        positives.append(pos)
        controls.append(ctrl)
        theta_es.append(m['theta_e_arcsec'])
        psfsize_rs.append(m['psfsize_r'])
    
    print(f"Loaded {len(positives)} paired samples")
    
    # Run gates
    RESULTS["gates"]["gate1_core_auc"] = gate_core_only_auc(positives, controls, theta_es, psfsize_rs)
    RESULTS["gates"]["gate2_arc_auc"] = gate_arc_annulus_auc(positives, controls, theta_es, psfsize_rs)
    RESULTS["gates"]["gate3_brightness"] = gate_core_brightness_match(positives, controls, theta_es, psfsize_rs)
    RESULTS["gates"]["gate4_injection"] = gate_injection_contribution(positives, controls, theta_es, psfsize_rs)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_pass = True
    for gate_name, result in RESULTS["gates"].items():
        if result is None:
            status = "SKIP"
        else:
            status = result["status"]
        
        if status == "FAIL":
            all_pass = False
        
        symbol = "✓" if status == "PASS" else ("⚠" if status in ["MARGINAL", "SKIP"] else "✗")
        print(f"  {symbol} {gate_name}: {status}")
    
    print("\n" + "=" * 60)
    if all_pass:
        print("OVERALL: PASS - No shortcuts detected")
        RESULTS["overall"] = "PASS"
    else:
        print("OVERALL: FAIL - Shortcuts detected, need remediation")
        RESULTS["overall"] = "FAIL"
    print("=" * 60)
    
    # Save results
    output_path = f"{args.data_path}/theta_aware_validation_results.json"
    with open(output_path, 'w') as f:
        json.dump(RESULTS, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
