#!/usr/bin/env python3
"""
EMR job for building COSMOS source bank.
Runs on master node with multi-core support.
"""
import sys
import os
import json
import subprocess
from pathlib import Path

def main():
    print("=" * 70)
    print("Gen5 COSMOS Bank Builder - EMR Job")
    print("=" * 70)
    
    # Configuration
    config = {
        "cosmos_dir": "/home/hadoop/.galsim/COSMOS_23.5_training_sample",
        "out_h5": "/mnt/cosmos_workspace/cosmos_bank_20k_parametric_v1.h5",
        "n_sources": 20000,
        "stamp_size": 96,
        "src_pixscale_arcsec": 0.03,
        "seed": 42,
        "intrinsic_psf_fwhm_arcsec": 0.0,
        "denoise_sigma_pix": 0.5,
        "hlr_min_arcsec": 0.1,
        "hlr_max_arcsec": 1.5,
        "dtype": "float32",
        "max_tries": 100000
    }
    
    config_path = "/mnt/cosmos_workspace/cosmos_bank_config_20k_v1.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Config saved to: {config_path}")
    print(f"   Building {config['n_sources']} templates")
    print(f"   Output: {config['out_h5']}")
    
    # Download COSMOS catalog if needed (GalSim will auto-download)
    print("\n📥 Ensuring COSMOS catalog is available...")
    download_cmd = """
python3 << 'EOF'
import galsim
import os
import sys

# Force download if not exists
cosmos_dir = os.path.expanduser("~/.galsim/COSMOS_23.5_training_sample")
if not os.path.exists(cosmos_dir):
    print(f"Downloading COSMOS catalog to {cosmos_dir}...")
    try:
        cat = galsim.COSMOSCatalog(use_real=True)
        print(f"✅ COSMOS catalog ready: {len(cat)} galaxies")
    except Exception as e:
        print(f"❌ Failed to download COSMOS catalog: {e}")
        sys.exit(1)
else:
    print(f"✅ COSMOS catalog already exists: {cosmos_dir}")
EOF
"""
    
    result = subprocess.run(download_cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"❌ Error downloading COSMOS catalog:")
        print(result.stderr)
        sys.exit(1)
    
    # Build the bank (this is the CPU-intensive part)
    print("\n🔨 Building COSMOS bank (this will take 30-60 minutes)...")
    print("   Using all available CPU cores for GalSim rendering")
    
    build_cmd = f"""
python3 << 'EOF'
import sys
sys.path.insert(0, '/mnt/code')

from dhs_cosmos.sims.cosmos_source_loader import build_cosmos_bank, BuildConfig

cfg = BuildConfig(
    cosmos_dir="{config['cosmos_dir']}",
    out_h5="{config['out_h5']}",
    n_sources={config['n_sources']},
    stamp_size={config['stamp_size']},
    src_pixscale_arcsec={config['src_pixscale_arcsec']},
    seed={config['seed']},
    intrinsic_psf_fwhm_arcsec={config['intrinsic_psf_fwhm_arcsec']},
    denoise_sigma_pix={config['denoise_sigma_pix']},
    hlr_min_arcsec={config['hlr_min_arcsec']},
    hlr_max_arcsec={config['hlr_max_arcsec']},
    dtype="{config['dtype']}",
    max_tries={config['max_tries']}
)

print("Starting COSMOS bank build...")
build_cosmos_bank(cfg)
print("✅ COSMOS bank build complete!")
EOF
"""
    
    result = subprocess.run(build_cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"❌ Error building COSMOS bank:")
        print(result.stderr)
        sys.exit(1)
    
    # Validate the bank
    print("\n✅ Validating COSMOS bank...")
    validate_cmd = f"""
python3 << 'EOF'
import h5py
import numpy as np

with h5py.File("{config['out_h5']}", "r") as f:
    n = f["images"].shape[0]
    stamp_size = f.attrs["stamp_size"]
    pixscale = f.attrs["src_pixscale_arcsec"]
    hlr = f["meta/hlr_arcsec"][:]
    clump = f["meta/clumpiness"][:]
    
    print(f"✅ Bank validation:")
    print(f"   Sources: {{n}}")
    print(f"   Stamp size: {{stamp_size}}x{{stamp_size}}")
    print(f"   Pixel scale: {{pixscale:.4f}} arcsec/pix")
    print(f"   HLR range: {{np.min(hlr):.3f}} - {{np.max(hlr):.3f}} arcsec")
    print(f"   HLR median: {{np.median(hlr):.3f}} arcsec")
    print(f"   Clumpiness range: {{np.min(clump):.3f}} - {{np.max(clump):.3f}}")
    print(f"   Clumpiness median: {{np.median(clump):.3f}}")
EOF
"""
    
    result = subprocess.run(validate_cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"⚠️  Warning: validation failed (but bank may be OK)")
        print(result.stderr)
    
    # Upload to S3
    s3_bucket = "s3://darkhaloscope/cosmos/"
    print(f"\n📤 Uploading to S3: {s3_bucket}")
    
    upload_cmds = [
        f"aws s3 cp {config['out_h5']} {s3_bucket}",
        f"aws s3 cp {config_path} {s3_bucket}",
    ]
    
    for cmd in upload_cmds:
        print(f"   Running: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Upload failed: {result.stderr}")
            sys.exit(1)
        print(f"   ✅ Uploaded")
    
    print("\n" + "=" * 70)
    print("🎉 COSMOS Bank Creation Complete!")
    print("=" * 70)
    print(f"Local:  {config['out_h5']}")
    print(f"S3:     {s3_bucket}cosmos_bank_20k_parametric_v1.h5")
    print(f"Config: {s3_bucket}cosmos_bank_config_20k_v1.json")
    print("=" * 70)

if __name__ == "__main__":
    main()

