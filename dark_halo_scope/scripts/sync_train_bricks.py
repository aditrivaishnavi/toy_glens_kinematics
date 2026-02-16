#!/usr/bin/env python3
"""Sync training bricks from S3 to local coadd cache."""
import os
import subprocess
import sys
from datetime import datetime

BRICKS_FILE = "/tmp/train_bricks.txt"
DEST = "/lambda/nfs/darkhaloscope-training-dc/dr10/coadd_cache"
LOG = "/lambda/nfs/darkhaloscope-training-dc/logs/coadd_sync.log"

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} | {msg}"
    print(line)
    with open(LOG, "a") as f:
        f.write(line + "\n")

# Load bricks
with open(BRICKS_FILE) as f:
    bricks = [line.strip() for line in f if line.strip()]

total = len(bricks)
log(f"Starting sync of {total} training bricks")

# Check how many already synced
already_synced = 0
to_sync = []
for brick in bricks:
    marker = os.path.join(DEST, brick, f"legacysurvey-{brick}-image-r.fits.fz")
    if os.path.exists(marker):
        already_synced += 1
    else:
        to_sync.append(brick)

log(f"Already synced: {already_synced}, To sync: {len(to_sync)}")

# Sync remaining
synced = 0
errors = 0
for i, brick in enumerate(to_sync):
    src = f"s3remote:darkhaloscope/dr10/coadd_cache/{brick}/"
    dst = os.path.join(DEST, brick)
    
    try:
        result = subprocess.run(
            ["rclone", "copy", src, dst, "--quiet"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            synced += 1
        else:
            errors += 1
            if errors <= 5:
                log(f"Error syncing {brick}: {result.stderr[:100]}")
    except subprocess.TimeoutExpired:
        errors += 1
        log(f"Timeout syncing {brick}")
    except Exception as e:
        errors += 1
        log(f"Exception syncing {brick}: {e}")
    
    # Progress update
    if (i + 1) % 100 == 0:
        log(f"Progress: {i+1}/{len(to_sync)} synced, {errors} errors")

log(f"Completed: {synced} synced, {errors} errors, {already_synced} already present")
log(f"Total bricks available: {already_synced + synced}")
