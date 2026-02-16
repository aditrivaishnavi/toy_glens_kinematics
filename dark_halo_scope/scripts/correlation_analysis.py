#!/usr/bin/env python3
"""Analyze correlation between model predictions and lens properties."""
import numpy as np
from pathlib import Path
from astropy.io import fits

# Full p_lens results from evaluation
p_lens_results = {
    "SDSSJ1205+4910": 0.9100, "SDSSJ1106+5228": 0.7173, "SDSSJ2300+0022": 0.6690,
    "SDSSJ0252+0039": 0.4933, "BELLSJ1631+1854": 0.3990, "BELLSJ0903+4116": 0.3841,
    "SDSSJ1451-0239": 0.3434, "BELLSJ1545+2748": 0.3428, "BELLSJ1221+3806": 0.3417,
    "BELLSJ0847+2348": 0.2994, "SDSSJ0936+0913": 0.1920, "SDSSJ1143-0144": 0.1704,
    "SDSSJ0912+0029": 0.1463, "SDSSJ1432+6317": 0.1422, "SDSSJ1016+3859": 0.1325,
    "SDSSJ0029-0055": 0.1225, "SDSSJ1029+0420": 0.1121, "SDSSJ1142+1001": 0.1088,
    "BELLSJ0801+4727": 0.1082, "BELLSJ1601+2138": 0.0717,
}

# Einstein radii from SLACS/BELLS catalogs
theta_e_dict = {
    "SDSSJ0029-0055": 0.96, "SDSSJ0037-0942": 1.53, "SDSSJ0216-0813": 1.16,
    "SDSSJ0252+0039": 1.04, "SDSSJ0330-0020": 1.10, "SDSSJ0728+3835": 1.25,
    "SDSSJ0737+3216": 0.98, "SDSSJ0822+2652": 1.17, "SDSSJ0912+0029": 1.63,
    "SDSSJ0936+0913": 1.09, "SDSSJ0956+5100": 1.33, "SDSSJ0959+0410": 0.99,
    "SDSSJ1016+3859": 1.09, "SDSSJ1020+1122": 1.20, "SDSSJ1023+4230": 1.41,
    "SDSSJ1029+0420": 1.01, "SDSSJ1106+5228": 1.23, "SDSSJ1112+0826": 1.49,
    "SDSSJ1134+6027": 1.10, "SDSSJ1142+1001": 0.98, "SDSSJ1143-0144": 1.68,
    "SDSSJ1153+4612": 1.05, "SDSSJ1204+0358": 1.31, "SDSSJ1205+4910": 1.22,
    "SDSSJ1213+6708": 1.42, "SDSSJ1218+0830": 1.45, "SDSSJ1250+0523": 1.13,
    "SDSSJ1402+6321": 1.35, "SDSSJ1403+0006": 0.83, "SDSSJ1416+5136": 1.37,
    "SDSSJ1420+6019": 1.04, "SDSSJ1430+4105": 1.52, "SDSSJ1432+6317": 1.25,
    "SDSSJ1436-0000": 1.12, "SDSSJ1443+0304": 0.81, "SDSSJ1451-0239": 1.04,
    "SDSSJ1525+3327": 1.31, "SDSSJ1531-0105": 1.71, "SDSSJ1538+5817": 1.00,
    "SDSSJ1621+3931": 1.29, "SDSSJ1627-0053": 1.23, "SDSSJ1630+4520": 1.78,
    "SDSSJ1636+4707": 1.09, "SDSSJ2238-0754": 1.27, "SDSSJ2300+0022": 1.24,
    "SDSSJ2303+1422": 1.62, "SDSSJ2321-0939": 1.60, "SDSSJ2341+0000": 1.44,
    "BELLSJ0747+4448": 1.16, "BELLSJ0801+4727": 0.91, "BELLSJ0830+5116": 1.41,
    "BELLSJ0847+2348": 1.26, "BELLSJ0903+4116": 1.48, "BELLSJ0918+5104": 1.28,
    "BELLSJ1014+3920": 1.07, "BELLSJ1110+2808": 1.37, "BELLSJ1159+5820": 0.97,
    "BELLSJ1221+3806": 1.53, "BELLSJ1226+5457": 1.22, "BELLSJ1318+3942": 1.08,
    "BELLSJ1349+3612": 0.95, "BELLSJ1401+3845": 1.35, "BELLSJ1522+2910": 1.42,
    "BELLSJ1541+1812": 1.48, "BELLSJ1545+2748": 1.34, "BELLSJ1601+2138": 1.52,
    "BELLSJ1611+1705": 1.19, "BELLSJ1631+1854": 1.06,
}

anchor_dir = Path("/lambda/nfs/darkhaloscope-training-dc/anchor_cutouts/known_lenses")

print("=== CORRELATION ANALYSIS ===")
print()

data = []
for fits_file in sorted(anchor_dir.glob("*.fits")):
    name = fits_file.stem
    with fits.open(fits_file) as hdu:
        img = hdu[0].data.astype(np.float32)
    r_max = img[1].max()  # r-band max
    central_r = img[1, 28:36, 28:36].max()  # central 8x8 max
    
    # Get p_lens if we have it
    p = p_lens_results.get(name, np.nan)
    te = theta_e_dict.get(name, np.nan)
    
    data.append({
        "name": name,
        "p_lens": p,
        "theta_e": te,
        "r_max": r_max,
        "central_r_max": central_r
    })

# Sort by p_lens
data = sorted(data, key=lambda x: x["p_lens"] if not np.isnan(x["p_lens"]) else -1, reverse=True)

# Print table
print(f"{'Name':<30} {'p_lens':<8} {'theta_e':<8} {'r_max':<10} {'central_r':<10}")
print("-" * 70)
for d in data[:20]:
    print(f"{d['name']:<30} {d['p_lens']:<8.4f} {d['theta_e']:<8.2f} {d['r_max']:<10.4f} {d['central_r_max']:<10.4f}")

print()
print("... bottom 10 ...")
for d in data[-10:]:
    p_str = f"{d['p_lens']:.4f}" if not np.isnan(d['p_lens']) else "N/A"
    print(f"{d['name']:<30} {p_str:<8} {d['theta_e']:<8.2f} {d['r_max']:<10.4f} {d['central_r_max']:<10.4f}")

# Compute correlation
valid = [(d["p_lens"], d["r_max"]) for d in data if not np.isnan(d["p_lens"])]
if len(valid) > 2:
    p_vals = [v[0] for v in valid]
    r_vals = [v[1] for v in valid]
    corr = np.corrcoef(p_vals, r_vals)[0, 1]
    print()
    print(f"=== CORRELATION: p_lens vs r_max ===")
    print(f"Pearson r = {corr:.3f}")
    if corr > 0.5:
        print("STRONG positive correlation - model responds to image brightness")
    elif corr > 0.3:
        print("MODERATE positive correlation")
    else:
        print("WEAK or no correlation")

# Also check theta_e correlation
valid_te = [(d["p_lens"], d["theta_e"]) for d in data if not np.isnan(d["p_lens"]) and not np.isnan(d["theta_e"])]
if len(valid_te) > 2:
    p_vals = [v[0] for v in valid_te]
    te_vals = [v[1] for v in valid_te]
    corr_te = np.corrcoef(p_vals, te_vals)[0, 1]
    print()
    print(f"=== CORRELATION: p_lens vs theta_e ===")
    print(f"Pearson r = {corr_te:.3f}")
