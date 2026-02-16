#!/usr/bin/env python3
"""Build comparison table and detection-rate figure from D04 matched results.

Reads the CSV outputs from the D04 matched comparison grids:
  1. grid_no_poisson/selection_function.csv     -- baseline
  2. grid_poisson_fixed/selection_function.csv  -- fixed Poisson
  3. poisson_fixed_clip20_combined/results.csv  -- bright-arc diagnostic

Produces:
  - Lensed-magnitude-stratified completeness table (LaTeX + markdown)
  - Multi-configuration detection-rate figure
  - Summary statistics JSON

Usage:
    python scripts/build_d04_comparison.py \
        --d04-dir results/D04_20260214_matched_comparison/ \
        --out-dir results/D04_20260214_matched_comparison/analysis/
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_grid_csv(path: str) -> pd.DataFrame:
    """Load a selection_function.csv and add derived columns."""
    df = pd.read_csv(path)
    # Ensure numeric
    for col in ["n_injections", "n_detected", "completeness"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def aggregate_by_lensed_mag(df: pd.DataFrame, threshold: float = 0.3):
    """Compute injection-weighted completeness by lensed magnitude bin."""
    # Filter to lensed_* bins and specific threshold
    mask = (
        df["source_mag_bin"].str.startswith("lensed_") &
        (df["threshold"] == threshold) &
        (df["threshold_type"] == "fixed") &
        (df["n_injections"] > 0)
    )
    lensed = df[mask].copy()
    rows = []
    for smb in sorted(lensed["source_mag_bin"].unique()):
        sub = lensed[lensed["source_mag_bin"] == smb]
        n_inj = int(sub["n_injections"].sum())
        n_det = int(sub["n_detected"].sum())
        c = n_det / n_inj if n_inj > 0 else float("nan")
        lo, hi = wilson_ci(n_det, n_inj)
        rows.append({
            "lensed_mag_bin": smb,
            "n_injections": n_inj,
            "n_detected": n_det,
            "completeness": c,
            "ci_lo": lo,
            "ci_hi": hi,
        })
    return pd.DataFrame(rows)


def aggregate_marginal(df: pd.DataFrame, threshold: float = 0.3):
    """Compute marginal completeness from source_mag_bin='all' rows."""
    mask = (
        (df["source_mag_bin"] == "all") &
        (df["threshold"] == threshold) &
        (df["threshold_type"] == "fixed") &
        (df["n_injections"] > 0)
    )
    sub = df[mask]
    n_inj = int(sub["n_injections"].sum())
    n_det = int(sub["n_detected"].sum())
    c = n_det / n_inj if n_inj > 0 else float("nan")
    lo, hi = wilson_ci(n_det, n_inj)
    return {"n_injections": n_inj, "n_detected": n_det,
            "completeness": c, "ci_lo": lo, "ci_hi": hi}


def aggregate_by_theta_e(df: pd.DataFrame, threshold: float = 0.3):
    """Compute completeness by theta_E from 'all' mag bin."""
    mask = (
        (df["source_mag_bin"] == "all") &
        (df["threshold"] == threshold) &
        (df["threshold_type"] == "fixed") &
        (df["n_injections"] > 0)
    )
    sub = df[mask]
    rows = []
    for te in sorted(sub["theta_e"].unique()):
        te_sub = sub[sub["theta_e"] == te]
        n_inj = int(te_sub["n_injections"].sum())
        n_det = int(te_sub["n_detected"].sum())
        c = n_det / n_inj if n_inj > 0 else float("nan")
        lo, hi = wilson_ci(n_det, n_inj)
        rows.append({
            "theta_e": te,
            "n_injections": n_inj,
            "n_detected": n_det,
            "completeness": c,
            "ci_lo": lo,
            "ci_hi": hi,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def make_lensed_mag_comparison_figure(
    baseline_mag: pd.DataFrame,
    poisson_mag: pd.DataFrame,
    out_path: str,
):
    """Bar chart comparing completeness by lensed magnitude: baseline vs Poisson."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # Merge on lensed_mag_bin
    merged = pd.merge(
        baseline_mag, poisson_mag,
        on="lensed_mag_bin", suffixes=("_baseline", "_poisson"),
        how="outer"
    ).sort_values("lensed_mag_bin")

    x = np.arange(len(merged))
    width = 0.35

    bars1 = ax.bar(x - width/2, merged["completeness_baseline"] * 100,
                   width, label="No Poisson", color="#4C72B0", alpha=0.85)
    bars2 = ax.bar(x + width/2, merged["completeness_poisson"] * 100,
                   width, label="Poisson (fixed)", color="#DD8452", alpha=0.85)

    # Error bars
    for bars_obj, lo_col, hi_col, c_col in [
        (bars1, "ci_lo_baseline", "ci_hi_baseline", "completeness_baseline"),
        (bars2, "ci_lo_poisson", "ci_hi_poisson", "completeness_poisson"),
    ]:
        if lo_col in merged.columns and hi_col in merged.columns:
            yerr_lo = (merged[c_col] - merged[lo_col]) * 100
            yerr_hi = (merged[hi_col] - merged[c_col]) * 100
            for i, bar in enumerate(bars_obj):
                if not np.isnan(merged.iloc[i][c_col]):
                    ax.errorbar(bar.get_x() + bar.get_width()/2,
                                bar.get_height(),
                                yerr=[[yerr_lo.iloc[i]], [yerr_hi.iloc[i]]],
                                fmt='none', color='black', capsize=3, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(merged["lensed_mag_bin"], rotation=45, ha="right")
    ax.set_ylabel("Completeness (%)")
    ax.set_xlabel("Lensed Apparent Magnitude Bin")
    ax.set_title("D04: Completeness by Lensed Magnitude — Baseline vs Fixed Poisson")
    ax.legend(loc="upper right")
    ax.set_ylim(0, None)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved lensed-mag comparison figure: {out_path}")


def make_theta_e_comparison_figure(
    baseline_te: pd.DataFrame,
    poisson_te: pd.DataFrame,
    out_path: str,
):
    """Line plot comparing completeness by theta_E: baseline vs Poisson."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for label, df, color, marker in [
        ("No Poisson", baseline_te, "#4C72B0", "o"),
        ("Poisson (fixed)", poisson_te, "#DD8452", "s"),
    ]:
        ax.errorbar(
            df["theta_e"], df["completeness"] * 100,
            yerr=[(df["completeness"] - df["ci_lo"]) * 100,
                  (df["ci_hi"] - df["completeness"]) * 100],
            label=label, color=color, marker=marker, markersize=6,
            capsize=3, linewidth=1.5
        )

    ax.set_xlabel(r"$\theta_E$ (arcsec)")
    ax.set_ylabel("Completeness (%)")
    ax.set_title(r"D04: Completeness by $\theta_E$ — Baseline vs Fixed Poisson")
    ax.legend()
    ax.set_ylim(0, None)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved theta_E comparison figure: {out_path}")


# --------------------------------------------------------------------------- #
# Markdown / LaTeX table builders
# --------------------------------------------------------------------------- #

def build_markdown_table(baseline_mag, poisson_mag) -> str:
    """Build a markdown comparison table."""
    merged = pd.merge(
        baseline_mag, poisson_mag,
        on="lensed_mag_bin", suffixes=("_bl", "_pn"), how="outer"
    ).sort_values("lensed_mag_bin")

    lines = [
        "| Lensed Mag Bin | N_inj (BL) | C_BL (%) | 95% CI (BL) | N_inj (Pn) | C_Pn (%) | 95% CI (Pn) | Delta (pp) |",
        "|:---|---:|---:|:---|---:|---:|:---|---:|",
    ]
    for _, row in merged.iterrows():
        c_bl = row.get("completeness_bl", float("nan"))
        c_pn = row.get("completeness_pn", float("nan"))
        delta = (c_pn - c_bl) * 100 if not (np.isnan(c_bl) or np.isnan(c_pn)) else float("nan")
        lo_bl = row.get("ci_lo_bl", float("nan"))
        hi_bl = row.get("ci_hi_bl", float("nan"))
        lo_pn = row.get("ci_lo_pn", float("nan"))
        hi_pn = row.get("ci_hi_pn", float("nan"))
        lines.append(
            f"| {row['lensed_mag_bin']} "
            f"| {int(row.get('n_injections_bl', 0)):,} "
            f"| {c_bl*100:.2f} "
            f"| [{lo_bl*100:.2f}, {hi_bl*100:.2f}] "
            f"| {int(row.get('n_injections_pn', 0)):,} "
            f"| {c_pn*100:.2f} "
            f"| [{lo_pn*100:.2f}, {hi_pn*100:.2f}] "
            f"| {delta:+.2f} |"
        )
    return "\n".join(lines)


def build_latex_table(baseline_mag, poisson_mag) -> str:
    """Build a LaTeX table for the paper."""
    merged = pd.merge(
        baseline_mag, poisson_mag,
        on="lensed_mag_bin", suffixes=("_bl", "_pn"), how="outer"
    ).sort_values("lensed_mag_bin")

    lines = [
        r"\begin{table}",
        r"\centering",
        r"\caption{Injection-recovery completeness by lensed apparent magnitude, "
        r"comparing baseline (no Poisson noise) and physically correct Poisson noise. "
        r"Both grids use identical parameters: "
        r"$\theta_E \in [0.5, 2.5]''$ (step 0.2), "
        r"PSF FWHM $\in [0.9, 1.8]''$ (step 0.15), "
        r"depth $\in [22.5, 24.5]$ mag (step 0.5), "
        r"500 injections per cell, threshold $p > 0.3$.}",
        r"\label{tab:completeness_poisson}",
        r"\begin{tabular}{lrcrrc}",
        r"\hline",
        r"Lensed mag & \multicolumn{2}{c}{Baseline} & \multicolumn{2}{c}{Poisson} & $\Delta$ \\",
        r" & $C$ (\%) & 95\% CI & $C$ (\%) & 95\% CI & (pp) \\",
        r"\hline",
    ]
    for _, row in merged.iterrows():
        c_bl = row.get("completeness_bl", float("nan"))
        c_pn = row.get("completeness_pn", float("nan"))
        delta = (c_pn - c_bl) * 100 if not (np.isnan(c_bl) or np.isnan(c_pn)) else float("nan")
        lo_bl = row.get("ci_lo_bl", float("nan"))
        hi_bl = row.get("ci_hi_bl", float("nan"))
        lo_pn = row.get("ci_lo_pn", float("nan"))
        hi_pn = row.get("ci_hi_pn", float("nan"))
        lines.append(
            f"{row['lensed_mag_bin']} & "
            f"{c_bl*100:.2f} & [{lo_bl*100:.2f}, {hi_bl*100:.2f}] & "
            f"{c_pn*100:.2f} & [{lo_pn*100:.2f}, {hi_pn*100:.2f}] & "
            f"{delta:+.2f} \\\\"
        )
    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Build D04 comparison table and figure")
    ap.add_argument("--d04-dir", required=True, help="D04 results directory")
    ap.add_argument("--out-dir", default=None, help="Output directory (default: d04-dir/analysis)")
    ap.add_argument("--threshold", type=float, default=0.3, help="Score threshold")
    args = ap.parse_args()

    d04_dir = args.d04_dir
    out_dir = args.out_dir or os.path.join(d04_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    # Load grid CSVs
    baseline_csv = os.path.join(d04_dir, "grid_no_poisson", "selection_function.csv")
    poisson_csv = os.path.join(d04_dir, "grid_poisson_fixed", "selection_function.csv")

    if not os.path.exists(baseline_csv):
        print(f"ERROR: Baseline CSV not found: {baseline_csv}")
        sys.exit(1)
    if not os.path.exists(poisson_csv):
        print(f"ERROR: Poisson CSV not found: {poisson_csv}")
        sys.exit(1)

    print(f"Loading baseline: {baseline_csv}")
    df_baseline = load_grid_csv(baseline_csv)
    print(f"  {len(df_baseline)} rows")

    print(f"Loading Poisson-fixed: {poisson_csv}")
    df_poisson = load_grid_csv(poisson_csv)
    print(f"  {len(df_poisson)} rows")

    thr = args.threshold
    print(f"\n--- Marginal completeness (threshold={thr}) ---")

    m_bl = aggregate_marginal(df_baseline, thr)
    m_pn = aggregate_marginal(df_poisson, thr)
    print(f"  Baseline:       {m_bl['completeness']:.4f} ({m_bl['n_detected']}/{m_bl['n_injections']}) "
          f"CI=[{m_bl['ci_lo']:.4f}, {m_bl['ci_hi']:.4f}]")
    print(f"  Poisson (fixed): {m_pn['completeness']:.4f} ({m_pn['n_detected']}/{m_pn['n_injections']}) "
          f"CI=[{m_pn['ci_lo']:.4f}, {m_pn['ci_hi']:.4f}]")
    delta = (m_pn["completeness"] - m_bl["completeness"]) * 100
    print(f"  Delta: {delta:+.2f} pp")

    # Lensed-magnitude stratified
    print(f"\n--- Lensed-magnitude stratified completeness ---")
    mag_bl = aggregate_by_lensed_mag(df_baseline, thr)
    mag_pn = aggregate_by_lensed_mag(df_poisson, thr)
    print("  Baseline:")
    for _, r in mag_bl.iterrows():
        print(f"    {r['lensed_mag_bin']}: {r['completeness']*100:.2f}% ({r['n_detected']}/{r['n_injections']})")
    print("  Poisson (fixed):")
    for _, r in mag_pn.iterrows():
        print(f"    {r['lensed_mag_bin']}: {r['completeness']*100:.2f}% ({r['n_detected']}/{r['n_injections']})")

    # theta_E stratified
    print(f"\n--- theta_E stratified completeness ---")
    te_bl = aggregate_by_theta_e(df_baseline, thr)
    te_pn = aggregate_by_theta_e(df_poisson, thr)
    print("  Baseline:")
    for _, r in te_bl.iterrows():
        print(f"    theta_E={r['theta_e']:.2f}: {r['completeness']*100:.2f}% ({r['n_detected']}/{r['n_injections']})")
    print("  Poisson (fixed):")
    for _, r in te_pn.iterrows():
        print(f"    theta_E={r['theta_e']:.2f}: {r['completeness']*100:.2f}% ({r['n_detected']}/{r['n_injections']})")

    # Build tables
    md_table = build_markdown_table(mag_bl, mag_pn)
    latex_table = build_latex_table(mag_bl, mag_pn)

    md_path = os.path.join(out_dir, "completeness_comparison_table.md")
    with open(md_path, "w") as f:
        f.write("# D04 Completeness Comparison: Baseline vs Fixed Poisson\n\n")
        f.write(f"Threshold: p > {thr}\n\n")
        f.write("## Marginal Completeness\n\n")
        f.write(f"- **Baseline (no Poisson):** {m_bl['completeness']*100:.2f}% "
                f"({m_bl['n_detected']}/{m_bl['n_injections']}) "
                f"CI=[{m_bl['ci_lo']*100:.2f}%, {m_bl['ci_hi']*100:.2f}%]\n")
        f.write(f"- **Poisson (fixed):** {m_pn['completeness']*100:.2f}% "
                f"({m_pn['n_detected']}/{m_pn['n_injections']}) "
                f"CI=[{m_pn['ci_lo']*100:.2f}%, {m_pn['ci_hi']*100:.2f}%]\n")
        f.write(f"- **Delta:** {delta:+.2f} pp\n\n")
        f.write("## By Lensed Apparent Magnitude\n\n")
        f.write(md_table)
        f.write("\n\n## By Einstein Radius\n\n")
        f.write("### Baseline\n\n")
        for _, r in te_bl.iterrows():
            f.write(f"- theta_E={r['theta_e']:.2f}\": C={r['completeness']*100:.2f}% "
                    f"({r['n_detected']}/{r['n_injections']})\n")
        f.write("\n### Poisson (fixed)\n\n")
        for _, r in te_pn.iterrows():
            f.write(f"- theta_E={r['theta_e']:.2f}\": C={r['completeness']*100:.2f}% "
                    f"({r['n_detected']}/{r['n_injections']})\n")
    print(f"\n  Saved markdown table: {md_path}")

    latex_path = os.path.join(out_dir, "completeness_comparison_table.tex")
    with open(latex_path, "w") as f:
        f.write(latex_table)
    print(f"  Saved LaTeX table: {latex_path}")

    # Build figures
    fig_mag_path = os.path.join(out_dir, "completeness_by_lensed_mag.png")
    make_lensed_mag_comparison_figure(mag_bl, mag_pn, fig_mag_path)

    fig_te_path = os.path.join(out_dir, "completeness_by_theta_e.png")
    make_theta_e_comparison_figure(te_bl, te_pn, fig_te_path)

    # Summary JSON
    summary = {
        "d04_dir": d04_dir,
        "threshold": thr,
        "marginal_baseline": m_bl,
        "marginal_poisson_fixed": m_pn,
        "delta_pp": delta,
        "n_lensed_mag_bins": len(mag_bl),
        "n_theta_e_bins": len(te_bl),
    }
    summary_path = os.path.join(out_dir, "d04_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved summary JSON: {summary_path}")

    # Also try to load bright-arc diagnostic results if present
    ba_dir = os.path.join(d04_dir, "poisson_fixed_clip20_combined")
    if os.path.exists(ba_dir):
        for fn in sorted(os.listdir(ba_dir)):
            if fn.endswith(".json"):
                jp = os.path.join(ba_dir, fn)
                with open(jp) as f:
                    bright_arc_data = json.load(f)
                print(f"\n--- Bright-arc diagnostic ({fn}) ---")
                print(f"  Poisson: {bright_arc_data.get('add_poisson_noise', '?')}")
                print(f"  Clip range: {bright_arc_data.get('clip_range_override', 'default')}")
                rbb = bright_arc_data.get("results_by_bin", {})
                for mag_bin, vals in sorted(rbb.items()):
                    dr03 = vals.get("detection_rate_p03", float("nan"))
                    dr05 = vals.get("detection_rate_p05", float("nan"))
                    n = vals.get("n_scored", 0)
                    snr = vals.get("median_arc_snr", float("nan"))
                    lo, hi = wilson_ci(int(round(dr03 * n)), n)
                    print(f"    {mag_bin}: {dr03*100:.1f}% (p>0.3) [{lo*100:.1f}%, {hi*100:.1f}%]"
                          f"  |  {dr05*100:.1f}% (p>0.5)  |  n={n}, SNR={snr:.0f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
