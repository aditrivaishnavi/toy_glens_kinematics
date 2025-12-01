# src/utils/lensibility.py

"""
Lensibility scoring utilities.

This module defines:
- GalaxyMetrics: container for quality/compactness metrics.
- lensibility_score: rule-based score + tier + notes for lens suitability.

Intended usage:
- Centralize all "is this galaxy a good kinematic source for lensing?"
  logic in one place, so other scripts can reuse it consistently.
"""

from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class GalaxyMetrics:
    """Minimal set of metrics needed to assess lens suitability."""

    file: str
    status: str

    flux_max: float
    flux_mean: float

    vel_min: float
    vel_max: float
    vel_std: float
    vel_grad: float

    flux_vel_corr: float

    mask_fraction: float
    frac_valid_in_1_5Re: float

    flag_low_rotation: bool
    flag_heavily_masked: bool
    flag_low_flux: bool
    usable_flag: bool  # from earlier pipeline ("usable" column)


def lensibility_score(m: GalaxyMetrics) -> Dict[str, Any]:
    """
    Compute a lens suitability ("lensibility") score for a galaxy.

    Returns
    -------
    result : dict
        {
          "score": float in [0, 100],
          "tier": str in {"elite", "good", "borderline", "reject"},
          "hard_reject": bool,
          "notes": List[str],
        }

    Design:
    - Hard rejects for clearly unusable cases (bad status, flags, etc.).
    - Otherwise, accumulate a score based on:
        vel_grad, frac_valid_in_1_5Re, vel_std, mask_fraction, flux_max,
        flux_mean, flux_vel_corr (diagnostic-level).
    - Weights are chosen heuristically but transparently.
    """

    notes: List[str] = []
    score = 0.0
    hard_reject = False

    # --- 0. Basic sanity checks ---
    if m.status.upper() != "OK":
        notes.append("status != OK → reject")
        return {
            "score": 0.0,
            "tier": "reject",
            "hard_reject": True,
            "notes": notes,
        }

    # If earlier pipeline already declared this unusable, honor that.
    if not m.usable_flag:
        notes.append("usable_flag is False → reject by upstream criteria")
        hard_reject = True

    # Hard flags from diagnostics
    if m.flag_low_rotation:
        notes.append("flag_low_rotation is True")
        hard_reject = True

    if m.flag_heavily_masked:
        notes.append("flag_heavily_masked is True (mask_fraction > 0.7)")
        hard_reject = True

    if m.flag_low_flux:
        notes.append("flag_low_flux is True (max flux too small)")
        hard_reject = True

    # Hard flux sanity: almost no emission
    if m.flux_max < 50.0:
        notes.append(f"flux_max={m.flux_max:.1f} < 50 → too weak emission")
        hard_reject = True

    # Hard kinematic sanity
    if m.frac_valid_in_1_5Re < 0.6:
        notes.append(
            f"frac_valid_in_1.5Re={m.frac_valid_in_1_5Re:.2f} < 0.60 → poor coverage"
        )
        hard_reject = True

    if m.vel_grad < 2.5:
        notes.append(f"vel_grad={m.vel_grad:.2f} < 2.5 km/s/pix → too weak gradient")
        hard_reject = True

    if m.vel_std < 50.0:
        notes.append(f"vel_std={m.vel_std:.1f} < 50 km/s → nearly flat field")
        hard_reject = True

    if hard_reject:
        return {
            "score": 0.0,
            "tier": "reject",
            "hard_reject": True,
            "notes": notes,
        }

    # --- 1. vel_grad: main sensitivity metric (0–30 points) ---
    if m.vel_grad >= 5.0:
        score += 30.0
        notes.append(f"vel_grad={m.vel_grad:.2f} ≥ 5 → strong rotation (+30)")
    elif m.vel_grad >= 3.0:
        score += 22.0
        notes.append(f"vel_grad={m.vel_grad:.2f} in [3,5) → good rotation (+22)")
    else:  # 2.5–3 (we already excluded <2.5 above)
        score += 12.0
        notes.append(f"vel_grad={m.vel_grad:.2f} in [2.5,3) → borderline rotation (+12)")

    # --- 2. frac_valid_in_1.5Re: coverage in region that matters (0–30 points) ---
    if m.frac_valid_in_1_5Re >= 0.9:
        score += 30.0
        notes.append(
            f"frac_valid_in_1.5Re={m.frac_valid_in_1_5Re:.2f} ≥ 0.90 → excellent coverage (+30)"
        )
    elif m.frac_valid_in_1_5Re >= 0.8:
        score += 24.0
        notes.append(
            f"frac_valid_in_1.5Re={m.frac_valid_in_1_5Re:.2f} in [0.80,0.90) → very good coverage (+24)"
        )
    else:  # in [0.6, 0.8)
        score += 16.0
        notes.append(
            f"frac_valid_in_1.5Re={m.frac_valid_in_1_5Re:.2f} in [0.60,0.80) → acceptable coverage (+16)"
        )

    # --- 3. vel_std: amplitude of velocity field (0–20 points) ---
    if m.vel_std >= 120.0:
        score += 20.0
        notes.append(f"vel_std={m.vel_std:.1f} ≥ 120 → very strong signal (+20)")
    elif m.vel_std >= 70.0:
        score += 14.0
        notes.append(f"vel_std={m.vel_std:.1f} in [70,120) → good amplitude (+14)")
    else:  # in [50,70)
        score += 8.0
        notes.append(f"vel_std={m.vel_std:.1f} in [50,70) → modest amplitude (+8)")

    # --- 4. mask_fraction: global sparsity (0–10 points) ---
    if m.mask_fraction <= 0.4:
        score += 10.0
        notes.append(
            f"mask_fraction={m.mask_fraction:.2f} ≤ 0.40 → low masking globally (+10)"
        )
    elif m.mask_fraction <= 0.6:
        score += 6.0
        notes.append(
            f"mask_fraction={m.mask_fraction:.2f} in (0.40,0.60] → typical MaNGA masking (+6)"
        )
    elif m.mask_fraction <= 0.7:
        score += 2.0
        notes.append(
            f"mask_fraction={m.mask_fraction:.2f} in (0.60,0.70] → heavy masking but acceptable (+2)"
        )
    else:
        # >0.7 would have been rejected earlier, so we shouldn't reach here
        notes.append(
            f"mask_fraction={m.mask_fraction:.2f} > 0.70 → would have been hard-rejected"
        )

    # --- 5. Flux sanity (0–5 points) ---
    # We already hard-rejected flux_max < 50; here we just mildly reward stronger Hα.
    if m.flux_max >= 200.0:
        score += 5.0
        notes.append(f"flux_max={m.flux_max:.1f} ≥ 200 → strong line emission (+5)")
    elif m.flux_max >= 100.0:
        score += 3.0
        notes.append(
            f"flux_max={m.flux_max:.1f} in [100,200) → decent line emission (+3)"
        )
    else:
        score += 1.0
        notes.append(
            f"flux_max={m.flux_max:.1f} in [50,100) → minimal acceptable emission (+1)"
        )

    # --- 6. flux_vel_corr: diagnostic-level (penalty only if extreme) ---
    corr_abs = abs(m.flux_vel_corr)
    if corr_abs <= 0.2:
        score += 3.0
        notes.append(
            f"|flux_vel_corr|={corr_abs:.3f} ≤ 0.20 → orthogonal flux/vel (disk-like) (+3)"
        )
    elif corr_abs <= 0.4:
        notes.append(
            f"|flux_vel_corr|={corr_abs:.3f} in (0.20,0.40] → mild correlation (possible bar / asymmetry)"
        )
    else:
        score -= 5.0
        notes.append(
            f"|flux_vel_corr|={corr_abs:.3f} > 0.40 → strong correlation (bar/merger?) (−5)"
        )

    # --- Clip score to [0, 100] ---
    if score < 0.0:
        score = 0.0
    if score > 100.0:
        score = 100.0

    # --- 7. Map numeric score to tier ---
    if score >= 80.0:
        tier = "elite"
    elif score >= 60.0:
        tier = "good"
    elif score >= 40.0:
        tier = "borderline"
    else:
        tier = "reject"

    return {
        "score": float(score),
        "tier": tier,
        "hard_reject": False,
        "notes": notes,
    }

