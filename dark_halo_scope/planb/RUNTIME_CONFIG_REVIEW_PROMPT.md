# Runtime Configuration Review Request

**Context**: Following your review of our selection functions (which we have implemented), we now have the runtime configuration for building anchor and contaminant sets. Please review and confirm this is sound.

---

## Data Collection Targets

| Dataset | Target N | Sources |
|---------|----------|---------|
| **Anchors** | 100-150 | SLACS/BELLS (68 existing) + Huang+2020 LS ML (50+) + Jacobs+2019 (30+) |
| **Contaminants** | 170-200 | Galaxy Zoo rings (50), spirals (50), mergers (30), spikes (20), edge-on (20) |

## Anchor Sources

| Source | N | θ_E Range | Expected Tier-A % | How to Get |
|--------|---|-----------|-------------------|------------|
| SLACS/BELLS | 68 | 0.8-2.5" | ~20% (HST-confirmed, often faint) | Already on NFS |
| Huang+2020 LS ML | 50+ | 0.5-3" | ~80% (ground-based discovery) | Paper appendix |
| Jacobs+2019 | 30+ | 0.5-2" | ~70% (ground-based discovery) | VizieR |

## Contaminant Sources

| Category | N | Source | Selection |
|----------|---|--------|-----------|
| Rings | 50 | Galaxy Zoo 2 | `p_ring > 0.5` |
| Spirals | 50 | Galaxy Zoo 2 | `p_spiral > 0.8, p_edge_on < 0.3` |
| Mergers | 30 | Galaxy Zoo Mergers | `P_merg > 0.6` |
| Spikes | 20 | Gaia DR3 bright stars | Cross-match with DR10 |
| Edge-on | 20 | Galaxy Zoo 2 | `p_edge_on > 0.8` |

## Cutout Download

```python
# Legacy Survey DR10 cutout service
url = f"https://www.legacysurvey.org/viewer/fits-cutout?ra={ra}&dec={dec}&layer=ls-dr10&pixscale=0.262&bands=grz&size=64"
```

- Size: 64×64 pixels at 0.262"/pixel (matches training)
- Bands: g, r, z (matches training)
- Total: ~350 cutouts, ~50 MB

## Arc Visibility Measurement

For anchors without published arc SNR, we will measure ourselves:

```python
def arc_visibility_snr(cutout, inner_r=4, outer_r=16):
    """Measure arc SNR in annulus region of r-band."""
    # Background from outer region
    bg = np.median(cutout[r > outer_r])
    noise = np.std(cutout[r > outer_r])
    
    # Arc signal in annulus
    annulus = (r >= inner_r) & (r < outer_r)
    signal = np.sum(cutout[annulus] - bg)
    
    return signal / (noise * np.sqrt(annulus.sum()))
```

## Selection Function (Implemented)

```python
AnchorSelectionFunction(
    theta_e_min=0.5,
    theta_e_max=3.0,
    arc_snr_min=2.0,
    on_missing_arc_visibility="TIER_B",  # Defensive
    on_missing_dr10_flag="EXCLUDE",       # Defensive
)

ContaminantSelectionFunction(
    valid_categories={"ring", "spiral", "merger", "spike", "edge_on"},
    exclude_anchor_matches=True,          # Coordinate cross-match
    anchor_match_radius_arcsec=3.0,
    on_missing_dr10_flag="EXCLUDE",
)
```

## Execution Environment

| Task | Where | Why |
|------|-------|-----|
| Catalog queries | Local | VizieR/astroquery, no GPU |
| Cutout download | Local → NFS | Network I/O |
| Arc measurement | Lambda (GPU) | Faster processing |
| Model evaluation | Lambda (GPU) | Needs trained model |

## Timeline

| Step | Duration |
|------|----------|
| Build anchor catalog | 1 hr |
| Build contaminant catalog | 1 hr |
| Download cutouts | 1-2 hrs |
| Measure arc visibility | 30 min |
| Apply selection function | 5 min |
| **Total** | **3-4 hrs** |

---

## Questions

1. **Are the target counts sufficient?** (100+ anchors, 170+ contaminants)

2. **Are the Galaxy Zoo selection thresholds appropriate?** (`p_ring > 0.5`, `p_spiral > 0.8`, etc.)

3. **Is the arc visibility measurement method reasonable?** (annulus SNR in r-band)

4. **Any concerns with the Legacy Survey cutout service?** (rate limits, authentication, etc.)

5. **Anything missing from this plan?**

---

Please confirm this configuration is sound or flag any issues.
