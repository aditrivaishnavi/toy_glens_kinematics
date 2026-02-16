# Research Plan 4: Deep Ground-Based Sources (Gen6)

## Core Thesis

> "Does training with ground-based deep-field source templates improve sim-to-real transfer compared to HST-derived sources?"

---

## Scientific Justification

HST COSMOS templates at 0.03"/pix contain morphological detail that disappears when downsampled to 0.262"/pix, but:
- PSF characteristics still mismatch (HST diffraction vs ground seeing)
- Noise properties differ (space vs ground background)
- Texture statistics are survey-specific

Ground-based deep cutouts have:
- Same PSF regime as target survey
- Same noise characteristics
- Naturally matched texture statistics

---

## What Gen6 Is

Use cutouts from ground-based deep fields (HSC Ultra-Deep, Legacy DR10 deep regions, DES Deep Fields) as source templates.

```python
def build_deep_source_bank(
    fits_dir,              # Directory with deep cutouts
    out_npz,               # Output bank
    n_sources=20000,
    stamp_size=96,
    src_pixscale_arcsec,   # Source pixel scale (e.g., 0.168 for HSC)
    target_pixscale_arcsec=0.262,
):
    # 1. Read FITS files
    # 2. Background subtract
    # 3. Resample to target pixel scale
    # 4. Pad/crop to stamp_size
    # 5. Save with metadata
```

---

## Data Requirements

| Survey | Pixel Scale | Depth | Area | Status |
|--------|-------------|-------|------|--------|
| HSC Ultra-Deep | 0.168"/pix | ~27 mag | ~10 sq deg | Public, accessible |
| Legacy DR10 Deep | 0.262"/pix | ~25 mag | ? | Need to identify regions |
| DES Deep Fields | 0.263"/pix | ~26 mag | ~27 sq deg | Public |

**BLOCKER:** Need to acquire and prepare deep cutouts before Gen6 can proceed.

---

## Cost-Effort Analysis

| Factor | Assessment |
|--------|------------|
| **Data requirement** | HIGH - must acquire deep FITS |
| **Engineering effort** | Medium (bank builder exists) |
| **Compute cost** | Same as baseline |
| **Publication value** | High if sim-to-real jump demonstrated |

---

## Challenges

1. **Data acquisition:** Deep field cutouts not readily available
   - **Mitigation:** Start with HSC public data, ~10 sq deg

2. **Limited area:** Deep fields are small, limiting morphology diversity
   - **Mitigation:** Use multiple surveys to increase diversity

3. **Survey-specific artifacts:** Deep cutouts may contain survey-specific patterns
   - **Mitigation:** Quality filtering, visual inspection

4. **Pixel scale mismatch:** Most deep surveys are ~0.17"/pix, target is 0.262"/pix
   - **Mitigation:** Resample during bank building (code exists)

5. **Selection effects:** Deep field galaxies may differ from DR10 foreground
   - **Mitigation:** Match size/brightness distribution to expected sources

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Deep data unavailable | Medium | Critical | Skip Gen6 entirely |
| Data too noisy | Low | Medium | Quality filtering |
| Artifacts leak in | Medium | Medium | Visual QA on bank |
| No improvement over Gen5 | Medium | High | Still publishable as negative result |

---

## When Gen6 Becomes Worth It

Gen6 is **high-upside, high-cost**. Pursue only if:

1. Deep data is readily accessible (public HSC/DES)
2. Small pilot bank shows clear sim-to-real improvement
3. Time allows (adds 2-6+ weeks)

**Recommendation:** Treat as optional "bonus arm" after Gen5-Prime + Gen7 + Gen8 are complete.

---

## Success Criteria

- Gen6 outperforms Gen5-COSMOS on real anchors
- Improvement persists across θ_E and PSF strata
- Bank contains >10k diverse templates

---

## Timeline (if pursued)

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Acquire deep data | 1-2 weeks | FITS files |
| Build source bank | 2-3 days | NPZ bank |
| Quality validation | 1-2 days | Verified clean bank |
| Training + evaluation | 3-4 days | Results |

**Total: ~2-4 weeks (if data available)**

---

## Contingency

If Gen6 cannot be pursued:
- Focus on Gen7 + Gen8 ablations
- Acknowledge "domain-matched sources" as future work in paper
- The ablation story remains complete without Gen6
