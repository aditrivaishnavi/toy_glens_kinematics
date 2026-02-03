# Data Variants Documentation

This directory documents all data variants used in Dark Halo Scope experiments.

## What is a Data Variant?

A data variant is a specific configuration of the data generation pipeline (Phases 2-4) that produces training/test data. Each variant is defined by:

1. **Parent Sample**: Which galaxies are selected (Phase 2/3)
2. **Injection Grid**: What lens parameters are simulated (Phase 4a)
3. **Injection Method**: How lenses are injected (Phase 4c) including:
   - PSF model (Gaussian vs Moffat)
   - Source morphology (Parametric vs COSMOS)
   - Control type (Paired vs Unpaired)

## Variant Summary Table

| Variant | Status | Gen | PSF | Source | Controls | Key Issue |
|---------|--------|-----|-----|--------|----------|-----------|
| [v3_color_relaxed](v3_color_relaxed.md) | Deprecated | 1 | Gaussian | Sersic | **Paired** | Shortcut learning |
| [v4_sota](v4_sota.md) | Active | 2 | Gaussian | Sersic | Unpaired | Gaussian PSF |
| [v4_sota_moffat](v4_sota_moffat.md) | Active | 3,4 | Moffat | Sersic | Unpaired | Smooth sources |
| [v5_cosmos_source](v5_cosmos_source.md) | Planned | 5 | Moffat | COSMOS | Unpaired | - |

## Version Lineage

```
v3_color_relaxed (Gen1)
    │
    ├── Fixed paired controls
    │
    ▼
v4_sota (Gen2)
    │
    ├── Added Moffat PSF
    │
    ▼
v4_sota_moffat (Gen3, Gen4)
    │
    ├── Adding real source morphology
    │
    ▼
v5_cosmos_source (Gen5) [PLANNED]
```

## Creating a New Variant

When creating a new data variant:

1. Create a markdown file: `v{N}_{name}.md`
2. Document ALL parameters that differ from parent variant
3. Include the exact Phase 4c command used
4. Record S3 locations for reproducibility
5. Update this README with the new variant

## Reproducibility Requirements

Every data variant MUST document:
- [ ] Parent sample selection criteria
- [ ] Complete injection grid parameters
- [ ] Phase 4c command with all flags
- [ ] S3 locations for manifests and stamps
- [ ] Any filtering applied (resolvability, arc_snr, etc.)

