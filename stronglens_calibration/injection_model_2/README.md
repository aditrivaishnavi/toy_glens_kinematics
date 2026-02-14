# Injection Model 2: Deflector-Conditioned Injection

## Overview

Model 2 addresses the fundamental limitation of Model 1 (parametric Sersic on random hosts):
the measured completeness is not representative of the CNN's selection function on its
intended target population because the injection is not conditioned on the deflector-host context.

**Key change:** Instead of injecting arcs onto random negative hosts with independent lens
parameters, Model 2:

1. **Selects LRG-like hosts** (DEV/SER morphology from Tractor) that resemble real
   strong-lens deflectors
2. **Conditions lens q and PA on the host light** by measuring second moments of the
   r-band cutout
3. Uses our existing validated SIE+shear injection engine (`dhs.injection_engine`)

## Provenance

- `host_matching.py` — Adapted from LLM-suggested starter code, thoroughly reviewed and
  hardened with:
  - Input validation and shape checks
  - Robust NaN/Inf/all-zero handling
  - Negative eigenvalue guards
  - Configurable q_min and q_scatter
  - 26 passing unit tests
- The LLM's suggested `injection_engine.py` was **rejected** due to 3 confirmed mathematical
  errors in the SIE deflection formulation:
  1. Incorrect denominator (`psi + q^2` instead of `psi`)
  2. Swapped atan/atanh for x/y components
  3. Wrong prefactor (`q` instead of `sqrt(q)`)
- Our existing engine (`dhs.injection_engine`) is validated against lenstronomy with 28
  passing tests and is used unchanged.

## Directory Structure

```
injection_model_2/
  README.md                    # This file
  host_matching.py             # Host moment estimation & lens param conditioning
  host_selection.py            # LRG-like host selection from manifest
  scripts/
    selection_function_grid_v2.py   # Fixed grid runner with Model 2 support
    host_conditioning_diagnostic.py # LRG vs random host comparison experiment
  tests/
    test_host_matching.py      # 26 unit tests for host moment estimation
  validation/
    (results from Model 2 runs will go here)
```

## Usage

```bash
# Run tests
cd stronglens_calibration/injection_model_2
python -m unittest discover -s tests -v

# Run Model 2 selection function grid (on Lambda)
cd /lambda/nfs/.../code
PYTHONPATH=. python stronglens_calibration/injection_model_2/scripts/selection_function_grid_v2.py \
    --checkpoint checkpoints/paperIV_efficientnet_v2_s_v4_finetune/best.pt \
    --manifest manifests/training_parity_70_30_v1.parquet \
    --host-split val \
    --model 2 \
    --out-dir results/selection_function_model2

# Run LRG vs random host diagnostic
PYTHONPATH=. python stronglens_calibration/injection_model_2/scripts/host_conditioning_diagnostic.py \
    --checkpoint checkpoints/paperIV_efficientnet_v2_s_v4_finetune/best.pt \
    --manifest manifests/training_parity_70_30_v1.parquet \
    --host-split val \
    --out-dir results/host_conditioning_diagnostic
```

## Paper Framing

- **Model 1** is reframed as an ablation demonstrating that naive injection fails
  and quantifying the host-context effect
- **Model 2** is the primary selection function result
- If Model 2 closes the injection-real gap, the paper story is:
  "Host context dominates CNN selection functions; deflector-conditioned injection is required"
- If it does NOT close the gap, we need to investigate preprocessing/appearance statistics
  before Model 3

## References

- Kormann et al. (1994): SIE mass model
- Koopmans et al. (2006): mass-light alignment in SLACS lenses
- Holder & Schechter (2003): external shear distributions
- MNRAS_RAW_NOTES.md Section 9.3: Model 2 specification
