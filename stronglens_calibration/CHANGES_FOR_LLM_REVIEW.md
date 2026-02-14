# Changes Made by Cursor LLM (2026-02-13) — For Independent Review

This document lists every file modified or created during the 2026-02-13 session,
what was changed, and why. The changes address 6 findings from an earlier LLM
review of this codebase. **None of these changes have been validated by the
project owner yet. An independent LLM review is requested before any training
runs.**

---

## Summary of Findings Being Addressed

| # | Finding | Severity |
|---|---------|----------|
| 1 | Normalization annulus (20, 32) hardcoded — wrong for 101x101 stamps | HIGH |
| 2 | Injection priors in code differ from paper description | MEDIUM |
| 3 | Model 2 q_lens conditioning described wrong in docs | LOW (docs only) |
| 4 | Beta_frac geometry may explain ~30% bright-arc ceiling | MEDIUM |
| 5 | Missing Poisson noise on arc photons | LOW |
| 6 | Gaussian PSF vs real PSF | LOW |

Findings 5 and 6 were NOT addressed (deferred — low priority).

---

## Files Modified

### 1. `dhs/utils.py` — Annulus revert + warning (Finding #1)

**What changed:**
- Added `import logging` and a `logger` at module level
- Added a 15-line KNOWN ISSUE comment block at the top of the file explaining
  the annulus bug, why we can't change the default, and the plan
- `normalize_outer_annulus()`: REVERTED back to hardcoded defaults `r_in=20,
  r_out=32` (matching all trained models). Previously this session had changed
  it to use `default_annulus_radii()` which would have broken all existing models.
- Added a `logger.warning()` inside `normalize_outer_annulus()` that fires when
  `r_out / half < 0.70` and image is > 64px. This warns at runtime that the
  annulus is suboptimal but does NOT change behavior.
- `default_annulus_radii()` function: KEPT (it was added earlier). Computes
  `r_in = 0.65 * R, r_out = 0.90 * R` where `R = min(H,W)//2`. Available for
  future retraining but NOT wired as default.

**Why:** The annulus (20, 32) sits at 40-63% of half-width on 101x101 stamps,
overlapping galaxy light. But all existing trained models expect this
normalization. Changing the default without retraining makes scores wrong.

**Risk:** The warning is `logger.warning()` which may be noisy in production
runs that process many images.

### 2. `dhs/preprocess.py` — Annulus kwargs passthrough (Finding #1)

**What changed:**
- `preprocess_stack()` signature: added two optional parameters
  `annulus_r_in: float | None = None` and `annulus_r_out: float | None = None`
- Inside the function: builds `annulus_kwargs` dict from these params and passes
  to `normalize_outer_annulus(**annulus_kwargs)`
- If both are None (default), behavior is IDENTICAL to before (uses hardcoded 20, 32)

**Why:** Allows new training configs to specify corrected annulus without changing
the default behavior for existing models.

**Risk:** If someone passes only `annulus_r_in` without `annulus_r_out` (or vice
versa), the function will pass one kwarg and let the other use the default. This
could create a mismatched annulus. No validation for this case exists.

### 3. `dhs/data.py` — DatasetConfig annulus fields (Finding #1)

**What changed:**
- `DatasetConfig` dataclass: added two fields `annulus_r_in: float = 0.0` and
  `annulus_r_out: float = 0.0`
- `LensDataset.__getitem__()`: if `annulus_r_in > 0`, adds it to `crop_kwargs`
  which gets passed to `preprocess_stack()`
- Same for `annulus_r_out`

**Why:** Allows YAML training configs to specify annulus radii, which flow
through DatasetConfig -> LensDataset -> preprocess_stack -> normalize_outer_annulus.

**Risk:** The field name `crop_kwargs` is now misleading since it also carries
annulus parameters. The `> 0` check means you cannot explicitly set annulus to 0
(but 0 would be meaningless anyway).

### 4. `tests/test_preprocess_regression.py` — Behavioral annulus tests (Finding #1)

**What changed:**
- Checksum test: restored to `EXPECTED_CHECKSUM = "7e25b9e366471bda"` (matching
  the reverted annulus defaults)
- `test_raw_robust_zero_centered_annulus`: now tests with explicit (20, 32) —
  the actual annulus used by the code
- Removed old `TestAnnulusGuards` and `TestNormalizationStability` classes
- Added new `TestAnnulusBehavioral` class with 4 tests:
  - `test_annulus_galaxy_flux_fraction_101x101`: asserts corrected annulus < 10%
    galaxy flux AND old annulus > 10% (documents the bug)
  - `test_corrected_annulus_improvement_over_old`: asserts >= 2x improvement
  - `test_preprocessing_outer_sky_near_zero`: prints sky medians (informational,
    no assertion — documents the KNOWN ISSUE)
  - `test_default_annulus_radii_is_in_outer_region`: tests the formula for
    multiple image sizes

**Why:** Old tests tested the helper function `default_annulus_radii()`, not the
actual preprocessing behavior. New tests document the known issue while passing.

**Risk:** `test_preprocessing_outer_sky_near_zero` has no assertion — it only
prints. It cannot catch regressions.

### 5. `tests/test_injection_priors.py` — NEW FILE (Finding #2)

**What changed:** Created from scratch. 17 tests that:
- Use Python AST parsing to read default parameter values from
  `dhs/injection_engine.py` WITHOUT importing torch
- Compare those defaults against `configs/injection_priors.yaml`
- Verify registry completeness (all expected keys present)

**Why:** Prevents code-to-paper drift. If someone changes a default in the code
without updating the YAML (or vice versa), a test fails.

**Risk:** AST parsing is fragile — it only handles simple constant defaults
(numbers, tuples, None). Complex defaults (e.g., function calls, class
instances) would be silently skipped via `_SENTINEL`. The `clumps_n_range` and
`clumps_frac_range` values in the YAML are NOT validated against the code because
they appear inside the function body, not as parameter defaults.

### 6. `configs/injection_priors.yaml` — NEW FILE (Finding #2)

**What changed:** Created from scratch. Documents the exact parameter values used
by `sample_source_params()` and `sample_lens_params()`, plus Model 2 conditioning
and preprocessing parameters.

**Why:** Single source of truth for paper writing. Validated by test_injection_priors.py.

**Risk:** The `clumps_n_range` and `clumps_frac_range` are documented here but
NOT validated by the test (see above). If someone changes those in the code, this
file could drift silently.

### 7. `scripts/beta_frac_ceiling_diagnostic.py` — NEW FILE (Finding #4)

**What changed:** Created from scratch. Two parts:
- Part 1 (math-only): computes CDF of beta_frac under area-weighted sampling.
  Confirms P(beta_frac < 0.55) = 29.5%, matching the observed ~30% ceiling.
- Part 2 (requires torch): runs injection experiments with varying beta_frac_max,
  measures detection rates via `arc_annulus_snr`.

**Why:** The LLM's original diagnostic scripts (in `review_deliverables/`) were
broken — wrong function names, wrong API signatures, wrong units. This is a
working replacement.

**Risk:** Part 2 uses a blank host (background noise only, no galaxy). This is
unrealistic — real hosts have bright central galaxies that affect arc detection.
Results from Part 2 should be treated as an upper bound on detection rates. Also,
`psfdepth_r = 100.0` is hardcoded as a proxy — real values vary per host.

### 8. `configs/paperIV_efficientnet_v2_s_v5_annulus_fix.yaml` — NEW FILE

**What changed:** Training config identical to `paperIV_efficientnet_v2_s.yaml`
except `annulus_r_in: 32.5` and `annulus_r_out: 45.0`, with a different output dir.

**Why:** Ready-to-use config for retraining with corrected annulus.

**Risk:** Has not been tested (no training run executed).

### 9. `configs/paperIV_bottlenecked_resnet_v2_annulus_fix.yaml` — NEW FILE

**What changed:** Same as above but for BottleneckedResNet architecture.

### 10. `docs/RETRAIN_PLAN_ANNULUS_FIX.md` — NEW FILE

**What changed:** Step-by-step retraining plan with launch commands, validation
criteria, risks, and definition of done.

### 11. `docs/MNRAS_RAW_NOTES.md` — Updated (Findings #1, #2, #3)

**What changed:**
- Section 4.2: Replaced hardcoded annulus description with corrected values +
  ERRATUM note explaining the history
- Added Section 9.3.1: Full injection prior table with exact code values, and
  note about Model 2 Gaussian additive conditioning (not multiplicative uniform)

### 12. `llm_review_package/LLM_PROMPT_COMPREHENSIVE_REVIEW_20260213.md` — Updated

**What changed:** Updated preprocessing description to include KNOWN ISSUE about
the annulus.

---

## Files NOT Modified (but relevant)

- `dhs/injection_engine.py` — No changes. Priors documented in YAML but code untouched.
- `dhs/train.py` — No changes. Config parsing at `dhs/scripts/run_experiment.py`
  uses `DatasetConfig(**cfg["dataset"])` which will pick up the new `annulus_r_in`
  and `annulus_r_out` fields from YAML automatically (dataclass unpacking).
- `injection_model_2/host_matching.py` — No changes. Model 2 conditioning
  documented correctly in YAML but code untouched.
- `scripts/selection_function_grid.py` — NOT updated to pass annulus kwargs.
  **This is a gap**: if you run the selection function with the new model, the
  scoring preprocessing must also use the new annulus. This script currently
  calls `preprocess_stack()` without annulus kwargs, so it would use (20, 32).

---

## Key Assertion: What Needs Retraining vs What Doesn't

**Claim: Pre-training data (cutouts, manifests, negative sampling) do NOT need to
be regenerated.**

Reasoning: The cutout files (.npz) store raw nanomaggy pixel values. The
normalization annulus is applied at preprocessing time (during training and
scoring), not during cutout generation. So the raw data is independent of the
annulus choice.

**Claim: All trained models (v1–v4) need retraining with the corrected annulus.**

Reasoning: The model learned feature representations that depend on the specific
normalization. With (20, 32) on 101x101, the galaxy center is normalized by
galaxy+sky statistics (inflated MAD). With (32.5, 45.0), it's normalized by
sky-dominated statistics (correct MAD). The same raw pixel values produce
different normalized inputs. A model trained on one normalization cannot be
reliably scored with a different normalization.

**Claim: The selection function grid must be rerun after retraining.**

Reasoning: The selection function measures completeness by injecting arcs into
hosts, preprocessing them with the model's normalization, and scoring with the
model. Both the normalization and the model weights change. Old selection function
results describe the old model's sensitivity under the old normalization — they
don't transfer.

---

## Test Results (2026-02-13)

All tests pass with the current code:

```
$ python -m unittest tests.test_preprocess_regression tests.test_injection_priors -v
...
Ran 42 tests in 0.151s
OK
```

Key behavioral test outputs:
- Current annulus (20,32) on 101x101: 19.9% galaxy flux (BAD)
- Corrected annulus (32.5,45.0) on 101x101: 6.4% galaxy flux (GOOD)
- Sky median (r>40) after preprocessing: -2.6 (KNOWN ISSUE: should be ~0)
- Checksum: 7e25b9e366471bda (locked to trained models)
