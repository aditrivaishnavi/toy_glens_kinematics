# PROMPT 18 (v2): Full Code Review — Corrected Injection Priors & D06 Rerun

**Date**: 2026-02-16
**Revision**: v2 — incorporates fixes from LLM1's first-pass review (see Appendix A)
**Context**: Pre-submission code review for MNRAS paper "The morphological barrier: quantifying the injection realism gap for CNN strong lens finders in DESI Legacy Survey DR10"

---

## CRITICAL ISSUE DISCOVERED

We discovered an **internal inconsistency** in the paper that no previous LLM review caught:

**The paper (v10) Section 3.2 correctly describes the fixed injection priors (K-corrected colours, narrowed beta_frac/n/Re, disabled clumps, background-texture noise). But ALL result numbers in Section 4 were generated with the OLD buggy priors.**

Specifically, these paper claims all used old/buggy priors:
- Section 4.2: "marginal completeness 3.41% (3755/110,000)" — **old priors**
- Table 2: No-Poisson 3.41% vs Poisson 2.37% at p>0.3 — **old priors**
- Table 3: Completeness by theta_E — **old priors**
- Section 4.3: Linear probe AUC 0.996 with beta_frac (0.1, 0.3) injections — **old priors**
- Table 4: Bright-arc detection rates, 6 conditions — **old priors**
- Section 4.4: Gain sweep at 1e12 — **old priors**
- All numbers in Abstract and Conclusions — **old priors**

The only results NOT affected are:
- Section 4.1: Tier-A recall (89.3%) — uses real lenses, no injection
- Table 1: Dataset composition — no injection

### Why LLM reviews missed this

Previous LLM reviews (Prompts 1-17) focused on:
- Code correctness within individual scripts
- Paper writing quality and scientific framing
- Experimental design (Poisson noise, clip range, gain sweep)

No review compared the paper's Section 3.2 prior descriptions against the actual default values used when the D01-D05 experiments were executed. The prior corrections in `injection_engine.py` happened *after* all D01-D05 results were generated. The paper Section 3.2 was then updated to match the corrected code, but the results were never regenerated.

---

## THE 6 PRIOR CORRECTIONS

All changes are in `dhs/injection_engine.py::sample_source_params()`:

| Parameter | OLD (D01-D05) | NEW (D06) | Rationale |
|-----------|---------------|-----------|-----------|
| `g_minus_r_mu_sigma` | N(0.8, 0.3) | N(1.15, 0.30) | K-correction: sources at z~1-3 have red observer-frame g-r. Measured arc annulus of 388 Tier-A: mean=1.23, std=0.39 |
| `r_minus_z_mu_sigma` | N(0.4, 0.2) | N(0.85, 0.20) | Same K-correction issue. Measured: mean=0.95, std=0.27 |
| `beta_frac_range` | (0.05, 0.35) | (0.10, 0.40) | Old range biased toward near-zero offsets producing compact images. Narrowed for realistic arc morphologies |
| `re_arcsec_range` | (0.1, 1.5) | (0.15, 0.50) | Sources < 0.15" unresolvable at ~1" seeing (produce point-like binary star artefacts). Max 0.50 avoids unrealistically extended sources |
| `n_range` | (0.5, 4.0) | (0.5, 2.0) | High Sérsic n > 2.5 produces concentrated cores that look point-like when lensed |
| `clumps_prob` | 0.6 | 0.0 | Clumps created unrealistic multi-blob morphologies not seen in real lensed sources. Disabled entirely |

Additionally, `add_sky_noise` was added to `inject_sis_shear()`:
- **What it does**: Adds per-band Gaussian noise (sigma from host's outer-annulus MAD) to the arc stamp before compositing onto the host.
- **What it is NOT**: It is NOT modelling additional sky photons. The host cutout already contains real sky/read noise.
- **Why**: Empirical texture matching — without it, bright injected arcs appear anomalously smooth against the noisy host. The physically correct arc-flux noise is the Poisson shot noise (`add_poisson_noise`); this Gaussian layer is an additional texture-matching step. The paper describes this as "background-texture matching noise".

---

## FIXES APPLIED FROM LLM1 FIRST-PASS REVIEW

Three blocking issues were identified by the first-pass reviewer. All three have been fixed in the code included in this package:

### Fix 1: Per-injection RNG for Poisson and background-texture noise (CRITICAL)

**Problem**: `torch.poisson()` and `torch.randn_like()` in `inject_sis_shear()` used the GLOBAL torch RNG. This meant:
- Results were non-reproducible across re-runs
- Paired comparisons (same seed, different noise condition) drew DIFFERENT noise realisations, invalidating "only Poisson differs" claims

**Fix**: Both noise operations now use the existing per-injection `rng = torch.Generator(device).manual_seed(seed)` that was already created for clump sampling:
- `torch.poisson(arc_electrons, generator=rng)` (line ~575)
- `torch.randn(shape, generator=rng, device=device, dtype=dtype)` (line ~601)

This ensures: (a) full reproducibility given the same seed, (b) paired conditions sharing the same seed get identical background-texture noise, so only the Poisson term differs, and (c) the gain=1e12 control will converge to the no-Poisson baseline injection-by-injection (not just in distribution).

### Fix 2: Stale D06 header comments

**Problem**: Header comments in `run_d06_corrected_priors.sh` listed wrong values for 2 of 6 priors:
- `re_arcsec_range: (0.3, 0.8)` → should be `(0.15, 0.50)`
- `n_range: (0.5, 2.5)` → should be `(0.5, 2.0)`

**Fix**: Header comments now match `configs/injection_priors.yaml` exactly.

### Fix 3: "Sky noise" naming clarification

**Problem**: Docstring and comments called it "sky noise", which is physically misleading since the host already contains sky noise. Adding Gaussian noise to the arc effectively double-counts background variance.

**Fix**: Docstring now explicitly states this is "background-texture matching noise", not physical sky modelling. Code comments explain the rationale and distinguish it from the Poisson shot noise.

### LLM1 error corrected

The first-pass reviewer claimed D06 was missing the "Poisson + clip=20" condition (Q3.1 point 1). This was **incorrect** — experiment [4] in the D06 driver (`ba_poisson_clip20`, lines 140-152) is exactly that condition. The reviewer listed only 5 of the 6 bright-arc experiments and missed it.

---

## CODE CHANGES FOR D06

### 1. `selection_function_grid.py` — Grid completeness (110k injections)

**Changes made:**
- Added `--add-sky-noise` CLI flag → passed to `inject_sis_shear(add_sky_noise=...)`
- Added `--save-cutouts` and `--save-cutouts-dir` CLI flags → saves each injection as `.npz` with a `cutout_metadata.parquet` index
- Updated metadata `source_model` from `"Sersic + optional clumps"` to `"Sersic (clumps disabled, clumps_prob=0.0)"`
- New function parameters: `add_sky_noise`, `save_cutouts`, `save_cutouts_dir`

**Cutout saving logic:**
- Each injection saved as `cell{XXXXX}_inj{YYYYY}.npz` containing `injected` (3,H,W) and `injection_only` (3,H,W) arrays
- Metadata parquet includes: `inj_id`, `cell_idx`, `theta_e`, `psf_bin`, `depth_bin`, `src_r_mag`, `lensed_r_mag`, `arc_snr`, `inj_seed`, `cutout_path`
- Estimated disk: ~11 GB per condition (110k × ~100 KB compressed)

### 2. `feature_space_analysis.py` — Linear probe AUC

**Changes made:**
- Added `--add-sky-noise`, `--add-poisson-noise`, `--gain-e-per-nmgy` CLI flags
- Function `extract_embeddings_from_injections()` now accepts and passes through `add_sky_noise`, `add_poisson_noise`, `gain_e_per_nmgy` to `inject_sis_shear()`
- Saved metadata now includes these flags

**Note on hardcoded beta_frac ranges:**
- Low beta_frac: (0.1, 0.3) — within the new engine default (0.10, 0.40). This is fine.
- High beta_frac: (0.7, 1.0) — intentionally far outside the realistic range, used as a control to test whether the CNN distinguishes arc morphology from non-arc morphology.

### 3. `bright_arc_injection_test.py` — Already refactored

This script was already refactored to 2-phase (generate/score) with `--add-sky-noise` in a previous session. No changes needed for D06.

### 4. `run_d06_corrected_priors.sh` — New driver script

Replaces `run_d05_full_reeval.sh`. Key differences:
- `--add-sky-noise` on ALL injection experiments (experiments 1-9)
- `--save-cutouts` on grid experiments (experiments 7-8)
- No explicit `--beta-frac-range` for "restricted" bright-arc experiments → engine default (0.10, 0.40) is used instead of old (0.1, 0.55)
- Bright-arc uses `--phase both` (2-phase: generate then score)
- Tier-A scoring (experiment 10) is unchanged (no injection involved)

### 5. `dhs/injection_engine.py` — Core engine fixes (this revision)

- Poisson noise: `torch.poisson(arc_electrons, generator=rng)` — uses per-injection seeded generator
- Background-texture noise: `torch.randn(shape, generator=rng, ...)` — uses same generator
- Docstring updated: `add_sky_noise` parameter description now explains it is background-texture matching, not physical sky noise modelling

---

## FILES IN THIS PACKAGE

| File | Description |
|------|-------------|
| `dhs/injection_engine.py` | Core injection engine with corrected defaults + RNG fix |
| `configs/injection_priors.yaml` | Prior registry synced to engine defaults |
| `scripts/selection_function_grid.py` | Grid script with add_sky_noise + save_cutouts |
| `scripts/feature_space_analysis.py` | Linear probe with add_sky_noise |
| `sim_to_real_validations/bright_arc_injection_test.py` | 2-phase bright-arc test |
| `scripts/run_d06_corrected_priors.sh` | D06 driver (all 10 experiments, header corrected) |
| `paper/mnras_merged_draft_v10.tex` | Paper v10 for context on claims |
| `tests/test_injection_priors.py` | Unit test: engine defaults match YAML registry |

---

## SPECIFIC REVIEW QUESTIONS

Please answer each question with YES/NO first, then explain.

### Q1: Code Correctness

**Q1.1**: In `injection_engine.py`, is the RNG fix correct? Specifically:
- Does `torch.poisson(arc_electrons, generator=rng)` produce deterministic Poisson draws given the same seed?
- Does `torch.randn(shape, generator=rng, device=device, dtype=dtype)` produce deterministic Gaussian draws?
- Is there any code path between `rng.manual_seed(seed)` (line ~482) and the noise section (line ~575+) that consumes random numbers from `rng` non-deterministically? (Note: `_add_clumps` uses `rng` but with `clumps_prob=0.0` it draws zero random numbers.)
- If `clumps_prob > 0` is re-enabled in the future, would the clump draws shift the Poisson/texture noise draws? Is this acceptable?

**Q1.2**: In `selection_function_grid.py`, is the `add_sky_noise` parameter correctly threaded from CLI → `run_selection_function()` → `inject_sis_shear()` call?

**Q1.3**: In `selection_function_grid.py`, is the cutout saving logic correct? Specifically:
- Does `injection_chw.numpy()` give the right tensor (injection-only, not the full injected image)?
- Is the `inj_id` naming scheme deterministic and unique?
- Is `inj_seed` recorded correctly for reproducibility?

**Q1.4**: In `feature_space_analysis.py`, are `add_sky_noise`, `add_poisson_noise`, and `gain_e_per_nmgy` all correctly passed from CLI → `extract_embeddings_from_injections()` → `inject_sis_shear()`?

### Q2: Experimental Design Validity

**Q2.1**: With the RNG fix, paired conditions now share identical background-texture noise realisations (same seed → same `rng` state). Does this fully resolve the paired-comparison concern? Are there any remaining sources of non-determinism?

**Q2.2**: The gain sweep sanity test (gain=1e12) should now converge to the "no Poisson" baseline **injection-by-injection** (not just in distribution), since both share the same seeded background-texture noise. Confirm this is correct.

**Q2.3**: The bright-arc "restricted" beta_frac is now (0.10, 0.40) instead of (0.1, 0.55). This narrows the range. Is this valid for the paper's experimental design? The paper will be updated to reflect the new range.

### Q3: Consistency Checks

**Q3.1**: Does the D06 driver script (`run_d06_corrected_priors.sh`) correctly implement all 6 bright-arc conditions from Table 4? Verify experiments [1]-[6] match: (1) baseline, (2) Poisson, (3) clip=20, (4) Poisson+clip=20, (5) unrestricted bf, (6) gain=1e12.

**Q3.2**: After D06 runs, which specific paper sections/tables need to be updated with new numbers? List each one.

**Q3.3**: Are there any implicit assumptions in the code that still reference old prior values? For example, hardcoded ranges, comments mentioning old defaults, or validation checks that would fail with the new priors?

### Q4: Anything Else

**Q4.1**: Are there any remaining bugs, edge cases, or failure modes that could produce incorrect results?

**Q4.2**: The background-texture noise is now described as empirical texture matching rather than physical sky modelling. Is this framing scientifically defensible for a referee? Should the paper quantify the effect (e.g., "adds ~X% to the per-pixel noise variance in the arc region")?

**Q4.3**: Previous LLM reviews (Prompts 1-17) missed the paper-code prior inconsistency. Looking at the code now, are there any OTHER inconsistencies between what the paper claims and what the code actually does?

---

## INSTRUCTION TO REVIEWER

Please be direct and factual. Do NOT give a rosy assessment. If you find issues, state them plainly with file name, line number, and what needs to change. If everything looks correct, say so clearly. We are about to run a 30+ hour experiment and need confidence that the code is right before committing to it.

---

## APPENDIX A: LLM1 First-Pass Review Summary

The first-pass review identified 3 blocking issues (all fixed above) and 1 error:

| Finding | Status | Action |
|---------|--------|--------|
| Poisson/sky noise uses global RNG, breaks paired comparisons | **FIXED** | Added `generator=rng` to both noise operations |
| D06 header comments list wrong re_arcsec_range and n_range | **FIXED** | Corrected to match injection_priors.yaml |
| "sky noise" naming misleading (double-counts background) | **FIXED** | Renamed to "background-texture matching noise" in docstring |
| "D06 missing Poisson+clip=20 condition" (Q3.1) | **REVIEWER ERROR** | Experiment [4] in D06 IS Poisson+clip=20 |
