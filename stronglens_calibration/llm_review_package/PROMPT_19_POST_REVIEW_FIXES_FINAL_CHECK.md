# PROMPT 19: Post-Review Fixes — Final Check Before D06 Execution

**Date**: 2026-02-16
**Context**: Two independent LLM reviewers (LLM1 and LLM3) reviewed the D06 code package (Prompt 18). This prompt documents their findings, the fixes applied, and asks for a final confirmation before committing to a 30+ hour experiment run.

---

## WHAT PROMPT 18 ASKED

We discovered that the paper (v10) Section 3.2 describes corrected injection priors (K-corrected colours, narrowed beta_frac/n/Re, disabled clumps), but ALL result numbers in Section 4 were generated with the OLD buggy priors. D06 is a full rerun of all 10 experiments with corrected priors. We asked reviewers to check the code for correctness before running.

---

## REVIEWER FINDINGS AND FIXES APPLIED

### Finding 1: Poisson and sky noise not seeded per injection (LLM1 Q4.1, LLM3 Q4.1)

**Both reviewers independently flagged this.**

**Problem**: `torch.poisson()` and `torch.randn_like()` in `inject_sis_shear()` used the GLOBAL torch RNG. This meant:
- Results were non-reproducible across re-runs
- Paired comparisons (same seed, different noise condition) drew different noise realisations, invalidating "only Poisson differs" claims
- The gain=1e12 sanity check would not converge to baseline injection-by-injection

**Fix applied** (`dhs/injection_engine.py`, lines ~575 and ~601):
- `torch.poisson(arc_electrons, generator=rng)` — uses the per-injection generator already created at line ~482
- `torch.randn(shape, generator=rng, device=device, dtype=dtype)` — same generator
- Added inline comment block explaining the fix and its rationale

### Finding 2: `add_sky_noise` is physically wrong — double-counts background noise (LLM1 Q4.2, LLM3 Q2.1/Q4.2)

**Both reviewers independently flagged this.**

**Problem**: The paper (v10, lines 180-183) explicitly states: *"the injected images already contain the survey's background and host-galaxy noise and artefacts. The hypothesised missing texture is primarily the shot noise associated with the added arc flux itself."*

But `add_sky_noise` adds a fresh Gaussian noise field to the injection stamp before compositing onto the already-noisy host. This:
- Double-counts background variance (host already has sky noise)
- Contradicts the paper's own physical story
- Confounds the Poisson experiment (no longer testing only shot noise)
- Makes injected arcs noisier than real arcs

**Fix applied**: Removed `--add-sky-noise` from ALL experiments in `run_d06_corrected_priors.sh`. D06 now uses only `--add-poisson-noise` where applicable, which is the physically justified noise term. The `add_sky_noise` feature remains in the engine code for potential future experiments but is not used in D06.

### Finding 3: Stale D06 header comments (LLM1 Q3.3)

**Problem**: Header comments listed wrong values for 2 of 6 priors:
- `re_arcsec_range: (0.3, 0.8)` → actual: `(0.15, 0.50)`
- `n_range: (0.5, 2.5)` → actual: `(0.5, 2.0)`

**Fix applied**: Corrected to match `configs/injection_priors.yaml` exactly.

### Finding 4: `.numpy()` without `.cpu()` is GPU-unsafe (LLM3 Q1.2 Caveat B)

**Problem**: Three scripts call `.numpy()` directly on torch tensors. This works today because tensors are on CPU, but will break if anyone moves injection to GPU.

**Fix applied**: Changed to `.detach().cpu().numpy()` in:
- `scripts/selection_function_grid.py` (2 call sites)
- `scripts/feature_space_analysis.py` (1 call site)
- `sim_to_real_validations/bright_arc_injection_test.py` (2 call sites)

### Finding 5: LLM1 claimed D06 missing "Poisson+clip=20" condition (LLM1 Q3.1)

**This was a reviewer error.** Experiment [4] in D06 (`ba_poisson_clip20`, lines 140-152) IS the Poisson+clip=20 condition. The reviewer listed only 5 of 6 experiments and missed it. No fix needed.

### Finding 6: LLM3 claimed D06 header comments wrong (LLM3 Q3.3)

**Already fixed** in Finding 3 above. LLM3 reviewed the v1 package before the header was corrected.

---

## D06 EXPERIMENT DESIGN (FINAL)

All 10 experiments, **no `--add-sky-noise`**, only Poisson shot noise where specified:

| # | Experiment | Poisson | Clip | beta_frac | Notes |
|---|-----------|---------|------|-----------|-------|
| 1 | Bright-arc baseline | No | 10 | default (0.10-0.40) | Control |
| 2 | Bright-arc Poisson | Yes (gain=150) | 10 | default | Shot noise effect |
| 3 | Bright-arc clip=20 | No | 20 | default | Clip effect |
| 4 | Bright-arc Poisson+clip=20 | Yes | 20 | default | Interaction |
| 5 | Bright-arc unrestricted | No | 10 | (0.1-1.0) | Morphology control |
| 6 | Gain sweep | Yes (gain=1e12) | 10 | default | Sanity check |
| 7 | Grid baseline | No | 10 | default | 110k injections, save cutouts |
| 8 | Grid Poisson | Yes (gain=150) | 10 | default | 110k injections, save cutouts |
| 9 | Linear probe | No | 10 | (0.1-0.3) low / (0.7-1.0) high | AUC separation |
| 10 | Tier-A scoring | N/A | 10 | N/A | No injection, unchanged |

**Key properties:**
- All injection experiments use corrected engine defaults (6 prior corrections)
- Poisson noise is per-injection seeded via `generator=rng` — deterministic and paired
- Gain=1e12 should converge to baseline injection-by-injection (same seed → same geometry, Poisson ≈ 0)
- No `add_sky_noise` — consistent with paper's statement that host already contains survey noise

---

## FILES IN THIS PACKAGE

| File | Description | What changed since Prompt 18 |
|------|-------------|------------------------------|
| `dhs/injection_engine.py` | Core injection engine | RNG fix: `generator=rng` on Poisson and sky noise |
| `configs/injection_priors.yaml` | Prior registry | No change |
| `scripts/selection_function_grid.py` | Grid script | `.detach().cpu().numpy()` safety fix |
| `scripts/feature_space_analysis.py` | Linear probe | `.detach().cpu().numpy()` safety fix |
| `sim_to_real_validations/bright_arc_injection_test.py` | 2-phase bright-arc | `.detach().cpu().numpy()` safety fix |
| `scripts/run_d06_corrected_priors.sh` | D06 driver | Removed `--add-sky-noise`, fixed header comments |
| `paper/mnras_merged_draft_v10.tex` | Paper v10 | No change (for context) |
| `tests/test_injection_priors.py` | Unit test | No change |

---

## SPECIFIC REVIEW QUESTIONS

Please answer each with YES/NO first, then explain.

**Q1**: Is the RNG fix correct? Specifically: does `torch.poisson(arc_electrons, generator=rng)` and `torch.randn(shape, generator=rng, ...)` produce deterministic draws given the same `seed`? Is there any code path between `rng.manual_seed(seed)` (line ~482) and the noise section (line ~575+) that could consume random numbers non-deterministically?

**Q2**: With `add_sky_noise` removed, is the D06 experiment design now consistent with the paper's physical story (paper v10 lines 180-183)? The paper says the missing ingredient is shot noise from arc flux. D06 tests exactly this with `--add-poisson-noise`.

**Q3**: Does the gain=1e12 sanity check now work correctly? With per-injection seeded RNG and no sky noise, gain=1e12 should make Poisson noise vanish, and the injection should be identical to the no-Poisson baseline for the same seed. Confirm.

**Q4**: Look at the D06 experiment table above. Are all 6 bright-arc conditions (matching paper Table 4) present? Are the grid baseline/Poisson pair present? Is the linear probe present?

**Q5**: Are there any remaining bugs, edge cases, or paper-code inconsistencies that could produce incorrect results or mislead a referee?

---

## INSTRUCTION TO REVIEWER

This is the final check before a 30+ hour experiment run. Be direct. If you see any remaining issue, state it with file name, line number, and what to fix. If everything is correct, say "CLEAR TO RUN" explicitly.
