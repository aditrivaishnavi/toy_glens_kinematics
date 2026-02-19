# PROMPT 21 — Paper v11 Final Scientific Review

**Date**: 2026-02-16
**Context**: This is the final review request before submission to MNRAS. Three independent LLM reviewers (Prompt 20) confirmed the D06 results are scientifically sound. Paper v11 incorporates all D06 numbers and the revised Poisson narrative. Your task is a referee-level review.

---

## Background

### What changed from v10 to v11

Paper v10 had an internal inconsistency: Section 3.2 described corrected injection priors (K-corrected colours, narrowed beta_frac, Re, n, disabled clumps), but all result numbers in Section 4 came from old, less realistic priors. D06 fixed this by re-running ALL 10 experiments with corrected priors.

### D06 key results (corrected priors)

**Grid completeness (110,000 injections, p>0.3):**
- No Poisson: 5.18% (5697/110,000) [was 3.41% in D05]
- Poisson: 3.79% (4174/110,000) [was 2.37%]
- Deficit: -1.38 pp [was -1.04 pp]

**Bright-arc detection rates (N=200/bin, θ_E=1.5, β_frac [0.10,0.40]):**

| Mag | Baseline | Poisson | clip=20 | P+clip20 | Unrestr | g=1e12 |
|-----|----------|---------|---------|----------|---------|--------|
| 18-19 | 29.0% | 29.0% | 45.0% | 42.5% | 20.5% | 29.0% |
| 19-20 | 33.5% | 33.5% | 46.5% | 42.5% | 14.5% | 34.0% |
| 20-21 | 38.5% | **41.5%** | 49.5% | 46.5% | 15.0% | 38.5% |
| 21-22 | 33.0% | **43.5%** | 37.5% | 32.5% | 12.0% | 33.0% |
| 22-23 | 23.5% | **32.5%** | 21.0% | 17.5% | 6.5% | 23.5% |
| 23-24 | 11.0% | **12.5%** | 4.0% | 3.5% | 2.5% | 11.0% |
| 24-25 | 2.0% | 1.5% | 1.0% | 1.0% | 0.5% | 2.0% |
| 25-26 | 0.5% | 0.5% | 0.0% | 0.0% | 0.0% | 0.5% |

**Critical finding**: Poisson noise INCREASES detection at mag 20-23 (bold), contradicting v10's "degrades at every bin".

**Poisson delta by θ_E (grid, p>0.3):**

| θ_E | No-Poisson | Poisson | Δ (pp) |
|-----|-----------|---------|--------|
| 0.50 | 0.22% | 0.47% | **+0.25** |
| 0.75 | 0.33% | 0.80% | **+0.47** |
| 1.00 | 1.53% | 2.34% | **+0.81** |
| 1.25 | 3.39% | 3.79% | **+0.40** |
| 1.50 | 5.59% | 4.60% | -0.99 |
| 1.75 | 6.52% | 4.91% | -1.61 |
| 2.00 | 7.99% | 5.66% | -2.33 |
| 2.25 | 8.02% | 5.22% | -2.80 |
| 2.50 | 8.33% | 5.08% | -3.25 |
| 2.75 | 7.90% | 4.73% | -3.17 |
| 3.00 | 7.15% | 4.14% | -3.01 |

Crossover between θ_E=1.25 (Poisson helps) and θ_E=1.50 (Poisson hurts).

**Linear probe:**
- AUC: 0.997 ± 0.003 [was 0.996 ± 0.004]
- Median score: real Tier-A = 0.995, inj low-bf = 0.191 [was 0.110]
- Fréchet distance per layer: 0.14 → 1.45 → 10.9 → 47.2 (330× increase)
- Penultimate FD: real vs low-bf = 215.0, real vs high-bf = 219.1

**Tier-A recall**: 89.3% (100/112) — unchanged.

### Paired Poisson delta analysis (NEW diagnostic)

Per-injection paired analysis of baseline vs Poisson bright-arc parquets (same seed, same hosts):

| Mag | N | Mean Δp | Med Δp | %pos | Gained | Lost | Net | Mechanism |
|-----|---|---------|--------|------|--------|------|-----|-----------|
| 18-19 | 200 | +0.004 | -0.004 | 42% | 7 | 7 | 0 | threshold_scatter |
| 19-20 | 200 | +0.008 | -0.007 | 36% | 11 | 11 | 0 | threshold_scatter |
| 20-21 | 200 | +0.038 | -0.000 | 46% | 25 | 19 | +6 | mixed |
| 21-22 | 200 | **+0.089** | +0.006 | 57% | **27** | **6** | **+21** | **systematic_uplift** |
| 22-23 | 200 | **+0.050** | -0.000 | 50% | **21** | **3** | **+18** | **systematic_uplift** |
| 23-24 | 200 | +0.011 | +0.000 | 57% | 8 | 5 | +3 | mixed |
| 24-25 | 200 | -0.002 | -0.000 | 49% | 0 | 1 | -1 | mixed |
| 25-26 | 200 | -0.002 | +0.000 | 59% | 1 | 1 | 0 | threshold_scatter |

Key: at mag 21-22, mean delta = +0.089 with 27 injections crossing 0.3 upward and only 6 downward. This is systematic uplift, not scatter. The median score nearly doubles (0.085 → 0.161).

### D05 prior discrepancy (VERIFIED)

The D06 run script header originally claimed incorrect "old" prior values. Verified from git history:

| Parameter | Actual D05 default | D06 value |
|-----------|-------------------|-----------|
| beta_frac_range | (0.1, 1.0) | (0.10, 0.40) |
| re_arcsec_range | (0.05, 0.50) | (0.15, 0.50) |
| n_range | (0.5, 4.0) | (0.5, 2.0) |
| g-r colour | N(0.2, 0.25) | N(1.15, 0.30) |
| r-z colour | N(0.1, 0.25) | N(0.85, 0.20) |
| clumps_prob | 0.6 | 0.0 |

D05 bright-arc used explicit `--beta-frac-range 0.1 0.55` (not the engine default).

### Code provenance

SHA256 checksums recorded for all executed code on lambda3:
- injection_engine.py: b94bbcd...
- selection_function_grid.py: c561dd7...
- bright_arc_injection_test.py: d7cf763...
- best.pt checkpoint: bb4c0d5...
- Full provenance in `provenance.json` (included in zip).

---

## Review Questions

### Q1: Number consistency (YES/NO + specifics)

Read paper v11 (`mnras_merged_draft_v11.tex`) end-to-end. Does every numerical claim match the D06 results provided above? Flag ANY mismatches. Check:
- Abstract (5.18%, 3.79%, 29-39%, AUC 0.997)
- Table 3 (thresholds)
- Table 4 (θ_E completeness)
- Table 5 (probe metrics, Fréchet distances)
- Bright-arc table (all 48 cells)
- Conclusions (all 5 items)
- Figure captions (peak θ_E, magnitude bins)
- Any remaining D05 values not updated

### Q2: Poisson narrative coherence (YES/NO + assessment)

Is the new "dual mechanism" Poisson narrative internally consistent throughout the paper? Check:
- Abstract says "regime-dependent" ✓/✗
- Section 4.4.1 now acknowledges limitation of first-principles prediction ✓/✗
- Section 4.4.2 reports regime-dependent bright-arc result ✓/✗
- Section 4.4.4 reports θ_E crossover ✓/✗
- Section 4.4.5 reports "destructive interference" not "amplified damage" ✓/✗
- Section 4.4.6 describes dual mechanism ✓/✗
- Conclusions item (iv) matches ✓/✗
- No sentence claims "degrades at every bin" or "rules out shot noise" ✓/✗

### Q3: Scientific defensibility

Would a referee accept the dual-mechanism explanation? Specifically:
- Is the "Poisson adds realistic texture" claim adequately supported by the paired delta analysis?
- Is the crossover at θ_E ≈ 1.3 well-explained?
- Is the Fréchet distance progression correctly interpreted (mid-level features, not pixel stats)?
- Are there any overclaims or underclaims?

### Q4: Missing content

Is there anything the paper should add given the D06 results? For example:
- Should the θ_E-stratified Poisson table be a formal numbered table?
- Should the paired delta analysis results be in a table or supplementary material?
- Should score histograms be included as a figure?
- Are any of the new findings (Fréchet progression, geometry dominance from unrestricted β_frac) underreported?

### Q5: Remaining weaknesses

What are the top 2-3 weaknesses a referee would target? How would you address them in the text? Be specific about what changes to make and where.

### Q6: Submission readiness

Is paper v11 ready for MNRAS submission? YES/NO. If NO, list the specific changes needed (ranked by priority).

---

## Files in zip

- `paper/mnras_merged_draft_v11.tex` — the paper
- `scripts/analyze_d06_results.py` — D06 analysis script
- `scripts/analyze_poisson_diagnostics.py` — Poisson diagnostics script
- `scripts/run_d06_corrected_priors.sh` — D06 driver script (corrected header)
- `scripts/store_d06_provenance.sh` — provenance script
- `dhs/injection_engine.py` — injection engine with corrected priors + RNG fix
- `configs/injection_priors.yaml` — corrected prior config
- `results/D06_20260216_corrected_priors/analysis/d06_analysis_summary.json`
- `results/D06_20260216_corrected_priors/poisson_diagnostics/d06_poisson_diagnostics.json`
- `results/D06_20260216_corrected_priors/provenance.json`
