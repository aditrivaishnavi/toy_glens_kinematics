# PROMPT 24 — Paper v14: Figure 5 Redesign and Final Submission Review

## Background

Paper v13 was reviewed by three independent LLMs (Prompt 23). All three gave a **conditional YES** for submission readiness, blocked on a single issue: the Figure 5 caption made score-contrast claims ("p > 0.99 for real, p < 0.01 for injections") that were contradicted by the audit data. Specifically:

- At compact theta_E = 0.75, the separation held (real ~0.98-1.00, inj ~0.002-0.02).
- At medium/extended theta_E >= 1.50, injections scored 0.91-0.99 and some real lenses scored < 0.5.

The root cause was a design flaw: the v13 figure **paired** real lenses with brightness-matched injections and **cherry-picked** the highest-scored injections, creating a misleading forced comparison. The injections are independent parametric simulations with no derivation from the specific real lenses they were paired with.

## What changed from v13 to v14

### The only change: Figure 5 completely redesigned

**Old design (v13):** Paired brightness-matched real vs injection cutouts at three theta_E, 4 pairs each. Injections selected by highest score (cherry-picked). Caption overclaimed score contrast.

**New design (v14):** Two **independent** gallery panels with no pairing or brightness matching:

- **Panel (a):** 12 randomly sampled Tier-A validation lenses (from the full pool of 112), sorted by r-band magnitude. Each shows its CNN detection probability. Selection method: `np.random.RandomState(42).choice(112, 12, replace=False)`, sorted by mag.

- **Panel (b):** 12 randomly sampled parametric Sersic injections from the D06 grid (no-Poisson baseline), 4 per theta_E regime (0.75, 1.50, 2.50 arcsec). Each shows its CNN detection probability. Selection method: `np.random.RandomState(42).choice(pool, 4, replace=False)` from a pool of ~200 injections per theta_E.

**Caption changes:** The new caption explicitly states:
- The panels are from independent datasets with no pairing
- Real lenses exhibit a range of CNN scores (some above 0.9, some below detection threshold)
- Injections are almost uniformly scored p ~ 0
- This is consistent with the 5.18% marginal grid completeness

### Audit data summary (from comparison_audit_v2.json)

**Panel (a) — Real Tier-A lenses (12 of 112):**

| r_mag | CNN score p |
|-------|-------------|
| 16.3 | 0.490 |
| 17.5 | 0.596 |
| 18.0 | 0.967 |
| 18.1 | 0.986 |
| 18.1 | 0.001 |
| 18.2 | 0.001 |
| 18.4 | 0.002 |
| 18.5 | 0.391 |
| 18.5 | 0.966 |
| 18.8 | 0.998 |
| 19.6 | 0.999 |
| 19.9 | 1.000 |

Note: 5 of 12 score below p = 0.5. This is expected — these are confirmed real lenses from the Tier-A catalog, but the CNN (trained as a general lens finder) does not detect all of them. The random sample honestly shows this diversity.

**Panel (b) — Parametric injections (4 per theta_E):**

| theta_E | Scores |
|---------|--------|
| 0.75 | 0.000002, 0.000012, 0.0009, 0.00008 |
| 1.50 | 0.000002, 0.000012, 0.000016, 0.000004 |
| 2.50 | 0.0001, 0.004, 0.00001, 0.00005 |

All 12 injections score p < 0.005. This is the honest picture of the morphological barrier — the vast majority of parametric injections are rejected by the CNN regardless of Einstein radius.

**Key contrast with v13:** The v13 figure cherry-picked the highest-scored injections (0.91-0.99 at theta_E >= 1.5), giving a misleading impression that injections frequently fool the CNN. The v14 random sample shows they almost never do.

---

## Review questions — please answer each with YES/NO first

### Q1: Is the Figure 5 redesign scientifically honest?

- Does the independent gallery approach avoid the misleading pairing of v13?
- Is the random sampling methodology (seed=42, no score-based selection) appropriate?
- Is the caption accurate given the audit data?
- Does showing low-scoring real lenses add or detract from the paper's message?

### Q2: Does the new figure strengthen or weaken the paper?

The v14 figure now shows: (1) the CNN reliably rejects parametric injections across all theta_E, and (2) real lenses have diverse scores. Is this a clearer illustration of the morphological barrier than the v13 paired comparison?

### Q3: Number consistency (spot check)

Since only the Figure 5 caption changed, verify the new caption's claims:
- "5.18 per cent marginal grid completeness" — consistent with Table 4?
- Score descriptions ("range of CNN scores", "almost uniformly p ~ 0") — consistent with audit data?

### Q4: Remaining issues from v13 still resolved?

Confirm the following fixes from v13 are still present in v14:
- Abstract uses m_source (not m_lensed) for bright-arc Poisson result
- Table 8 theta_E values are exact grid steps
- Bonferroni correction sentence present
- Figure 1 caption says four bins
- Table 5 full-width

### Q5: Top 3 remaining weaknesses for referee

What are the three most likely referee objections? Be specific. Has the Figure 5 redesign addressed or created any new vulnerabilities?

### Q6: Submission readiness

Is v14 submission-ready (YES/NO)? If NO, list the specific blocking items.

---

## Files included in zip

- `paper/mnras_merged_draft_v14.tex` — the paper source
- `paper/mnras_merged_draft_v14.pdf` — compiled PDF
- `paper/fig1_completeness.pdf` through `paper/fig5_comparison.pdf` — all 5 figures
- `paper/generate_all_figures.py` — figures 1-4 generation script
- `scripts/generate_comparison_figure.py` — figure 5 generation script (v2, independent galleries)
- `results/D06_20260216_corrected_priors/comparison_figure/comparison_audit_v2.json` — audit trail for Figure 5
- `results/D06_20260216_corrected_priors/poisson_diagnostics/d06_poisson_diagnostics.json` — paired delta data
- `results/D06_20260216_corrected_priors/provenance.json` — code provenance checksums
