# Research Plan 5: Uber Mixture and Ensemble (Late-Stage)

## Core Thesis

> "Does combining multiple source modes and artifact profiles in a single training set produce a more robust model than any individual component?"

---

## When to Pursue This

**PREREQUISITE:** Ablations for Gen5-Prime, Gen7, Gen8 must be complete first.

**Why this order matters:**
1. Without ablations, you cannot attribute gains to specific components
2. Reviewers will ask "which helped?" - Uber alone cannot answer
3. Ablations are publishable; soup is not

---

## What Uber Is

Combine multiple source modes and artifact profiles in weighted mixture:

```python
DEFAULT_UBER = MixerConfig(
    source_modes=["cosmos", "deep", "hybrid"],
    source_probs=[0.34, 0.33, 0.33],
    artifact_profiles=["none", "mild", "strong"],
    artifact_probs=[0.15, 0.70, 0.15],
)

# Each sample is assigned deterministically by task_id
def assign_modes(task_id, cfg):
    source_mode = categorical_from_hash(task_id, cfg.source_probs)
    artifact_profile = categorical_from_hash(task_id, cfg.artifact_probs)
    return source_mode, artifact_profile
```

---

## Ensemble (Optional Addition)

**What it is:** Train N models (different seeds or ablation variants), combine predictions.

**When useful:**
- Reduces variance
- Can improve precision at fixed recall
- Robustness check: improvements should persist across models

**When NOT useful for paper:**
- Muddles attribution of gains
- Complicates selection function interpretation
- Not needed for core scientific contribution

**Recommendation:** Single model for main results. Ensemble only as final practical improvement, reported in appendix.

---

## Ablation-Informed Weights

After ablations complete, set Uber weights based on what helped:

| If Ablation Shows... | Then Uber Weights |
|----------------------|-------------------|
| Gen7 helps, Gen8 neutral | cosmos=0.4, hybrid=0.6, artifacts=mild |
| Gen8 helps, Gen7 neutral | cosmos=1.0, artifacts=mixed |
| Both help | cosmos=0.34, hybrid=0.33, deep=0.33, artifacts=mild |
| Neither helps | Skip Uber, use best single variant |

---

## Per-Row Tracking

Store metadata for stratified analysis:

```python
# Each parquet row includes:
{
    "source_mode": "cosmos|deep|hybrid",
    "artifact_profile": "none|mild|strong",
    "gen_variant": "uber_v1",
}
```

This enables:
- Post-hoc analysis of which combinations perform best
- Selection function stratified by source_mode
- Debugging if certain combinations underperform

---

## Cost-Effort Analysis

| Factor | Assessment |
|--------|------------|
| **Prerequisite** | Ablations complete |
| **Data requirement** | Uses existing + Gen7/Gen8 code |
| **Engineering effort** | Low (mixer code exists) |
| **Compute cost** | Same as single training |
| **Publication value** | "Best practical model" but not core contribution |

---

## Challenges

1. **Attribution problem:** Cannot claim "Uber is novel"
   - **Mitigation:** Present as "practical synthesis", not contribution

2. **Selection function complexity:** Mixture complicates completeness estimates
   - **Mitigation:** Stratify by source_mode, report separately

3. **Hyperparameter soup:** Tuning mixture weights is ad-hoc
   - **Mitigation:** Use ablation results to inform weights

---

## Success Criteria

- Uber ≥ best individual variant on all metrics
- Selection function remains interpretable when stratified
- Per-source-mode analysis reveals no pathological behavior

---

## Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Ablations complete | (prerequisite) | Clear winners identified |
| Set Uber weights | 1 day | Config file |
| Training | 2-3 days | Uber model |
| Stratified analysis | 1-2 days | Per-mode metrics |

**Total: ~1 week (after ablations)**

---

## Paper Placement

- **Main paper:** Ablation results (Gen5-Prime, Gen7, Gen8)
- **Appendix/brief section:** "For practical use, we combine..."
- **Code release:** Include Uber config for practitioners
