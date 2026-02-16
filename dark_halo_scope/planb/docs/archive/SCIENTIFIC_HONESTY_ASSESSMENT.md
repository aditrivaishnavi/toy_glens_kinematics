# Scientific Honesty Assessment: Training Data Design

**Date:** 2026-02-06
**Author:** Investigation notes
**Status:** Critical decision point

---

## The Core Question

> "Do we need negative examples with similar brightness? Is this even possible in the real world?"

---

## The Honest Answer

### 1. In Reality, Lens Detection is NOT a Paired Comparison

**What we're simulating:**
- Take LRG_A, add synthetic arc → "positive"
- Take LRG_A without arc → "negative"
- Train classifier to distinguish them

**What real detection requires:**
- Given an image of LRG_X (never seen before)
- Determine: is there a lensed source behind it?
- You don't have a "control" version of LRG_X

**The fundamental mismatch:**
Our training task is "spot the difference" between two versions of the same galaxy.
The real task is "classify this galaxy" with no reference.

### 2. The Inner Image Problem is Real Physics... But Irrelevant

Yes, the inner (counter) image exists. Yes, it adds flux to the core. Yes, this is physically correct.

**But in real observations:**
- The inner image is highly demagnified (typically 10-100x fainter than outer arc)
- It's superimposed on the bright LRG core (signal-to-noise is poor)
- It's often completely invisible
- Real lens detections are based on the OUTER arc, not the inner image

**Our simulation includes something that's physically correct but observationally irrelevant.**

### 3. What Published Papers Actually Do

Looking at how the field handles this:

| Approach | Method | Shortcut Risk |
|----------|--------|---------------|
| **Metcalf et al. 2019** | Inject arcs into random LRGs, compare to different random LRGs | Low (unpaired) |
| **Lanusse et al. 2018** | Fully synthetic (lens galaxy + arc vs just lens galaxy) | Similar to ours |
| **Petrillo et al. 2017** | Real confirmed lenses vs real non-lenses | Lowest (but limited data) |
| **Our approach** | Same LRG ± arc | High (paired shortcut) |

The **safest** approaches use:
1. Unpaired negatives (different galaxies)
2. Real confirmed lenses when available
3. Simulations where positives and negatives have overlapping feature distributions

### 4. The Scientifically Defensible Options

#### Option A: Unpaired Training (RECOMMENDED)

- **Positives:** LRG + injected arc
- **Negatives:** DIFFERENT LRGs (no arc injection)

This forces the model to learn:
- What arcs look like (morphology, color, position)
- NOT "this specific LRG is brighter than its control version"

**Pros:**
- Matches real detection task
- No paired shortcut
- Scientifically defensible

**Cons:**
- Must ensure positives and negatives have similar LRG property distributions
- More complex data pipeline

#### Option B: Aggressive Augmentation

Keep paired training but:
- Strong brightness augmentation (random scaling)
- Core masking during training
- Force model to ignore absolute brightness

**Pros:**
- Can reuse existing data
- Core dropout addresses the shortcut

**Cons:**
- Masking the core throws away real information (what if inner image IS visible?)
- Augmentation may not fully break the shortcut

#### Option C: Dual-Stream with Gate

- Train on paired data
- Include a "core leakage gate" that FAILS if core alone is predictive
- Only accept models that pass the gate

**Pros:**
- Explicit scientific control
- Documents the concern

**Cons:**
- Current models fail this gate (Core LR AUC = 0.95)
- May never pass without fundamental changes

---

## What Should We Do Honestly?

### The Truth We Must Accept

1. **Paired training creates an artificial shortcut.** This is not speculation - our Core LR AUC = 0.95 proves it.

2. **The shortcut is NOT a bug.** It's a consequence of the experimental design (paired positives/negatives from same galaxy).

3. **The inner image is physically real but practically irrelevant.** Real detections don't rely on it.

4. **A model that achieves high AUROC on paired data may fail on unpaired data.** We haven't tested this.

### The Scientifically Defensible Path

1. **Acknowledge the limitation** in any paper or report
2. **Test on unpaired data** to measure real generalization
3. **Consider unpaired training** as the scientifically correct approach
4. **Use core dropout as mitigation** if paired training is necessary

---

## The Hard Question

> "Is this even possible in the real world?"

**The inner image adding core flux?** Yes, physically possible but rarely detectable.

**A non-lens with similar brightness?** Yes, absolutely. LRG intrinsic brightness variation spans orders of magnitude.

**Training that doesn't exploit this shortcut?** Yes, with unpaired design.

---

## Recommendation

**If we want scientific rigor:**

1. Keep current training as "Experiment A" (paired, with known shortcut)
2. Create "Experiment B" with unpaired negatives
3. Compare performance on:
   - Paired test set (current)
   - Unpaired test set (simulated)
   - Real confirmed lenses (if available)
4. Report all results honestly

**The minimum acceptable outcome:**
- Document that paired training has a known shortcut
- Show that core dropout mitigates it (or doesn't)
- Test generalization to unpaired data

---

## Questions That Remain

1. Do we have access to real confirmed lenses for validation?
2. Can we construct an unpaired negative set from our LRG pool?
3. What do the ablation runs (no_hardneg, no_coredrop, minimal) tell us about shortcut reliance?
4. Should we pause and redesign before continuing?
