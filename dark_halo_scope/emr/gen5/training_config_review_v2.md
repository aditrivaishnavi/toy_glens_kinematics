# Gen5-Prime Training Configuration Review v2

## CRITICAL UPDATE: Core Leakage Gate FAILED

### What I Found

I ran the core leakage LR gate as recommended. Results:

| Stratum | Core-only AUC | Threshold | Status |
|---------|---------------|-----------|--------|
| Overall | **0.90** | < 0.70 | **FAIL** |
| theta < 1" | 0.88 | - | FAIL |
| 1" ≤ theta < 2" | 0.85 | - | FAIL |
| theta ≥ 2" | 0.74 | - | FAIL |

**Even for large Einstein radii (theta ≥ 2", arc at 10+ pixels from center), core-only features achieve 0.74 AUC.**

### Root Cause Investigation

| Finding | Evidence |
|---------|----------|
| Stamp core ALWAYS brighter | 100% of samples (15/15 across 5 bricks) |
| Same LRG, multiple injections | All theta≥2" samples from same (ra, dec) position |
| Low correlation with arc_snr | 0.29 - core difference NOT from arc flux |
| Systematic offset | stamp_core ~15% brighter than ctrl_core on average |

**Conclusion:** The injection process adds flux to the ENTIRE image, not just the arc region. This creates a non-physical shortcut that LR can exploit.

---

## Implications for Training

### Option A: 6-Channel Approach (BLOCKED)

The original plan `concat(stamp, ctrl)` is **not viable** because:
1. Model will learn trivial shortcut (core brightness comparison)
2. At inference, we don't have true ctrl for real lenses
3. Core leakage gate failed at 0.90 AUC

### Option B: 3-Channel with ctrl for Hard Negatives Only

Use ctrl_stamp as **additional negative examples**, not as input:

```python
# Training batch construction:
# Positives: stamp (LRG + arc), label = 1
# Negatives: ctrl_stamp (base LRG), label = 0

# Model sees 3-channel input only
# ctrl provides hard negatives (same position, different label)
```

**Pros:**
- No inference-time ctrl needed
- Hard negatives improve discrimination
- Avoids core shortcut (model must learn arc features to distinguish)

**Cons:**
- Loses paired differential information

### Option C: 3-Channel with Residual View

Compute `stamp - ctrl` as a "residual" image showing only the arc:

```python
# Preprocessing:
residual = stamp - ctrl  # Shows only injected arc

# Training:
# Positives: residual (visible arc), label = 1
# Negatives: noise-like residuals from ctrl-ctrl pairs, label = 0
```

**Pros:**
- Explicitly isolates arc signal
- No background/core confounds

**Cons:**
- Requires ctrl at inference (problem for real lenses)
- May be too "easy" - model learns to detect any non-zero residual

### Option D: Fix the Data Pipeline (Recommended Long-Term)

Investigate why injection adds flux to entire image:
1. Check if injection code has background offset
2. Verify sky subtraction consistency between stamp and ctrl
3. Re-run injection with flux-conserving method

**This is the cleanest solution but requires pipeline debugging.**

---

## Revised Recommendation: NO-GO for 6-Channel

The core leakage gate failed definitively. **I cannot proceed with the 6-channel approach.**

### Proposed Path Forward

1. **Immediate:** Switch to Option B (3-channel, ctrl as hard negatives)
2. **Parallel:** Investigate Option D (fix injection pipeline)
3. **Defer:** Option C until inference strategy is clarified

---

## Updated Training Configuration (Option B)

```python
config = {
    # -------------------------------------------------------------------------
    # CHANGED: MODEL INPUT
    # -------------------------------------------------------------------------
    
    "input_channels": 3,  # Changed from 6
    # Reason: 6-channel failed core leakage gate (0.90 AUC)
    # Now using single 3-channel image
    
    "use_ctrl_as_negative": True,  # NEW
    # ctrl_stamp becomes a hard negative example (label=0)
    # Same LRG position, different label - forces model to learn arc features
    
    # -------------------------------------------------------------------------
    # DATA CONSTRUCTION
    # -------------------------------------------------------------------------
    
    "batch_composition": {
        "positives": "stamp_npz",      # LRG + arc, label=1
        "negatives": "ctrl_stamp_npz", # Base LRG, label=0
        "ratio": 1.0,                  # 50% positives, 50% negatives
    },
    # Each batch has balanced pos/neg from same LRG positions
    
    # -------------------------------------------------------------------------
    # EVERYTHING ELSE UNCHANGED
    # -------------------------------------------------------------------------
    
    "model_name": "resnet18",
    "pretrained": True,  # NOW POSSIBLE with 3-channel input
    "batch_size": 256,
    "epochs": 50,
    "learning_rate": 1e-3,
    "optimizer": "AdamW",
    # ... rest unchanged ...
}
```

### Why This Works

| Concern | How Addressed |
|---------|---------------|
| Core leakage | Model sees stamp OR ctrl, not both - can't compare cores |
| No inference ctrl | Model trained on 3-channel only, no ctrl needed at inference |
| Hard negatives | ctrl is challenging - same LRG, must detect arc to distinguish |
| Pretrained weights | ImageNet pretrained ResNet18 works with 3-channel |

---

## Questions for You

### Q1: Should I Proceed with Option B?

Given the core leakage failure, Option B (3-channel, ctrl as hard negative) is the safest path forward.

**Trade-off:** We lose the explicit paired differential learning, but we:
- Avoid the shortcut
- Match inference scenario (3-channel only)
- Can use pretrained weights

**Do you approve Option B?**

### Q2: Should I Investigate the Injection Pipeline?

The core brightness offset suggests a bug or design issue in the injection code. Options:

- (A) Proceed with Option B now, debug pipeline later
- (B) Pause training, investigate and fix injection first
- (C) Accept this is inherent to injection (PSF spreading) and design around it

**Which approach do you prefer?**

### Q3: What About Real Lens Inference?

With Option B, the model is trained to distinguish:
- Positive: LRG + arc (synthetic)
- Negative: LRG only (real base images)

At inference on real lens candidates:
- We have a single image (potential lens)
- Model predicts P(lens present)

**This matches deployment scenario correctly. Do you agree?**

### Q4: Arc_snr=0 Samples in Option B

With ctrl as negatives, arc_snr=0 positives (injected arc with SNR≈0) become:
- Very similar to their paired ctrl negative
- Hard positives that model must learn subtle features from

**Should I:**
- (A) Include arc_snr=0 positives (curriculum learning opportunity)
- (B) Exclude arc_snr=0 positives (too similar to negatives, may confuse)

---

## Summary

| Gate | Previous Status | Current Status |
|------|-----------------|----------------|
| Arc localization | PASS | PASS |
| Split integrity | PASS | PASS |
| Core leakage | NOT TESTED | **FAIL (0.90 AUC)** |
| 6-channel viability | Assumed OK | **BLOCKED** |

**My Recommendation:** 

**NO-GO for 6-channel approach.**

**GO for Option B (3-channel, ctrl as hard negatives)** pending your approval.

---

## Self-Generated Additional Questions

### Q5: Is the Core Brightness Offset Physical or Artificial?

If the injection adds flux everywhere due to PSF convolution of the arc, this is physically correct (light spreads). But the offset is ~15% of core flux, which seems too high for an arc at theta=2".

**Should I compute expected PSF-spread flux contribution to verify?**

### Q6: Paired Loss Without Paired Input?

We originally wanted paired training to reduce sim-to-real gap. With Option B, we lose explicit pairing.

**Alternative:** Add a contrastive auxiliary loss:
```python
# Main loss: BCE for classification
# Aux loss: embedding(stamp) should be far from embedding(ctrl) for same position
```

**Would this recover some paired learning benefit?**

### Q7: What If Model Learns "Brightness = Lens"?

Even with 3-channel input, if stamp is systematically brighter than ctrl, and model only sees stamps as positives and ctrls as negatives, it might learn:
- High brightness → lens
- Low brightness → no lens

**Mitigation:** Normalize stamp and ctrl identically (per-sample to same mean/std).

**Should I enforce this normalization?**

### Q8: Validation on Real Lens Candidates

Ultimate test is performance on REAL known lenses (not synthetic).

**Plan:**
- After training, evaluate on:
  - Jacobs et al. confirmed lenses
  - Huang et al. candidates
  - Known non-lenses from same surveys

**Should this be a blocking criterion for deployment?**

---

## Final Decision Request

Please respond:

1. **Option B approved?** (3-channel, ctrl as hard negatives)
2. **Pipeline investigation priority?** (Now vs later)
3. **Arc_snr=0 handling?** (Include or exclude)
4. **Normalized brightness?** (Enforce same mean/std for stamp and ctrl)

Upon your approval, I will implement Option B and proceed with training.
