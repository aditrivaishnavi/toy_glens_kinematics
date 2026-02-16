# Follow-up Questions: Code Issues and Clarifications

Thank you for the excellent review. Your key insight that **r<10 blur would destroy the arc signal** caught a critical error in our approach. We agree with the delta shuffling + residual view strategy.

We have specific questions about implementation details and a few issues we found in the provided code.

---

## Part A: Issues Found in Provided Code

### Issue 1: Channel Dimension Mismatch

**Problem:** `shortcut_resistant_augmentation` returns 2C channels (original + residual), but the model expects C channels.

```python
def shortcut_resistant_augmentation(batch_x, theta_pix, psf_fwhm_pix):
    # ... 
    batch_x = add_residual_view(batch_x, sigma=3.0)  # (B,C,H,W) -> (B,2C,H,W)
    return batch_x
```

**Our model:**
```python
# ConvNeXt-Tiny expects 3 input channels
m = convnext_tiny(weights=None)
# First conv layer: Conv2d(3, 96, kernel_size=4, stride=4)
```

**Questions:**
1. Should we modify the model's first layer to accept 6 channels instead of 3?
2. Or should residual view be a separate branch that gets fused later?
3. What's the recommended architecture modification?

**Our suggestion:**
```python
# Option A: Modify first conv
m.features[0][0] = nn.Conv2d(6, 96, kernel_size=4, stride=4)

# Option B: Two-branch early fusion
class DualBranchInput(nn.Module):
    def __init__(self):
        self.raw_conv = nn.Conv2d(3, 48, kernel_size=4, stride=4)
        self.resid_conv = nn.Conv2d(3, 48, kernel_size=4, stride=4)
    def forward(self, x):  # x: (B, 6, H, W)
        raw = x[:, :3]
        resid = x[:, 3:]
        return torch.cat([self.raw_conv(raw), self.resid_conv(resid)], dim=1)
```

**Does Option A or B make more sense? Or is there a better approach?**

---

### Issue 2: Device Handling in validate_no_shortcuts

**Problem:** The function mixes numpy arrays with torch model calls without explicit device handling.

```python
def validate_no_shortcuts(model, x: np.ndarray, ...):
    # ...
    with torch.no_grad():
        t = torch.from_numpy(hardneg_x).float().to(next(model.parameters()).device)
        p = torch.sigmoid(model(t)).detach().cpu().numpy().ravel()
```

**Questions:**
1. Should we add explicit device parameter?
2. How should we handle batch size limits for GPU memory?

**Our suggestion:**
```python
def validate_no_shortcuts(
    model,
    x: np.ndarray,
    y: np.ndarray,
    theta_over_fwhm: np.ndarray,
    arc_snr: np.ndarray,
    hardneg_x: np.ndarray | None = None,
    device: str = "cuda",
    batch_size: int = 64,
):
    # ... batch processing for memory efficiency
```

**Does this make sense?**

---

### Issue 3: Hard Negative Ratio and Training Loop Integration

**Problem:** The delta shuffling code generates hard negatives, but integration with training loop is unclear.

**Current understanding:**
```python
# Per epoch:
# - 40% paired positives (LRG + arc)
# - 40% paired controls (same LRG, no arc)  
# - 20% hard negatives (ctrl + shuffled delta)
```

**Questions:**
1. Should hard negatives be generated on-the-fly in the dataloader, or pre-computed?
2. What's the recommended label for hard negatives? y=0 (same as controls)?
3. Should we use the same pair_id for positive, control, and hard_negative from the same LRG?

**Our suggestion:**
```python
class PairedDataset(Dataset):
    def __getitem__(self, idx):
        pos, ctrl = self.load_pair(idx)
        
        # Random choice: return (positive, 1), (control, 0), or (hard_neg, 0)
        r = np.random.random()
        if r < 0.4:
            return pos, 1
        elif r < 0.8:
            return ctrl, 0
        else:
            delta = pos - ctrl
            shuffled = azimuthal_shuffle(delta)
            hard_neg = ctrl + shuffled
            return hard_neg, 0
```

**Is this the right approach?**

---

### Issue 4: azimuthal_shuffle Implementation Detail

**Problem:** The provided `azimuthal_shuffle` operates on the full delta, but should we preserve the inner core?

```python
def azimuthal_shuffle(delta: np.ndarray, nbins: int = 24, r_max: float | None = None) -> np.ndarray:
    # Currently shuffles ALL radii including the innermost pixels
```

**Questions:**
1. Should we skip shuffling for r < 2 pixels (the true lens core)?
2. Or does it not matter because delta is already small in the core?

**Our data shows:**
- For large θ_E (1.3-2.5"): core injection contribution = +8%
- For all samples: core injection contribution = +67%

**So the core DOES have injection flux for most samples.**

**Our suggestion:**
```python
def azimuthal_shuffle(delta, nbins=24, r_max=None, r_min=0.0):
    # Skip shuffling for r < r_min (preserve innermost structure)
    for b in range(nbins):
        if edges[b] < r_min:
            out[:, m] = delta[:, m]  # Don't shuffle inner core
            continue
        # ... shuffle as before
```

**Does this make sense, or should we shuffle everything?**

---

### Issue 5: Residual Sigma Value

**You recommended:** sigma = 3.0 pixels for the residual view.

**Our PSF:** FWHM = 1.32" → 5.04 pixels → sigma = 2.14 pixels

**Question:** Is sigma=3.0 optimal, or should it be:
- a) PSF-matched (sigma ≈ 2.1 pixels)
- b) Slightly larger than PSF (sigma = 3.0, your recommendation)
- c) Adaptive based on per-image PSF size

**Trade-off we see:**
- Too small sigma: residual still contains radial structure
- Too large sigma: might blur out the arc signal

---

## Part B: Clarification Questions

### Question 6: Stratified Gate Thresholds

You recommend:
```
radial_profile_auc_x_ge_1 <= 0.60
core_auc_x_ge_1 <= 0.60
annulus_auc_x_ge_1 >= 0.75
hardneg_mean_p <= 0.05
arc_occlusion_drop >= 0.30
```

**Questions:**
1. What if we pass 4/5 gates but fail one? Should we iterate or is partial pass acceptable?
2. Are these thresholds empirically derived from other papers, or theoretical?
3. How strict should we be for publication? (e.g., ≤0.55 vs ≤0.60)

---

### Question 7: Annulus AUC Definition

In `validate_no_shortcuts`, you use:
```python
fa = aperture_stats_features(x1, r0=4.0, r1=12.0)  # Fixed annulus
```

**But θ_E varies from 2 to 10 pixels in our data.**

**Questions:**
1. Should the annulus be adaptive: r0 = θ_E - k*PSF, r1 = θ_E + k*PSF?
2. Or is fixed r0=4, r1=12 a reasonable proxy for "where arcs typically are"?

**Our suggestion:**
```python
def adaptive_annulus_features(x, theta_pix, psf_pix, k=1.5):
    r0 = np.maximum(theta_pix - k * psf_pix, 2.0)
    r1 = theta_pix + k * psf_pix
    # Per-sample annulus
```

**Does this overcomplicate things?**

---

### Question 8: Training with Residual View - Initialization

When we modify the model to accept 6 channels:
```python
m.features[0][0] = nn.Conv2d(6, 96, kernel_size=4, stride=4)
```

**Questions:**
1. Should we initialize the new weights randomly?
2. Or copy ImageNet weights for channels 0-2 and initialize 3-5 separately?
3. Does using pretrained weights for the first 3 channels help?

**Our suggestion:**
```python
# Copy pretrained weights for RGB channels
old_weight = m.features[0][0].weight.data  # (96, 3, 4, 4)
new_conv = nn.Conv2d(6, 96, kernel_size=4, stride=4)
new_conv.weight.data[:, :3] = old_weight
new_conv.weight.data[:, 3:] = old_weight  # Initialize residual same as raw
m.features[0][0] = new_conv
```

**Is this a good initialization strategy?**

---

### Question 9: Arc Occlusion Test Implementation

You defined:
```python
# crude annulus occlusion: zero r in [4,12]
m = (r >= 4.0) & (r < 12.0)
x_occ = x_pos.copy()
x_occ[:, :, m] = 0.0
```

**Questions:**
1. Should we zero or fill with outer-region statistics (mean/noise)?
2. Zeroing creates a sharp discontinuity - does this affect the test validity?
3. Should occlusion be applied BEFORE or AFTER residual view computation?

**Our concern:** Zeroing in raw image then computing residual = raw - blur creates artifacts at the boundary.

**Our suggestion:**
```python
# Option A: Fill with outer noise (smoother)
outer_mean = x_pos[:, :, r > 16].mean()
outer_std = x_pos[:, :, r > 16].std()
x_occ[:, :, m] = np.random.normal(outer_mean, outer_std, m.sum())

# Option B: Apply occlusion after residual view
x_resid = add_residual_view(x_pos)
x_resid_occ = x_resid.copy()
x_resid_occ[:, :, m] = 0.0  # Only occlude in final representation
```

**Which is more appropriate for the occlusion sensitivity test?**

---

### Question 10: Curriculum Learning

You mentioned:
> "start with more hard negatives early (forces morphology learning), then anneal to a stable mix"

**Questions:**
1. What's a good starting ratio? 50% hard negatives?
2. What's the final ratio after annealing? 20%?
3. Over how many epochs should we anneal?

**Our suggestion (for 30 epoch training):**
```python
def get_hardneg_prob(epoch, total_epochs=30):
    # Start at 0.5, anneal to 0.2 over first 15 epochs
    if epoch < 15:
        return 0.5 - (0.3 * epoch / 15)
    return 0.2
```

**Does this schedule make sense?**

---

## Part C: Summary of What We Plan to Implement

Based on your feedback, our implementation plan:

1. **Delta shuffling hard negatives** - generate on-the-fly in dataloader
2. **Residual view** - 6-channel input (3 raw + 3 residual), sigma=3.0
3. **Adaptive center perturbation** - radius = min(0.5*θ, 2*PSF_sigma), mode=noise
4. **Photometric jitter** - gain in [0.9, 1.1]
5. **Stratified validation** - separate gates for x ≥ 1.0 vs x < 0.8
6. **Model modification** - Conv2d(6, 96) with pretrained init for first 3 channels

**Questions:**
1. Is this the right priority order?
2. Are we missing anything critical?
3. Should we implement ALL of these before retraining, or iterate?

---

## Part D: Our Current Data Summary (for reference)

| Metric | Value |
|--------|-------|
| Training samples | 1.38M positives + 1.38M controls |
| θ_E range | 0.5" - 2.5" (mean 1.35") |
| PSF FWHM range | 0.97" - 1.60" (mean 1.32") |
| θ_E/FWHM range | 0.4 - 1.9 (mean ~1.0) |
| Samples with x ≥ 1.0 | ~50% |
| Current central-only AUC (paired) | 0.76 |
| Current arc-annulus AUC | 0.71 |
| Target: central-only AUC (x ≥ 1.0) | ≤ 0.60 |
| Target: arc-annulus AUC (x ≥ 1.0) | ≥ 0.75 |

---

Please address the specific code issues (1-5) and clarification questions (6-10) so we can implement correctly. Thank you!
