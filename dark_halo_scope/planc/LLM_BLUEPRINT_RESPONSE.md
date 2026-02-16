# LLM Blueprint Response

**Date:** 2026-02-07  
**Source:** External LLM review of IMPLEMENTATION_BLUEPRINT_REQUEST.md

---

## Core Blueprint

### Two Paper Options, Clarified

### Option 1: "Selection Functions and Failure Modes"

This is the more defensible 4-week paper, because you can scope it around:

* A clearly defined **audited pipeline** (training data construction + evaluation protocol)
* A selection function defined as: **detection probability conditioned on data-quality strata and lens observables**
* A failure taxonomy: contaminant categories + observing-condition dependence + spatial dependence

### Option 2: "Ensemble diversification via domain-specialized training"

This is a strong secondary contribution because the DR10 paper explicitly notes that base models trained on the same data have correlated predictions and suggests training on different subsets to increase diversity.
But in 4 weeks you should treat this as **one controlled experiment**, not a full thesis.

---

## What the DR10 Paper Implies You Should Copy Exactly

These are the highest-leverage "do what they did" pieces:

1. **Nonlens selection stratified by z-band exposure count** to avoid the "exposure-count confound"
2. **Nonlens:lens ratio ~100:1** in each bin
3. **Train/validation split 70/30** while maintaining the ratio
4. **101×101 pixel cutouts (~26″)**
5. Their deployed workflow uses **a very high threshold** (top 0.01%) prior to visual inspection

Those five choices are exactly the kinds of things reviewers will ask you to justify. Copying them reduces reviewer friction.

---

## Answers to Specific Questions

### 1) Architecture

**Baseline:** ResNet-18 first. Move to ResNet-34 only if you are clearly underfitting.
Reason: With 101×101 cutouts, ResNet-18 is usually sufficient; deeper nets mostly add capacity you do not have label-quality to exploit.

**EfficientNet:** B0 (or B1 if you have enough GPU RAM). Use ImageNet pretraining for faster convergence.

**Metadata branch:** Yes, but only if you can guarantee metadata correctness and avoid leakage. Keep it minimal:
* `nexp_z`, `psfsize_r` (or z), `psfdepth_r` (or z), `tractor_type` one-hot
* Do not include sky coordinates or brick IDs in the classifier (use them only for analysis).

**Input format:** 3-channel (g,r,z) at 101×101 (match DR10). z-only is a useful ablation, not the default.

### 2) Selection Function Methodology

Use **one global model** as the primary selection function.

* If you publish a selection function, it must be tied to a specific algorithm. So publish it for your baseline model.

Then add:
* **Domain-specialized ensemble** as a secondary analysis: show where specialized models improve recall at fixed FPR in weak strata.

**Completeness estimation (what you can defend):**
* Define strata in data-quality space: (z_nexp bin, PSF bin, depth bin, Tractor type).
* Within each stratum, compute recall on **Tier-A anchors only** (confirmed). Treat Tier-B as noisy positives, not ground truth.

**Uncertainty:**
* For per-stratum recall: bootstrap over Tier-A anchors (simple and honest). When N is tiny, also report exact binomial interval or Bayesian beta posterior; and mark strata with "insufficient data."

**Calibration:**
* You cannot estimate true prevalence from candidates; report:
  * prevalence-free metrics (ROC/PR)
  * reliability curves on your labeled sets but explicitly caveat "label = training label"
  * plus scenario-weighted calibration for a **claimed deployment prior** (example: 10,000:1)

### 3) Ensemble Diversification (Option 2)

Use **domain splits that correspond to the known confounds and real failure sources**:

1. z-band exposure count bins (because the confound is explicitly discussed)
2. seeing bins (PSF)
3. depth bins
4. Optionally morphology type (SER/DEV vs EXP/REX)

Do not do "geographic" splits as the main diversification mechanism unless you are explicitly studying spatial domain shift.

**How to measure diversity:**
* correlation of predicted probabilities on a large shared unlabeled set
* disagreement rate near threshold (most relevant for human review)
* ensemble entropy / variance

**Ensembling method:**
* Start with simple averaging.
* Only add a meta-learner if you can show it beats averaging on a held-out validation set without leakage.

### 4) Training Protocol

Minimum viable, defensible:
* Loss: BCEWithLogits + `pos_weight` (or focal loss if you see collapse on rare positives)
* Optimizer: AdamW, cosine LR schedule
* Augmentations: rotations (0/90/180/270), flips; mild Gaussian noise; mild intensity jitter
* Early stopping: monitor validation AUC
* Class imbalance: do not oversample positives aggressively; prefer weighting + good negatives.

### 5) Evaluation Protocol

**Tier-B label noise (your circularity problem):**
* Primary recall/completeness must be on Tier-A anchors only.
* Tier-B is used for:
  * training (with label smoothing, and/or grade weights)
  * "candidate consistency" analysis (how much do you reproduce Huang-like candidates)

**Grade weighting:**
* Yes. If you have "confident/probable," weight confident higher (or set probable target = 0.8 instead of 1.0).

**Failure modes:**
* Build contaminant sets explicitly and report FPR by category.
* Interpretability: GradCAM is fine as qualitative support; do not oversell it as causal.

---

## The Honest Framing That Will Survive MNRAS

### Best Framing for Impact and Honesty

Pick **Option B + C** combined:

**"Where do ML lens finders fail, and how does data quality shape their selection function?"**

### What Your Paper Can Honestly Answer

* "Given confirmed anchors, what is the detection probability across observing conditions?"
* "What contaminant classes dominate false positives at deployment-like thresholds?"
* "How sensitive are results to domain shift (seeing/depth/exposure/morphology)?"
* "Does domain-specialized training increase diversity and improve recovery in hard regimes?"

### What You Cannot Honestly Claim

* "Fraction of all strong lenses in DR10 that are detected," unless you have an external, unbiased lens population model or extensive spectroscopy.

---

## 4-Week Plan with Checkpoints

### Week 1: Data + Metadata Correctness

Deliverables:
* Positive catalog ingested (Tier-A and Tier-B separated)
* Local Tractor metadata match validated on a random sample
* Stratified negative catalog built (match z_nexp distribution per lens type)

Checkpoint tests:
* Are your negatives matched by z-exposure bin (and optionally type)?
* Confirm cutout size exactly 101×101

### Week 2: Baseline Model + Sanity

Deliverables:
* ResNet-18 baseline trained
* Held-out test set with stable AUC and reasonable FPR at high thresholds

Checkpoint tests:
* Train/val curves are stable (no collapse)
* Top-K predictions are not dominated by obvious artifacts

### Week 3: Selection Function + Failure Taxonomy

Deliverables:
* Recall vs (PSF, depth, nexp) using Tier-A anchors only
* Bootstrapped CIs
* FPR by contaminant category

Checkpoint tests:
* In best strata, Tier-A recall is meaningfully higher than in worst strata

### Week 4: Ensemble Diversification + Paper Figures

Deliverables:
* Domain-specialized models (one split axis)
* Diversity metrics + performance delta vs baseline
* Paper-quality figures: selection heatmaps, reliability diagram, failure galleries

---

## Reviewer Objections and Preemptions

1. **"You trained on candidates found by similar models."**
   Response: completeness/selection is evaluated on independent confirmed Tier-A anchors; candidates are treated as weak labels and analyzed separately.

2. **"Your negative sampling is not representative."**
   Response: negatives are stratified to match deployment confounds (z_nexp) and expanded toward realistic imbalance; additionally, failure modes are reported by contaminant type.

3. **"Selection function is model-dependent."**
   Response: yes; you provide it for a precisely specified baseline and show sensitivity via a domain-specialized ensemble.

4. **"Small-N anchors."**
   Response: strata with insufficient anchors are flagged; uncertainties are bootstrapped and/or binomial/Bayesian; spatial correlation is assessed via region holdouts.

---

## Science Impact Language

* **Cosmology / lens counts:** selection functions distort inferred abundance if not modeled; your paper provides a lookup for detection probability conditioned on data quality, enabling forward-modeling of observed counts.
* **Substructure:** if completeness depends on arc surface brightness/seeing, you are biased toward smooth/high-SNR arcs, which can bias substructure-sensitive samples.
* **Time-delay cosmography:** if selection favors certain configurations, the discovered sample is not representative.
* **Survey design:** your results translate to requirements on seeing/depth/exposure strategy for efficient lens discovery at fixed human-review budget.

---

## How to Use the Code Package

1. Build negative catalog from Tractor:
```bash
python data/download_negatives.py --tractor_catalog /path/to/tractor.parquet --out data/negatives --target 500000
```

2. Download negative cutouts:
```bash
python data/download_cutouts.py --catalog data/negatives/negatives_for_cutouts.csv --out data/neg_cutouts
```

3. Download positive cutouts:
```bash
python data/download_cutouts.py --catalog data/catalogs/positives.csv --out data/pos_cutouts
```

4. Prepare splits:
```bash
python data/prepare_dataset.py --pos_dir data/pos_cutouts --neg_dir data/neg_cutouts --out data/datasets/dr10_v1
```

5. Train baseline:
```bash
python training/train_baseline.py --data_root data/datasets/dr10_v1 --model resnet18 --epochs 30 --out runs/baseline_resnet18
```

6. Evaluate:
```bash
python evaluation/compute_completeness.py --data_root data/datasets/dr10_v1 --ckpt runs/baseline_resnet18/best.pt --out eval/completeness
python evaluation/compute_calibration.py --data_root data/datasets/dr10_v1 --ckpt runs/baseline_resnet18/best.pt --out eval/calibration
python evaluation/analyze_failures.py --data_root data/datasets/dr10_v1 --ckpt runs/baseline_resnet18/best.pt --out eval/failures
```

---

## Final Realism Check

**4 weeks is realistic** for Option 1 plus one controlled diversification experiment, if:
* Tractor metadata access is already solved
* you keep the selection-function definition narrow (data-quality axes + Tier-A anchors)
* you avoid trying to prove "true DR10 completeness"

---

*Response saved from external LLM review, 2026-02-07*
