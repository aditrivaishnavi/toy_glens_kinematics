# Project Status: Comprehensive Accounting

**Date**: 2026-02-07
**Last Updated**: 07:00 UTC

---

## Executive Summary

We are building a defensible gravitational lens classifier that does NOT rely on shortcuts (core brightness).

**Key Discovery**: Residual radial profile preprocessing + unpaired training eliminates the core brightness shortcut.

**Current Status**: 5 full-scale experiments running on 5 GPUs (~16-30 hours to completion)

---

## Phase 0: Foundation Lock ✅ COMPLETE

| Task | Status | Notes |
|------|--------|-------|
| Lock Evaluation Protocol | ✅ Done | `evaluation_protocol.yaml` |
| Define Anchor Set | ⚠️ Partial | Tier-A anchors exist, need validation |
| Define Contaminant Set | ⚠️ Partial | Need to curate |
| Lock Data Splits | ✅ Done | No brick overlap verified |
| Verify Paired Data Integrity | ✅ Done | 100% pass rate |
| Code Verification | ✅ Done | All unit tests pass |

---

## Phase 1: Baseline Training ✅ COMPLETE (with issues)

| Task | Status | Result |
|------|--------|--------|
| Train Gen5-Prime Baseline | ✅ Done | AUROC=0.9913 |
| Core LR AUC Gate | ❌ FAILED | 0.9497 (threshold: <0.65) |
| Core Masked Drop Gate | ✅ PASSED | 0.08% (threshold: <10%) |
| Ablation: no_hardneg | ✅ Done | AUROC=0.9950, Core LR=0.9497 FAIL |
| Ablation: no_coredrop | ✅ Done | AUROC=0.9967, Core LR=0.9497 FAIL |
| Ablation: minimal | ✅ Done | Running/completed |

**Finding**: Core brightness shortcut is baked into the PAIRED DATA, not a training artifact.

---

## Phase 2: Data Fix Investigation ✅ COMPLETE

| Task | Status | Result |
|------|--------|--------|
| Root cause analysis | ✅ Done | Arc flux overlaps with lens galaxy core |
| External LLM review | ✅ Done | Multiple reviews integrated |
| Residual preprocessing discovery | ✅ Done | **KEY BREAKTHROUGH** |
| Unpaired training proposal | ✅ Done | LRG-disjoint sampling |
| Gate validation | ✅ Done | paired+residual OR unpaired+residual+dropout |

**Key Finding**: `residual_radial_profile` preprocessing breaks the shortcut by removing azimuthal brightness patterns.

---

## Phase 3: Mini Experiments (10% data) ✅ COMPLETE

| Exp | Config | Status | AUROC | Core LR | Gates |
|-----|--------|--------|-------|---------|-------|
| A1 | Paired + Residual | ✅ Done | 0.9730 | 0.9347 | ❌ FAIL |
| B1 | Unpaired + Residual | ✅ Done | **0.9777** | 0.50 | ✅ **PASS** |
| B2 | Unpaired + Residual + dropout r=5 | ⏭️ Skipped | - | - | - |
| B3 | Unpaired + Residual + scheduled | ⏭️ Skipped | - | - | - |
| B4 | Unpaired + Residual + dropout r=3 | ⏭️ Skipped | - | - | - |

**Decision**: B1 passed gates → skipped B2/B3/B4 in mini phase, but now running all at full scale.

---

## Phase 4: Full-Scale Training 🔄 IN PROGRESS

| Exp | Config | GPU | Status | ETA |
|-----|--------|-----|--------|-----|
| A1 | Paired + Residual | lambda | 🔄 Running | ~30 hrs (Feb 8, 11am) |
| B1 | Unpaired + Residual | lambda4 | 🔄 Running | ~16 hrs (Feb 7, 9pm) |
| B2 | Unpaired + Residual + dropout r=5, p=0.5 | lambda2 | 🔄 Running | ~16 hrs (Feb 7, 9pm) |
| B3 | Unpaired + Residual + scheduled masking | lambda3 | 🔄 Running | ~16 hrs (Feb 7, 9pm) |
| B4 | Unpaired + Residual + dropout r=3, p=0.3 | lambda5 | 🔄 Running | ~16 hrs (Feb 7, 9pm) |

**All 5 experiments started: 2026-02-07 05:07 UTC**

---

## Phase 5: Post-Training Evaluation ⏳ PENDING

| Task | Status | Depends On |
|------|--------|------------|
| Aggregate results from all 5 experiments | ⏳ Pending | Phase 4 |
| Compare A1 vs B1 (paired vs unpaired) | ⏳ Pending | Phase 4 |
| Compare B1/B2/B3/B4 (masking variants) | ⏳ Pending | Phase 4 |
| Select best configuration | ⏳ Pending | Phase 4 |
| Final test set evaluation | ⏳ Pending | Phase 4 |
| Core sensitivity curves | ⏳ Pending | Phase 4 |
| θ_E stratified analysis | ⏳ Pending | Phase 4 |

---

## Phase 6: Gen7/Gen8 Ablations ⏳ NOT STARTED

### Important: Two Orthogonal Dimensions

The experiments have **two independent dimensions** that can be combined:

| Dimension | Purpose | Options |
|-----------|---------|---------|
| **Data Structure** | Avoid shortcuts | Paired vs **Unpaired** |
| **Source Realism** | Improve sim-to-real | Gen5 vs Gen6 vs Gen7 vs Gen8 |

**Current experiments (Phase 4)** fix the data structure but still use Gen5 sources.

**Phase 6** will test source realism improvements on the **fixed (unpaired) data structure**.

### The Full Experiment Matrix

```
                        Data Structure
                        Paired       Unpaired + Residual
                       ┌────────────┬─────────────────────┐
             Gen5      │ ❌ FAILED   │ 🔄 B1 (running)     │
             (COSMOS)  │ Core=0.95  │ Core=0.50 (mini)    │
Source       ──────────┼────────────┼─────────────────────┤
Realism      Gen7      │ ❌ Would    │ ⏳ Phase 6          │
             (Hybrid)  │ also fail  │ NEW EXPERIMENT      │
             ──────────┼────────────┼─────────────────────┤
             Gen8      │ ❌ Would    │ ⏳ Phase 6          │
             (Artifacts)│ also fail │ NEW EXPERIMENT      │
                       └────────────┴─────────────────────┘
```

### Phase 6 Tasks

| Task | Status | Notes |
|------|--------|-------|
| Fix Gen7 code issues (clump_flux_frac) | ⏳ Pending | Semantics bug identified |
| **Generate Gen7 data with UNPAIRED sampling** | ⏳ Pending | New EMR job needed |
| Train B1-Gen7: Unpaired + Residual + Gen7 sources | ⏳ Pending | Compare to B1 |
| **Generate Gen8 artifacts with UNPAIRED sampling** | ⏳ Pending | New EMR job needed |
| Train B1-Gen8: Unpaired + Residual + Gen8 artifacts | ⏳ Pending | Compare to B1 |
| Calibrate DR10 artifact rates for Gen8 | ⏳ Pending | - |
| Compare Gen5 vs Gen7 vs Gen8 (all unpaired) | ⏳ Pending | - |

### Phase 6 Experiments (Future)

| Exp | Data Structure | Source | Preprocessing | Status |
|-----|----------------|--------|---------------|--------|
| B1-Gen5 | Unpaired | Gen5 (COSMOS) | Residual | 🔄 Phase 4 (running) |
| B1-Gen7 | Unpaired | **Gen7 (Procedural)** | Residual | ⏳ Phase 6 |
| B1-Gen8 | Unpaired | Gen5 + **Gen8 artifacts** | Residual | ⏳ Phase 6 |
| B1-Uber | Unpaired | **Mixed Gen5+Gen7+Gen8** | Residual | ⏳ Phase 6 |

**Key Point**: Gen7/Gen8 require generating NEW datasets with unpaired LRG sampling. The current Gen5 paired data cannot be reused.

---

## Phase 7: Real Data Evaluation ⏳ NOT STARTED

| Task | Status | Notes |
|------|--------|-------|
| Anchor recall evaluation | ⏳ Pending | Tier-A anchors |
| Contaminant rejection | ⏳ Pending | Need contaminant catalog |
| DR10 scoring pipeline | ⏳ Pending | EMR job exists |
| Selection function analysis | ⏳ Pending | By θ_E, PSF, depth |

---

## Phase 8: Publication ⏳ NOT STARTED

| Task | Status | Notes |
|------|--------|-------|
| Generate paper figures | ⏳ Pending | - |
| Write results section | ⏳ Pending | - |
| Bootstrap confidence intervals | ⏳ Pending | - |
| Multi-seed reproducibility | ⏳ Pending | - |

---

## Data Assets (Existing)

| Asset | Location | Size | Status |
|-------|----------|------|--------|
| v5_cosmos_paired (original) | S3 + local | 128 GB | ✅ Available |
| Unpaired manifest V2 (metadata) | S3 + NFS | 13 MB | ✅ Available |
| Unpaired manifest V1 full (blobs) | NFS + lambda4 | 68 GB | ✅ Available |
| Unpaired manifest V1 mini (10%) | lambda4 | 7 GB | ✅ Available |
| Mini B1 results | NFS | ~200 MB | ✅ Backed up |

---

## Datasets To Create

### Summary Table

| Dataset | Type | Status | Blocking | Can Start Now? | Plan |
|---------|------|--------|----------|----------------|------|
| **Anchor Set** | Evaluation | ⚠️ Needs validation | Phase 7 (anchor recall) | ✅ YES | [ANCHOR_CONTAMINANT_PLAN.md](ANCHOR_CONTAMINANT_PLAN.md) |
| **Contaminant Set** | Evaluation | ❌ Not curated | Phase 7 (contaminant FPR) | ✅ YES | [ANCHOR_CONTAMINANT_PLAN.md](ANCHOR_CONTAMINANT_PLAN.md) |
| **DR10 Artifact Calibration** | Reference | ❌ Not done | Gen8 dataset | ✅ YES | - |
| **Gen7 Dataset (Unpaired)** | Training | ❌ Not created | Gen7 training | ⚠️ After code fix | - |
| **Gen8 Dataset (Unpaired)** | Training | ❌ Not created | Gen8 training | ⚠️ After calibration | - |

---

### Dependency Tree

```
                     ┌─────────────────────────────────────┐
                     │   PHASE 4: Gen5 Unpaired Training   │
                     │        (Currently Running)          │
                     └──────────────┬──────────────────────┘
                                    │
                                    ▼
                     ┌─────────────────────────────────────┐
                     │   PHASE 5: Evaluate & Select Best   │
                     │     Compare A1/B1/B2/B3/B4          │
                     └──────────────┬──────────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
│ GEN7 PATH           │ │ GEN8 PATH           │ │ REAL DATA PATH      │
├─────────────────────┤ ├─────────────────────┤ ├─────────────────────┤
│                     │ │                     │ │                     │
│ Fix Gen7 code       │ │ DR10 artifact       │ │ Validate anchors    │
│ (clump_flux_frac)   │ │ calibration         │ │ (can do NOW)        │
│       │             │ │       │             │ │       │             │
│       ▼             │ │       ▼             │ │       ▼             │
│ Generate Gen7       │ │ Generate Gen8       │ │ Curate contaminants │
│ dataset (EMR)       │ │ dataset (EMR)       │ │ (can do NOW)        │
│       │             │ │       │             │ │       │             │
│       ▼             │ │       ▼             │ │       ▼             │
│ Train B1-Gen7       │ │ Train B1-Gen8       │ │ PHASE 7: Real eval  │
│       │             │ │       │             │ │ - Anchor recall     │
│       ▼             │ │       ▼             │ │ - Contaminant FPR   │
│ Compare to B1-Gen5  │ │ Compare to B1-Gen5  │ │ - Selection function│
└─────────────────────┘ └─────────────────────┘ └─────────────────────┘
              │                     │                     │
              └─────────────────────┼─────────────────────┘
                                    ▼
                     ┌─────────────────────────────────────┐
                     │   PHASE 8: Publication              │
                     │   - Figures, writing, CIs           │
                     └─────────────────────────────────────┘
```

---

### Dataset Details

#### 1. Anchor Set (Real Lenses) - ⚠️ NEEDS VALIDATION

**Purpose**: Measure sim-to-real transfer (can model find real lenses?)

**Requirements**:
- Minimum 30 real lenses (SLACS, BELLS, SL2S, etc.)
- All must have DR10 coverage
- All theta_e > 0.5"
- Columns: name, ra, dec, theta_e_arcsec, source

**Current State**: File may exist (`tier_a_anchors.csv`), needs validation

**Action**: Run `python planb/phase0_foundation/validate_anchors.py`

**Can Start**: ✅ NOW (parallel with training)

**Unblocks**: Phase 7 anchor recall evaluation

---

#### 2. Contaminant Set (Known Non-Lenses) - ❌ NOT CURATED

**Purpose**: Measure false positive rate on challenging non-lenses

**Requirements**:
- ≥50 ring galaxies
- ≥50 face-on spirals
- ≥30 mergers/interacting pairs
- No overlap with anchor set

**Current State**: NOT CREATED

**Action**: Curate from Galaxy Zoo, SDSS, or other catalogs

**Can Start**: ✅ NOW (parallel with training)

**Unblocks**: Phase 7 contaminant rejection evaluation

---

#### 3. DR10 Artifact Calibration - ❌ NOT DONE

**Purpose**: Measure real artifact rates to configure Gen8

**Measurements Needed**:
- Cosmic ray rate per stamp
- PSF anisotropy distribution
- Astrometric jitter magnitude
- Saturation spike frequency

**Current State**: NOT DONE

**Action**: Sample DR10 images and measure artifact rates

**Can Start**: ✅ NOW (parallel with training)

**Unblocks**: Gen8 dataset generation

---

#### 4. Gen7 Dataset (Unpaired + Procedural Sources) - ❌ NOT CREATED

**Purpose**: Test if procedural source realism improves sim-to-real

**Requirements**:
- Hybrid Sersic + clumps sources
- UNPAIRED LRG sampling (critical!)
- Same size as current dataset (~800K train)

**Dependencies**:
1. Fix Gen7 code (clump_flux_frac semantics)
2. Phase 5 decision (confirm we want to proceed)

**Action**: Modify EMR pipeline, run generation job

**Can Start**: ⚠️ After Gen7 code fix + Phase 5 decision

**Unblocks**: Gen7 training (B1-Gen7)

---

#### 5. Gen8 Dataset (Unpaired + Artifacts) - ❌ NOT CREATED

**Purpose**: Test if domain randomization improves robustness

**Requirements**:
- Gen5 COSMOS sources + Gen8 artifacts
- UNPAIRED LRG sampling (critical!)
- Calibrated artifact rates from DR10
- Same size as current dataset

**Dependencies**:
1. DR10 artifact calibration (see #3)
2. Phase 5 decision (confirm we want to proceed)

**Action**: Modify EMR pipeline, run generation job

**Can Start**: ⚠️ After DR10 calibration + Phase 5 decision

**Unblocks**: Gen8 training (B1-Gen8)

---

### What To Do NOW (While Training Runs)

These can be done in parallel with Phase 4 training:

| Task | Effort | Tool | Unblocks | Plan Doc |
|------|--------|------|----------|----------|
| 1. Validate anchor set | 1 hr | Python script | Phase 7 | **[ANCHOR_CONTAMINANT_PLAN.md](ANCHOR_CONTAMINANT_PLAN.md)** |
| 2. Curate contaminant set | 2-3 hrs | Manual + Python | Phase 7 | **[ANCHOR_CONTAMINANT_PLAN.md](ANCHOR_CONTAMINANT_PLAN.md)** |
| 3. DR10 artifact calibration | 2-3 hrs | Python/EMR | Gen8 dataset | - |
| 4. Fix Gen7 code (clump_flux_frac) | 1 hr | Python | Gen7 dataset | - |
| 5. Prepare results aggregation script | 30 min | Python | Phase 5 | - |

**Detailed Plan**: See [ANCHOR_CONTAMINANT_PLAN.md](ANCHOR_CONTAMINANT_PLAN.md) for step-by-step instructions.

---

## Code Assets

| Module | Location | Status |
|--------|----------|--------|
| `planb/unpaired_experiment/` | All instances | ✅ Synced |
| `planb/phase1_baseline/` | lambda | ✅ Available |
| `planb/tests/` | All instances | ✅ 50 tests pass |

---

## Documentation Structure

```
planb/
├── README.md                           # Main entry point
├── PROJECT_STATUS.md                   # This file - current status (single source of truth)
├── WORKLOG.md                          # Session log
├── ANCHOR_CONTAMINANT_PLAN.md          # Detailed plan for real data evaluation sets
├── SELECTION_FUNCTION_REVIEW_PROMPT.md # LLM review request for selection functions
│
├── evaluation/                         # Evaluation modules (NEW)
│   ├── __init__.py
│   ├── anchor_set.py                   # AnchorSet + AnchorSelectionFunction
│   └── contaminant_set.py              # ContaminantSet + ContaminantSelectionFunction
│
└── docs/
    ├── reference/                      # Contracts, exit criteria (4 files)
    └── archive/                        # Historical investigation + reviews (11 files)
```

**Total: 5 active docs at root + 3 new modules + 15 archived docs in subdirectories**

---

## Summary: What Remains

### Immediate (next 16-30 hours)
- [x] 5 full-scale experiments running (Gen5 + unpaired)
- [ ] Monitor training progress
- [ ] Results aggregation script ready

### Short-term (after Phase 4)
- [ ] Compare all 5 experiments (A1/B1/B2/B3/B4)
- [ ] Select best unpaired config (B1/B2/B3/B4)
- [ ] Final test evaluation

### Medium-term: Gen7/Gen8 with Unpaired Data
- [ ] Generate **new Gen7 dataset** with unpaired LRG sampling (EMR job)
- [ ] Train B1-Gen7: Unpaired + Residual + Gen7 procedural sources
- [ ] Generate **new Gen8 dataset** with unpaired LRG sampling + artifacts (EMR job)
- [ ] Train B1-Gen8: Unpaired + Residual + Gen8 artifacts
- [ ] Compare Gen5 vs Gen7 vs Gen8 (all on unpaired data)

### Medium-term: Real Data Evaluation
- [ ] Real data evaluation (anchors, contaminants)
- [ ] DR10 scoring pipeline

### Long-term
- [ ] Paper figures and writing
- [ ] Reproducibility (multi-seed)

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Training fails/crashes | Low | High | Checkpoints every 5 epochs |
| All B variants fail gates | Low | High | A1 as fallback, document limitation |
| Network/instance failure | Medium | Medium | NFS backup, can resume |
| Results not significant | Low | High | Report as negative result |

---

## Timeline Estimate

| Phase | Duration | Start | End |
|-------|----------|-------|-----|
| Phase 4 (full training Gen5) | 16-30 hrs | Feb 7, 5am | Feb 8, 11am |
| Phase 5 (evaluation) | 2-4 hrs | Feb 8, 11am | Feb 8, 3pm |
| Phase 6a (Gen7 data generation) | 4-8 hrs | After Phase 5 | EMR job |
| Phase 6b (Gen7 training) | 16-20 hrs | After 6a | - |
| Phase 6c (Gen8 data generation) | 4-8 hrs | Parallel with 6b | EMR job |
| Phase 6d (Gen8 training) | 16-20 hrs | After 6c | - |
| Phase 7 (real data eval) | 2-3 days | After Phase 6 | - |
| Phase 8 (publication) | 1-2 weeks | After Phase 7 | - |

**Earliest completion of Phase 4 (Gen5 unpaired): Feb 8, 2026**
**Earliest completion of Phase 6 (Gen7/Gen8): Feb 10-11, 2026**
