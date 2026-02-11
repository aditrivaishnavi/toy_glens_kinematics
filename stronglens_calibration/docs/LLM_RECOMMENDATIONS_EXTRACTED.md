# LLM Recommendations Extracted from conversation_with_llm.txt (Lines 1–3460)

Every concrete recommendation, decision, or action item, organized by topic.  
**D** = Decision (already decided) | **A** = Action (needs to be done) | **~** = approximate line range.

---

## 1. Negative Sampling (N1/N2 Pools, Ratios, Thresholds, Types)

| # | Item | Type | Lines |
|---|------|------|-------|
| 1.1 | **N1:N2 ratio = 85:15** within negatives | D | ~1049–1053 |
| 1.2 | **100:1 negative:positive** per (nobs_z, type) bin; N2 implemented as **oversampled stream** rather than matching true prevalence | D | ~1049–1061 |
| 1.3 | **Deprecate paired controls** as primary negative source; keep as diagnostic pool only | D | ~1065–1070 |
| 1.4 | **Both** automatic mining + manual curation for N2 sourcing | D | ~1073–1078 |
| 1.5 | **Tractor for high-recall mining**: edge-on = EXP/DEV + ellipticity; blue clumpy = colors + size; spiral/ring/merger proxies weak from Tractor alone | D | ~1082–1098 |
| 1.6 | **Exclusion radius**: 5 arcsec from catalog position; also exclude within **(5″ + 2×θE_max)** when θE available (~11″ for θE_max=3″) | D | ~1101–1108 |
| 1.7 | **Maskbits**: exclude bright star, saturation, bleeding trails, large contiguous masked fraction; exclude if **masked pixel fraction > 1–2%** in stamp center | A | ~1109–1120 |
| 1.8 | **Paper IV parity**: `z_mag_limit = 20.0` (not 21.0) | D | ~1880–1897 |
| 1.9 | **Paper IV parity**: N1 TYPE set = **SER, DEV, REX only** (exclude EXP for strict parity) | D | ~1868–1878, ~2004–2010 |
| 1.10 | **TYPE_BINS = ["SER", "DEV", "REX"]** for stratification (remove EXP for Paper IV parity) | D | ~2497–2501 |
| 1.11 | **5.8% N2 is acceptable**; do not distort survey to force 15%; oversample N2 at batch sampler level to reach training mix | D | ~1997–2002 |
| 1.12 | **85:15 as global constraint**, not per-stratum; sample N2 across all strata; allocate N1 per-stratum for 100:1 | D | ~2507–2517 |
| 1.13 | **16.5% N2 is acceptable**; do not try to hit exactly 15% | D | ~2877–2882 |
| 1.14 | **Edge-on proxy**: loosen ellipticity_min 0.55 → 0.50 (non-blocking); target edge_on ~2–4% of N2 | A | ~2884–2889 |
| 1.15 | **Remove bright_core** category if redundant with ring_proxy/large_galaxy | D | ~2890–2893 |
| 1.16 | **Add merger/neighbor proxy** (high value): Tractor fitbits/geometric "bright neighbor within 3–6″" | A | ~2897–2910 |
| 1.17 | **Add spiral-arm/ringlike proxy**: g-r ≤ 0.7, shape_r ≥ 0.8, optionally EXP | A | ~2911–2914 |
| 1.18 | **Add AGN/QSO host proxy**: high sersic + compact; best in cutout-validation stage | A | ~2915–2918 |
| 1.19 | **Add bright-star artifact neighborhood** category (curated pool) | A | ~2919–2922 |
| 1.20 | **N2 confuser redefs**: ring_proxy = DEV+flux; edge-on = ellipticity+size; blue_clumpy = relax to g-r ≤ 0.7, r ≤ 20.5 | D | ~1949–1962 |
| 1.21 | **Minimum exposures**: nobs_g ≥ 3 AND nobs_r ≥ 3 AND nobs_z ≥ 3 (Paper IV) | D | ~1672–1675 |
| 1.22 | **DECaLS footprint**: δ > -18° AND δ < 32° (both bounds) | D | ~1682–1695 |

---

## 2. Spatial Splits (HEALPix, Nside, Allocations)

| # | Item | Type | Lines |
|---|------|------|-------|
| 2.1 | **nside=128** for split assignment (64 acceptable but 128 safer) | D | ~1129–1133 |
| 2.2 | **Do NOT hard-stratify** split by PSF/depth; assign splits by HEALPix deterministically; then verify balance | D | ~1136–1141 |
| 2.3 | Split: **hash(healpix_index)** → train/val/test; verify no healpix cell in multiple splits | D | ~1151–1158 |
| 2.4 | **70/15/15** split (train/val/test) | D | ~490–491 |
| 2.5 | Paper IV uses **70/30** train/val (not 70/15/15) | D | ~2860 |
| 2.6 | **Split assignment ordering bug**: use explicit order `["train","val","test"]`, not `sorted(allocations.items())` | A | ~2521–2525, ~3205–3208 |
| 2.7 | **Deterministic stratified sampling**: use `row_number() over (partition by ... order by hash(galaxy_id, seed)) <= target`; NOT `.sample() + limit()` | A | ~2527–2538 |
| 2.8 | **assign_split** must iterate splits in explicit order, not sorted by name | A | ~2780–2787 |
| 2.9 | For within-stratum: `hash(galaxy_id, seed)` or `(brickname, objid)` for stable ordering | D | ~2785–2787 |

---

## 3. Label Handling (Tier Weights, Sample Weights)

| # | Item | Type | Lines |
|---|------|------|-------|
| 3.1 | **Use sample weights in loss** as primary; not only label smoothing | D | ~1190–1197 |
| 3.2 | **Literature confirmed**: label=1.0, weight=1.0 | D | ~1198–1208 |
| 3.3 | **Tier-A confident**: label=1.0, weight=0.9–1.0 | D | ~1198–1208 |
| 3.4 | **Tier-B probable**: label=1.0, weight=0.3–0.6 (depending on purity) | D | ~1198–1208 |
| 3.5 | **Mild label smoothing** (1.0 → 0.95) for Tier-B only, optionally | D | ~1198–1208 |
| 3.6 | **SLACS/BELLS**: do NOT use as training positives if "invisible in DR10"; put in evaluation/stress-test only; training only if pass arc-visibility selection | D | ~1210–1218 |
| 3.7 | **Overlap prevention**: explicit ID exclusion + spatial split; build canonical ID (ra,dec rounded 0.1″ + catalog name) | D | ~1228–1236 |
| 3.8 | **822 DESI DR1 spectroscopic matches**: treat as **confirmed positives** if genuinely spectroscopically selected/confirmed; otherwise intermediate tier with explicit criteria | D | ~2978–2980 |
| 3.9 | **Tier weighting implementation**: labels y=1/0; weights: Literature/Tier-A strong=1.0; Tier-A confident=0.9–1.0; Tier-B=0.3–0.6 | D | ~2967–2974 |
| 3.10 | Avoid strong label smoothing AND strong down-weighting at once; if label smoothing for Tier-B only: target 0.8–0.9, weight ~0.6 | D | ~2967–2974 |
| 3.11 | **Candidates (Tier-B)**: sample weight by grade/probability; label smoothing (confident=0.95, probable=0.7–0.8); exclude from evaluation anchors | D | ~868–877 |

---

## 4. Injection Realism (Flux, PSF, Noise, Acceptance Criteria)

| # | Item | Type | Lines |
|---|------|------|-------|
| 4.1 | **Photometric calibration**: draw source mags from empirical; convert to counts via DR10 ZP; apply per-band PSF + DR10 noise | A | ~262–267 |
| 4.2 | **Noise**: use empirical from cutout (inverse-variance, blank-sky annuli, or MAD) not generic Gaussian | A | ~268–273 |
| 4.3 | **PSF**: band-dependent, spatially varying; Moffat with params from DR10 if full model unavailable | A | ~274–281 |
| 4.4 | **Source population**: match color, size after seeing, surface brightness; bootstrap from faint blue DR10 galaxies or HST priors | A | ~282–290 |
| 4.5 | **Match 6–10 observable targets** between injections and real: annulus SNR, SB, color, arc width, residual amplitude, radial location | A | ~432–439 |
| 4.6 | **Inner images**: include with realistic visibility OR suppress; run both as ablation; make choice explicit | A | ~459–464 |
| 4.7 | **Arc annulus SNR**: median within **0.5×–2×** vs real; 10th and 90th percentiles within same | D | ~1239–1243 |
| 4.8 | **Color (g-r, r-z)**: median within **±0.2 mag** | D | ~1254–1256 |
| 4.9 | **Noise histogram KS test**: p-value > **0.05** for background pixels | D | ~1246–1248 |
| 4.10 | **Visual sanity**: blinded 50-image panel; injections not systematically cleaner than real | D | ~1258–1260 |
| 4.11 | **All diagnostics must pass** for GO; rule-based GO/NO-GO | D | ~1250–1261 |
| 4.12 | **Source magnitude**: r-band 22–26 (unlensed); μ ~ 5–30; annulus SNR ~0–5 majority, tail to ~10 | D | ~1231–1236 |
| 4.13 | **Source morphology**: bracket with smooth Sersic (baseline) + clumpy; COSMOS optional third | D | ~1238–1243 |
| 4.14 | **PSF**: per-cutout when possible; Moffat beta 2.5, 3.5, 4.5 if bracketing | D | ~1246–1251 |
| 4.15 | **Noise**: outer annulus, mask sources via sigma-clipping; use MAD not std | D | ~1246–1251 |
| 4.16 | **Inner images**: primary = include with realistic visibility; ablation = suppressed variant | D | ~1253–1257 |
| 4.17 | **GO if**: injected vs real diagnostics match; completeness stable across regions; contaminant FPR identifies known confusers | D | ~523–527 |
| 4.18 | **NO-GO if**: cannot match arc SB/annulus SNR without arbitrary tuning; completeness wildly sensitive to priors | D | ~528–532 |
| 4.19 | **Injection base**: negative cutouts only; no positives; no training cutouts as hosts | D | ~3365–3366 |

---

## 5. Training Protocol (Architecture, Epochs, Batch, Augmentation, Loss)

| # | Item | Type | Lines |
|---|------|------|-------|
| 5.1 | **ResNet18** as primary baseline (fast, stable, good for ablations) | D | ~1267–1271 |
| 5.2 | Add **EfficientNet-B0** only if incremental gain on independent validation | D | ~1267–1271 |
| 5.3 | **Single ResNet18** first; avoid full Paper IV ensemble initially | D | ~1273–1277 |
| 5.4 | **20–40 epochs** until plateau; state early stopping on spatial-val | D | ~1279–1283 |
| 5.5 | **Batch size**: 256 for 64×64×3; 128–192 for 101×101 on 24GB; use AMP | D | ~1285–1288 |
| 5.6 | **Safe augmentations**: rotations/flips, small translations, mild additive noise, mild PSF blur jitter | D | ~1290–1298 |
| 5.7 | **Risky**: aggressive brightness/contrast jitter (can create shortcuts); heavy normalization changes | D | ~1295–1298 |
| 5.8 | **Train on 101×101** (Paper IV parity); 64×64 as ablation | D | ~2938–2943 |
| 5.9 | **Normalization**: per-image per-band median/MAD or robust std, clip, optionally asinh | D | ~1908–1914 |
| 5.10 | **Paper IV cutouts**: 101×101 pixels | D | ~2855 |

---

## 6. Evaluation Protocol (Completeness, Calibration, Failure Modes)

| # | Item | Type | Lines |
|---|------|------|-------|
| 6.1 | **Independent evaluation sets** from spectroscopic DESI searches (single-fiber, pairwise) to break circularity | A | ~821–822 |
| 6.2 | **Calibration**: report ECE/RC curves; quantify drift across strata | A | ~823–824 |
| 6.3 | **Do NOT report ECE/MCE for single-class strata** (e.g. Tier-A only); misleading | D | ~3270 |
| 6.4 | **Overall ECE** dominated by negatives; does not certify positive-class calibration | D | ~3267–3268 |
| 6.5 | **Bootstrap intervals** (68% and/or 95%) for test AUC, test recall @ 0.5, Tier-A recall | A | ~3347+ |
| 6.6 | **Split disjointness**: verify `galaxy_id` and `npz_path` across train/val/test | A | ~3351–3355 |
| 6.7 | **"Held-out Tier-A anchors"** OK; avoid "independent spectroscopic validation" unless external catalog | D | ~3358–3359 |
| 6.8 | **Positive-focused calibration**: reliability diagram for high-score region, or calibration error restricted to bins with positive counts | A | ~3340–3345 |
| 6.9 | **FPR stratified by N2 confuser category** (important for failure-mode analysis) | A | ~895–896 |
| 6.10 | **Failure mode taxonomy**: FPR spikes when X; FN spikes when Y; reproducible definitions and counts | A | ~597–598 |
| 6.11 | **Shortcut gates**: annulus-only and core-only classifier checks (4.7, 4.8); can be "future work" for first submission | D | ~3328 |
| 6.12 | **Tier-A recall @ 0.5 with bootstrap** for n=48 | A | ~3347+ |

---

## 7. Selection Function Grid (θE, PSF, Depth Bins, Injections per Cell)

| # | Item | Type | Lines |
|---|------|------|-------|
| 7.1 | **θE**: 0.5–3.0″ in **0.25″** steps → 11 bins | D | ~1299–1304 |
| 7.2 | **PSF FWHM**: 0.9–1.8″ in **0.15″** steps → 7 bins | D | ~1299–1304 |
| 7.3 | **Depth** (psfdepth_r or galdepth_r): 22.5–24.5 in **0.5 mag** steps → 5 bins | D | ~1299–1304 |
| 7.4 | **Optional fourth axis**: host type (Tractor TYPE bins) | D | ~1304 |
| 7.5 | **Minimum 200 injections per cell** for stable binomial error bars | D | ~1310–1312 |
| 7.6 | **Sparse cells**: mark "insufficient" below Nmin; merge adjacent bins; do not smooth unless justified | D | ~1314–1318 |
| 7.7 | **Uncertainty**: Bayesian binomial (Beta posterior) per cell; 68% intervals (95% in appendix) | D | ~1320–1324 |
| 7.8 | **Injections**: after training only (selection-function measurement); optionally stress test on injected val set | D | ~2985–2991 |
| 7.9 | **Total cells**: 11 × 7 × 5 = 385; ≥77,000 injections total | D | ~3340–3342 |
| 7.10 | **nobs_z bins**: {1,2}, {3,5}, {6,10}, {11+} or similar; show sensitivity to binning | D | ~904–908 |

---

## 8. Paper Deliverables and Claims

| # | Item | Type | Lines |
|---|------|------|-------|
| 8.1 | **Minimum viable novelty**: (1) quantitative completeness map C(θE, PSF, depth, host type) with uncertainty; (2) failure-mode taxonomy tied to covariates; (3) reproducible audit protocol + public lookup/code | D | ~565–573 |
| 8.2 | **Positioning**: "They build samples and confirm lenses; we quantify what a given detector would select and miss, and why" | D | ~582–589 |
| 8.3 | **Best framing**: "Where do ML lens finders fail, and why?" | D | ~804–807 |
| 8.4 | **Must-have figures**: (1) data/split schematic; (2) score distributions by stratum; (3) selection function heatmaps C(θE, PSF); (4) failure mode gallery with counts; (5) independent validation performance table | D | ~1341–1347 |
| 8.5 | **Claims to AVOID**: overall precision in survey; cosmology constraints; "complete" lens sample; outperforming Huang without matched protocol | D | ~1349–1353 |
| 8.6 | **One-sentence novelty**: "We provide a detector-audit framework for DR10 strong-lens searches, including injection-calibrated completeness surfaces and a condition- and confuser-resolved false-positive taxonomy, enabling bias-aware use of ML lens catalogs." | D | ~1355–1357 |
| 8.7 | **Journal**: MNRAS reasonable for methods+selection bias; ApJ plausible; optimize correctness first | D | ~1359–1361 |
| 8.8 | **Mistakes section**: state plainly uncalibrated initial injections; quantitative realism gates; prior sensitivity; no claim outside validated envelope | D | ~519–524 |
| 8.9 | **Output**: release code, selection-function tables, evaluation catalogs, QA plots; limitations, prior sensitivity, circularity mitigation | D | ~515–517 |

---

## 9. GO/NO-GO Criteria

| # | Item | Type | Lines |
|---|------|------|-------|
| 9.1 | **GO if**: (1) injected vs real diagnostics match within tolerances; (2) completeness stable across sky regions; (3) contaminant FPR identifies known confusers | D | ~523–527 |
| 9.2 | **NO-GO if**: (1) cannot match arc SB and annulus SNR without arbitrary tuning; (2) completeness wildly sensitive to injection choices | D | ~528–532 |
| 9.3 | **GO for EMR negative sampling** if: δ bounds, nobs_g/r/z≥3, z<20 enforced; exclusion working | D | ~3198–3201 |
| 9.4 | **Blocking before EMR**: (1) split assignment ordering bug; (2) TYPE bin policy (SER/DEV/REX for parity) | A | ~3203–3210 |
| 9.5 | **Non-blocking**: edge-on underrepresentation; merger/neighbor proxy in next iteration | D | ~3212–3215 |
| 9.6 | **Post-run validation**: filter parity counters; per-stratum availability; exclusion sanity; distribution sanity vs positives | A | ~3217–3221 |
| 9.7 | **Injection GO/NO-GO**: all diagnostics (SNR, color, noise KS, visual) must pass | D | ~1250–1261 |

---

## 10. Schema, Cutouts, and Implementation Details

| # | Item | Type | Lines |
|---|------|------|-------|
| 10.1 | **Cutout size**: 101×101 for training/eval (Paper IV); 64×64 as center-crop ablation | D | ~1313, ~2938–2943 |
| 10.2 | **Require g,r,z** for training; exclude if missing band; for first pass exclude missing-band cutouts | D | ~1328–1334 |
| 10.3 | **NaN in one band**: exclude if central; if outer rim only, can keep but track | D | ~1336–1338 |
| 10.4 | **Schema additions**: (1) cutout URL, timestamp, layer version; (2) injection metadata (source mag, θE, SNR, etc.); (3) mask fraction, bright star distance; (4) healpix multi-nside, region_id | A | ~944–960 |
| 10.5 | **Annulus for arc detection**: r = 4–16 px (1.0″–4.2″) primary; 20–40 px wrong for θE 0.5–3″ | D | ~2595–2605 |
| 10.6 | **Core radii for shortcuts**: compute at r=4, 8, 12 px | D | ~2587–2592 |
| 10.7 | **Shortcut AUC**: red flag >0.70; yellow >0.60; bootstrap CI; flag if lower CI > 0.60 (yellow) or > 0.70 (red) | D | ~2607–2613 |
| 10.8 | **Shortcut features to add**: Laplacian/DoG energy, azimuthal asymmetry, arc-annulus color, saturation fraction, edge artifact score | A | ~2615–2625 |
| 10.9 | **2% NaN in central 50×50** OK; add stricter core NaN (0 or <0.1%); track bandwise NaN | D | ~2575–2581 |
| 10.10 | **Band order**: verify via provenance/header + unit test, not brightness heuristics | D | ~2583–2588 |
| 10.11 | **Normalization validation**: store per-band p1, p50, p99, MAD, clip fraction, outer-annulus background | A | ~2627–2634 |
| 10.12 | **run_info.json** next to best.pt: commit hash, config path, command | A | ~3325 |
| 10.13 | **requirements.txt** or conda env with pinned versions for reproducibility | A | ~3294 |
| 10.14 | **Path override** (env var or --data_root) for portability | A | ~3312 |
| 10.15 | **Preprocessing consistency**: unit test or checksum on reference cutout | A | ~3308 |

---

## 11. Miscellaneous Decisions and Actions

| # | Item | Type | Lines |
|---|------|------|-------|
| 11.1 | **Option 1 (hybrid)** as core paper: real-image detector + calibrated injection selection function | D | ~382–383 |
| 11.2 | **Phase 4c** = selection-function engine and stress-test tool, NOT primary training source | D | ~970–979 |
| 11.3 | **Candidates**: use only for image-domain realism (marginal distributions), not as positive labels for calibration | D | ~281–289 |
| 11.4 | **Paper IV nobs_z**: simple bins {1,2,3,4+}; show sensitivity to modest binning; report robustness | D | ~904–908 |
| 11.5 | **Per-cutout PSF** preferred over brick-average; quantify error if fallback | D | ~921–925 |
| 11.6 | **Paper IV parity report**: starting count, per-filter counts, final per-stratum counts | A | ~1727–1732 |
| 11.7 | **Paper IV alignment**: match broad source categories; do not claim exact replication of 1,372 | D | ~1221–1225 |
| 11.8 | **Stratum shortfall**: record explicitly; backfill from adjacent strata (same TYPE, neighboring nobs_z) rather than silent change | D | ~2491–2495 |
| 11.9 | **Critical path**: train baseline (1–2 weeks); independent eval set; injection realism diagnostics | D | ~960–968 |
| 11.10 | **Cut for scope**: full ensemble meta-learner; full-survey inference | D | ~966–967 |
| 11.11 | **Keep for scope**: real baseline, strict spatial split, independent validation, validated injection selection function, failure taxonomy | D | ~1600–1605 |
