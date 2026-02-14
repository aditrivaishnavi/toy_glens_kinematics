# Actionable Recommendations Inventory
## Extracted from second_conversation_with_llm.txt (LLM1 + LLM2)

---

## Per-Question Inventory

### Q1.1 — Is the annulus bug actually impactful for median/MAD normalization?

| Field | Value |
|-------|-------|
| LLM1 Verdict | CONCERN |
| LLM2 Verdict | CONCERN |
| Concrete Action Item | No |
| Action (if yes) | — |
| Category | — |

---

### Q1.2 — For real DR10 LRGs with R_e ~ 4–12 px, would sky median still be −2.6?

| Field | Value |
|-------|-------|
| LLM1 Verdict | CONCERN |
| LLM2 Verdict | CONCERN |
| Concrete Action Item | Yes |
| Action | **LLM2:** "Test with a de Vaucouleurs profile (n=4) at R_e = 4, 8, and 12 px to bound the real impact." |
| Category | Test improvement |

---

### Q1.3 — Is the formula r_in = 0.65R, r_out = 0.90R principled or arbitrary?

| Field | Value |
|-------|-------|
| LLM1 Verdict | CONCERN |
| LLM2 Verdict | CONCERN |
| Concrete Action Item | Yes |
| Action | **LLM1:** "measure host-light contamination in candidate annuli on a representative host set and pick radii where contamination is acceptably small, then freeze those radii in config." **LLM2:** "For the MNRAS paper: The fixed formula is acceptable if you document the assumption (galaxy light negligible at >65% of stamp half-width) and show it holds for the actual host distribution." |
| Category | Documentation fix / paper item |

---

### Q1.4 — Will the logger.warning() flood logs during training?

| Field | Value |
|-------|-------|
| LLM1 Verdict | FAIL (with [NEW BUG]) |
| LLM2 Verdict | FAIL |
| Concrete Action Item | Yes |
| Action | **LLM1:** "warn once (module-level static flag), or make it `logger.debug`, or only warn if the caller did not explicitly pass annulus radii." **LLM2:** "Either (a) emit the warning once at the start of training using a module-level flag, or (b) use logger.warning() only on the first call (add a `_warned_annulus = False` sentinel), or (c) remove the warning entirely since the KNOWN ISSUE comment already documents it." **LLM2 NEW BUG:** "use a module-level `_warned` flag or downgrade to `logger.debug()`." |
| Category | Code fix |

---

### Q1.5 — Verify no code path expects the None-accepting signature

| Field | Value |
|-------|-------|
| LLM1 Verdict | PASS |
| LLM2 Verdict | PASS |
| Concrete Action Item | No |
| Action | — |
| Category | — |

---

### Q1.6 — What happens when r_in > r_out? (partial annulus specification)

| Field | Value |
|-------|-------|
| LLM1 Verdict | FAIL (with [NEW BUG]) |
| LLM2 Verdict | FAIL |
| Concrete Action Item | Yes |
| Action | **LLM1:** "in preprocess_stack, enforce (annulus_r_in is None) == (annulus_r_out is None) or raise ValueError. Also enforce r_in < r_out and a minimum pixel count in the annulus." **LLM2:** "Add validation in preprocess_stack or normalize_outer_annulus: `assert r_in < r_out, f\"r_in ({r_in}) must be < r_out ({r_out})\"`." **LLM2 NEW BUG:** "add `assert r_in < r_out` at the top of `normalize_outer_annulus`." |
| Category | Code fix |

---

### Q1.7 — Python 3.10+ compatibility for float \| None

| Field | Value |
|-------|-------|
| LLM1 Verdict | PASS |
| LLM2 Verdict | PASS |
| Concrete Action Item | No |
| Action | — |
| Category | — |

---

### Q1.8 — Sentinel 0.0 for "use default" — is it clearly documented?

| Field | Value |
|-------|-------|
| LLM1 Verdict | CONCERN |
| LLM2 Verdict | CONCERN |
| Concrete Action Item | Yes |
| Action | **LLM1:** "explicitly documented in config comments to avoid silent intent mismatch. Cleaner: use None instead of 0.0 and serialize null in YAML." **LLM2:** "could be clearer with an explicit comment in the YAML template." |
| Category | Documentation fix / Code fix |

---

### Q1.9 — v5 YAML dataset keys match DatasetConfig fields exactly?

| Field | Value |
|-------|-------|
| LLM1 Verdict | PASS |
| LLM2 Verdict | PASS |
| Concrete Action Item | No |
| Action | — |
| Category | — |

---

### Q1.10 — Old configs (v1–v4) still work with new DatasetConfig?

| Field | Value |
|-------|-------|
| LLM1 Verdict | PASS |
| LLM2 Verdict | PASS |
| Concrete Action Item | No |
| Action | — |
| Category | — |

---

### Q1.11 — Verify full forwarding chain YAML → normalize_outer_annulus

| Field | Value |
|-------|-------|
| LLM1 Verdict | PASS |
| LLM2 Verdict | PASS |
| Concrete Action Item | No |
| Action | — |
| Category | — |

---

### Q1.12 — Should test_preprocessing_outer_sky_near_zero have an assertion?

| Field | Value |
|-------|-------|
| LLM1 Verdict | FAIL |
| LLM2 Verdict | CONCERN |
| Concrete Action Item | Yes |
| Action | **LLM1:** "If you expect 'near zero', assert a tolerance. If you expect a trained-distribution offset (like −2.6), assert that instead (with tolerance) and explicitly justify why that is the desired invariant." **LLM2:** "once retraining with (32.5, 45.0) is complete, this test should be updated with an assertion like abs(sky_median) < 0.5. Add a commented-out assertion with a TODO marker: # TODO: after retraining, assert abs(sky_median) < 0.5." |
| Category | Test improvement |

---

### Q1.13 — How representative is the synthetic exp galaxy (scale=8 px)?

| Field | Value |
|-------|-------|
| LLM1 Verdict | CONCERN |
| LLM2 Verdict | CONCERN |
| Concrete Action Item | Yes |
| Action | **LLM2:** "The test is valuable as a regression guard and documentation tool, but the specific numbers should not be cited in the paper as representative." |
| Category | Documentation fix / paper item |

---

### Q1.14 — Is the checksum cross-platform stable?

| Field | Value |
|-------|-------|
| LLM1 Verdict | CONCERN |
| LLM2 Verdict | CONCERN |
| Concrete Action Item | Yes |
| Action | **LLM1:** "Prefer: numeric tolerances on summary statistics, or hashing after quantization (e.g., round to 1e-5), or storing a small golden array and using np.allclose with defined tolerances." **LLM2:** "Test on the actual Lambda GH200 hardware before locking the checksum. If it fails, increase the tolerance (round to 5 decimals) or use a norm-based invariant instead (np.allclose with rtol=1e-5)." |
| Category | Test improvement |

---

### Q1.15 — AST parsing and computed expressions

| Field | Value |
|-------|-------|
| LLM1 Verdict | CONCERN |
| LLM2 Verdict | CONCERN |
| Concrete Action Item | Yes |
| Action | **LLM1:** "if _SENTINEL is returned for any default, fail the test with a message 'complex defaults are not supported by the validator'." **LLM2:** "Add a test that asserts the set of extracted defaults matches the expected set, so a missing key causes a failure: self.assertEqual(set(self.defaults.keys()), expected_keys)." |
| Category | Test improvement / Code fix |

---

### Q1.16 — clumps_n_range and clumps_frac_range in YAML not validated

| Field | Value |
|-------|-------|
| LLM1 Verdict | FAIL (with [NEW BUG]) |
| LLM2 Verdict | FAIL |
| Concrete Action Item | Yes |
| Action | **LLM1:** "add clumps_n_range and clumps_frac_range as explicit parameters to sample_source_params (and include them in the test registry match)." "The YAML values are not used — the code hard-codes them." **LLM2:** "Either (a) make these function parameter defaults so AST parsing can validate them, or (b) add a separate test that imports the module, calls sample_source_params with a controlled RNG, and checks clump ranges." **LLM1:** "Make injection_priors.yaml reflect actual code-used parameters (especially clumps), and extend tests to validate them." |
| Category | Code fix / Test improvement |

---

### Q1.17 — g-r color: Gaussian vs Uniform discrepancy

| Field | Value |
|-------|-------|
| LLM1 Verdict | CONCERN |
| LLM2 Verdict | PASS |
| Concrete Action Item | Yes |
| Action | **LLM1:** "Any older documentation claiming Uniform must be corrected." **LLM2:** "Ensure MNRAS paper describes the prior as g−r ~ N(0.2, 0.25), not uniform." |
| Category | Documentation fix / paper item |

---

### Q1.18 — Are blank-host detection rates meaningful?

| Field | Value |
|-------|-------|
| LLM1 Verdict | CONCERN |
| LLM2 Verdict | CONCERN |
| Concrete Action Item | Yes |
| Action | **LLM2:** "Note in the script output that these are upper bounds. The script's docstring already says this ('unrealistic')." |
| Category | Documentation fix |

---

### Q1.19 — Does SIS approximation hold for SIE?

| Field | Value |
|-------|-------|
| LLM1 Verdict | CONCERN |
| LLM2 Verdict | CONCERN |
| Concrete Action Item | No |
| Action | — |
| Category | — |

---

### Q1.20 — How to distinguish geometry ceiling vs coincidence?

| Field | Value |
|-------|-------|
| LLM1 Verdict | PASS |
| LLM2 Verdict | CONCERN |
| Concrete Action Item | Yes |
| Action | **LLM1:** "Hold source brightness fixed and sample beta_frac in bins (or fix beta_frac deterministically). Measure detection probability vs beta_frac and compare to the predicted multi-image fraction under the same (q, γ) prior (compute caustic area numerically for your lens model), not SIS. Separately rerun with much larger clip_range to rule out clipping artifacts as the 'ceiling'." **LLM2:** "The beta_frac diagnostic script should output P(detected | beta_frac) in bins, not just a binary threshold comparison." |
| Category | Code fix / Test improvement |

---

### Q1.21 — Why not finetune from v4?

| Field | Value |
|-------|-------|
| LLM1 Verdict | CONCERN |
| LLM2 Verdict | CONCERN |
| Concrete Action Item | Yes |
| Action | **LLM1:** "Start from v4 weights, swap preprocessing, finetune with low LR for 10–30 epochs. If it fails to recover, then consider scratch training." **LLM2:** "Run both: (a) v5 from scratch, and (b) v5-ft finetuning from v4 with the v4 recipe (60 epochs cosine, lr=5e-5). Compare convergence and final AUC." |
| Category | Deferred / paper item / training config |

---

### Q1.22 — v5 missing freeze_backbone_epochs and warmup_epochs

| Field | Value |
|-------|-------|
| LLM1 Verdict | FAIL |
| LLM2 Verdict | CONCERN (LLM2 also labels [NEW BUG #1]) |
| Concrete Action Item | Yes |
| Action | **LLM1:** "If you want a controlled experiment isolating annulus effects, you should match v2/v4 training protocol." **LLM2:** "v5 should explicitly include freeze_backbone_epochs: 5 and warmup_epochs: 5 to match the v2 recipe that led to the best results." **LLM2 NEW BUG:** "add freeze_backbone_epochs: 5 and warmup_epochs: 5 to the v5 config." |
| Category | Code fix (config) |

---

### Q1.23 — Should v5 do two-phase training?

| Field | Value |
|-------|-------|
| LLM1 Verdict | CONCERN |
| LLM2 Verdict | CONCERN |
| Concrete Action Item | Yes |
| Action | **LLM1:** "either replicate the same two-phase protocol for v5, or at minimum, evaluate v5 at the comparable 'end-of-phase-1' point and then optionally finetune." **LLM2:** "Plan for v5 (phase 1) + v6 (phase 2 finetune from v5 best)." |
| Category | Deferred / paper item |

---

### Q2.1 — Float16 precision with gradient accumulation

| Field | Value |
|-------|-------|
| LLM1 Verdict | PASS |
| LLM2 Verdict | PASS |
| Concrete Action Item | No |
| Action | — |
| Category | — |

---

### Q2.2 — Does scoring disable augmentation?

| Field | Value |
|-------|-------|
| LLM1 Verdict | PASS |
| LLM2 Verdict | PASS |
| Concrete Action Item | No |
| Action | — |
| Category | — |

---

### Q3.1 — Scoring scripts don't pass annulus kwargs

| Field | Value |
|-------|-------|
| LLM1 Verdict | FAIL |
| LLM2 Verdict | FAIL |
| Concrete Action Item | Yes |
| Action | **LLM1:** "store preprocessing in the checkpoint and force scoring to use it." Provides code sketch for PreprocessSpec, preprocess_stack, train.py checkpoint saving, and dhs/scoring.py load_model_and_spec. "This eliminates the entire class of 'scripts forgot an argument' failures." **LLM2:** "store annulus in checkpoint metadata" — provides code sketch for train.py checkpoint format, dhs/scoring_utils.py load_model_and_preprocess_config, and usage in scoring scripts. |
| Category | Code fix (architecture) |

---

### Q3.2 — HWC/CHW format consistency

| Field | Value |
|-------|-------|
| LLM1 Verdict | PASS |
| LLM2 Verdict | PASS |
| Concrete Action Item | Yes (minor) |
| Action | **LLM2:** "Consistent, but worth documenting." |
| Category | Documentation fix |

---

### Q3.3 — Training vs scoring preprocess_stack kwargs identical?

| Field | Value |
|-------|-------|
| LLM1 Verdict | CONCERN |
| LLM2 Verdict | CONCERN |
| Concrete Action Item | Yes |
| Action | **LLM1:** "This is another reason to checkpoint the preprocessing spec and always load it." (Same fix as Q3.1) **LLM2:** "kwargs are functionally identical for current setup. However, the scoring scripts don't pass annulus kwargs (the Q3.1 bug)." |
| Category | Code fix (covered by Q3.1) |

---

### Q3.4 — Checksum / float32 cross-platform bit-identity

| Field | Value |
|-------|-------|
| LLM1 Verdict | CONCERN |
| LLM2 Verdict | CONCERN |
| Concrete Action Item | Yes |
| Action | **LLM1:** Same as Q1.14 — prefer tolerance-based tests. **LLM2:** "Test on Lambda GH200 before locking checksum. Consider a tolerance-based check as a fallback." |
| Category | Test improvement |

---

### Q3.5 — Mixed precision training vs float32 inference

| Field | Value |
|-------|-------|
| LLM1 Verdict | PASS |
| LLM2 Verdict | PASS |
| Concrete Action Item | No |
| Action | — |
| Category | — |

---

### Q3.6 — Manifest absolute paths and data integrity

| Field | Value |
|-------|-------|
| LLM1 Verdict | CONCERN |
| LLM2 Verdict | CONCERN |
| Concrete Action Item | Yes |
| Action | **LLM1:** "add a cutout hash or dataset version id at manifest build time." **LLM2:** "Add a SHA256 column to the manifest for at least a random 1% of cutouts, and verify at training startup." |
| Category | Code fix |

---

### Q3.7 — Failed injection handling in selection function grid

| Field | Value |
|-------|-------|
| LLM1 Verdict | PASS |
| LLM2 Verdict | PASS |
| Concrete Action Item | Yes |
| Action | **LLM1:** "you should report n_failed per cell in outputs" — verify it's already done. |
| Category | Test improvement / Documentation |

---

### Q4.1 — Audit dhs/preprocess.py

| Field | Value |
|-------|-------|
| LLM1 Verdict | CONCERN |
| LLM2 Verdict | PASS with CONCERNS |
| Concrete Action Item | Yes |
| Action | **LLM1:** "Centralize via checkpointed spec." (Q3.1 fix). "Major edge-case bug: partial annulus specification" (Q1.6 fix). **LLM2:** "Low risk" for arc contamination with corrected annulus — no new action beyond Q1.6/Q3.1. |
| Category | Code fix (covered by Q1.6, Q3.1) |

---

### Q4.2 — Audit dhs/model.py

| Field | Value |
|-------|-------|
| LLM1 Verdict | PASS |
| LLM2 Verdict | PASS |
| Concrete Action Item | Yes (minor) |
| Action | **LLM2:** "build_efficientnet_v2_s accepts in_ch != 3 but only replaces the first conv — pretrained weights for the first conv are lost when in_ch != 3. Not a bug since in_ch=3 is always used, but worth noting." |
| Category | Documentation fix |

---

### Q4.3 — Audit dhs/data.py

| Field | Value |
|-------|-------|
| LLM1 Verdict | CONCERN |
| LLM2 Verdict | PASS with CONCERN |
| Concrete Action Item | Yes |
| Action | **LLM1:** Training vs scoring mismatch — covered by Q3.1. **LLM2:** "The _get_file_and_label method uses row.get(self.dcfg.sample_weight_col, 1.0) — if sample_weight_col is None, it attempts row.get(None, 1.0) which works but is semantically odd." |
| Category | Code fix (minor) / Documentation |

---

### Q4.4 — Audit dhs/transforms.py

| Field | Value |
|-------|-------|
| LLM1 Verdict | PASS |
| LLM2 Verdict | PASS |
| Concrete Action Item | No |
| Action | — |
| Category | — |

---

### Q4.5 — Audit injection_model_2/host_matching.py

| Field | Value |
|-------|-------|
| LLM1 Verdict | PASS |
| LLM2 Verdict | PASS |
| Concrete Action Item | Yes (minor) |
| Action | **LLM2:** "The PA convention (atan2(vy, vx)) gives angle from x-axis. The docstring says 'E of N convention' but the code measures from x-axis (E of E?). This may cause a ~π/2 PA offset relative to standard astronomical convention. However, since the CNN is trained on these same stamps, this is consistent within the pipeline." |
| Category | Documentation fix / paper item |

---

### Q4.6 — Audit selection_function_grid_v2.py

| Field | Value |
|-------|-------|
| LLM1 Verdict | CONCERN |
| LLM2 Verdict | PASS with KNOWN GAP |
| Concrete Action Item | Yes |
| Action | **LLM1:** Preprocessing parameter drift — Q3.1. **LLM2:** "The nearest_bin function snaps continuous values to the nearest grid center. Hosts at boundary between two bins could be assigned to either. For a fine grid, this could cause some hosts to land in unexpected bins. Not a bug, but could be improved with proper bin edges." |
| Category | Code fix (minor, optional) |

---

### Q4.7 — Audit dhs/injection_engine.py

| Field | Value |
|-------|-------|
| LLM1 Verdict | PASS |
| LLM2 Verdict | PASS |
| Concrete Action Item | Yes |
| Action | **LLM1:** "b_n ≈ 2n − 1/3 is an approximation; acceptable, but document it." |
| Category | Documentation fix |

---

### Q5.1 — Preprocessing identity: real vs injected cutouts

| Field | Value |
|-------|-------|
| LLM1 Verdict | CONCERN |
| LLM2 Verdict | PASS |
| Concrete Action Item | No (LLM2 says no bias) |
| Action | — |
| Category | — |

---

### Q5.2 — SIE deflection vs Kormann et al. 1994

| Field | Value |
|-------|-------|
| LLM1 Verdict | PASS |
| LLM2 Verdict | PASS |
| Concrete Action Item | No |
| Action | — |
| Category | — |

---

### Q5.3 — Sersic normalization vs Graham & Driver 2005

| Field | Value |
|-------|-------|
| LLM1 Verdict | PASS |
| LLM2 Verdict | PASS |
| Concrete Action Item | No |
| Action | — |
| Category | — |

---

### Q5.4 — Area-weighted sampling beta_frac = sqrt(uniform(...))

| Field | Value |
|-------|-------|
| LLM1 Verdict | PASS |
| LLM2 Verdict | PASS |
| Concrete Action Item | No |
| Action | — |
| Category | — |

---

### Q5.5 — Augmentation leakage into inference/scoring

| Field | Value |
|-------|-------|
| LLM1 Verdict | PASS |
| LLM2 Verdict | PASS |
| Concrete Action Item | No |
| Action | — |
| Category | — |

---

### Q5.6 — Bright-arc 30% ceiling: preprocessing artifact?

| Field | Value |
|-------|-------|
| LLM1 Verdict | CONCERN |
| LLM2 Verdict | CONCERN |
| Concrete Action Item | Yes |
| Action | **LLM1:** "rerun bright-arc test with clip_range=50 and compare." "Geometry explanation remains strong: beta_frac diagnostic should be done under the actual SIE+shear priors and evaluated as a curve vs beta_frac, not a single threshold." **LLM2:** "The 30% ceiling is partially explained by preprocessing clipping (saturating bright arcs to ±10), partially by β_frac geometry." Suggests testing with larger clip_range. |
| Category | Test improvement / Deferred |

---

### Q5.7 — Flux conservation through preprocessing

| Field | Value |
|-------|-------|
| LLM1 Verdict | CONCERN |
| LLM2 Verdict | CONCERN |
| Concrete Action Item | No |
| Action | Both clarify that preprocessing is not flux-conserving by design; this is expected. No fix needed. |
| Category | — |

---

## [NEW BUG] Labels Summary

| ID | Location | Description |
|----|----------|--------------|
| NEW BUG (LLM1) | Q1.4 | logger.warning() fires every call for 101×101 — will flood logs |
| NEW BUG (LLM1) | Q1.6 | preprocess_stack allows partial annulus specification → empty mask → saturated garbage |
| NEW BUG (LLM1) | Q1.16 | YAML clump priors are ignored; code hard-codes clump ranges |
| NEW BUG (LLM2) | Q1.22 | v5 config silently omits freeze_backbone_epochs and warmup_epochs |
| NEW BUG (LLM2) | Q1.6 | No r_in < r_out validation |
| NEW BUG (LLM2) | Q1.4 | Warning log flood |

---

## Additional Recommendations (Summary / "Stop the Bleeding" Sections)

### LLM1 — "Stop the Bleeding" Patch Set (4 items)

1. **Add preprocessing spec to checkpoints** and require scoring to use it (code sketch provided).
2. **In preprocess_stack**, enforce both annulus radii set together and r_in < r_out.
3. **Remove or warn-once** the normalize_outer_annulus warning.
4. **Make injection_priors.yaml reflect actual code-used parameters** (especially clumps), and extend tests to validate them.

### LLM2 — Additional Bugs Section (3 items)

1. **[NEW BUG #1] — v5 config missing freeze/warmup settings (Q1.22)**  
   Fix: add `freeze_backbone_epochs: 5` and `warmup_epochs: 5` to the v5 config.

2. **[NEW BUG #2] — No r_in < r_out validation (Q1.6)**  
   Fix: add `assert r_in < r_out` at the top of `normalize_outer_annulus`.

3. **[NEW BUG #3] — Warning log flood (Q1.4)**  
   Fix: use a module-level `_warned` flag or downgrade to `logger.debug()`.

### LLM1 Top 10 — Cross-Reference

- Issues 1–10 map to Q3.1, Q1.6, Q1.4, Q1.16, Q1.15, Q1.22, Q1.12, Q1.14/Q3.4, Q3.3, Q3.6.

### LLM2 Top 10 — Cross-Reference

- Issues 1–10 map to Q3.1, Q1.22, Q1.6, Q1.4, Q1.16, Q1.14/Q3.4, Q5.6a, Q1.21/Q1.23, Q1.15, Q3.6.

---

## Categorized Action Summary

### Code Fixes
- Q1.4: Fix warning log flood (warn once or debug)
- Q1.6: Enforce r_in < r_out, both annulus params set together
- Q1.16: Add clumps_n_range and clumps_frac_range to sample_source_params; wire from YAML
- Q1.22: Add freeze_backbone_epochs: 5, warmup_epochs: 5 to v5 config
- Q3.1: Store preprocessing spec in checkpoint; scoring loads and uses it
- Q3.6: Add cutout hash/SHA256 or version tag to manifest

### Test Improvements
- Q1.2: Test with de Vaucouleurs at R_e = 4, 8, 12 px
- Q1.12: Add assertion to test_preprocessing_outer_sky_near_zero (after retrain)
- Q1.14/Q3.4: Replace bitwise checksum with tolerance-based test or np.allclose
- Q1.15: Fail test when _SENTINEL returned; add expected-keys assertion
- Q1.16: Add test for clump ranges (AST or runtime)
- Q1.20: Output P(detected | beta_frac) in bins in diagnostic script
- Q5.6: Rerun bright-arc test with clip_range=50; evaluate as curve vs beta_frac

### Documentation Fixes
- Q1.3: Document annulus assumption for MNRAS paper
- Q1.8: Document sentinel 0.0; consider explicit YAML comment
- Q1.13: Don't cite test numbers as representative in paper
- Q1.17: Correct documentation to Gaussian N(0.2, 0.25)
- Q1.18: Note blank-host rates are upper bounds in script output
- Q3.2: Document HWC/CHW conversions
- Q4.2: Note in_ch != 3 behavior for build_efficientnet_v2_s
- Q4.5: Document PA convention (x-axis vs E of N)
- Q4.7: Document b_n approximation

### Deferred / Paper Items
- Q1.21: Try finetuning from v4 before scratch training
- Q1.23: Plan v5 + v6 two-phase training
- Q1.3: Measure contamination and pick radii for representative host set
