# LLM Review Prompt Sequence — Instructions

**Date:** 2026-02-13
**Total prompts:** 4 (ask in order, each session is independent)

## Sequencing Strategy

| # | File | Focus | Why first/next |
|---|------|-------|----------------|
| 1 | `PROMPT_1_CODE_AUDIT.md` | Code changes + pipeline integrity | If code is broken, nothing else matters. Must validate BEFORE retraining. |
| 2 | `PROMPT_2_INJECTION_PHYSICS.md` | 30% ceiling + injection physics + training evaluation | If the injection APPROACH is flawed, retraining won't help. Must diagnose before committing GPU time. |
| 3 | `PROMPT_3_RETRAIN_DECISION.md` | Annulus decision + data pipeline + hostile referee prep | Armed with Prompt 1+2 answers, decide the plan. |
| 4 | `PROMPT_4_ROADMAP_AND_CODE.md` | Honest assessment + next steps + **FULLY FIXED CODE** | The deliverable prompt. Asks for complete, runnable code. |

## How to Use

1. **Each prompt is self-contained.** It has its own project context section, so a fresh LLM session has enough background. You don't need to paste prior answers.

2. **Attach the zip to EVERY prompt.** Each prompt references `stronglens_calibration_for_review_20260213.zip`. Attach it to every session.

3. **Feed answers forward (optional but recommended).** If Prompt 1 reveals a bug, mention it when pasting Prompt 2. Example: "The reviewer in session 1 found that [X]. Please factor this into your analysis."

4. **Prompt 4 asks for code.** The LLM should provide complete, fixed files — not patches, not pseudocode. If the LLM cannot provide downloadable links, ask it to output the full file contents in code blocks.

## What's in the Zip

`stronglens_calibration_for_review_20260213.zip` contains:
- `CHANGES_FOR_LLM_REVIEW.md` — detailed changelog for the 2026-02-13 session
- Complete `dhs/` package (model, preprocess, data, train, injection_engine, etc.)
- Complete `configs/` (all YAML training configs + injection_priors.yaml)
- Complete `tests/` (preprocessing regression + injection priors)
- Complete `injection_model_1/` and `injection_model_2/`
- Complete `sim_to_real_validations/`
- Complete `scripts/`, `docs/`, `common/`

## Question Count Per Prompt

| Prompt | Lines | Questions | Focus |
|--------|-------|-----------|-------|
| 1 | 296 | 46 | Code changes, pipeline consistency, full code audit, verification requests |
| 2 | 245 | 30 | 30% ceiling, injection physics, CNN behavior, training evaluation |
| 3 | 278 | 40 | Retraining decision, cheap experiments, data pipeline, hostile referee, publishability, literature |
| 4 | 236 | 15 + 10 code deliverables | Honest assessment, next steps (Model 2.5/B/C/D), **FULLY FIXED CODE** |
| **Total** | **1,055** | **131 + 10 code deliverables** | — |

## Cross-Reference: Every Original Question Is Covered

All 23 questions (Q1-Q23) + 7 verification requests from the comprehensive prompt,
and all ~80 questions from the supplementary prompt, are allocated across the 4
prompts. Nothing is skipped.

| Original Source | Covered In |
|----------------|------------|
| Comprehensive Q1-Q3 (LLM's failed prediction) | Prompt 2, Section 1 |
| Comprehensive Q4-Q5 (30% ceiling) | Prompt 2, Section 2 |
| Comprehensive Q6-Q10 (injection physics) | Prompt 2, Section 3 |
| Comprehensive Q11-Q14 (Model 2.5/B/C/D) | Prompt 4, Section 2 |
| Comprehensive Q15, Q17 (CNN behavior) | Prompt 2, Section 5 |
| Comprehensive Q16 (preprocessing identity) | Prompt 1, Section 5 |
| Comprehensive Q18 (code audit) | Prompt 1, Section 4 |
| Comprehensive Q19-Q21 (publishability) | Prompt 3, Section 5 + Prompt 4 Q2.5 |
| Comprehensive Q22-Q23 (literature) | Prompt 3, Section 6 |
| Comprehensive Verification 1-7 | Prompt 1, Section 5 |
| Supplementary A1-A7 (code changes) | Prompt 1, Sections 1.1-1.7 |
| Supplementary B1-B4 (data pipeline) | Prompt 3, Section 3 |
| Supplementary C1-C3 (training pipeline) | Prompt 1, Section 2 |
| Supplementary D1-D3 (retraining) | Prompt 3, Section 1 |
| Supplementary E1-E4 (hostile referee) | Prompt 3, Section 4 |
| Supplementary F1-F3 (pipeline consistency) | Prompt 1, Section 3 |
| Supplementary G1-G2 (cheap experiments) | Prompt 3, Section 2 |
| Supplementary H-train.1-7 (training eval) | Prompt 2, Sections 4+6 |
| Supplementary I1-I10 (assessment) | Prompt 4, Section 1 |
