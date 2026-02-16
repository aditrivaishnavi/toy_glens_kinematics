# Short Experiment Matrix (Exact Runs + Expected Outcomes)

## Gates
G1: Core-only LR AUC (central 10x10 pixels)         target < 0.65
G2: Radial-profile-only LR AUC (azimuthal profile)  target < 0.65

If either gate fails, synthetic AUROC is not interpretable for sim-to-real.

## Runs (minimum set)
R0: paired_baseline (no mitigations)
- Expect: very high synthetic AUROC; G1 fails.

R1: paired_core_dropout (p=0.5)
- Expect: synthetic AUROC high; G1 often still fails.

R3: unpaired_matched_raw
- Expect: AUROC drops vs paired; G1 and G2 should pass if matching is adequate.

R4: unpaired_matched_raw_hardneg (azimuthal shuffle prob=0.3)
- Expect: similar AUROC; improved contaminant robustness.

R5: unpaired_matched_residual
- Expect: stronger gate passing (G1/G2 lower); may improve real-lens behavior.

R6: unpaired_matched_residual_hardneg
- Expect: best robustness; slight AUROC cost possible.

## Stop/Go after R3
- GO to anchor/contaminant evaluation if: test AUROC >= 0.75 AND G1,G2 pass.
- Otherwise: fix matching bins or switch to residual mode before spending compute.
