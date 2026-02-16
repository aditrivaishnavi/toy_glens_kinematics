# Review deliverables

This folder contains two runnable diagnostics scripts meant to resolve the main mystery quickly:

1) `diagnostics/bright_arc_ceiling_audit.py`
   - Tests whether the bright-arc ceiling is an artifact of robust normalization and clipping.

2) `diagnostics/embedding_mismatch_probe.py`
   - Quantifies whether real positives and synthetic injections occupy different regions in the CNN feature space.

Both scripts assume the repo layout in `stronglens_calibration_full_codebase.zip` and that `cutout_path` entries
in your parquet manifest exist locally.

Run from the repo root with:

```
PYTHONPATH=. python diagnostics/bright_arc_ceiling_audit.py --help
PYTHONPATH=. python diagnostics/embedding_mismatch_probe.py --help
```
