# Gen6 / Gen7 / Gen8 / Uber dataset code bundle

## Contents

## Installation (important for Spark)

Option A (recommended): build a wheel and install on all nodes

- On the driver:
  - `python -m pip install -U build`
  - `python -m build`
- Distribute the wheel to executors (EMR bootstrap, or `--py-files dist/*.whl`).

Option B: zip distribution

- `zip -r dhs_gen.zip dhs_gen`
- `spark-submit --py-files dhs_gen.zip your_job.py`


- `dhs_gen/deep_sources/*` (Gen6)
- `dhs_gen/hybrid_sources/*` (Gen7)
- `dhs_gen/domain_randomization/*` (Gen8)
- `dhs_gen/uber/mixer.py` (uber mix assignment)
- `dhs_gen/validation/quality_checks.py` (stamp QC)

## Scientific recommendation: separate gens vs uber
High-confidence approach for publishable, interpretable results:

1) Run separate ablations first:
   - Gen6 only (deep sources, no artifacts)
   - Gen7 only (hybrid sources, no artifacts)
   - Gen8 only (domain randomization on top of your best source mode)

2) Evaluate each on the same locked real-data anchor set at fixed FPR thresholds.

3) Only after you have evidence that multiple help, build an uber_mix dataset with fixed weights.
   Keep per-row columns (source_mode, artifact_profile, gen_variant) so you can stratify metrics and
   still interpret effects.

This gives you both:
- a clean ablation section for MNRAS/ApJ/AAS,
- and a practical final model trained on a richer mixture.

## Quality checks
Validate stamp parquet:
```bash
python -m dhs_gen.validation.quality_checks --parquet "/path/to/stamps/*.parquet" --out-json qc.json
```
