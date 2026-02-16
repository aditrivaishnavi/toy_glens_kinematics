#!/usr/bin/env python3
"""
Generate comprehensive LLM review report after Phase 1 gates.
Includes all code, results, and specific questions.
"""
import json
from datetime import datetime, timezone

# Load all gate results
gates = {}
for i in ["1_1", "1_2", "1_3", "1_4"]:
    try:
        with open(f"/lambda/nfs/darkhaloscope-training-dc/gate_{i}_results.json") as f:
            gates[f"gate_{i}"] = json.load(f)
    except FileNotFoundError:
        gates[f"gate_{i}"] = {"status": "NOT RUN"}

report = f"""# Phase 1 Sanity Gates - LLM Review Report

**Generated:** {datetime.now(timezone.utc).isoformat()}

## Summary

| Gate | Status | Key Finding |
|------|--------|-------------|
| 1.1 Quality Distributions | {gates['gate_1_1'].get('overall_passed', 'N/A')} | See details below |
| 1.2 Bandset Audit | {gates['gate_1_2'].get('overall_passed', 'N/A')} | See details below |
| 1.3 Null-Injection | {gates['gate_1_3'].get('overall_passed', 'N/A')} | See details below |
| 1.4 SNR Ablation | {gates['gate_1_4'].get('overall_passed', 'N/A')} | See details below |

## Detailed Results

### Gate 1.1: Class-Conditional Quality Distributions
```json
{json.dumps(gates['gate_1_1'], indent=2)}
```

### Gate 1.2: Bandset Audit
```json
{json.dumps(gates['gate_1_2'], indent=2)}
```

### Gate 1.3: Null-Injection Test
```json
{json.dumps(gates['gate_1_3'], indent=2)}
```

### Gate 1.4: SNR Ablation Check
```json
{json.dumps(gates['gate_1_4'], indent=2)}
```

## Questions for LLM Review

1. Are the pass criteria for each gate appropriate?
2. Any concerns about the KS test p-values in Gate 1.1?
3. Is the null-injection test sufficient to rule out shortcuts?
4. Should we proceed to Phase 2, or investigate any findings first?
5. Are there other quality checks we should run?

## Scripts Used

All scripts are saved in `dark_halo_scope/scripts/gate_1_*.py`
"""

with open("/lambda/nfs/darkhaloscope-training-dc/phase1_llm_report.md", "w") as f:
    f.write(report)

print("Report saved to phase1_llm_report.md")
print(report)
