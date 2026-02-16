"""
Uber dataset mixer: assign source_mode + artifact_profile deterministically per task and store in manifests.

Recommended scientific approach:
- First run separate ablations: gen6_deep, gen7_hybrid, gen8_domain
- Then, if multiple help on the real anchor set, create uber_mix with fixed weights.

CLI (pyarrow-based):
    python -m dhs_gen.uber.mixer --in-parquet in.parquet --out-parquet out.parquet --mode uber
"""
from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from ..utils import categorical_from_hash


@dataclass
class MixerConfig:
    source_modes: List[str]
    source_probs: List[float]
    artifact_profiles: List[str]
    artifact_probs: List[float]
    salt: str = "mix_v1"


DEFAULT_UBER = MixerConfig(
    source_modes=["cosmos", "deep", "hybrid"],
    source_probs=[0.34, 0.33, 0.33],
    artifact_profiles=["none", "mild", "strong"],
    artifact_probs=[0.15, 0.70, 0.15],
    salt="uber_v1",
)


def assign_modes(task_id: str, cfg: MixerConfig) -> Dict[str, str]:
    sidx = categorical_from_hash(task_id, np.array(cfg.source_probs), salt=cfg.salt + "_src")
    aidx = categorical_from_hash(task_id, np.array(cfg.artifact_probs), salt=cfg.salt + "_art")
    return {"source_mode": cfg.source_modes[sidx], "artifact_profile": cfg.artifact_profiles[aidx]}


def add_modes_to_manifest_pyarrow(
    in_parquet: str,
    out_parquet: str,
    cfg: MixerConfig,
    task_id_col: str = "task_id",
    gen_variant: str = "uber_mix",
) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as e:
        raise RuntimeError("pyarrow required: pip install pyarrow") from e

    tbl = pq.read_table(in_parquet)
    # Avoid duplicate columns if rerunning the mixer on an already-mixed manifest.
    for col in ["source_mode", "artifact_profile", "gen_variant"]:
        while col in tbl.column_names:
            idx = list(tbl.column_names).index(col)
            tbl = tbl.remove_column(idx)
    if task_id_col not in tbl.column_names:
        raise KeyError(f"Missing column {task_id_col}")

    task_ids = [str(x) for x in tbl[task_id_col].to_pylist()]
    src = []
    art = []
    for tid in task_ids:
        m = assign_modes(tid, cfg)
        src.append(m["source_mode"])
        art.append(m["artifact_profile"])

    tbl2 = tbl.append_column("source_mode", pa.array(src, pa.string()))
    tbl2 = tbl2.append_column("artifact_profile", pa.array(art, pa.string()))
    tbl2 = tbl2.append_column("gen_variant", pa.array([gen_variant] * len(task_ids), pa.string()))
    pq.write_table(tbl2, out_parquet, compression="zstd")


def validate_manifest_distribution(parquet_path: str) -> Dict[str, object]:
    try:
        import pyarrow.parquet as pq
    except Exception as e:
        raise RuntimeError("pyarrow required: pip install pyarrow") from e

    cols = ["task_id", "source_mode", "artifact_profile", "region_split"]
    tbl = pq.read_table(parquet_path, columns=[c for c in cols if c in pq.read_schema(parquet_path).names])
    out: Dict[str, object] = {"n": int(tbl.num_rows)}

    for c in ["source_mode", "artifact_profile", "region_split"]:
        if c in tbl.column_names:
            vals = np.array(tbl[c].to_pylist(), dtype=object)
            u, cnt = np.unique(vals, return_counts=True)
            out[c] = {str(x): int(y) for x, y in zip(u, cnt)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-parquet", required=True)
    ap.add_argument("--out-parquet", required=True)
    ap.add_argument("--mode", choices=["uber", "gen6", "gen7", "gen8"], default="uber")
    ap.add_argument("--salt", default=None)
    args = ap.parse_args()

    if args.mode == "uber":
        cfg = DEFAULT_UBER
        gen_variant = "uber_mix"
    elif args.mode == "gen6":
        cfg = MixerConfig(["deep"], [1.0], ["none"], [1.0], salt="gen6")
        gen_variant = "gen6_deep"
    elif args.mode == "gen7":
        cfg = MixerConfig(["hybrid"], [1.0], ["none"], [1.0], salt="gen7")
        gen_variant = "gen7_hybrid"
    else:
        cfg = MixerConfig(["cosmos"], [1.0], ["mild"], [1.0], salt="gen8")
        gen_variant = "gen8_domain"

    if args.salt is not None:
        cfg.salt = args.salt

    add_modes_to_manifest_pyarrow(args.in_parquet, args.out_parquet, cfg=cfg, gen_variant=gen_variant)
    rep = validate_manifest_distribution(args.out_parquet)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
