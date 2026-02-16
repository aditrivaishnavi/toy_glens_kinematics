"""
Deterministic template selection for deep-source banks (Spark-friendly).

CLI:
    python -m dhs_gen.deep_sources.deep_source_sampler --bank deep_bank.npz --task-id 12345
"""
from __future__ import annotations
import argparse
import json
import numpy as np
from ..utils import blake2b_u64


def load_bank(npz_path: str):
    z = np.load(npz_path, allow_pickle=True)
    images = z["images"]
    meta = json.loads(str(z["meta"]))
    return images, meta


def choose_index(task_id: str, n: int, salt: str = "") -> int:
    return int(blake2b_u64(task_id, salt=salt) % n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--task-id", required=True)
    ap.add_argument("--salt", default="")
    args = ap.parse_args()
    images, meta = load_bank(args.bank)
    idx = choose_index(args.task_id, int(images.shape[0]), salt=args.salt)
    img = images[idx].astype(np.float32)
    print(json.dumps({"index": idx, "sum": float(img.sum()), "file": meta["file"][idx]}))


if __name__ == "__main__":
    main()
