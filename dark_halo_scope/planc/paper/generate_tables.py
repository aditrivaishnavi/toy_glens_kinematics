from __future__ import annotations
import argparse, json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_json", required=True)
    ap.add_argument("--out_tex", required=True)
    args = ap.parse_args()
    s = json.loads(Path(args.summary_json).read_text())
    rec = s.get("test_recall", {})
    lines = [
        "\\begin{tabular}{lccc}",
        "\\hline",
        "Metric & Value & 2.5\\% & 97.5\\% \\",
        "\\hline",
        f"Recall (thr={s.get('threshold',0.5)}) & {rec.get('recall',0):.3f} & {rec.get('ci_low',0):.3f} & {rec.get('ci_high',0):.3f} \\",
        "\\hline",
        "\\end{tabular}",
    ]
    Path(args.out_tex).write_text("\n".join(lines), encoding="utf-8")

if __name__ == "__main__":
    main()
