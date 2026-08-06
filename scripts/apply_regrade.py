#!/usr/bin/env python
"""Apply the corrected math grading to stored results, non-destructively.

Adds `success_symbolic` alongside the original `success` field for every math
task. Nothing is deleted: the defective grading stays on disk so the
difference is auditable. `hbws/results.py` prefers the corrected field.

Zero API cost -- every model response was already stored.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
EXP = ROOT / "experiments"

from hbws import verify  # noqa: E402

GOLD = {}
for f in ROOT.glob("data/math_*.jsonl"):
    for line in open(f):
        r = json.loads(line)
        GOLD[r["id"]] = r["gold_answer"]

_CACHE = {}


def graded(pred, gold):
    k = (pred, gold)
    if k not in _CACHE:
        _CACHE[k] = verify.math_equal(pred, gold)
    return _CACHE[k]


def main():
    files = [p for p in EXP.rglob("results_seed*.jsonl") if "math" in p.parent.name]
    changed_files = flips = total = 0
    for p in files:
        rows = [json.loads(l) for l in open(p)]
        touched = False
        for r in rows:
            g = GOLD.get(r["task_id"])
            if g is None:
                continue
            pred = verify.extract_boxed(r["solution"])
            ok = pred is not None and graded(pred, g)
            total += 1
            if ok != bool(r["success"]):
                flips += 1
            if r.get("success_symbolic") != ok:
                r["success_symbolic"] = ok
                touched = True
        if touched:
            with open(p, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            changed_files += 1
    print(f"regraded {total} math answers across {len(files)} files")
    print(f"  grade changed on {flips} ({flips/max(1,total):.2%})")
    print(f"  files updated: {changed_files}")
    print("original `success` field preserved; `success_symbolic` added")


if __name__ == "__main__":
    main()
