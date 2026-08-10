#!/usr/bin/env python
"""The three-arm provenance causal test: breakage by arm, on the baseline-
correct population Table 1 uses (baseline correct in every one of ITS OWN
3 seeds).

Companion to scripts/run_provenance_causal.py, which produced the raw data:
three arms sharing identical downstream structure (verify, then refine on
rejection, up to 3 iterations), differing ONLY in where the first draft comes
from (explicit reuse / same-policy regeneration / different-policy
regeneration), all with cross-arm caching disabled. Because all three run the
real verify+refine loop, breakage here is the DIRECTLY OBSERVED rate for each
arm -- not an estimate built from a leak probability and a separately-measured
acceptance rate, unlike the exploratory arm-B/arm-C reanalyses this test was
built to go beyond.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
SEEDS = [0, 1, 2]

BASE = {"code": "envelope_test/direct_code_loose", "math": "envelope_test/cot_math_loose"}
ARMS = [
    ("arm1_assign", "explicit reuse", "provenance_causal/arm1_assign_{}_loose"),
    ("arm2_samepolicy", "same-policy regen", "provenance_causal/arm2_samepolicy_{}_loose"),
    ("arm3_diffpolicy", "different-policy regen", "provenance_causal/arm3_diffpolicy_{}_loose"),
]


def per_task(rel_path: str) -> dict:
    acc = defaultdict(list)
    for s in SEEDS:
        p = EXP / rel_path / f"results_seed{s}.jsonl"
        for r in map(json.loads, open(p)):
            acc[r["task_id"]].append(bool(r.get("success_symbolic", r["success"])))
    return acc


def main():
    print("=" * 78)
    print("Three-arm provenance causal test: breakage by arm (baseline-correct pop.)")
    print("=" * 78)
    rows = {}
    for fam, base_path in BASE.items():
        B = per_task(base_path)
        baseline_correct = [t for t, v in B.items() if len(v) == 3 and all(v)]
        print(f"\n{fam}: baseline-correct n={len(baseline_correct)}")
        for key, label, tmpl in ARMS:
            A = per_task(tmpl.format(fam))
            wrong = tot = 0
            for t in baseline_correct:
                for v in A.get(t, []):
                    tot += 1
                    wrong += not v
            brk = wrong / tot if tot else float("nan")
            rows[f"{fam}_{key}"] = {"n_tasks": len(baseline_correct), "pairs": tot,
                                     "breakage": brk}
            print(f"  {label:24s}: pairs={tot:4d}  breakage={brk:.4f}")

    out = EXP / "provenance_causal_analysis.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
