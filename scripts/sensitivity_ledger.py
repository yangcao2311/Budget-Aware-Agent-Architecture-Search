#!/usr/bin/env python
"""Sensitivity analysis for the ledger-defect confound.

4.7% of frozen-test executions terminated with `budget_exceeded:settle_overrun`
--- a defect of the pre-tiktoken token estimator, which under-counted prompt
tokens and so occasionally settled above its own reservation. Those tasks were
scored as failures. The rate is uneven across arms (incumbent 6.5%, vanilla
4.5%, baseline 3.6%), so it could in principle bias the comparisons.

This recomputes every headline contrast on the subset of tasks where NEITHER
arm hit the defect in ANY seed, and reports both numbers side by side. If the
conclusions move, the affected runs must be repeated.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
SEEDS = [0, 1, 2]
NB = 10000


def load(dirname):
    """task_id -> (mean success, hit_defect_in_any_seed)"""
    succ = defaultdict(list)
    bad = set()
    for s in SEEDS:
        p = EXP / dirname / f"results_seed{s}.jsonl"
        if not p.exists():
            continue
        for r in map(json.loads, open(p)):
            succ[r["task_id"]].append(bool(r["success"]))
            if str(r["status"]).startswith("budget_exceeded"):
                bad.add(r["task_id"])
    return {t: sum(v) / len(v) for t, v in succ.items()}, bad


def contrast(sd, bd, clean_only, seed=0):
    S, sbad = load(sd)
    B, bbad = load(bd)
    ids = sorted(set(S) & set(B))
    if clean_only:
        ids = [t for t in ids if t not in sbad and t not in bbad]
    if not ids:
        return None
    d = [S[t] - B[t] for t in ids]
    n = len(d)
    rng = random.Random(seed)
    bo = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(NB))
    easy = [1 - S[t] for t in ids if B[t] == 1.0]
    return {"delta": sum(d) / n, "lo": bo[int(.025 * NB)], "hi": bo[int(.975 * NB)],
            "brk": sum(easy) / len(easy) if easy else 0.0, "n": n}


T = "envelope_test"
CASES = [
    ("C1 code loose  vanilla-direct", f"{T}/verify_refine_3_code_loose", f"{T}/direct_code_loose"),
    ("C1 math loose  vanilla-cot", f"{T}/verify_refine_3_math_loose", f"{T}/cot_math_loose"),
    ("C2 code tight  vanilla-direct", f"{T}/verify_refine_3_code_tight", f"{T}/direct_code_tight"),
    ("C3 code loose  no-signal", "envelope_test_mask0.0_k1/verify_refine_3_code_loose",
     "envelope_test_mask0.0_k1/direct_code_loose"),
    ("C4 code tight  protected", f"{T}/incumbent_refine_code_tight", f"{T}/direct_code_tight"),
    ("C4 code loose  protected", f"{T}/incumbent_refine_code_loose", f"{T}/direct_code_loose"),
    ("C4 code unseen protected", f"{T}/incumbent_refine_code_unseen", f"{T}/direct_code_unseen"),
    ("C4 math tight  protected", f"{T}/incumbent_refine_cot_math_tight", f"{T}/cot_math_tight"),
    ("C4 math loose  protected", f"{T}/incumbent_refine_cot_math_loose", f"{T}/cot_math_loose"),
    ("C4 math unseen protected", f"{T}/incumbent_refine_cot_math_unseen", f"{T}/cot_math_unseen"),
]


def main():
    print("Sensitivity to the ledger defect: all tasks vs. defect-free tasks only")
    print("=" * 92)
    print(f"{'contrast':32s}{'ALL: delta [CI]':>28s}{'brk':>7s} | "
          f"{'CLEAN: delta [CI]':>28s}{'brk':>7s}{'  n':>5s}")
    flips = []
    for name, sd, bd in CASES:
        a = contrast(sd, bd, clean_only=False)
        c = contrast(sd, bd, clean_only=True)
        if not a or not c:
            print(f"{name:32s}  MISSING")
            continue
        astr = f"{a['delta']:+.3f} [{a['lo']:+.3f},{a['hi']:+.3f}]"
        cstr = f"{c['delta']:+.3f} [{c['lo']:+.3f},{c['hi']:+.3f}]"
        print(f"{name:32s}{astr:>28s}{a['brk']:>7.3f} | "
              f"{cstr:>28s}{c['brk']:>7.3f}{c['n']:>5d}")
        # a conclusion "flips" if significance or sign of the effect changes
        sig_a = a["lo"] > 0 or a["hi"] < 0
        sig_c = c["lo"] > 0 or c["hi"] < 0
        if sig_a != sig_c or (a["delta"] > 0) != (c["delta"] > 0):
            flips.append(name)
        # C4's guarantee: breakage must stay <= 0.02 in both views
        if "protected" in name and (a["brk"] > 0.02 or c["brk"] > 0.02):
            flips.append(name + " [breakage bound]")
    print("=" * 92)
    if flips:
        print(f"CONCLUSIONS AFFECTED ({len(flips)}): " + "; ".join(flips))
    else:
        print("No headline conclusion changes sign, significance, or violates the "
              "breakage bound\nunder the defect-free subset.")


if __name__ == "__main__":
    main()
