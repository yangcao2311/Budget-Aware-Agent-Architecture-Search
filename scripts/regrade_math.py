#!/usr/bin/env python
"""Re-grade every stored math answer with symbolic equivalence, and report
how the paper's contrasts move.

The grader in `verify.math_equal` falls back to sympy when string and numeric
comparison both fail. That fallback silently never fired: sympy was not
installed in the environment that produced the frozen results, and the import
is inside a try/except. Re-grading is possible at zero API cost because every
model response is stored.

This is a grader defect, not a protocol change: it is applied identically to
every arm, and no model is re-run. Both gradings are reported.
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
EXP = ROOT / "experiments"
SEEDS = [0, 1, 2]
NB = 10000

from hbws import verify  # noqa: E402


def gold_map():
    g = {}
    for f in ROOT.glob("data/math_*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            g[r["id"]] = r["gold_answer"]
    return g


GOLD = gold_map()


def grade_string_only(sol, gold):
    p = verify.extract_boxed(sol)
    if p is None:
        return False
    a, b = verify._norm(p), verify._norm(gold)
    if a == b:
        return True
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return False


def per_task(dirname, symbolic):
    acc = defaultdict(list)
    for s in SEEDS:
        p = EXP / dirname / f"results_seed{s}.jsonl"
        if not p.exists():
            continue
        for r in map(json.loads, open(p)):
            g = GOLD.get(r["task_id"])
            if g is None:
                acc[r["task_id"]].append(bool(r["success"]))
                continue
            ok = (verify.grade_math(r["solution"], g) if symbolic
                  else grade_string_only(r["solution"], g))
            acc[r["task_id"]].append(ok)
    return {t: sum(v) / len(v) for t, v in acc.items()}


def contrast(sd, bd, symbolic, seed=0):
    S, B = per_task(sd, symbolic), per_task(bd, symbolic)
    ids = sorted(set(S) & set(B))
    if not ids:
        return None
    d = [S[t] - B[t] for t in ids]
    n = len(d)
    rng = random.Random(seed)
    bo = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(NB))
    easy = [1 - S[t] for t in ids if B[t] == 1.0]
    hard = [S[t] for t in ids if B[t] == 0.0]
    return {"succ": sum(S[t] for t in ids) / n, "base": sum(B[t] for t in ids) / n,
            "delta": sum(d) / n, "lo": bo[int(.025 * NB)], "hi": bo[int(.975 * NB)],
            "brk": sum(easy) / len(easy) if easy else 0.0,
            "rep": sum(hard) / len(hard) if hard else 0.0}


CASES = []
for tier in ["tight", "unseen", "loose"]:
    CASES.append((f"C1/C4 math {tier} vanilla",
                  f"envelope_test/verify_refine_3_math_{tier}",
                  f"envelope_test/cot_math_{tier}"))
    CASES.append((f"C4  math {tier} protected",
                  f"envelope_test/incumbent_refine_cot_math_{tier}",
                  f"envelope_test/cot_math_{tier}"))
CASES.append(("OOD math loose protected",
              "envelope_ood/incumbent_refine_cot_math_loose",
              "envelope_ood/cot_math_loose"))
CASES.append(("OOD math loose vanilla",
              "envelope_ood/verify_refine_3_math_loose",
              "envelope_ood/cot_math_loose"))


def main():
    print("Math re-grading: string/numeric only  vs  + symbolic equivalence")
    print("=" * 96)
    print(f"{'contrast':30s}{'STRING: delta [CI]':>28s}{'brk':>7s} | "
          f"{'SYMBOLIC: delta [CI]':>28s}{'brk':>7s}")
    changed = []
    for name, sd, bd in CASES:
        a = contrast(sd, bd, symbolic=False)
        b = contrast(sd, bd, symbolic=True)
        if not a or not b:
            continue
        astr = f"{a['delta']:+.3f} [{a['lo']:+.3f},{a['hi']:+.3f}]"
        bstr = f"{b['delta']:+.3f} [{b['lo']:+.3f},{b['hi']:+.3f}]"
        print(f"{name:30s}{astr:>28s}{a['brk']:>7.3f} | {bstr:>28s}{b['brk']:>7.3f}")
        sa = a["lo"] > 0 or a["hi"] < 0
        sb = b["lo"] > 0 or b["hi"] < 0
        if sa != sb or (a["delta"] > 0) != (b["delta"] > 0):
            changed.append(name)
        if "protected" in name and b["brk"] > 0.02:
            changed.append(name + " [breakage bound]")
    base = contrast("envelope_test/verify_refine_3_math_loose",
                    "envelope_test/cot_math_loose", symbolic=True)
    base0 = contrast("envelope_test/verify_refine_3_math_loose",
                     "envelope_test/cot_math_loose", symbolic=False)
    print("=" * 96)
    print(f"math baseline accuracy: {base0['base']:.3f} (string) -> "
          f"{base['base']:.3f} (symbolic)")
    if changed:
        print(f"CONCLUSIONS AFFECTED ({len(changed)}): " + "; ".join(changed))
    else:
        print("No math conclusion changes sign, significance, or violates the "
              "breakage bound.")


if __name__ == "__main__":
    main()
