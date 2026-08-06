#!/usr/bin/env python
"""Measure the verifier's false-rejection rate and test the bound

    b <= Pr(reject | incumbent correct)                                   (3)

against observed breakage, across every condition already on disk. No new
inference: the rate is recoverable from stored traces, because a protected
workflow that exits after [generate, verify] had its draft accepted, while one
that reaches a refine node had it rejected.

The bound is derived, not fitted, so this is a genuine test: a single condition
with breakage above its own false-rejection rate would falsify it.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
SEEDS = [0, 1, 2]
NB = 10000


def per_task(dirname):
    acc = defaultdict(list)
    for s in SEEDS:
        p = EXP / dirname / f"results_seed{s}.jsonl"
        if p.exists():
            for r in map(json.loads, open(p)):
                acc[r["task_id"]].append(
                    bool(r.get("success_symbolic", r["success"])))
    return {t: sum(v) / len(v) for t, v in acc.items()}


def cluster_boot_upper(per_task_vals, seed=0):
    """One-sided 95% upper bound by task-clustered bootstrap. Execution seeds
    of a task are not independent, so the task is the resampling unit."""
    if not per_task_vals:
        return float("nan")
    rng = random.Random(seed)
    n = len(per_task_vals)
    boots = sorted(sum(per_task_vals[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(NB))
    return boots[int(0.95 * NB)]


def measure(struct, base, tag="envelope_test"):
    """Returns per-task breakage and per-task false-rejection indicators,
    both restricted to tasks the baseline solves in every seed."""
    B = per_task(f"{tag}/{base}")
    brk, rej = [], []
    per_task_reject = defaultdict(list)
    per_task_break = defaultdict(list)
    for s in SEEDS:
        p = EXP / tag / struct / f"results_seed{s}.jsonl"
        if not p.exists():
            continue
        for r in map(json.loads, open(p)):
            if B.get(r["task_id"], 0.0) != 1.0:
                continue
            types = [t["type"] for t in r.get("trace", [])]
            if "verify" not in types:
                continue
            per_task_reject[r["task_id"]].append(1.0 if "refine" in types else 0.0)
            ok = bool(r.get("success_symbolic", r["success"]))
            per_task_break[r["task_id"]].append(0.0 if ok else 1.0)
    rej = [sum(v) / len(v) for v in per_task_reject.values()]
    brk = [sum(v) / len(v) for v in per_task_break.values()]
    if not rej:
        return None
    return {
        "n_tasks": len(rej),
        "reject": sum(rej) / len(rej),
        "reject_ub": cluster_boot_upper(rej),
        "breakage": sum(brk) / len(brk),
        "breakage_ub": cluster_boot_upper(brk),
    }


CONDS = [
    ("code, oracle tests, loose", "incumbent_refine_code_loose",
     "direct_code_loose", "envelope_test"),
    ("code, oracle tests, tight", "incumbent_refine_code_tight",
     "direct_code_tight", "envelope_test"),
    ("code, 50% tests, loose", "verify_refine_3_code_loose",
     "direct_code_loose", "envelope_test_mask0.5_k1"),
    ("code, NO tests, loose", "verify_refine_3_code_loose",
     "direct_code_loose", "envelope_test_mask0.0_k1"),
    ("math, self-check, loose", "incumbent_refine_cot_math_loose",
     "cot_math_loose", "envelope_test"),
    ("math, self-check, tight", "incumbent_refine_cot_math_tight",
     "cot_math_tight", "envelope_test"),
    ("math OOD, self-check", "incumbent_refine_cot_math_loose",
     "cot_math_loose", "envelope_ood"),
    ("code OOD, NO tests", "incumbent_refine_code_loose",
     "direct_code_loose", "envelope_ood"),
    ("code OOD, tests restored", "incumbent_refine_code_loose",
     "direct_code_loose", "envelope_ood_visible"),
    ("BBH, self-check, loose", "incumbent_refine_logic_loose",
     "direct_logic_loose", "envelope_logic_prospective"),
]


def main():
    print("=" * 88)
    print("Testing the derived bound  breakage <= Pr(reject | incumbent correct)")
    print("Rates are over baseline-correct TASKS; upper bounds are one-sided 95%,")
    print("task-clustered (execution seeds within a task are not independent).")
    print("=" * 88)
    print(f"{'condition':30s}{'tasks':>7s}{'false-rej':>11s}{'[95% ub]':>10s}"
          f"{'breakage':>10s}{'[95% ub]':>10s}{'  bound':>8s}")
    rows, violations = [], []
    for name, st, ba, tag in CONDS:
        m = measure(st, ba, tag)
        if not m:
            print(f"{name:30s}  (no data)")
            continue
        holds = m["breakage"] <= m["reject"] + 1e-9
        if not holds:
            violations.append(name)
        rows.append((name, m))
        print(f"{name:30s}{m['n_tasks']:>7d}{m['reject']:>11.3f}"
              f"{m['reject_ub']:>10.3f}{m['breakage']:>10.3f}"
              f"{m['breakage_ub']:>10.3f}{'  holds' if holds else '  VIOLATED':>8s}")
    print("=" * 88)
    if violations:
        print(f"BOUND VIOLATED in {len(violations)}: " + "; ".join(violations))
    else:
        span = [m["reject"] for _, m in rows]
        print(f"Bound holds in {len(rows)}/{len(rows)} conditions, over the "
              f"full false-rejection range {min(span):.3f} to {max(span):.3f}.")
        tight = [(n, m) for n, m in rows if 0.05 < m["reject"] < 0.95]
        for n, m in tight:
            print(f"  near-tight: {n} -- breakage {m['breakage']:.3f} against "
                  f"bound {m['reject']:.3f}")
    json.dump([{"condition": n, **m} for n, m in rows],
              open(EXP / "false_rejection_table.json", "w"), indent=2)
    print(f"written to {EXP / 'false_rejection_table.json'}")


if __name__ == "__main__":
    main()
