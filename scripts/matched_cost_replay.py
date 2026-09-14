#!/usr/bin/env python3
"""Cost-matched offline replay using only the frozen three-seed pools.

This is an expected-cost comparison, not a new inference run.  For each
domain we compute the stored k=1, k=2 and k=3 prefix policies, then mix the
two adjacent prefixes whose expected logged dollar cost brackets the
reference-preserving arm.  The mixture weight is fixed from cost only; it is
independent of labels and selection outcomes.  Per-task fractional accuracy
is the expectation of that randomized policy, and task-clustered bootstrap
resampling gives its paired interval against reference preservation.
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hbws import verify
from hbws.data import load_split

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments" / "envelope_test"
SEEDS = [0, 1, 2]
NB = 10000


def load_records(rel):
    out = defaultdict(dict)
    for s in SEEDS:
        with open(EXP / rel / f"results_seed{s}.jsonl") as f:
            for line in f:
                r = json.loads(line)
                out[r["task_id"]][s] = r
    return out


def choose_math(by_seed, k):
    answers = {}
    for s in SEEDS[:k]:
        sol = by_seed[s].get("solution") or ""
        boxed = verify.extract_boxed(sol)
        answers[s] = verify._norm(boxed) if boxed else None
    counts = Counter(v for v in answers.values() if v is not None)
    if not counts:
        return 0
    best = max(counts.values())
    winners = sorted(s for s in SEEDS[:k]
                     if answers[s] in {a for a, c in counts.items() if c == best})
    return winners[0]


def choose_code(by_seed, k, task):
    passers = []
    for s in SEEDS[:k]:
        ok, _ = verify.run_code_tests(by_seed[s].get("solution") or "",
                                       task["feedback_tests"])
        if ok:
            passers.append(s)
    return passers[0] if passers else 0


def prefix_accuracy(family, raw, tasks, k):
    out = {}
    for tid, by_seed in raw.items():
        if len(by_seed) != 3 or tid not in tasks:
            continue
        chosen = (choose_code(by_seed, k, tasks[tid]) if family == "code"
                  else choose_math(by_seed, k))
        r = by_seed[chosen]
        out[tid] = float(r.get("success_symbolic", r.get("success", False)))
    return out


def mean_seed_cost(raw):
    return [sum(by[s]["budget"]["usd"] for by in raw.values() if s in by) /
            sum(1 for by in raw.values() if s in by) for s in SEEDS]


def ref_accuracy(raw):
    return {tid: sum(float(r.get("success_symbolic", r.get("success", False)))
                   for r in by.values()) / len(by)
            for tid, by in raw.items() if len(by) == 3}


def bootstrap(vals, seed=0):
    n = len(vals)
    rng = random.Random(seed)
    point = sum(vals) / n
    draws = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(NB))
    return point, draws[int(.025 * NB)], draws[int(.975 * NB)]


def run(family, base_rel, ref_rel, target_cost):
    raw, ref_raw = load_records(base_rel), load_records(ref_rel)
    tasks = {t["id"]: t for t in load_split(family, "test")[:150]}
    costs = mean_seed_cost(raw)
    prefixes = {k: prefix_accuracy(family, raw, tasks, k) for k in [1, 2, 3]}
    prefix_costs = {k: sum(costs[:k]) for k in [1, 2, 3]}
    if target_cost <= prefix_costs[1]:
        lo, hi = 1, 1
        q = 0.0
    elif target_cost <= prefix_costs[2]:
        lo, hi = 1, 2
        q = (target_cost - prefix_costs[lo]) / (prefix_costs[hi] - prefix_costs[lo])
    else:
        lo, hi = 2, 3
        q = (target_cost - prefix_costs[lo]) / (prefix_costs[hi] - prefix_costs[lo])
    common = sorted(set(prefixes[lo]) & set(prefixes[hi]) & set(ref_raw))
    mixed = {t: (1 - q) * prefixes[lo][t] + q * prefixes[hi][t] for t in common}
    ref = ref_accuracy(ref_raw)
    diffs = [ref[t] - mixed[t] for t in common]
    return {
        "family": family,
        "n_tasks": len(common),
        "target_cost": target_cost,
        "seed_costs": costs,
        "prefix_costs": prefix_costs,
        "lower_k": lo,
        "upper_k": hi,
        "upper_prefix_probability": q,
        "lower_prefix_accuracy": sum(prefixes[lo][t] for t in common) / len(common),
        "upper_prefix_accuracy": sum(prefixes[hi][t] for t in common) / len(common),
        "mixed_accuracy": sum(mixed.values()) / len(mixed),
        "reference_accuracy": sum(ref[t] for t in common) / len(common),
        "paired_diff_ref_minus_mixed": bootstrap(diffs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    rows = [
        run("code", "direct_code_loose", "incumbent_refine_code_loose", 0.002514702222222222),
        run("math", "cot_math_loose", "incumbent_refine_cot_math_loose", 0.015322166666666666),
    ]
    payload = {"description": "expected-cost-matched offline replay", "rows": rows}
    print(json.dumps(payload, indent=2))
    if args.out:
        args.out.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
