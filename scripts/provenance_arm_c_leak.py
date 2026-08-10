#!/usr/bin/env python
"""Arm C's accepting-path exposure: a genuinely different drafting policy.

Companion to scripts/regen_leak.py (arm B, same-policy regeneration, zero
cost). Arm C needed one real generation run per task
(scripts/run_provenance_arm.py: single generate node, prompt_id="solve_cot",
temperature=0.7 -- the actual first-draft config of the vanilla
verify_refine_3 workflow -- no verify, no refine).

leak = Pr(arm-C draft wrong | baseline correct), conditioned on the SAME
baseline-correct population Table 1 uses (baseline correct in every seed),
task-clustered point estimate and bootstrap CI (one value per task, averaged
over that task's 3 arm-C seeds, then averaged over tasks -- not pooled over
all (task, seed) pairs, which would let baseline-noisy tasks and easy tasks
count unequally).

accept|I wrong (code only, zero additional cost): code's verifier is the
oracle test suite, a real subprocess execution, not an LLM call, so it costs
nothing to ask what it would have decided on arm C's WRONG drafts. Math's
verifier is an LLM self-check and is not computed here -- that would need new
inference we have not been asked to spend.
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
from hbws.data import load_split  # noqa: E402

DOMAINS = [
    ("code", "envelope_test/direct_code_loose", "provenance_arm/cot_hot_code_loose"),
    ("math", "envelope_test/cot_math_loose", "provenance_arm/cot_hot_math_loose"),
]


def per_task(path_tmpl):
    acc = defaultdict(list)
    for s in SEEDS:
        p = EXP / f"{path_tmpl}/results_seed{s}.jsonl"
        for r in map(json.loads, open(p)):
            acc[r["task_id"]].append(bool(r.get("success_symbolic", r["success"])))
    return acc


def cluster_ci(vals, seed=0):
    if not vals:
        return float("nan"), float("nan"), float("nan")
    n = len(vals)
    rng = random.Random(seed)
    point = sum(vals) / n
    boots = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(NB))
    return point, boots[int(0.025 * NB)], boots[int(0.975 * NB)]


def code_accept_given_wrong():
    """Zero-cost: re-run the oracle feedback tests (what the verify node would
    see, not the held-out grading tests) against arm-C's WRONG code drafts."""
    tasks = {t["id"]: t for t in load_split("code", "test")[:150]}
    B = per_task("envelope_test/direct_code_loose")
    acc = tot = 0
    for s in SEEDS:
        for r in map(json.loads, open(EXP / f"provenance_arm/cot_hot_code_loose/results_seed{s}.jsonl")):
            vals = B.get(r["task_id"])
            if not vals or all(vals):
                continue  # only tasks where baseline fails every seed
            ok = bool(r.get("success_symbolic", r["success"]))
            if ok:
                continue  # only arm-C's WRONG drafts
            tot += 1
            passed, _ = verify.run_code_tests(r["solution"], tasks[r["task_id"]]["feedback_tests"])
            acc += passed
    return (acc / tot, tot) if tot else (float("nan"), 0)


def main():
    print("=" * 90)
    print("Arm C (different drafting policy): accepting-path exposure, zero-cost reanalysis")
    print("=" * 90)
    rows = {}
    for fam, base_path, armc_path in DOMAINS:
        B = per_task(base_path)
        C = per_task(armc_path)
        baseline_correct = [t for t, v in B.items() if len(v) == 3 and all(v)]
        vals = []
        for t in baseline_correct:
            cs = C.get(t)
            if cs and len(cs) == 3:
                vals.append(sum(1 for x in cs if not x) / 3)
        point, lo, hi = cluster_ci(vals)
        rows[fam] = dict(n_baseline_correct=len(baseline_correct), leak=point, lo=lo, hi=hi)
        print(f"{fam:6s}: baseline-correct n={len(baseline_correct):3d}  "
              f"leak={point:.4f}  [95% CI {lo:.4f}, {hi:.4f}]")

    acc_rate, acc_n = code_accept_given_wrong()
    rows["code"]["accept_given_wrong"] = acc_rate
    rows["code"]["accept_n"] = acc_n
    combo = rows["code"]["leak"] * acc_rate if acc_n else float("nan")
    rows["code"]["combo"] = combo
    print(f"\ncode: accept|I wrong (oracle feedback tests, zero-cost) = {acc_rate:.4f}  (n={acc_n})")
    print(f"code: leak * accept|I wrong (accepting-path exposure estimate) = {combo:.4f}")
    print("\nmath: accept|I wrong not computed -- math's verifier is an LLM self-check call,")
    print("      which would need new inference not yet authorized.")

    out = EXP / "provenance_arm_c_leak.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
