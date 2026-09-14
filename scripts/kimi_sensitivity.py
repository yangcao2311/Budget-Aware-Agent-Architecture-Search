#!/usr/bin/env python3
"""Recompute the Kimi K3 table with effective n, task-clustered CIs, and
the requested provider-failure sensitivity view.

No inference is run here.  The primary analysis is complete-case: rows with a
model response or a reservation rejection are retained, while provider-error
rows are omitted from the matched task x seed set.  Reservation rejections are
already scored as model failures by the ledger and therefore remain in the
primary analysis.  The sensitivity analysis retains every row and scores a
provider error as wrong (0), so it is a worst-case check on the small residual
provider-failure fraction.

Within each task, paired seeds are classified as repair, breakage, or unchanged.
Tasks are then weighted equally.  This retains partially baseline-correct tasks
and makes the reported identity exact. Confidence intervals resample task
clusters and recompute the conditional-rate ratios in each bootstrap draw.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
SEEDS = (0, 1, 2)
NB = 10000
VALID_STATUSES = {"completed", "reserve_rejected"}

CONDITIONS = {
    "code, oracle, loose": (
        "kimi_envelope_test/incumbent_refine_code_loose",
        "kimi_envelope_test/direct_code_loose",
    ),
    "code, oracle, tight": (
        "kimi_envelope_test/incumbent_refine_code_tight",
        "kimi_envelope_test/direct_code_tight",
    ),
    "math, self-check, loose": (
        "kimi_envelope_test/incumbent_refine_cot_math_loose",
        "kimi_envelope_test/cot_math_loose",
    ),
    "math, self-check, tight": (
        "kimi_envelope_test/incumbent_refine_cot_math_tight",
        "kimi_envelope_test/cot_math_tight",
    ),
    "BBH, self-check, loose": (
        "kimi_envelope_logic_prospective/incumbent_refine_logic_loose",
        "kimi_envelope_logic_prospective/direct_logic_loose",
    ),
    "code OOD, no tests": (
        "kimi_envelope_ood/incumbent_refine_code_loose",
        "kimi_envelope_ood/direct_code_loose",
    ),
    "math OOD, self-check": (
        "kimi_envelope_ood/incumbent_refine_cot_math_loose",
        "kimi_envelope_ood/cot_math_loose",
    ),
    "code OOD, tests restored": (
        "kimi_envelope_ood_visible/incumbent_refine_code_loose",
        "kimi_envelope_ood_visible/direct_code_loose",
    ),
    "code, 50% tests, loose": (
        "kimi_envelope_test_mask0.5_k1/verify_refine_3_code_loose",
        "kimi_envelope_test_mask0.5_k1/direct_code_loose",
    ),
    "code, no tests, loose": (
        "kimi_envelope_test_mask0.0_k1/verify_refine_3_code_loose",
        "kimi_envelope_test_mask0.0_k1/direct_code_loose",
    ),
}


def load_rows(dirname: str, include_provider_failures: bool):
    """Return (task, seed) -> (correct, status), with one row per seed."""
    rows = {}
    for seed in SEEDS:
        path = EXP / dirname / f"results_seed{seed}.jsonl"
        with path.open() as fh:
            for row in map(json.loads, fh):
                status = row.get("status", "")
                valid = status in VALID_STATUSES
                if not (valid or include_provider_failures):
                    continue
                # Provider errors have no model response and are deliberately
                # scored as wrong only in the sensitivity view.
                correct = (
                    bool(row.get("success_symbolic", row.get("success", False)))
                    if valid
                    else False
                )
                rows[(row["task_id"], seed)] = (correct, status)
    return rows


def _percentile_ci(values, seed=0):
    if not values:
        return float("nan"), float("nan"), float("nan")
    n = len(values)
    point = sum(values) / n
    rng = random.Random(seed)
    draws = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(NB)
    )
    return point, draws[int(0.025 * NB)], draws[int(0.975 * NB)]


def _ratio_ci(values, seed=0):
    """Bootstrap a ratio of task-level numerator/denominator contributions."""
    if not values:
        return float("nan"), float("nan"), float("nan")
    n = len(values)

    def ratio(sample):
        num = sum(values[i][0] for i in sample)
        den = sum(values[i][1] for i in sample)
        return num / den if den else float("nan")

    point = ratio(range(n))
    rng = random.Random(seed)
    draws = []
    for _ in range(NB):
        value = ratio([rng.randrange(n) for _ in range(n)])
        if value == value:
            draws.append(value)
    draws.sort()
    return point, draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))]


def metrics(protected_dir: str, baseline_dir: str, failure_as_wrong=False):
    """Compute rates and task-clustered CIs for one condition."""
    protected = load_rows(protected_dir, failure_as_wrong)
    baseline = load_rows(baseline_dir, failure_as_wrong)
    paired = sorted(set(protected) & set(baseline))
    by_task = defaultdict(list)
    for task, seed in paired:
        by_task[task].append(seed)

    task_values = {}
    for task, seeds in by_task.items():
        keys = [(task, seed) for seed in seeds]
        vals = [(baseline[k][0], protected[k][0]) for k in keys]
        n = len(vals)
        b = sum(x for x, _ in vals) / n
        p = sum(x for _, x in vals) / n
        task_values[task] = {
            "baseline": b,
            "workflow": p,
            "repair": (sum((not b0) and w0 for b0, w0 in vals) / n,
                       sum(not b0 for b0, _ in vals) / n),
            "breakage": (sum(b0 and (not w0) for b0, w0 in vals) / n,
                         sum(b0 for b0, _ in vals) / n),
            "net": p - b,
        }

    repair_values = [v["repair"] for v in task_values.values()]
    breakage_values = [v["breakage"] for v in task_values.values()]
    net_values = [v["net"] for v in task_values.values()]
    baseline_p = sum(v["baseline"] for v in task_values.values()) / len(task_values)
    repair = _ratio_ci(repair_values)
    breakage = _ratio_ci(breakage_values)
    net = _percentile_ci(net_values)
    identity_delta = (1 - baseline_p) * repair[0] - baseline_p * breakage[0]
    if abs(identity_delta - net[0]) > 1e-12:
        raise AssertionError("repair--breakage identity drift")

    statuses = Counter(status for _, status in protected.values())
    return {
        "n_eff": len(paired),
        "n_tasks": len(task_values),
        "n_repair": sum(v["repair"][1] for v in task_values.values()),
        "n_breakage": sum(v["breakage"][1] for v in task_values.values()),
        "baseline": baseline_p,
        "repair": repair,
        "breakage": breakage,
        "net": net,
        "protected_status": dict(statuses),
        "failure_as_wrong": failure_as_wrong,
    }


def _fmt(ci, signed=False):
    point, lo, hi = ci
    if signed:
        return f"{point:+.3f} [{lo:+.3f},{hi:+.3f}]"
    return f"{point:.3f} [{lo:.3f},{hi:.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args()
    report = {}
    for name, (protected, baseline) in CONDITIONS.items():
        report[name] = {
            "complete_case": metrics(protected, baseline, False),
            "failure_as_wrong": metrics(protected, baseline, True),
        }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    print("Kimi K3: complete-case analysis (provider errors omitted)")
    print("condition | n_eff | n_rep/n_brk | repair [95% CI] | breakage [95% CI] | net [95% CI]")
    for name, result in report.items():
        m = result["complete_case"]
        print(
            f"{name} | {m['n_eff']} | {m['n_repair']}/{m['n_breakage']} | "
            f"{_fmt(m['repair'])} | {_fmt(m['breakage'])} | {_fmt(m['net'], True)}"
        )
    print("\nFailure-as-wrong sensitivity (provider errors scored as 0)")
    print("condition | n_eff | repair | breakage | net | changed pairs")
    for name, result in report.items():
        cc, fw = result["complete_case"], result["failure_as_wrong"]
        changed = fw["n_eff"] - cc["n_eff"]
        print(
            f"{name} | {fw['n_eff']} | {_fmt(fw['repair'])} | "
            f"{_fmt(fw['breakage'])} | {_fmt(fw['net'], True)} | +{changed}"
        )


if __name__ == "__main__":
    main()
