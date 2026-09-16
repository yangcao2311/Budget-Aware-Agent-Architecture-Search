#!/usr/bin/env python3
"""Analysis for the GLM-4-Flash budget-parity follow-up: assign vs.
same-policy regeneration on math/tight and math/loose, with drafting cost
excluded from the shared verify/refine suffix budget.

tight and loose are analyzed completely separately -- this is NOT treated
as a single-factor tight-vs-loose experiment (both cells use the identical
SUFFIX_CAPS/DRAFT_EXTRA_* sizing from run_glm_budget_parity.py, so any
remaining tight/loose difference reflects only which stored reference
population was used, not a budget dose-response). No task is filtered by
correctness or by the arms' observed effect.

Task-clustered 95% CIs (task, not task-seed pair, is the bootstrap unit,
matching scripts/analyze_cell_control.py's convention). For rare/zero
events (e.g. a breakage or repair count of 0 or 1 in a 141/309-row
denominator), the percentile bootstrap is explicitly flagged as
uninformative rather than reported as if it certified zero risk -- see
`ci_is_degenerate`.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
TAG = "glm_budget_parity_20260916"
REF_PREFIX = "glm4flash_repaired_matrix_clean_20260910_envelope_test"
SEEDS = (0, 1, 2)
TIERS = ("tight", "loose")
ARMS = ("assign", "same_policy")


def load(path: Path) -> dict[str, dict]:
    out = {}
    for line in path.read_text().splitlines():
        if line:
            row = json.loads(line)
            out[row["task_id"]] = row
    return out


def success(row: dict) -> bool:
    return bool(row.get("success_symbolic", row.get("success", False)))


def first_gate(row: dict) -> tuple[str, dict]:
    trace = row.get("trace") or []
    for i, item in enumerate(trace):
        if item.get("type") != "verify":
            continue
        if "reserve_rejected" in item:
            return "verify_gated", {}
        following = trace[i + 1] if i + 1 < len(trace) else {}
        if following.get("type") == "refine":
            return "reject", following
        return "pass", {}
    return "no_verify", {}


def ci(values: list[float], seed: int, draws: int = 10000) -> dict:
    if not values:
        return {"point": None, "lo": None, "hi": None, "ci_is_degenerate": True}
    point = sum(values) / len(values)
    n_events = sum(1 for v in values if v > 0)
    rng = random.Random(seed)
    n = len(values)
    samples = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n
                     for _ in range(draws))
    lo, hi = samples[int(.025 * draws)], samples[int(.975 * draws)]
    # A rare/zero-event percentile bootstrap collapses to a near-point
    # interval (frequently [0,0] or [point,point]-ish) that looks like a
    # tight, reassuring CI but is not informative evidence that the true
    # rate is bounded away from something larger -- flag it rather than
    # report it uncritically (task instruction: avoid claiming risk is
    # "ruled out" this way).
    degenerate = n_events <= 2 or (hi - lo) < (1.0 / n)
    return {"point": point, "lo": lo, "hi": hi, "ci_is_degenerate": degenerate,
            "n_events": n_events, "n_clusters": n}


def analyze_cell(tier: str) -> dict:
    reference = {s: load(EXP / REF_PREFIX / f"cot_math_{tier}" / f"results_seed{s}.jsonl")
                for s in SEEDS}
    task_ids = sorted(reference[0])
    assert all(sorted(reference[s]) == task_ids for s in SEEDS), (
        f"{tier}: reference task ids differ across seeds")

    arms_out = {}
    for arm in ARMS:
        rows = {s: load(EXP / TAG / f"{tier}_{arm}" / f"results_seed{s}.jsonl")
               for s in SEEDS}
        counts = Counter()
        task_delta, task_repair, task_break = [], [], []
        total_calls = 0
        total_tokens = 0
        for tid in task_ids:
            deltas, repairs, breaks = [], [], []
            for s in SEEDS:
                ref_row = reference[s][tid]
                row = rows[s][tid]
                ref_ok, arm_ok = success(ref_row), success(row)
                deltas.append(float(arm_ok) - float(ref_ok))
                repairs.append(float((not ref_ok) and arm_ok))
                breaks.append(float(ref_ok and not arm_ok))
                counts["reference_correct"] += int(ref_ok)
                counts["reference_wrong"] += int(not ref_ok)
                counts["repair"] += int((not ref_ok) and arm_ok)
                counts["breakage"] += int(ref_ok and not arm_ok)
                gate, _ = first_gate(row)
                counts[gate] += 1
                if gate == "reject":
                    counts["reject_reference_correct"] += int(ref_ok)
                total_calls += row.get("budget", {}).get("llm_calls", 0)
                total_tokens += (row.get("budget", {}).get("in_tokens", 0)
                                + row.get("budget", {}).get("out_tokens", 0))
            task_delta.append(sum(deltas) / len(SEEDS))
            task_repair.append(sum(repairs) / len(SEEDS))
            task_break.append(sum(breaks) / len(SEEDS))

        n = len(task_ids) * len(SEEDS)
        arms_out[arm] = {
            "rows": n,
            "accuracy": sum(success(rows[s][tid]) for tid in task_ids for s in SEEDS) / n,
            "delta_vs_reference_ci95": ci(task_delta, 20260916),
            "repair_events": counts["repair"],
            "repair_denominator": counts["reference_wrong"],
            "repair_rate": (counts["repair"] / counts["reference_wrong"]
                           if counts["reference_wrong"] else None),
            "repair_rate_ci95": ci(task_repair, 20260917),
            "breakage_events": counts["breakage"],
            "breakage_denominator": counts["reference_correct"],
            "breakage_rate": (counts["breakage"] / counts["reference_correct"]
                              if counts["reference_correct"] else None),
            "breakage_rate_ci95": ci(task_break, 20260918),
            "control_flow": {k: counts[k] for k in
                            ("pass", "reject", "reject_reference_correct",
                             "verify_gated", "no_verify")},
            "total_llm_calls": total_calls,
            "total_tokens": total_tokens,
        }

    assign, same_policy = arms_out["assign"], arms_out["same_policy"]
    pairwise = {
        "same_policy_minus_assign_repairs": same_policy["repair_events"] - assign["repair_events"],
        "same_policy_minus_assign_breakages": same_policy["breakage_events"] - assign["breakage_events"],
        "same_policy_extra_llm_calls_from_drafting": (
            same_policy["total_llm_calls"] - assign["total_llm_calls"]),
        "same_policy_extra_tokens_from_drafting": (
            same_policy["total_tokens"] - assign["total_tokens"]),
    }
    return {"tier": tier, "n_tasks": len(task_ids), "arms": arms_out,
            "pairwise": pairwise}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    report = {
        "condition": "glm4flash_budget_parity_20260916",
        "note": "tight and loose analyzed separately; not a single-factor "
                "tight-vs-loose comparison (both use identical suffix "
                "caps). Drafting cost is reported in full "
                "(total_llm_calls/total_tokens include the same_policy "
                "arm's extra draft call) even though it is excluded from "
                "the shared verify/refine suffix budget by construction.",
        "bootstrap_unit": "task", "bootstrap_draws": 10000,
        "cells": {tier: analyze_cell(tier) for tier in TIERS},
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
