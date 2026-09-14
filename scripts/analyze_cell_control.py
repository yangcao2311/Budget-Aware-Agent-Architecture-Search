#!/usr/bin/env python3
"""Analyze one frozen baseline-plus-three-arm control cell.

The task, not the task--seed pair, is the bootstrap unit.  The report includes
the first-verifier/first-refinement reachability audit needed to distinguish a
provenance intervention from ledger gating.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parent.parent
SEEDS = (0, 1, 2)
ARMS = ("arm1_assign", "arm2_samepolicy", "arm3_diffpolicy")


def load(directory: Path) -> dict[str, dict[int, dict]]:
    result: dict[str, dict[int, dict]] = defaultdict(dict)
    for seed in SEEDS:
        path = directory / f"results_seed{seed}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        for line in path.read_text().splitlines():
            if line:
                row = json.loads(line)
                result[row["task_id"]][seed] = row
    return result


def success(row: dict) -> bool:
    return bool(row.get("success_symbolic", row.get("success", False)))


def first_gate(row: dict) -> tuple[str, dict]:
    trace = row.get("trace") or []
    for index, item in enumerate(trace):
        if item.get("type") != "verify":
            continue
        if "reserve_rejected" in item:
            return "verify_gated", {}
        following = trace[index + 1] if index + 1 < len(trace) else {}
        if following.get("type") == "refine":
            return "reject", following
        return "pass", {}
    return "no_verify", {}


def ci(values: list[float], seed: int, draws: int = 10000) -> list[float]:
    point = sum(values) / len(values)
    rng = random.Random(seed)
    n = len(values)
    samples = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n
                     for _ in range(draws))
    return [point, samples[int(.025 * draws)], samples[int(.975 * draws)]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--causal-dir", type=Path, required=True)
    parser.add_argument("--arm-suffix", required=True,
                        help="Suffix after each arm directory, e.g. _math_tight_nonbinding")
    parser.add_argument("--condition", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = load(args.baseline_dir)
    task_ids = sorted(baseline)
    if len(task_ids) != 150 or any(set(baseline[t]) != set(SEEDS) for t in task_ids):
        raise AssertionError("baseline must contain 150 tasks x 3 seeds")
    arm_data = {
        arm: load(args.causal_dir / f"{arm}{args.arm_suffix}") for arm in ARMS
    }
    if any(set(data) != set(task_ids) or
           any(set(data[t]) != set(SEEDS) for t in task_ids)
           for data in arm_data.values()):
        raise AssertionError("every arm must contain the same 150 tasks x 3 seeds")

    report = {
        "condition": args.condition,
        "bootstrap_unit": "task",
        "bootstrap_draws": 10000,
        "n_tasks": 150,
        "n_seeds": 3,
        "reference": {},
        "arms": {},
        "pairwise": {},
    }
    base_rows = [baseline[t][s] for t in task_ids for s in SEEDS]
    report["reference"] = {
        "accuracy": sum(success(row) for row in base_rows) / 450,
        "correct": sum(success(row) for row in base_rows),
        "status_counts": dict(Counter(row.get("status") for row in base_rows)),
    }

    for arm_index, arm in enumerate(ARMS):
        counts = Counter()
        task_delta, task_repair, task_break = [], [], []
        rows_this_arm = []
        for task in task_ids:
            deltas = []
            repairs = []
            breaks = []
            for seed in SEEDS:
                base = baseline[task][seed]
                row = arm_data[arm][task][seed]
                rows_this_arm.append(row)
                base_ok, arm_ok = success(base), success(row)
                deltas.append(float(arm_ok) - float(base_ok))
                repairs.append(float((not base_ok) and arm_ok))
                breaks.append(float(base_ok and not arm_ok))
                counts["reference_correct"] += int(base_ok)
                counts["reference_wrong"] += int(not base_ok)
                counts["repair"] += int((not base_ok) and arm_ok)
                counts["breakage"] += int(base_ok and not arm_ok)
                counts["byte_identical_to_reference"] += int(
                    (row.get("solution") or "") == (base.get("solution") or ""))
                gate, initial_refine = first_gate(row)
                counts[gate] += 1
                if gate == "reject":
                    counts["reject_reference_correct"] += int(base_ok)
                    key = ("initial_refinement_gated" if
                           "reserve_rejected" in initial_refine else
                           "initial_refinement_executed")
                    counts[key] += 1
                if base_ok and not arm_ok:
                    if gate in ("verify_gated", "no_verify"):
                        counts["preverification_breakage"] += 1
                    elif gate == "pass":
                        counts["acceptance_path_breakage"] += 1
                    else:
                        counts["rejection_path_breakage"] += 1
            task_delta.append(sum(deltas) / 3)
            task_repair.append(sum(repairs) / 3)
            task_break.append(sum(breaks) / 3)
        report["arms"][arm] = {
            "rows": 450,
            "status_counts": dict(Counter(row.get("status") for row in rows_this_arm)),
            "accuracy": sum(success(row) for row in rows_this_arm) / 450,
            "delta_ci95": ci(task_delta, 20260911 + arm_index),
            "repair_events": counts["repair"],
            "repair_denominator": counts["reference_wrong"],
            "repair_rate": counts["repair"] / counts["reference_wrong"],
            "repair_rate_per_pair_ci95": ci(task_repair, 20261011 + arm_index),
            "breakage_events": counts["breakage"],
            "breakage_denominator": counts["reference_correct"],
            "breakage_rate": counts["breakage"] / counts["reference_correct"],
            "breakage_rate_per_pair_ci95": ci(task_break, 20261111 + arm_index),
            "final_output_byte_identity_to_reference": (
                counts["byte_identical_to_reference"] / 450),
            "control_flow": {key: counts[key] for key in (
                "verify_gated", "no_verify", "pass", "reject",
                "reject_reference_correct", "initial_refinement_executed",
                "initial_refinement_gated", "preverification_breakage",
                "acceptance_path_breakage", "rejection_path_breakage")},
            "calls": sum(row.get("budget", {}).get("llm_calls", 0)
                         for row in rows_this_arm),
            "tokens": sum(row.get("budget", {}).get("in_tokens", 0) +
                          row.get("budget", {}).get("out_tokens", 0)
                          for row in rows_this_arm),
        }

    for comparator in ("arm2_samepolicy", "arm3_diffpolicy"):
        assign = report["arms"]["arm1_assign"]
        other = report["arms"][comparator]
        repair_gap = other["repair_events"] - assign["repair_events"]
        break_gap = other["breakage_events"] - assign["breakage_events"]
        report["pairwise"][f"assign_vs_{comparator}"] = {
            "other_minus_assign_repairs": repair_gap,
            "other_minus_assign_breakages": break_gap,
            "lambda_star": repair_gap / break_gap if break_gap > 0 else None,
        }

    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
