#!/usr/bin/env python3
"""Analyze the frozen repaired-executor GLM 2x2 provenance matrix.

The unit of resampling is a task.  Each task contributes up to three paired
reference/workflow executions.  Provider/runtime failures are reported and
never silently converted to wrong answers.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
SEEDS = (0, 1, 2)
ARMS = ("arm1_assign", "arm2_samepolicy", "arm3_diffpolicy")
STRUCTURE = {"code": "direct", "math": "cot"}


def rows(path: Path) -> dict[str, dict[int, dict]]:
    out: dict[str, dict[int, dict]] = defaultdict(dict)
    for seed in SEEDS:
        file = path / f"results_seed{seed}.jsonl"
        if not file.exists():
            continue
        for raw in file.read_text().splitlines():
            if raw:
                row = json.loads(raw)
                out[row["task_id"]][seed] = row
    return out


def valid(row: dict | None) -> bool:
    return bool(row) and row.get("status") in {"completed", "reserve_rejected"}


def ci_task(values: list[float], seed: int, draws: int = 10000) -> list[float | None]:
    if not values:
        return [None, None, None]
    point = sum(values) / len(values)
    rng = random.Random(seed)
    n = len(values)
    boot = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n
                  for _ in range(draws))
    return [point, boot[int(.025 * draws)], boot[int(.975 * draws)]]


def path(row: dict) -> str:
    if row.get("success"):
        return "correct"
    trace = row.get("trace") or []
    if any(x.get("type") == "refine" for x in trace):
        return "rejection"
    if any(x.get("type") == "verify" and "reserve_rejected" not in x
           for x in trace):
        return "acceptance"
    return "pre_verification"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    report = {"design": "2 domains x 2 budgets x (reference + 3 arms) x "
                        "150 tasks x 3 seeds; task-cluster bootstrap",
              "cells": {}}
    for family in ("code", "math"):
        for tier in ("tight", "loose"):
            key = f"{family}/{tier}"
            base_dir = EXP / f"{args.prefix}envelope_test/{STRUCTURE[family]}_{family}_{tier}"
            base = rows(base_dir)
            arm_rows = {arm: rows(EXP / args.tag / f"{arm}_{family}_{tier}")
                        for arm in ARMS}
            task_ids = sorted(base)
            cell = {"n_tasks": len(task_ids), "n_seeds": len(SEEDS),
                    "reference": {}, "arms": {}}
            baseline_pairs = [base[t][s] for t in task_ids for s in SEEDS
                              if s in base[t]]
            cell["reference"] = {
                "rows": len(baseline_pairs),
                "accuracy": sum(bool(r.get("success")) for r in baseline_pairs)
                            / max(1, len(baseline_pairs)),
                "status_counts": dict(Counter(r.get("status") for r in baseline_pairs)),
                "calls": sum(r.get("budget", {}).get("llm_calls", 0) for r in baseline_pairs),
                "tokens": sum(r.get("budget", {}).get("in_tokens", 0) +
                              r.get("budget", {}).get("out_tokens", 0)
                              for r in baseline_pairs),
            }
            for ai, arm in enumerate(ARMS):
                task_stats = []
                statuses = Counter()
                calls = tokens = 0
                for task_id in task_ids:
                    pairs = []
                    for seed in SEEDS:
                        b = base[task_id].get(seed)
                        a = arm_rows[arm].get(task_id, {}).get(seed)
                        if a:
                            statuses[a.get("status")] += 1
                            calls += a.get("budget", {}).get("llm_calls", 0)
                            tokens += (a.get("budget", {}).get("in_tokens", 0) +
                                       a.get("budget", {}).get("out_tokens", 0))
                        if valid(b) and valid(a):
                            pairs.append((b, a))
                    if pairs:
                        task_stats.append({
                            "p": sum(bool(b.get("success")) for b, _ in pairs) / len(pairs),
                            "a": sum(bool(a.get("success")) for _, a in pairs) / len(pairs),
                            "repair_num": sum((not b.get("success")) and a.get("success") for b, a in pairs),
                            "repair_den": sum(not b.get("success") for b, _ in pairs),
                            "break_num": sum(b.get("success") and not a.get("success") for b, a in pairs),
                            "break_den": sum(bool(b.get("success")) for b, _ in pairs),
                            "acceptance_break": sum(b.get("success") and path(a) == "acceptance" for b, a in pairs),
                            "rejection_break": sum(b.get("success") and path(a) == "rejection" for b, a in pairs),
                        })
                rn = sum(x["repair_num"] for x in task_stats)
                rd = sum(x["repair_den"] for x in task_stats)
                bn = sum(x["break_num"] for x in task_stats)
                bd = sum(x["break_den"] for x in task_stats)
                cell["arms"][arm] = {
                    "effective_tasks": len(task_stats),
                    "matched_pairs": sum(x["repair_den"] + x["break_den"] for x in task_stats),
                    "status_counts": dict(statuses),
                    "accuracy": sum(x["a"] for x in task_stats) / max(1, len(task_stats)),
                    "accuracy_delta_ci95": ci_task([x["a"] - x["p"] for x in task_stats],
                                                   1000 + ai),
                    "repair_events": rn, "repair_denominator": rd,
                    "repair_rate": rn / rd if rd else None,
                    "breakage_events": bn, "breakage_denominator": bd,
                    "breakage_rate": bn / bd if bd else None,
                    "acceptance_path_breakage_events": sum(x["acceptance_break"] for x in task_stats),
                    "rejection_path_breakage_events": sum(x["rejection_break"] for x in task_stats),
                    "calls": calls, "tokens": tokens,
                }
            a = cell["arms"]["arm1_assign"]
            d = cell["arms"]["arm3_diffpolicy"]
            # Per-input expected value is (repairs - lambda*breakages)/N.
            repair_gap = d["repair_events"] - a["repair_events"]
            break_gap = d["breakage_events"] - a["breakage_events"]
            cell["lambda_star_assign_vs_diffpolicy"] = (
                repair_gap / break_gap if break_gap > 0 else None)
            defects = {}
            for label, item in [("reference", cell["reference"]), *cell["arms"].items()]:
                defects[label] = sum(v for k, v in item["status_counts"].items()
                                     if k not in {"completed", "reserve_rejected"})
            cell["executor_defects"] = defects
            report["cells"][key] = cell
    out = EXP / args.output
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
