#!/usr/bin/env python
"""Replay code-verifier masking on the same stored reference output.

This diagnostic uses no model calls. It separates verifier false rejection
from draft regeneration by running both the full and 50%-masked visible test
suites against each archived baseline output that passes the hidden grader.
"""
from __future__ import annotations

import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hbws.data import load_split
from hbws.verify import run_code_tests

SEEDS = (0, 1, 2)


def mask_tests(test_code: str, fraction: float) -> str:
    lines = [line for line in test_code.splitlines() if line.strip()]
    assertions = [line for line in lines if line.lstrip().startswith("assert")]
    other = [line for line in lines if not line.lstrip().startswith("assert")]
    keep = math.ceil(fraction * len(assertions))
    return "\n".join(other + assertions[:keep]) if keep else ""


def load_eligible_pairs() -> list[tuple[str, str, str]]:
    tasks = {task["id"]: task for task in load_split("code", "test")}
    pairs = []
    for seed in SEEDS:
        path = (ROOT / "experiments" / "envelope_test" /
                "direct_code_loose" / f"results_seed{seed}.jsonl")
        for row in map(json.loads, path.open()):
            correct = bool(row.get("success_symbolic", row["success"]))
            if correct:
                task = tasks[row["task_id"]]
                pairs.append((row["task_id"], row["solution"],
                              task["feedback_tests"]))
    return pairs


def evaluate_pair(pair: tuple[str, str, str]) -> dict:
    task_id, solution, full_tests = pair
    full_pass, _ = run_code_tests(solution, full_tests)
    half_pass, _ = run_code_tests(solution, mask_tests(full_tests, 0.5))
    return {
        "task_id": task_id,
        "full_reject": not full_pass,
        "half_reject": not half_pass,
    }


def summarize() -> dict:
    pairs = load_eligible_pairs()
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(evaluate_pair, pairs))
    full = sum(row["full_reject"] for row in rows)
    half = sum(row["half_reject"] for row in rows)
    inversion = sum((not row["full_reject"]) and row["half_reject"]
                    for row in rows)
    return {
        "baseline_correct_task_seed_pairs": len(rows),
        "full_test_rejections": full,
        "full_test_frr": full / len(rows),
        "half_test_rejections": half,
        "half_test_frr": half / len(rows),
        "full_pass_half_reject": inversion,
    }


def main() -> None:
    print(json.dumps(summarize(), indent=2))


if __name__ == "__main__":
    main()
