#!/usr/bin/env python3
"""Regrade stored GLM-4-Flash code outputs with the current held-out grader.

No model endpoint is called.  Source artifacts remain immutable; regraded rows
are written under a separate final-analysis prefix.  Provider-error rows are
retained as errors and are never converted into model failures.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hbws.data import load_split
from hbws.verify import run_code_tests


ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
SOURCE = EXP / "glm4flash_envelope_test" / "direct_code_loose"
DEST = EXP / "glm4flash_final_envelope_test" / "direct_code_loose"


def regrade(row: dict, tests_by_id: dict[str, str]) -> dict:
    row = dict(row)
    if row.get("status") == "completed":
        row["success"], row["grader_feedback"] = run_code_tests(
            row.get("solution") or "", tests_by_id[row["task_id"]]
        )
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    tests_by_id = {t["id"]: t["grading_tests"] for t in load_split("code", "test")}
    DEST.mkdir(parents=True, exist_ok=True)
    for seed in (0, 1, 2):
        src = SOURCE / f"results_seed{seed}.jsonl"
        rows = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(lambda row: regrade(row, tests_by_id), rows))
        rows.sort(key=lambda row: row["task_id"])
        out = DEST / f"results_seed{seed}.jsonl"
        out.write_text("".join(json.dumps(row) + "\n" for row in rows))
        completed = [row for row in rows if row.get("status") == "completed"]
        summary = {
            "run": "glm4flash_final_envelope_test/direct_code_loose",
            "seed": seed,
            "n": len(rows),
            "completed": len(completed),
            "provider_errors": len(rows) - len(completed),
            "successes": sum(bool(row.get("success")) for row in completed),
            "success_rate_completed": (
                sum(bool(row.get("success")) for row in completed) / len(completed)
                if completed else None
            ),
        }
        (DEST / f"summary_seed{seed}.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary))

    seed_maps = []
    for seed in (0, 1, 2):
        path = DEST / f"results_seed{seed}.jsonl"
        seed_maps.append({row["task_id"]: row for row in map(json.loads, open(path))})
    stable = {
        task_id for task_id in set.intersection(*(set(rows) for rows in seed_maps))
        if all(rows[task_id].get("status") == "completed" and
               rows[task_id].get("success") for rows in seed_maps)
    }
    print(f"stable_baseline_correct_tasks={len(stable)}")


if __name__ == "__main__":
    main()
