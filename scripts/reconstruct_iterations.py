#!/usr/bin/env python3
"""Offline per-iteration reconstruction for the factorial budget ablation.

Combines each row's own `trace` (node visit order + verify-tool-call
reservation status, already saved by the existing, unmodified
hbws.runner) with the raw per-call log from glm_iteration_logger.py
(reservation request/refusal, exact request, exact response text per
LLM call) to produce, for every task/seed/cell, a complete per-iteration
record:

  - candidate: the full candidate text at that iteration (from the call
    log's response_text -- never inferred from the row's final `solution`)
  - verdict: whether the masked verifier accepted it (inferred from trace
    node order, the same logic scripts/analyze_*.py's first_gate uses
    elsewhere in this repo)
  - correct: the candidate's correctness under the task's REAL grader
    (grading_tests, held out from the masked verifier and from the
    runner entirely -- graded here, offline, via the existing,
    unmodified hbws.verify.run_code_tests)
  - in_tokens / out_tokens / llm_calls for that iteration
  - the reservation request made before the iteration, and whether it
    was refused

This makes any tighter/looser cap combination replayable purely from
these saved artifacts -- no further GLM call is needed for a different
cap question against this same roster/seed set.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hbws import verify
from hbws.data import load_split

EXP = ROOT / "experiments"
TAG = "factorial_budget_ablation_20260917"
CELLS = ("A", "B", "C", "D")
SEEDS = (0, 1, 2)


def load_grading_tests() -> dict[str, str]:
    return {t["id"]: t["grading_tests"] for t in load_split("code", "test")[:150]}


def load_results(cell: str) -> dict[str, dict[int, dict]]:
    out: dict[str, dict[int, dict]] = defaultdict(dict)
    for seed in SEEDS:
        path = EXP / f"{TAG}/cell_{cell}" / f"results_seed{seed}.jsonl"
        for line in path.read_text().splitlines():
            if line:
                row = json.loads(line)
                out[row["task_id"]][seed] = row
    return out


def load_calls(cell: str) -> dict[tuple[str, int], list[dict]]:
    path = EXP / f"{TAG}_{cell}_calls.jsonl"
    out: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for line in path.read_text().splitlines():
        if line:
            row = json.loads(line)
            out[(row["task_id"], row["seed"])].append(row)
    return out


def reconstruct_row(row: dict, calls: list[dict], grading_tests: str) -> dict:
    trace = row.get("trace") or []
    call_iter = iter(calls)
    iterations = []
    for idx, item in enumerate(trace):
        node_type = item.get("type")
        if node_type not in ("generate", "refine"):
            continue
        call = next(call_iter, None)
        entry = {"trace_index": idx, "node_type": node_type}
        if call is None:
            # Should not happen (every generate/refine trace entry has a
            # matching call-log row, admitted or refused); record the gap.
            entry.update({"candidate": None, "reservation_refused": None,
                         "note": "no matching call-log row found"})
            iterations.append(entry)
            continue
        entry["reservation_request"] = call.get("reservation_request")
        entry["reservation_refused"] = call.get("reservation_refused")
        entry["in_tokens"] = call.get("in_tokens")
        entry["out_tokens"] = call.get("out_tokens")
        entry["llm_calls"] = 0 if call.get("reservation_refused") else 1
        if call.get("reservation_refused"):
            entry["candidate"] = None
            entry["verdict"] = None
            entry["correct"] = None
            iterations.append(entry)
            continue
        candidate = call.get("response_text")
        entry["candidate"] = candidate
        # verdict: did the masked verify (immediately following, per the
        # existing suffix graph) accept this candidate? Inferred from
        # trace node order (same convention as scripts/analyze_*.py's
        # first_gate), not hardcoded to "always fails under full mask" --
        # this stays correct even if mask_frac is later changed for a
        # different replay.
        following_verify = trace[idx + 1] if idx + 1 < len(trace) else None
        if following_verify is not None and following_verify.get("type") == "verify":
            after_verify = trace[idx + 2] if idx + 2 < len(trace) else None
            entry["verdict"] = "rejected" if (after_verify and after_verify.get("type") == "refine") else "accepted"
        else:
            entry["verdict"] = "unknown_no_following_verify"
        ok, _ = verify.run_code_tests(candidate, grading_tests)
        entry["correct"] = ok
        iterations.append(entry)
    return {"task_id": row["task_id"], "final_status": row.get("status"),
            "final_success_under_grader": row.get("success"),
            "iterations": iterations}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=CELLS)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    grading_tests = load_grading_tests()
    results = load_results(args.cell)
    calls = load_calls(args.cell)

    with args.output.open("w") as f:
        for tid in sorted(results):
            for seed in SEEDS:
                row = results[tid][seed]
                recon = reconstruct_row(row, calls.get((tid, seed), []),
                                        grading_tests[tid])
                recon["seed"] = seed
                f.write(json.dumps(recon, sort_keys=True) + "\n")
    print("wrote", args.output)


if __name__ == "__main__":
    main()
