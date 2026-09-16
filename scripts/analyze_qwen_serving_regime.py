#!/usr/bin/env python3
"""Byte-identity analysis for the Qwen serving-regime intervention
(reviewer B1): does changing vLLM concurrency/batching (B) or prefix
caching (P) change output bytes relative to the serial reference (S),
same model/hardware/request? S2 (a repeat of S) checks time-drift /
reversibility of the comparison itself.

Two comparisons, both exact UTF-8 byte comparison (no strip/normalize):
  (a) first-draft identity: the node "g" response text, taken directly
      from each condition's qwen_call_logger.py call log (never inferred
      from the final `solution`), for every task/seed where a first draft
      was actually generated (i.e. excluding execution-error task/seeds
      that produced no draft at all).
  (b) first-accepted identity: the row's final `solution`, restricted to
      task/seeds where the first verify passed in BOTH the condition and
      the S reference (no refinement fired in either) -- otherwise a
      refine call's own regeneration, not the serving regime, could be
      the source of any byte difference.

Reports planned/generated/failed counts and denominators explicitly, task-
clustered 95% CIs (task is the bootstrap unit, matching
scripts/analyze_cell_control.py's convention), and agreement/correctness/
rejection stats so a byte difference is never conflated with a correctness
or execution-error signal.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
TAG = "qwen_serving_regime_20260915"
SEEDS = (0, 1, 2)
N_TASKS = 150


def load_results(condition: str) -> dict[str, dict[int, dict]]:
    out: dict[str, dict[int, dict]] = defaultdict(dict)
    for seed in SEEDS:
        path = EXP / f"{TAG}/{condition}_samepolicy" / f"results_seed{seed}.jsonl"
        for line in path.read_text().splitlines():
            if line:
                row = json.loads(line)
                out[row["task_id"]][seed] = row
    return out


def load_first_drafts(condition: str) -> dict[tuple[str, int], str]:
    """(task_id, seed) -> node 'g' response_text, from the raw call log."""
    path = EXP / f"{TAG}_{condition}_calls.jsonl"
    drafts: dict[tuple[str, int], str] = {}
    for line in path.read_text().splitlines():
        if not line:
            continue
        row = json.loads(line)
        if row.get("node_id") == "g" and row.get("status") == "completed":
            key = (row["task_id"], row["seed"])
            drafts.setdefault(key, row["response_text"])
    return drafts


def first_gate(row: dict) -> str:
    trace = row.get("trace") or []
    for i, item in enumerate(trace):
        if item.get("type") != "verify":
            continue
        if "reserve_rejected" in item:
            return "verify_gated"
        following = trace[i + 1] if i + 1 < len(trace) else {}
        return "reject" if following.get("type") == "refine" else "pass"
    return "no_verify"


def success(row: dict) -> bool:
    return bool(row.get("success_symbolic", row.get("success", False)))


def ci(values: list[float], seed: int, draws: int = 10000) -> list[float]:
    if not values:
        return [None, None, None]
    point = sum(values) / len(values)
    rng = random.Random(seed)
    n = len(values)
    samples = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n
                     for _ in range(draws))
    return [point, samples[int(.025 * draws)], samples[int(.975 * draws)]]


def analyze_condition(condition: str, s_results, s_drafts, seed_for_ci: int) -> dict:
    cond_results = load_results(condition)
    cond_drafts = load_first_drafts(condition)
    task_ids = sorted(s_results)

    planned = N_TASKS * len(SEEDS)
    generated_drafts = 0
    failed_no_draft = 0
    draft_compared = 0
    draft_identical = 0
    accepted_compared = 0
    accepted_identical = 0
    status_counts = Counter()
    gate_counts = Counter()
    agreement_same_answer = 0
    both_correct = 0

    task_draft_identical: dict[str, list] = defaultdict(list)
    task_accepted_identical: dict[str, list] = defaultdict(list)

    for tid in task_ids:
        for seed in SEEDS:
            s_row = s_results[tid][seed]
            row = cond_results[tid][seed]
            status_counts[str(row.get("status"))] += 1

            key = (tid, seed)
            has_draft = key in cond_drafts
            has_s_draft = key in s_drafts
            if has_draft:
                generated_drafts += 1
            else:
                failed_no_draft += 1

            if has_draft and has_s_draft:
                draft_compared += 1
                same_bytes = cond_drafts[key] == s_drafts[key]
                draft_identical += int(same_bytes)
                task_draft_identical[tid].append(float(same_bytes))

            s_gate = first_gate(s_row)
            c_gate = first_gate(row)
            gate_counts[c_gate] += 1
            if s_gate == "pass" and c_gate == "pass":
                accepted_compared += 1
                same_bytes = (row.get("solution") or "") == (s_row.get("solution") or "")
                accepted_identical += int(same_bytes)
                task_accepted_identical[tid].append(float(same_bytes))

            if success(row) == success(s_row):
                agreement_same_answer += 1
            if success(row) and success(s_row):
                both_correct += 1

    task_draft_rate = [sum(v) / len(v) for v in task_draft_identical.values() if v]
    task_accepted_rate = [sum(v) / len(v) for v in task_accepted_identical.values() if v]

    return {
        "condition": condition,
        "planned_positions": planned,
        "generated_first_drafts": generated_drafts,
        "failed_no_draft": failed_no_draft,
        "status_counts": dict(status_counts),
        "first_verify_gate_counts": dict(gate_counts),
        "first_draft_identity": {
            "compared": draft_compared,
            "identical": draft_identical,
            "rate": draft_identical / draft_compared if draft_compared else None,
            "task_clustered_ci95": ci(task_draft_rate, seed_for_ci),
        },
        "first_accepted_identity": {
            "compared": accepted_compared,
            "identical": accepted_identical,
            "rate": accepted_identical / accepted_compared if accepted_compared else None,
            "task_clustered_ci95": ci(task_accepted_rate, seed_for_ci + 1),
        },
        "agreement_with_S_same_correctness_label": agreement_same_answer / planned,
        "both_correct_with_S": both_correct,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    s_results = load_results("S")
    s_drafts = load_first_drafts("S")

    report = {
        "reference_condition": "S",
        "n_tasks": N_TASKS,
        "n_seeds": len(SEEDS),
        "bootstrap_unit": "task",
        "bootstrap_draws": 10000,
        "conditions": {},
    }
    for i, condition in enumerate(("B", "P", "S2")):
        report["conditions"][condition] = analyze_condition(
            condition, s_results, s_drafts, seed_for_ci=20260916 + i)

    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
