#!/usr/bin/env python3
"""Analysis for the factorial budget ablation (LLM-call cap x
output-token cap), fully-masked code verifier, self-drafting arm.

Decomposes breakage into main effects and an interaction, using the
per-task-clustered paired bootstrap (each resample draws the same task
index across all four cells, since every cell shares the identical 150
tasks x 3 seeds). Requires scripts/reconstruct_iterations.py to have
already been run for all four cells (it supplies each row's first-draft
correctness, needed to define "reference-correct"/"reference-wrong" here
in the absence of any separate reference arm -- the row's OWN first
draft, pre-refine, is the reference point, matching wf_incumbent_refine's
"incumbent-protecting" design intent).

Cells: A=(tight call, tight token) B=(tight call, loose token)
       C=(loose call, tight token) D=(loose call, loose token)

  call_main_effect_on_breakage  = mean_breakage(C,D) - mean_breakage(A,B)
  token_main_effect_on_breakage = mean_breakage(B,D) - mean_breakage(A,C)
  interaction                   = (D - C) - (B - A)

Also reports the four terminal-path categories (pass / reject-then-
executed / verify_gated / no_verify) and the corner-reproduction check
(cell A must equal BUDGET_TIERS['tight'], cell D must equal
BUDGET_TIERS['loose'] -- already asserted by construction in
run_factorial_budget_ablation.py; repeated here empirically against the
manifest for a second, independent confirmation).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EXP = ROOT / "experiments"
TAG = "factorial_budget_ablation_20260917"
CELLS = ("A", "B", "C", "D")
SEEDS = (0, 1, 2)
N_TASKS = 150

TIGHT_CALL = {"A", "B"}
LOOSE_CALL = {"C", "D"}
TIGHT_TOKEN = {"A", "C"}
LOOSE_TOKEN = {"B", "D"}


def load_results(cell: str) -> dict[str, dict[int, dict]]:
    out: dict[str, dict[int, dict]] = defaultdict(dict)
    for seed in SEEDS:
        path = EXP / f"{TAG}/cell_{cell}" / f"results_seed{seed}.jsonl"
        for line in path.read_text().splitlines():
            if line:
                row = json.loads(line)
                out[row["task_id"]][seed] = row
    return out


def load_reconstruction(cell: str) -> dict[tuple[str, int], dict]:
    path = EXP / f"{TAG}_{cell}_iterations.jsonl"
    out = {}
    for line in path.read_text().splitlines():
        if line:
            row = json.loads(line)
            out[(row["task_id"], row["seed"])] = row
    return out


def success(row: dict) -> bool:
    return bool(row.get("success_symbolic", row.get("success", False)))


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


def ci(values: list[float], seed: int, draws: int = 10000) -> dict:
    if not values:
        return {"point": None, "lo": None, "hi": None, "ci_is_degenerate": True}
    point = sum(values) / len(values)
    n_events = sum(1 for v in values if abs(v) > 1e-12)
    rng = random.Random(seed)
    n = len(values)
    samples = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n
                     for _ in range(draws))
    lo, hi = samples[int(.025 * draws)], samples[int(.975 * draws)]
    degenerate = n_events <= 2 or (hi - lo) < (1.0 / n)
    return {"point": point, "lo": lo, "hi": hi, "ci_is_degenerate": degenerate,
            "n_events": n_events, "n_clusters": n, "note": "post-hoc descriptive"}


def per_cell_task_data(cell: str, results, recon) -> dict:
    """Per task: mean-over-seeds breakage/repair indicator, plus counts."""
    task_ids = sorted(results)
    counts = Counter()
    task_break, task_repair = {}, {}
    for tid in task_ids:
        breaks, repairs = [], []
        for seed in SEEDS:
            row = results[tid][seed]
            arm_ok = success(row)
            r = recon.get((tid, seed))
            first_iter = (r["iterations"][0] if r and r["iterations"] else None)
            ref_ok = bool(first_iter["correct"]) if first_iter and first_iter.get("correct") is not None else None
            gate = first_gate(row)
            counts["rows"] += 1
            counts[gate] += 1
            if ref_ok is None:
                counts["first_draft_reservation_refused"] += 1
                continue
            counts["reference_correct"] += int(ref_ok)
            counts["reference_wrong"] += int(not ref_ok)
            counts["breakage"] += int(ref_ok and not arm_ok)
            counts["repair"] += int((not ref_ok) and arm_ok)
            breaks.append(float(ref_ok and not arm_ok))
            repairs.append(float((not ref_ok) and arm_ok))
        task_break[tid] = sum(breaks) / len(breaks) if breaks else 0.0
        task_repair[tid] = sum(repairs) / len(repairs) if repairs else 0.0
    return {"task_ids": task_ids, "task_break": task_break,
            "task_repair": task_repair, "counts": dict(counts)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    per_cell = {}
    for cell in CELLS:
        results = load_results(cell)
        recon = load_reconstruction(cell)
        per_cell[cell] = per_cell_task_data(cell, results, recon)

    task_ids = per_cell["A"]["task_ids"]
    for cell in CELLS:
        assert per_cell[cell]["task_ids"] == task_ids, f"{cell}: task id set mismatch"

    def cell_series(metric: str, cell: str) -> list[float]:
        d = per_cell[cell][metric]
        return [d[t] for t in task_ids]

    def combo(metric: str, cells: set[str], sign: dict[str, float]) -> list[float]:
        n = len(task_ids)
        acc = [0.0] * n
        for c in cells:
            s = cell_series(metric, c)
            w = sign[c]
            for i in range(n):
                acc[i] += w * s[i]
        return acc

    # main effects and interaction, per task (averaged within group before
    # differencing, exactly matching the point-estimate formula in the
    # module docstring)
    call_effect_per_task = combo("task_break", {"A", "B", "C", "D"},
                                 {"A": -0.5, "B": -0.5, "C": 0.5, "D": 0.5})
    token_effect_per_task = combo("task_break", {"A", "B", "C", "D"},
                                  {"A": -0.5, "C": -0.5, "B": 0.5, "D": 0.5})
    interaction_per_task = combo("task_break", {"A", "B", "C", "D"},
                                 {"A": -1.0, "B": 1.0, "C": 1.0, "D": -1.0})

    report = {
        "condition": "factorial_budget_ablation_20260917",
        "family": "code", "mask_frac": 0.0,
        "workflow": "wf_incumbent_refine (self-drafting arm only)",
        "n_tasks": len(task_ids), "n_seeds": len(SEEDS),
        "cells": {c: per_cell[c]["counts"] for c in CELLS},
        "corner_reproduction_check": json.loads(
            (EXP / f"{TAG}_manifest.json").read_text())["corner_check"],
        "main_effects_and_interaction_on_breakage": {
            "call_cap_main_effect": ci(call_effect_per_task, 20260917),
            "token_cap_main_effect": ci(token_effect_per_task, 20260918),
            "interaction": ci(interaction_per_task, 20260919),
            "sign_convention": "positive = loose is worse (higher "
                               "breakage) than tight, for that axis",
        },
        "per_cell_breakage_repair": {
            c: {
                "breakage_events": per_cell[c]["counts"].get("breakage", 0),
                "breakage_denominator": per_cell[c]["counts"].get("reference_correct", 0),
                "breakage_rate": (per_cell[c]["counts"].get("breakage", 0)
                                  / per_cell[c]["counts"]["reference_correct"]
                                  if per_cell[c]["counts"].get("reference_correct") else None),
                "repair_events": per_cell[c]["counts"].get("repair", 0),
                "repair_denominator": per_cell[c]["counts"].get("reference_wrong", 0),
                "repair_rate": (per_cell[c]["counts"].get("repair", 0)
                               / per_cell[c]["counts"]["reference_wrong"]
                               if per_cell[c]["counts"].get("reference_wrong") else None),
                "terminal_path_categories": {
                    k: per_cell[c]["counts"].get(k, 0)
                    for k in ("pass", "reject", "verify_gated", "no_verify")
                },
                "first_draft_reservation_refused": per_cell[c]["counts"].get(
                    "first_draft_reservation_refused", 0),
            } for c in CELLS
        },
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
