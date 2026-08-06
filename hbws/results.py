"""Canonical result loader.

Math results carry two grades: `success` from the original grader (whose
symbolic-equivalence fallback silently never fired, because sympy was absent
from the environment that produced the frozen runs) and `success_symbolic`
from the corrected grader applied post hoc to the same stored responses.
Analyses use the corrected grade where it exists; the original is preserved
on disk so the difference stays auditable.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent / "experiments"
SEEDS = (0, 1, 2)


def row_success(r: dict, corrected: bool = True) -> bool:
    if corrected and "success_symbolic" in r:
        return bool(r["success_symbolic"])
    return bool(r["success"])


def per_task(dirname: str, corrected: bool = True, seeds=SEEDS) -> dict[str, float]:
    """task_id -> mean success over execution seeds."""
    acc = defaultdict(list)
    for s in seeds:
        p = EXP / dirname / f"results_seed{s}.jsonl"
        if not p.exists():
            continue
        for r in map(json.loads, open(p)):
            acc[r["task_id"]].append(row_success(r, corrected))
    return {t: sum(v) / len(v) for t, v in acc.items()}


def defect_ids(dirname: str, seeds=SEEDS) -> set[str]:
    """Tasks that hit a ledger settle-overrun in any seed."""
    bad = set()
    for s in seeds:
        p = EXP / dirname / f"results_seed{s}.jsonl"
        if not p.exists():
            continue
        for r in map(json.loads, open(p)):
            if str(r["status"]).startswith("budget_exceeded"):
                bad.add(r["task_id"])
    return bad
