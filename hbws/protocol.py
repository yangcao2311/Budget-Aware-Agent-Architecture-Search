"""Unified evaluation protocol: run one workflow over a task list, grade with
the held-out grader, and log per-task results + budget spend.

This is the ONLY entry point any method (baseline or searched candidate) may
use — identical model, caps, grader, and logging for everyone (方案 §5.3).
"""
from __future__ import annotations

import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import verify
from .ledger import BudgetCaps
from .runner import run_workflow

EXP_DIR = Path(__file__).resolve().parent.parent / "experiments"


def grade(task: dict, solution: str) -> bool:
    if task["family"] == "code":
        ok, _ = verify.run_code_tests(solution, task["grading_tests"])
        return ok
    return verify.grade_math(solution, task["gold_answer"])


def evaluate(wf: dict, tasks: list[dict], caps: BudgetCaps, *, run_name: str,
             seed: int = 0, use_cache: bool = True, workers: int = 6) -> dict:
    out_dir = EXP_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    def one(task):
        r = run_workflow(wf, task, BudgetCaps(**vars(caps)), use_cache=use_cache, seed=seed)
        r["success"] = grade(task, r["solution"])
        return r

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, t): t for t in tasks}
        for f in as_completed(futs):
            try:
                results.append(f.result())
            except Exception as e:
                results.append({"task_id": futs[f]["id"], "status": f"error:{e}",
                                "success": False, "solution": "",
                                "budget": {}, "trace": []})

    results.sort(key=lambda r: r["task_id"])
    with open(out_dir / f"results_seed{seed}.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    n = len(results)
    succ = sum(r["success"] for r in results)
    usd = [r["budget"].get("usd", 0) for r in results]
    calls = [r["budget"].get("llm_calls", 0) for r in results]
    toks = [r["budget"].get("in_tokens", 0) + r["budget"].get("out_tokens", 0) for r in results]
    secs = sorted(r["budget"].get("wall_sec", 0) for r in results)
    summary = {
        "run": run_name, "seed": seed, "n": n,
        "success_rate": round(succ / n, 4),
        "usd_per_task": round(sum(usd) / n, 5),
        "usd_per_success": round(sum(usd) / succ, 5) if succ else None,
        "llm_calls_per_task": round(sum(calls) / n, 2),
        "tokens_per_task": round(sum(toks) / n, 1),
        "p50_sec": round(secs[n // 2], 2) if n else 0,
        "p95_sec": round(secs[int(n * 0.95)] if n > 1 else secs[0], 2) if n else 0,
        "total_usd": round(sum(usd), 4),
        "wall_min": round((time.monotonic() - t0) / 60, 1),
        "over_budget": sum(1 for r in results if str(r["status"]).startswith("budget_exceeded")),
        "errors": sum(1 for r in results if str(r["status"]).startswith("error")),
    }
    with open(out_dir / f"summary_seed{seed}.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary
