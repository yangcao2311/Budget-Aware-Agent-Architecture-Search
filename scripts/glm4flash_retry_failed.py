#!/usr/bin/env python3
"""Backfill only failed GLM-4-Flash rows without touching completed rows.

The first official-API run exposed provider-side 429s.  This recovery pass is
deliberately narrow: it retries failed rows from the math baseline, writes the
new attempts to a separate directory, and merges a replacement only when the
retry completed (or was reserve-rejected).  Existing successful rows are never
called again and never overwritten.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hbws.data import load_split
from hbws.dsl import ENVELOPE_LIB
from hbws.ledger import BUDGET_TIERS
from hbws.protocol import evaluate

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
RUN_DIR = "glm4flash_envelope_test/cot_math_loose"
RETRY_DIR = "_glm4flash_retry/glm4flash_envelope_test/cot_math_loose"
PASS_LOG = EXP / "glm4flash_backfill_passes.jsonl"
GOOD = {"completed", "reserve_rejected"}


def append_pass(row: dict) -> None:
    with PASS_LOG.open("a") as f:
        f.write(json.dumps({"time": time.time(), **row}, sort_keys=True) + "\n")


def clamp_to_tier(wf: dict) -> dict:
    import copy

    wf = copy.deepcopy(wf)
    caps = BUDGET_TIERS["loose"]
    for node in wf["nodes"]:
        params = node.get("params")
        if params and "max_output_tokens" in params:
            share = caps.max_out_tokens // node["k"] if node["type"] == "vote" else caps.max_out_tokens
            params["max_output_tokens"] = max(64, min(params["max_output_tokens"], share))
    return wf


def refresh_summary(seed: int, rows: list[dict]) -> None:
    out_dir = EXP / RUN_DIR
    path = out_dir / f"summary_seed{seed}.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    n = len(rows)
    succ = sum(bool(r.get("success")) for r in rows)
    usd = [r.get("budget", {}).get("usd", 0) for r in rows]
    calls = [r.get("budget", {}).get("llm_calls", 0) for r in rows]
    toks = [r.get("budget", {}).get("in_tokens", 0) + r.get("budget", {}).get("out_tokens", 0) for r in rows]
    secs = sorted(r.get("budget", {}).get("wall_sec", 0) for r in rows)
    summary = {
        **old,
        "run": RUN_DIR,
        "seed": seed,
        "n": n,
        "success_rate": round(succ / n, 4) if n else 0,
        "usd_per_task": round(sum(usd) / n, 5) if n else 0,
        "usd_per_success": round(sum(usd) / succ, 5) if succ else None,
        "llm_calls_per_task": round(sum(calls) / n, 2) if n else 0,
        "tokens_per_task": round(sum(toks) / n, 1) if n else 0,
        "p50_sec": round(secs[n // 2], 2) if n else 0,
        "p95_sec": round(secs[int(n * 0.95)] if n > 1 else secs[0], 2) if n else 0,
        "total_usd": round(sum(usd), 4),
        "reserve_rejected": sum(r.get("status") == "reserve_rejected" for r in rows),
        "over_budget": sum(str(r.get("status", "")).startswith("budget_exceeded") for r in rows),
        "errors": sum(r.get("status") not in GOOD for r in rows),
    }
    path.write_text(json.dumps(summary, indent=2) + "\n")

    aggregate_path = EXP / "glm4flash_envelope_test_summary.jsonl"
    if aggregate_path.exists():
        aggregate = [json.loads(line) for line in aggregate_path.read_text().splitlines() if line.strip()]
        replaced = False
        for i, row in enumerate(aggregate):
            if row.get("run") == RUN_DIR and row.get("seed") == seed:
                aggregate[i] = {**row, **summary}
                replaced = True
        if not replaced:
            aggregate.append(summary)
        aggregate_path.write_text("".join(json.dumps(row) + "\n" for row in aggregate))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = ap.parse_args()

    tasks = {t["id"]: t for t in load_split("math", "test")[:150]}
    wf = clamp_to_tier(ENVELOPE_LIB["cot"]())
    append_pass({"phase": "start", "run": RUN_DIR, "seeds": args.seeds})

    for seed in args.seeds:
        authoritative = EXP / RUN_DIR / f"results_seed{seed}.jsonl"
        rows = [json.loads(line) for line in authoritative.read_text().splitlines() if line.strip()]
        bad_ids = [r["task_id"] for r in rows if r.get("status") not in GOOD]
        if not bad_ids:
            refresh_summary(seed, rows)
            print(f"seed={seed}: already complete", flush=True)
            continue

        retry_tasks = [tasks[task_id] for task_id in bad_ids]
        print(f"seed={seed}: retrying {len(retry_tasks)} failed rows", flush=True)
        append_pass({"phase": "seed_start", "seed": seed, "n_retry": len(retry_tasks)})
        evaluate(wf, retry_tasks, BUDGET_TIERS["loose"], run_name=RETRY_DIR,
                 seed=seed, use_cache=False, workers=args.workers)
        retry_path = EXP / RETRY_DIR / f"results_seed{seed}.jsonl"
        new_by_id = {
            r["task_id"]: r
            for r in (json.loads(line) for line in retry_path.read_text().splitlines() if line.strip())
            if r.get("status") in GOOD
        }
        merged = [new_by_id.get(r["task_id"], r) for r in rows]
        authoritative.write_text("".join(json.dumps(r) + "\n" for r in merged))
        refresh_summary(seed, merged)
        remaining = [r["task_id"] for r in merged if r.get("status") not in GOOD]
        append_pass({"phase": "seed_done", "seed": seed,
                     "n_replaced": len(new_by_id), "remaining": len(remaining),
                     "remaining_ids": remaining})
        print(f"seed={seed}: replaced={len(new_by_id)} remaining={len(remaining)}", flush=True)

    append_pass({"phase": "done", "run": RUN_DIR})


if __name__ == "__main__":
    main()
