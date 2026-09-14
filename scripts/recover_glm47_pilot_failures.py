#!/usr/bin/env python3
"""Recover the transport-failed GLM-4.7 baseline concurrency pilot.

Targets are frozen from execution status only.  Once DNS/provider access is
available, failed rows are rerun with the already-decided workers=2 and then
atomically overlaid on the authoritative baseline file.  The failed pilot and
attempt metadata remain internal operational records.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hbws.data import load_split
from hbws.dsl import ENVELOPE_LIB
from hbws.ledger import BUDGET_TIERS
from hbws.protocol import evaluate
from scripts.run_envelope import clamp_to_tier
import scripts.run_glm47_within_family as frozen


EXP = ROOT / "experiments"
SOURCE = EXP / "glm47flash_20260911_envelope_test/cot_math_tight/results_seed0.jsonl"
SOURCE_SUMMARY = SOURCE.with_name("summary_seed0.json")
REC = EXP / "glm47flash_20260913_pilot_recovery"
MANIFEST = REC / "manifest.json"
APPLIED = REC / "applied.json"
DECISION = EXP / "glm47flash_20260913_concurrency_decision.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def freeze() -> None:
    if not SOURCE.exists() or not SOURCE_SUMMARY.exists() or not DECISION.exists():
        raise RuntimeError("failed pilot result, summary, and concurrency decision are required")
    decision = json.loads(DECISION.read_text())
    if decision["workers_for_remaining_runs"] != 2:
        raise RuntimeError("the frozen operational rule did not select workers=2")
    failed = sorted(row["task_id"] for row in rows(SOURCE)
                    if str(row.get("status", "")).startswith("error"))
    if not failed:
        raise RuntimeError("pilot has no transport-failed rows to recover")
    sources = [
        Path(__file__), ROOT / "scripts/run_glm47_within_family.py",
        ROOT / "scripts/run_envelope.py", ROOT / "hbws/runner.py",
        ROOT / "hbws/llm.py", ROOT / "hbws/dsl.py",
        ROOT / "hbws/prompts.py", ROOT / "hbws/verify.py",
        ROOT / "hbws/ledger.py", ROOT / "hbws/protocol.py",
    ]
    payload = {
        "status": "frozen after failed pilot and before recovery calls",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": "pilot rows whose final status begins with error",
        "targets": failed,
        "n_targets": len(failed),
        "seed": 0,
        "workers": 2,
        "model": frozen.MODEL,
        "endpoint": frozen.ENDPOINT,
        "source_result_sha256": sha(SOURCE),
        "source_summary_sha256": sha(SOURCE_SUMMARY),
        "decision_sha256": sha(DECISION),
        "policy": "same task/workflow/caps/seed/no-cache; execution-status-only replacement",
        "source_hashes": {str(path.relative_to(ROOT)): sha(path) for path in sources},
    }
    REC.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists():
        old = json.loads(MANIFEST.read_text())
        payload["frozen_at_utc"] = old["frozen_at_utc"]
        if payload != old:
            raise RuntimeError("Refusing to alter frozen pilot recovery")
    else:
        MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("frozen", len(failed), "failed pilot rows")


def config(verify: bool = True) -> dict:
    cfg = json.loads(MANIFEST.read_text())
    if verify and not APPLIED.exists():
        if sha(SOURCE) != cfg["source_result_sha256"]:
            raise RuntimeError("frozen pilot result changed before recovery apply")
        if sha(SOURCE_SUMMARY) != cfg["source_summary_sha256"]:
            raise RuntimeError("frozen pilot summary changed before recovery apply")
        if sha(DECISION) != cfg["decision_sha256"]:
            raise RuntimeError("frozen concurrency decision changed")
        for name, digest in cfg["source_hashes"].items():
            if sha(ROOT / name) != digest:
                raise RuntimeError(f"frozen source changed: {name}")
    return cfg


def recovered() -> dict[str, dict]:
    out = {}
    for path in sorted(REC.glob("runs/pass*/results_seed0.jsonl")):
        for row in rows(path):
            if row.get("status") == "completed":
                out[row["task_id"]] = row
    return out


def run() -> None:
    cfg = config()
    done = recovered()
    remaining = [task_id for task_id in cfg["targets"] if task_id not in done]
    if not remaining:
        print("all recovery rows complete")
        return
    task_map = {task["id"]: task for task in load_split("math", "test")[:150]}
    tasks = [task_map[task_id] for task_id in remaining]
    workflow = clamp_to_tier(ENVELOPE_LIB["cot"](), BUDGET_TIERS["tight"])
    env = frozen.provider_env()
    env["LLM_ATTEMPT_LOG"] = str(REC / "attempts.jsonl")
    os.environ.update(env)
    pass_no = len(list((REC / "runs").glob("pass*")))
    name = f"glm47flash_20260913_pilot_recovery/runs/pass{pass_no:02d}"
    print("recover", len(tasks), "rows with workers=2", flush=True)
    evaluate(workflow, tasks, BUDGET_TIERS["tight"], run_name=name,
             seed=0, use_cache=False, workers=2)


def write_summary(final_rows: list[dict]) -> None:
    n = len(final_rows)
    succ = sum(bool(row.get("success")) for row in final_rows)
    budgets = [row.get("budget", {}) for row in final_rows]
    secs = sorted(float(b.get("wall_sec", 0)) for b in budgets)
    summary = {
        "run": str(SOURCE.parent.relative_to(EXP)), "seed": 0, "n": n,
        "success_rate": round(succ / n, 4),
        "usd_per_task": round(sum(float(b.get("usd", 0)) for b in budgets) / n, 5),
        "usd_per_success": (round(sum(float(b.get("usd", 0)) for b in budgets) / succ, 5)
                            if succ else None),
        "llm_calls_per_task": round(sum(int(b.get("llm_calls", 0)) for b in budgets) / n, 2),
        "tokens_per_task": round(sum(int(b.get("in_tokens", 0)) + int(b.get("out_tokens", 0))
                                     for b in budgets) / n, 1),
        "p50_sec": round(secs[n // 2], 2),
        "p95_sec": round(secs[min(int(n * .95), n - 1)], 2),
        "total_usd": round(sum(float(b.get("usd", 0)) for b in budgets), 4),
        "wall_min": round(sum(secs) / 60, 1),
        "reserve_rejected": sum(row.get("status") == "reserve_rejected" for row in final_rows),
        "over_budget": sum(str(row.get("status", "")).startswith("budget_exceeded")
                           for row in final_rows),
        "errors": sum(str(row.get("status", "")).startswith("error") for row in final_rows),
        "price_in_per_m": 0.0, "price_out_per_m": 0.0,
    }
    tmp = SOURCE_SUMMARY.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2) + "\n")
    tmp.replace(SOURCE_SUMMARY)


def apply() -> None:
    if APPLIED.exists():
        print("recovery already applied")
        return
    cfg = config()
    overlay = recovered()
    missing = [task_id for task_id in cfg["targets"] if task_id not in overlay]
    if missing:
        raise RuntimeError(f"recovery incomplete: {len(missing)} rows")
    original = rows(SOURCE)
    final_rows = [overlay.get(row["task_id"], row) for row in original]
    if len(final_rows) != 150 or len({row["task_id"] for row in final_rows}) != 150:
        raise RuntimeError("row-count invariant failed")
    tmp = SOURCE.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(json.dumps(row, sort_keys=True) for row in final_rows) + "\n")
    tmp.replace(SOURCE)
    write_summary(final_rows)
    payload = {
        "status": "applied", "applied_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_replaced": len(cfg["targets"]),
        "remaining_errors": sum(str(row.get("status", "")).startswith("error")
                                for row in final_rows),
        "final_result_sha256": sha(SOURCE),
    }
    APPLIED.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("overwrote", len(cfg["targets"]), "failed pilot rows")


def status() -> None:
    cfg = config(verify=False)
    done = recovered()
    unresolved = [task_id for task_id in cfg["targets"] if task_id not in done]
    print(json.dumps({"targets": len(cfg["targets"]), "unresolved": len(unresolved),
                      "applied": APPLIED.exists()}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "run", "apply", "status"))
    args = parser.parse_args()
    {"freeze": freeze, "run": run, "apply": apply, "status": status}[args.stage]()


if __name__ == "__main__":
    main()
