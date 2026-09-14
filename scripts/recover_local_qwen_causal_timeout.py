#!/usr/bin/env python3
"""Recover and overwrite the sole transport-error row in local Qwen causal."""
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
from hbws.dsl import wf_incumbent_refine
from hbws.ledger import BUDGET_TIERS
from hbws.protocol import evaluate
from hbws import llm
from openai import OpenAI
from scripts.run_local_qwen_control import CAUSAL_TAG, env


EXP = ROOT / "experiments"
SOURCE = (EXP / CAUSAL_TAG / "arm2_samepolicy_code_loose" /
          "results_seed2.jsonl")
SUMMARY = SOURCE.with_name("summary_seed2.json")
REC_TAG = "qwen25coder7b_local_causal_recovery_20260912"
REC = EXP / REC_TAG
MANIFEST = REC / "manifest.json"
APPLIED = REC / "applied.json"
TARGET = "mbpp_92"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def freeze() -> None:
    REC.mkdir(parents=True, exist_ok=True)
    original = rows(SOURCE)
    failures = sorted(
        row["task_id"] for row in original
        if str(row.get("status", "")).startswith("error")
    )
    if len(original) != 150 or failures != [TARGET]:
        raise RuntimeError(
            f"expected 150 rows and sole failure {TARGET}; got "
            f"n={len(original)}, failures={failures}")
    payload = {
        "status": "frozen before recovery call",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "target": TARGET,
        "arm": "same_policy_regenerate",
        "seed": 2,
        "selection": "sole row whose original status begins with error",
        "policy": "same task, workflow, seed, caps, workers=1, and no cache",
        "source_hash": digest(SOURCE),
        "summary_hash": digest(SUMMARY),
        "source_files": {
            "scripts/run_local_qwen_control.py": digest(
                ROOT / "scripts/run_local_qwen_control.py"),
            "hbws/dsl.py": digest(ROOT / "hbws/dsl.py"),
            "hbws/runner.py": digest(ROOT / "hbws/runner.py"),
            "hbws/protocol.py": digest(ROOT / "hbws/protocol.py"),
        },
    }
    if MANIFEST.exists():
        old = json.loads(MANIFEST.read_text())
        payload["frozen_at_utc"] = old["frozen_at_utc"]
        if payload != old:
            raise RuntimeError("refusing to alter frozen local recovery")
    else:
        MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("frozen", TARGET)


def config() -> dict:
    cfg = json.loads(MANIFEST.read_text())
    if not APPLIED.exists():
        if digest(SOURCE) != cfg["source_hash"]:
            raise RuntimeError("source result changed after recovery freeze")
        if digest(SUMMARY) != cfg["summary_hash"]:
            raise RuntimeError("source summary changed after recovery freeze")
    return cfg


def run() -> None:
    config()
    local_env = env()
    os.environ.update(local_env)
    os.environ["LLM_MAX_RETRIES"] = "5"
    os.environ["LLM_BACKOFF_SCHEDULE"] = "2,5,15,30"
    os.environ["LLM_ATTEMPT_LOG"] = str(REC / "attempts.jsonl")
    # This target's deterministic completion takes about 122 seconds on the
    # local Mac.  The shared client's 90-second default caused the original
    # transport timeout even though the server completed normally.  Extend
    # only the recovery process's transport wait; inference inputs are fixed.
    recovery_client = OpenAI(
        api_key=local_env["MLX_API_KEY"],
        base_url=local_env["MLX_BASE_URL"],
        timeout=600.0,
    )
    # evaluate() executes in a ThreadPoolExecutor, so assigning only the main
    # thread's thread-local client would leave the worker on the 90 s default.
    # Override the process-local factory instead; this script has one worker.
    llm._client = lambda: recovery_client
    task = next(task for task in load_split("code", "test")[:150]
                if task["id"] == TARGET)
    evaluate(wf_incumbent_refine(), [task], BUDGET_TIERS["loose"],
             run_name=REC_TAG, seed=2, use_cache=False, workers=1)


def recompute_summary(final_rows: list[dict]) -> None:
    n = len(final_rows)
    succ = sum(bool(row.get("success")) for row in final_rows)
    usd = [row.get("budget", {}).get("usd", 0) for row in final_rows]
    calls = [row.get("budget", {}).get("llm_calls", 0) for row in final_rows]
    toks = [row.get("budget", {}).get("in_tokens", 0)
            + row.get("budget", {}).get("out_tokens", 0)
            for row in final_rows]
    secs = sorted(row.get("budget", {}).get("wall_sec", 0)
                  for row in final_rows)
    old = json.loads(SUMMARY.read_text())
    recovery_summary = json.loads(
        (REC / "summary_seed2.json").read_text())
    summary = {
        "run": str(SOURCE.parent.relative_to(EXP)),
        "seed": 2,
        "n": n,
        "success_rate": round(succ / n, 4),
        "usd_per_task": round(sum(usd) / n, 5),
        "usd_per_success": round(sum(usd) / succ, 5) if succ else None,
        "llm_calls_per_task": round(sum(calls) / n, 2),
        "tokens_per_task": round(sum(toks) / n, 1),
        "p50_sec": round(secs[n // 2], 2),
        "p95_sec": round(secs[int(n * 0.95)], 2),
        "total_usd": round(sum(usd), 4),
        "wall_min": round(float(old.get("wall_min", 0))
                          + float(recovery_summary.get("wall_min", 0)), 1),
        "reserve_rejected": sum(row.get("status") == "reserve_rejected"
                                for row in final_rows),
        "over_budget": sum(str(row.get("status", "")).startswith(
            "budget_exceeded") for row in final_rows),
        "errors": sum(str(row.get("status", "")).startswith("error")
                      for row in final_rows),
        "price_in_per_m": 0.0,
        "price_out_per_m": 0.0,
    }
    temporary = SUMMARY.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(SUMMARY)


def apply_recovery() -> None:
    if APPLIED.exists():
        print("already applied")
        return
    cfg = config()
    recovered = rows(REC / "results_seed2.jsonl")
    if len(recovered) != 1 or recovered[0]["task_id"] != TARGET:
        raise RuntimeError("recovery output does not contain exactly the target")
    replacement = recovered[0]
    if replacement.get("status") != "completed":
        raise RuntimeError(f"recovery did not complete: {replacement['status']}")

    original_lines = [line for line in SOURCE.read_text().splitlines() if line]
    final_lines = []
    final_rows = []
    replaced = 0
    for line in original_lines:
        row = json.loads(line)
        if row["task_id"] == TARGET:
            final_lines.append(json.dumps(replacement, sort_keys=True))
            final_rows.append(replacement)
            replaced += 1
        else:
            final_lines.append(line)
            final_rows.append(row)
    if replaced != 1 or len(final_rows) != 150:
        raise RuntimeError("replacement row-count invariant failed")

    temporary = SOURCE.with_suffix(".jsonl.tmp")
    temporary.write_text("\n".join(final_lines) + "\n")
    temporary.replace(SOURCE)
    recompute_summary(final_rows)
    payload = {
        "status": "applied",
        "applied_at_utc": datetime.now(timezone.utc).isoformat(),
        "target": cfg["target"],
        "remaining_errors": 0,
        "final_source_hash": digest(SOURCE),
    }
    APPLIED.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("overwrote recovered row", TARGET)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "run", "apply", "all"))
    args = parser.parse_args()
    if args.stage in ("freeze", "all"):
        freeze()
    if args.stage in ("run", "all"):
        run()
    if args.stage in ("apply", "all"):
        apply_recovery()
