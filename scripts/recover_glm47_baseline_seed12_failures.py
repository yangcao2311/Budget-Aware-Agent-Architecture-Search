#!/usr/bin/env python3
"""Recover execution-status failures in GLM-4.7 baseline seeds 1 and 2."""
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
BASE = EXP / "glm47flash_20260911_envelope_test/cot_math_tight"
REC = EXP / "glm47flash_20260914_baseline_seed12_recovery"
MANIFEST = REC / "manifest.json"
APPLIED = REC / "applied.json"
DECISION = EXP / "glm47flash_20260913_concurrency_decision.json"
SEEDS = (1, 2)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def result_path(seed: int) -> Path:
    return BASE / f"results_seed{seed}.jsonl"


def summary_path(seed: int) -> Path:
    return BASE / f"summary_seed{seed}.json"


def freeze() -> None:
    decision = json.loads(DECISION.read_text())
    if decision["workers_for_remaining_runs"] != 2:
        raise RuntimeError("frozen concurrency rule did not select workers=2")
    targets, input_hashes = {}, {}
    for seed in SEEDS:
        rp, sp = result_path(seed), summary_path(seed)
        rows = read_rows(rp)
        if len(rows) != 150:
            raise RuntimeError(f"baseline seed {seed} is incomplete")
        targets[str(seed)] = sorted(
            row["task_id"] for row in rows
            if str(row.get("status", "")).startswith("error")
        )
        input_hashes[str(rp.relative_to(ROOT))] = sha(rp)
        input_hashes[str(sp.relative_to(ROOT))] = sha(sp)
    sources = [
        Path(__file__), ROOT / "scripts/run_glm47_within_family.py",
        ROOT / "scripts/run_envelope.py", ROOT / "hbws/runner.py",
        ROOT / "hbws/llm.py", ROOT / "hbws/dsl.py",
        ROOT / "hbws/prompts.py", ROOT / "hbws/verify.py",
        ROOT / "hbws/ledger.py", ROOT / "hbws/protocol.py",
    ]
    payload = {
        "status": "frozen after baseline completion and before recovery calls",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": "seed 1/2 rows whose final status begins with error",
        "targets": targets,
        "n_targets": sum(map(len, targets.values())),
        "workers": 2,
        "model": frozen.MODEL,
        "endpoint": frozen.ENDPOINT,
        "policy": "same task/workflow/caps/seed/no-cache; execution-status-only replacement",
        "decision_sha256": sha(DECISION),
        "input_hashes": input_hashes,
        "source_hashes": {str(path.relative_to(ROOT)): sha(path) for path in sources},
    }
    REC.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists():
        old = json.loads(MANIFEST.read_text())
        payload["frozen_at_utc"] = old["frozen_at_utc"]
        if payload != old:
            raise RuntimeError("Refusing to alter frozen seed 1/2 recovery")
    else:
        MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("frozen", payload["n_targets"], "failed baseline rows")


def config(verify: bool = True) -> dict:
    cfg = json.loads(MANIFEST.read_text())
    if verify and not APPLIED.exists():
        if sha(DECISION) != cfg["decision_sha256"]:
            raise RuntimeError("frozen concurrency decision changed")
        for name, digest in cfg["input_hashes"].items():
            if sha(ROOT / name) != digest:
                raise RuntimeError(f"frozen baseline input changed: {name}")
        for name, digest in cfg["source_hashes"].items():
            if sha(ROOT / name) != digest:
                raise RuntimeError(f"frozen source changed: {name}")
    return cfg


def recovered(seed: int) -> dict[str, dict]:
    out = {}
    for path in sorted(REC.glob(f"runs/seed{seed}_pass*/results_seed{seed}.jsonl")):
        for row in read_rows(path):
            if row.get("status") == "completed":
                out[row["task_id"]] = row
    return out


def run() -> None:
    cfg = config()
    task_map = {task["id"]: task for task in load_split("math", "test")[:150]}
    workflow = clamp_to_tier(ENVELOPE_LIB["cot"](), BUDGET_TIERS["tight"])
    env = frozen.provider_env()
    env["LLM_ATTEMPT_LOG"] = str(REC / "attempts.jsonl")
    os.environ.update(env)
    for seed in SEEDS:
        done = recovered(seed)
        remaining = [task_id for task_id in cfg["targets"][str(seed)]
                     if task_id not in done]
        if not remaining:
            continue
        tasks = [task_map[task_id] for task_id in remaining]
        pass_no = len(list((REC / "runs").glob(f"seed{seed}_pass*")))
        name = f"glm47flash_20260914_baseline_seed12_recovery/runs/seed{seed}_pass{pass_no:02d}"
        print("seed", seed, "recover", len(tasks), "rows with workers=2", flush=True)
        evaluate(workflow, tasks, BUDGET_TIERS["tight"], run_name=name,
                 seed=seed, use_cache=False, workers=2)


def summarize(seed: int, rows: list[dict]) -> None:
    n = len(rows)
    budgets = [row.get("budget", {}) for row in rows]
    succ = sum(bool(row.get("success")) for row in rows)
    secs = sorted(float(b.get("wall_sec", 0)) for b in budgets)
    summary = {
        "run": str(BASE.relative_to(EXP)), "seed": seed, "n": n,
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
        "reserve_rejected": sum(row.get("status") == "reserve_rejected" for row in rows),
        "over_budget": sum(str(row.get("status", "")).startswith("budget_exceeded") for row in rows),
        "errors": sum(str(row.get("status", "")).startswith("error") for row in rows),
        "price_in_per_m": 0.0, "price_out_per_m": 0.0,
    }
    sp = summary_path(seed)
    tmp = sp.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2) + "\n")
    tmp.replace(sp)


def apply() -> None:
    if APPLIED.exists():
        print("recovery already applied")
        return
    cfg = config()
    missing = []
    overlays = {}
    for seed in SEEDS:
        overlays[seed] = recovered(seed)
        missing.extend((seed, task_id) for task_id in cfg["targets"][str(seed)]
                       if task_id not in overlays[seed])
    if missing:
        raise RuntimeError(f"recovery incomplete: {len(missing)} rows")
    final_hashes = {}
    for seed in SEEDS:
        rp = result_path(seed)
        final = [overlays[seed].get(row["task_id"], row) for row in read_rows(rp)]
        if len(final) != 150 or len({row["task_id"] for row in final}) != 150:
            raise RuntimeError(f"row-count invariant failed for seed {seed}")
        tmp = rp.with_suffix(".jsonl.tmp")
        tmp.write_text("\n".join(json.dumps(row, sort_keys=True) for row in final) + "\n")
        tmp.replace(rp)
        summarize(seed, final)
        final_hashes[str(rp.relative_to(ROOT))] = sha(rp)
    payload = {
        "status": "applied", "applied_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_replaced": cfg["n_targets"],
        "remaining_errors": 0, "final_result_hashes": final_hashes,
    }
    APPLIED.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("overwrote", cfg["n_targets"], "failed baseline rows")


def status() -> None:
    cfg = config(verify=False)
    unresolved = []
    for seed in SEEDS:
        done = recovered(seed)
        unresolved.extend((seed, task_id) for task_id in cfg["targets"][str(seed)]
                          if task_id not in done)
    print(json.dumps({"targets": cfg["n_targets"], "unresolved": len(unresolved),
                      "applied": APPLIED.exists()}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "run", "apply", "status"))
    args = parser.parse_args()
    {"freeze": freeze, "run": run, "apply": apply, "status": status}[args.stage]()


if __name__ == "__main__":
    main()
