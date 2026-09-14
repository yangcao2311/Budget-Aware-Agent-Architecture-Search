#!/usr/bin/env python3
"""Recover transport-failed rows in the GLM math/tight nonbinding run.

The recovery target set is frozen only after the nine original arm/seed files
are complete.  Selection depends solely on execution status.  Recovery calls
use the original task, arm, seed, workflow, nonbinding caps, and cache setting.
Once every target has a completed replacement, ``apply`` atomically overlays
those rows onto the original result files and recomputes their summaries.

Provider-attempt logs and the recovery manifest remain internal audit records;
the authoritative result files contain the final complete 150-row cells.
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
from hbws.dsl import _wf_verify_refine, wf_assign_refine, wf_incumbent_refine_cot
from hbws.protocol import evaluate
from scripts.run_glm4flash_replication import load_key
from scripts.run_math_tight_nonbinding import BASELINE_PREFIX, CAPS, TAG
from scripts.run_provenance_causal import load_baseline_by_seed


EXP = ROOT / "experiments"
REC_TAG = "glm4flash_math_tight_nonbinding_recovery_20260912"
REC = EXP / REC_TAG
MANIFEST = REC / "manifest.json"
APPLIED = REC / "applied.json"
ARMS = {
    "arm1_assign": wf_assign_refine,
    "arm2_samepolicy": wf_incumbent_refine_cot,
    "arm3_diffpolicy": lambda: _wf_verify_refine(3),
}
SEEDS = (0, 1, 2)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def source_path(arm: str, seed: int) -> Path:
    return (EXP / TAG / f"{arm}_math_tight_nonbinding" /
            f"results_seed{seed}.jsonl")


def source_summary_path(arm: str, seed: int) -> Path:
    return source_path(arm, seed).with_name(f"summary_seed{seed}.json")


def freeze() -> None:
    REC.mkdir(parents=True, exist_ok=True)
    targets: dict[str, list[str]] = {}
    input_hashes: dict[str, str] = {}
    for arm in ARMS:
        for seed in SEEDS:
            path = source_path(arm, seed)
            if not path.exists():
                raise RuntimeError(f"original run is incomplete: missing {path}")
            rows = read_rows(path)
            if len(rows) != 150:
                raise RuntimeError(
                    f"original run is incomplete: {path} has {len(rows)} rows")
            key = f"{arm}/seed{seed}"
            targets[key] = sorted(
                row["task_id"] for row in rows
                if str(row.get("status", "")).startswith("error")
            )
            input_hashes[str(path.relative_to(ROOT))] = digest(path)
            summary = source_summary_path(arm, seed)
            if not summary.exists():
                raise RuntimeError(f"original summary is missing: {summary}")
            input_hashes[str(summary.relative_to(ROOT))] = digest(summary)

    sources = [
        Path(__file__),
        ROOT / "scripts/run_math_tight_nonbinding.py",
        ROOT / "hbws/dsl.py",
        ROOT / "hbws/runner.py",
        ROOT / "hbws/protocol.py",
        ROOT / "hbws/verify.py",
    ]
    payload = {
        "status": "frozen after original run and before recovery calls",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "glm-4-flash-250414",
        "endpoint": "https://open.bigmodel.cn/api/paas/v4",
        "selection": "rows whose original status begins with error",
        "policy": (
            "rerun only frozen failed rows with unchanged task, arm, seed, "
            "workflow, caps, and no-cache setting; workers=1"
        ),
        "targets": targets,
        "n_targets": sum(map(len, targets.values())),
        "input_hashes": input_hashes,
        "source_hashes": {
            str(path.relative_to(ROOT)): digest(path) for path in sources
        },
    }
    if MANIFEST.exists():
        old = json.loads(MANIFEST.read_text())
        payload["frozen_at_utc"] = old["frozen_at_utc"]
        if payload != old:
            raise RuntimeError("Refusing to alter frozen recovery targets")
    else:
        MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("frozen", payload["n_targets"], "transport-failed rows")


def config(*, verify_inputs: bool = True) -> dict:
    if not MANIFEST.exists():
        raise RuntimeError("run the freeze stage first")
    cfg = json.loads(MANIFEST.read_text())
    if verify_inputs and not APPLIED.exists():
        for name, sha in cfg["input_hashes"].items():
            if digest(ROOT / name) != sha:
                raise RuntimeError(f"frozen input changed: {name}")
        for name, sha in cfg["source_hashes"].items():
            if digest(ROOT / name) != sha:
                raise RuntimeError(f"frozen source changed: {name}")
    return cfg


def provider_env() -> None:
    os.environ.update({
        "LLM_PROVIDER": "glm",
        "GLM_API_KEY": load_key(),
        "GLM_BASE_URL": "https://open.bigmodel.cn/api/paas/v4",
        "GLM_MODEL": "glm-4-flash-250414",
        "LLM_DISABLE_SEED": "0",
        "LLM_PRICE_IN_PER_M": "0",
        "LLM_PRICE_OUT_PER_M": "0",
        "LLM_BACKOFF_SCHEDULE": "10,30,90",
        "LLM_MAX_RETRIES": "5",
        "LLM_ATTEMPT_LOG": str(REC / "attempts.jsonl"),
    })
    os.environ.pop("LLM_REASONING_EFFORT", None)


def recovered(arm: str, seed: int) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    pattern = f"runs/{arm}_seed{seed}_pass*/results_seed{seed}.jsonl"
    for path in sorted(REC.glob(pattern)):
        for row in read_rows(path):
            if row.get("status") == "completed":
                rows[row["task_id"]] = row
    return rows


def next_pass(arm: str, seed: int) -> int:
    return len(list((REC / "runs").glob(f"{arm}_seed{seed}_pass*")))


def run(workers: int) -> None:
    if workers != 1:
        raise RuntimeError("recovery is frozen at workers=1")
    cfg = config()
    provider_env()
    task_map = {task["id"]: task for task in load_split("math", "test")[:150]}
    references = load_baseline_by_seed(
        "math", "tight", BASELINE_PREFIX, SEEDS)
    (REC / "runs").mkdir(parents=True, exist_ok=True)

    for arm, builder in ARMS.items():
        for seed in SEEDS:
            targets = cfg["targets"][f"{arm}/seed{seed}"]
            done = recovered(arm, seed)
            remaining = [task_id for task_id in targets if task_id not in done]
            if not remaining:
                continue
            if arm == "arm1_assign":
                tasks = [
                    {**task_map[task_id],
                     "_assign_solution": references[seed][task_id][0]}
                    for task_id in remaining
                ]
            else:
                tasks = [task_map[task_id] for task_id in remaining]
            pass_no = next_pass(arm, seed)
            run_name = (f"{REC_TAG}/runs/"
                        f"{arm}_seed{seed}_pass{pass_no:02d}")
            print(arm, "seed", seed, "retry", len(tasks), flush=True)
            evaluate(builder(), tasks, CAPS, run_name=run_name, seed=seed,
                     use_cache=False, workers=1)
    print("recovery pass complete; rerun if status reports unresolved rows")


def recovery_wall_minutes(arm: str, seed: int) -> float:
    total = 0.0
    pattern = f"runs/{arm}_seed{seed}_pass*/summary_seed{seed}.json"
    for path in REC.glob(pattern):
        total += float(json.loads(path.read_text()).get("wall_min", 0))
    return total


def summarize(path: Path, rows: list[dict], added_wall_min: float) -> None:
    seed = int(path.stem.replace("results_seed", ""))
    old = json.loads(source_summary_path(path.parent.name.split("_math_")[0], seed).read_text())
    n = len(rows)
    succ = sum(bool(row.get("success")) for row in rows)
    usd = [row.get("budget", {}).get("usd", 0) for row in rows]
    calls = [row.get("budget", {}).get("llm_calls", 0) for row in rows]
    toks = [
        row.get("budget", {}).get("in_tokens", 0)
        + row.get("budget", {}).get("out_tokens", 0)
        for row in rows
    ]
    secs = sorted(row.get("budget", {}).get("wall_sec", 0) for row in rows)
    summary = {
        "run": str(path.parent.relative_to(EXP)),
        "seed": seed,
        "n": n,
        "success_rate": round(succ / n, 4),
        "usd_per_task": round(sum(usd) / n, 5),
        "usd_per_success": round(sum(usd) / succ, 5) if succ else None,
        "llm_calls_per_task": round(sum(calls) / n, 2),
        "tokens_per_task": round(sum(toks) / n, 1),
        "p50_sec": round(secs[n // 2], 2) if secs else 0,
        "p95_sec": round(secs[min(int(n * 0.95), n - 1)], 2) if secs else 0,
        "total_usd": round(sum(usd), 4),
        "wall_min": round(float(old.get("wall_min", 0)) + added_wall_min, 1),
        "reserve_rejected": sum(row.get("status") == "reserve_rejected"
                                for row in rows),
        "over_budget": sum(str(row.get("status", "")).startswith(
            "budget_exceeded") for row in rows),
        "errors": sum(str(row.get("status", "")).startswith("error")
                      for row in rows),
        "price_in_per_m": 0.0,
        "price_out_per_m": 0.0,
    }
    summary_path = path.with_name(f"summary_seed{seed}.json")
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(summary_path)


def apply_recovery() -> None:
    if APPLIED.exists():
        print("recovery already applied")
        return
    cfg = config()
    missing = []
    overlays: dict[tuple[str, int], dict[str, dict]] = {}
    for arm in ARMS:
        for seed in SEEDS:
            overlay = recovered(arm, seed)
            overlays[(arm, seed)] = overlay
            for task_id in cfg["targets"][f"{arm}/seed{seed}"]:
                if task_id not in overlay:
                    missing.append((arm, seed, task_id))
    if missing:
        raise RuntimeError(
            f"recovery incomplete: {len(missing)} rows; first={missing[:5]}")

    final_hashes = {}
    for arm in ARMS:
        for seed in SEEDS:
            path = source_path(arm, seed)
            original_lines = [line for line in path.read_text().splitlines() if line]
            original_rows = [json.loads(line) for line in original_lines]
            if len(original_rows) != 150 or len({
                    row["task_id"] for row in original_rows}) != 150:
                raise RuntimeError(f"row-count invariant failed for {path}")
            overlay = overlays[(arm, seed)]
            final_lines = []
            rows = []
            for line, row in zip(original_lines, original_rows):
                replacement = overlay.get(row["task_id"])
                if replacement is None:
                    final_lines.append(line)
                    rows.append(row)
                else:
                    final_lines.append(json.dumps(replacement, sort_keys=True))
                    rows.append(replacement)
            temporary = path.with_suffix(".jsonl.tmp")
            temporary.write_text("\n".join(final_lines) + "\n")
            temporary.replace(path)
            summarize(path, rows, recovery_wall_minutes(arm, seed))
            final_hashes[str(path.relative_to(ROOT))] = digest(path)

    payload = {
        "status": "applied",
        "applied_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_replaced": cfg["n_targets"],
        "remaining_errors": 0,
        "final_result_hashes": final_hashes,
    }
    APPLIED.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("overwrote", cfg["n_targets"], "failed rows with completed recoveries")


def status() -> None:
    cfg = config(verify_inputs=False)
    unresolved = []
    for arm in ARMS:
        for seed in SEEDS:
            done = recovered(arm, seed)
            unresolved.extend(
                (arm, seed, task_id)
                for task_id in cfg["targets"][f"{arm}/seed{seed}"]
                if task_id not in done
            )
    print(json.dumps({
        "targets": cfg["n_targets"],
        "unresolved": len(unresolved),
        "applied": APPLIED.exists(),
        "first_unresolved": unresolved[:10],
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "run", "apply", "status"))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    {
        "freeze": freeze,
        "run": lambda: run(args.workers),
        "apply": apply_recovery,
        "status": status,
    }[args.stage]()
