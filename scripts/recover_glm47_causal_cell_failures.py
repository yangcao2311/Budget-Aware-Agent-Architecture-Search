#!/usr/bin/env python3
"""Freeze and recover transport failures for one GLM-4.7 causal cell."""
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
import scripts.run_glm47_within_family as frozen
from scripts.run_provenance_causal import load_baseline_by_seed

EXP = ROOT / "experiments"
REC_ROOT = EXP / "glm47flash_20260914_causal_recovery"
BASELINE_PREFIX = frozen.PREFIX
ARMS = {
    "arm1_assign": wf_assign_refine,
    "arm2_samepolicy": wf_incumbent_refine_cot,
    "arm3_diffpolicy": lambda: _wf_verify_refine(3),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def source(arm: str, seed: int) -> Path:
    return EXP / frozen.TAG / f"{arm}_math_tight_nonbinding" / f"results_seed{seed}.jsonl"


def summary(arm: str, seed: int) -> Path:
    return source(arm, seed).with_name(f"summary_seed{seed}.json")


def cell_dir(arm: str, seed: int) -> Path:
    return REC_ROOT / f"{arm}_seed{seed}"


def manifest(arm: str, seed: int) -> Path:
    return cell_dir(arm, seed) / "manifest.json"


def applied(arm: str, seed: int) -> Path:
    return cell_dir(arm, seed) / "applied.json"


def freeze(arm: str, seed: int) -> None:
    rp, sp = source(arm, seed), summary(arm, seed)
    rows = read_rows(rp)
    if len(rows) != 150:
        raise RuntimeError("causal cell must contain exactly 150 rows")
    targets = sorted(row["task_id"] for row in rows
                     if str(row.get("status", "")).startswith("error"))
    if not targets:
        raise RuntimeError("cell has no execution-status failures")
    baseline = EXP / f"{BASELINE_PREFIX}envelope_test/cot_math_tight/results_seed{seed}.jsonl"
    sources = [
        Path(__file__), ROOT / "scripts/run_glm47_within_family.py",
        ROOT / "scripts/run_provenance_causal.py", ROOT / "hbws/runner.py",
        ROOT / "hbws/llm.py", ROOT / "hbws/dsl.py", ROOT / "hbws/prompts.py",
        ROOT / "hbws/verify.py", ROOT / "hbws/ledger.py", ROOT / "hbws/protocol.py",
    ]
    payload = {
        "status": "frozen after cell completion and before recovery calls",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "arm": arm, "seed": seed, "targets": targets, "n_targets": len(targets),
        "selection": "rows whose final status begins with error",
        "workers": 2, "model": frozen.MODEL, "endpoint": frozen.ENDPOINT,
        "policy": "same task/arm/workflow/caps/reference/seed/no-cache; status-only replacement",
        "source_result_sha256": sha(rp), "source_summary_sha256": sha(sp),
        "baseline_sha256": sha(baseline),
        "source_hashes": {str(path.relative_to(ROOT)): sha(path) for path in sources},
    }
    mp = manifest(arm, seed)
    mp.parent.mkdir(parents=True, exist_ok=True)
    if mp.exists():
        old = json.loads(mp.read_text())
        payload["frozen_at_utc"] = old["frozen_at_utc"]
        if payload != old:
            raise RuntimeError("Refusing to alter frozen cell recovery")
    else:
        mp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("frozen", len(targets), arm, "seed", seed, "failures")


def config(arm: str, seed: int, verify: bool = True) -> dict:
    cfg = json.loads(manifest(arm, seed).read_text())
    if verify and not applied(arm, seed).exists():
        if sha(source(arm, seed)) != cfg["source_result_sha256"]:
            raise RuntimeError("frozen causal result changed")
        if sha(summary(arm, seed)) != cfg["source_summary_sha256"]:
            raise RuntimeError("frozen causal summary changed")
        baseline = EXP / f"{BASELINE_PREFIX}envelope_test/cot_math_tight/results_seed{seed}.jsonl"
        if sha(baseline) != cfg["baseline_sha256"]:
            raise RuntimeError("frozen baseline reference changed")
        for name, digest in cfg["source_hashes"].items():
            if sha(ROOT / name) != digest:
                raise RuntimeError(f"frozen source changed: {name}")
    return cfg


def recovered(arm: str, seed: int) -> dict[str, dict]:
    out = {}
    for path in sorted(cell_dir(arm, seed).glob(f"runs/pass*/results_seed{seed}.jsonl")):
        for row in read_rows(path):
            if row.get("status") == "completed":
                out[row["task_id"]] = row
    return out


def run(arm: str, seed: int) -> None:
    cfg = config(arm, seed)
    done = recovered(arm, seed)
    remaining = [task_id for task_id in cfg["targets"] if task_id not in done]
    if not remaining:
        print("all recovery rows complete")
        return
    task_map = {task["id"]: task for task in load_split("math", "test")[:150]}
    tasks = [task_map[task_id] for task_id in remaining]
    if arm == "arm1_assign":
        refs = load_baseline_by_seed("math", "tight", BASELINE_PREFIX, (seed,))[seed]
        tasks = [{**task, "_assign_solution": refs[task["id"]][0]} for task in tasks]
    workflow = ARMS[arm]()
    env = frozen.provider_env()
    env["LLM_ATTEMPT_LOG"] = str(cell_dir(arm, seed) / "attempts.jsonl")
    os.environ.update(env)
    pass_no = len(list((cell_dir(arm, seed) / "runs").glob("pass*")))
    name = (f"glm47flash_20260914_causal_recovery/{arm}_seed{seed}/"
            f"runs/pass{pass_no:02d}")
    print("recover", len(tasks), arm, "seed", seed, "workers=2", flush=True)
    evaluate(workflow, tasks, frozen.CAPS, run_name=name, seed=seed,
             use_cache=False, workers=2)


def write_summary(arm: str, seed: int, rows: list[dict]) -> None:
    n = len(rows)
    budgets = [row.get("budget", {}) for row in rows]
    succ = sum(bool(row.get("success")) for row in rows)
    secs = sorted(float(b.get("wall_sec", 0)) for b in budgets)
    payload = {
        "run": str(source(arm, seed).parent.relative_to(EXP)), "seed": seed, "n": n,
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
    sp = summary(arm, seed)
    tmp = sp.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(sp)


def apply(arm: str, seed: int) -> None:
    ap = applied(arm, seed)
    if ap.exists():
        print("recovery already applied")
        return
    cfg = config(arm, seed)
    overlay = recovered(arm, seed)
    missing = [task_id for task_id in cfg["targets"] if task_id not in overlay]
    if missing:
        raise RuntimeError(f"recovery incomplete: {len(missing)} rows")
    rp = source(arm, seed)
    final = [overlay.get(row["task_id"], row) for row in read_rows(rp)]
    if len(final) != 150 or len({row["task_id"] for row in final}) != 150:
        raise RuntimeError("row-count invariant failed")
    tmp = rp.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(json.dumps(row, sort_keys=True) for row in final) + "\n")
    tmp.replace(rp)
    write_summary(arm, seed, final)
    payload = {"status": "applied", "applied_at_utc": datetime.now(timezone.utc).isoformat(),
               "n_replaced": cfg["n_targets"], "remaining_errors": 0,
               "final_result_sha256": sha(rp)}
    ap.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("overwrote", cfg["n_targets"], arm, "seed", seed, "failures")


def status(arm: str, seed: int) -> None:
    cfg = config(arm, seed, verify=False)
    done = recovered(arm, seed)
    unresolved = [task_id for task_id in cfg["targets"] if task_id not in done]
    print(json.dumps({"targets": cfg["n_targets"], "unresolved": len(unresolved),
                      "applied": applied(arm, seed).exists()}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "run", "apply", "status"))
    parser.add_argument("--arm", required=True, choices=tuple(ARMS))
    parser.add_argument("--seed", required=True, type=int, choices=(0, 1, 2))
    args = parser.parse_args()
    {"freeze": freeze, "run": run, "apply": apply, "status": status}[args.stage](
        args.arm, args.seed)


if __name__ == "__main__":
    main()
