#!/usr/bin/env python3
"""Frozen addendum for causal-arm provider/runtime failures.

Run the freeze stage only after the original causal matrix has finished.
Baseline failures are handled by recover_repaired_matrix_failures.py.  This
addendum freezes any additional task/seed whose baseline is valid but at least
one causal arm failed, reruns all three interventions for that pair, and
materializes a final causal directory without mutating raw logs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hbws.data import load_split
from hbws.dsl import wf_assign_refine
from hbws.ledger import BUDGET_TIERS
from hbws.protocol import evaluate
from scripts.recover_repaired_matrix_failures import (
    ARM_BUILDERS,
    CAUSAL_TAG,
    CELLS,
    CLEAN_CAUSAL,
    CLEAN_PREFIX,
    GOOD,
    REC as BASE_RECOVERY,
    SEEDS,
    arm_path,
    base_path,
    read_rows,
)

EXP = ROOT / "experiments"
REC = EXP / "glm4flash_repaired_matrix_causal_recovery_20260910"
MANIFEST = REC / "manifest.json"
FINAL_CAUSAL = "glm4flash_repaired_matrix_final_20260910_causal"
FINAL_ANALYSIS = "glm4flash_repaired_matrix_final_20260910_analysis.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> None:
    REC.mkdir(parents=True, exist_ok=True)
    baseline_cfg = json.loads((BASE_RECOVERY / "manifest.json").read_text())
    baseline_targets = {
        (key, task_id)
        for key, task_ids in baseline_cfg["targets"].items()
        for task_id in task_ids
    }
    targets: dict[str, list[str]] = {}
    input_hashes: dict[str, str] = {}
    for family, tier in CELLS:
        for seed in SEEDS:
            key = f"{family}/{tier}/seed{seed}"
            bad = set()
            base = {r["task_id"]: r for r in
                    read_rows(base_path(family, tier, seed))}
            for arm in ARM_BUILDERS:
                path = arm_path(arm, family, tier, seed)
                input_hashes[str(path.relative_to(ROOT))] = digest(path)
                bad.update(r["task_id"] for r in read_rows(path)
                           if r.get("status") not in GOOD)
            causal_only = sorted(
                task_id for task_id in bad
                if (key, task_id) not in baseline_targets
                and base[task_id].get("status") in GOOD
            )
            targets[key] = causal_only
            baseline_path = base_path(family, tier, seed)
            input_hashes[str(baseline_path.relative_to(ROOT))] = digest(baseline_path)
    sources = [
        Path(__file__),
        ROOT / "scripts/recover_repaired_matrix_failures.py",
        ROOT / "hbws/dsl.py",
        ROOT / "hbws/runner.py",
        ROOT / "hbws/protocol.py",
        ROOT / "hbws/verify.py",
    ]
    payload = {
        "status": "frozen after original causal pass and before addendum calls",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "glm-4-flash-250414",
        "reason": "valid baseline but provider/runtime failure in at least one causal arm",
        "policy": "rerun all three interventions for each frozen task/seed; originals immutable",
        "targets": targets,
        "n_target_pairs": sum(map(len, targets.values())),
        "input_hashes": input_hashes,
        "source_hashes": {str(path.relative_to(ROOT)): digest(path)
                          for path in sources},
        "source_causal_tag": CAUSAL_TAG,
        "baseline_clean_prefix": CLEAN_PREFIX,
        "baseline_recovered_causal_tag": CLEAN_CAUSAL,
        "final_causal_tag": FINAL_CAUSAL,
    }
    if MANIFEST.exists():
        old = json.loads(MANIFEST.read_text())
        payload["frozen_at_utc"] = old["frozen_at_utc"]
        if payload != old:
            raise RuntimeError("Refusing to alter frozen causal recovery")
    else:
        MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("frozen", payload["n_target_pairs"], "causal-only failed pairs")


def config() -> dict:
    cfg = json.loads(MANIFEST.read_text())
    for name, sha in cfg["input_hashes"].items():
        assert digest(ROOT / name) == sha, f"original input changed: {name}"
    for name, sha in cfg["source_hashes"].items():
        assert digest(ROOT / name) == sha, f"frozen source changed: {name}"
    return cfg


def provider_env() -> None:
    from run_glm4flash_replication import load_key
    os.environ.update(
        LLM_PROVIDER="glm", GLM_API_KEY=load_key(),
        GLM_BASE_URL="https://open.bigmodel.cn/api/paas/v4",
        GLM_MODEL="glm-4-flash-250414", LLM_DISABLE_SEED="0",
        LLM_PRICE_IN_PER_M="0", LLM_PRICE_OUT_PER_M="0",
        LLM_BACKOFF_SCHEDULE="10,30,90", LLM_MAX_RETRIES="5")


def recovered(arm: str, family: str, tier: str, seed: int) -> dict[str, dict]:
    rows = {}
    pattern = f"runs/{arm}_{family}_{tier}_seed{seed}_pass*/results_seed{seed}.jsonl"
    for path in sorted(REC.glob(pattern)):
        for row in read_rows(path):
            if row.get("status") in GOOD:
                rows[row["task_id"]] = row
    return rows


def run(workers: int) -> None:
    cfg = config()
    provider_env()
    for family, tier in CELLS:
        task_map = {t["id"]: t for t in load_split(family, "test")[:150]}
        for seed in SEEDS:
            key = f"{family}/{tier}/seed{seed}"
            targets = cfg["targets"][key]
            if not targets:
                continue
            base = {r["task_id"]: r for r in
                    read_rows(base_path(family, tier, seed))}
            for arm, builders in ARM_BUILDERS.items():
                done = recovered(arm, family, tier, seed)
                remaining = [task_id for task_id in targets if task_id not in done]
                if not remaining:
                    continue
                if arm == "arm1_assign":
                    tasks = [{**task_map[t], "_assign_solution": base[t]["solution"]}
                             for t in remaining]
                    wf = wf_assign_refine()
                else:
                    tasks = [task_map[t] for t in remaining]
                    wf = builders[family]()
                pass_no = len(list(REC.glob(
                    f"runs/{arm}_{family}_{tier}_seed{seed}_pass*")))
                name = (f"glm4flash_repaired_matrix_causal_recovery_20260910/"
                        f"runs/{arm}_{family}_{tier}_seed{seed}_pass{pass_no:02d}")
                print(arm, key, "retry", len(tasks), flush=True)
                evaluate(wf, tasks, BUDGET_TIERS[tier], run_name=name,
                         seed=seed, use_cache=False, workers=workers)
    print("causal recovery pass complete", flush=True)


def materialize() -> None:
    cfg = config()
    missing = []
    for family, tier in CELLS:
        for seed in SEEDS:
            key = f"{family}/{tier}/seed{seed}"
            targets = cfg["targets"][key]
            for arm in ARM_BUILDERS:
                overlay = recovered(arm, family, tier, seed)
                for task_id in targets:
                    if task_id not in overlay:
                        missing.append((key, arm, task_id))
                source = {r["task_id"]: r for r in
                          read_rows(arm_path(arm, family, tier, seed,
                                             clean=True))}
                source.update(overlay)
                out = (EXP / FINAL_CAUSAL / f"{arm}_{family}_{tier}" /
                       f"results_seed{seed}.jsonl")
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text("".join(
                    json.dumps(source[t], sort_keys=True) + "\n"
                    for t in sorted(source)))
    if missing:
        raise RuntimeError(f"causal recovery incomplete: {missing[:5]}")
    print("materialized final causal matrix")


def analyze() -> None:
    materialize()
    subprocess.run([
        sys.executable,
        "scripts/analyze_repaired_matrix.py",
        "--prefix",
        CLEAN_PREFIX,
        "--tag",
        FINAL_CAUSAL,
        "--output",
        FINAL_ANALYSIS,
    ], cwd=ROOT, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "run", "materialize", "analyze"))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    {
        "freeze": freeze,
        "run": lambda: run(args.workers),
        "materialize": materialize,
        "analyze": analyze,
    }[args.stage]()
