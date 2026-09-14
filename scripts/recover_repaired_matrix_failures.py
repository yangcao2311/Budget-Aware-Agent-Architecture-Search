#!/usr/bin/env python3
"""Frozen recovery of failed baseline rows in the repaired GLM 2x2 matrix.

The original matrix files are immutable.  A baseline provider failure also
invalidates the three causal rows for that task/seed because the assignment arm
would otherwise receive an empty reference.  This script therefore freezes the
affected task/seed pairs before recovery calls, reruns their baseline and all
three interventions at one worker, and materializes a separate clean matrix.
Completed original rows are copied byte-for-byte at the JSON-object level.
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
from hbws.dsl import ENVELOPE_LIB, _wf_verify_refine, wf_assign_refine
from hbws.dsl import wf_incumbent_refine, wf_incumbent_refine_cot
from hbws.ledger import BUDGET_TIERS
from hbws.protocol import evaluate
from scripts.run_envelope import clamp_to_tier

EXP = ROOT / "experiments"
REC = EXP / "glm4flash_repaired_matrix_recovery_20260910"
MANIFEST = REC / "manifest.json"
BASE_PREFIX = "glm4flash_repaired_code_20260909_"
CAUSAL_TAG = "glm4flash_repaired_code_20260909_causal"
CLEAN_PREFIX = "glm4flash_repaired_matrix_clean_20260910_"
CLEAN_CAUSAL = "glm4flash_repaired_matrix_clean_20260910_causal"
GOOD = {"completed", "reserve_rejected"}
SEEDS = (0, 1, 2)
CELLS = (("code", "tight"), ("code", "loose"),
         ("math", "tight"), ("math", "loose"))
BASE_STRUCTURE = {"code": "direct", "math": "cot"}
ARM_BUILDERS = {
    "arm1_assign": None,
    "arm2_samepolicy": {"code": wf_incumbent_refine,
                        "math": wf_incumbent_refine_cot},
    "arm3_diffpolicy": {"code": lambda: _wf_verify_refine(3),
                        "math": lambda: _wf_verify_refine(3)},
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x]


def base_path(family: str, tier: str, seed: int, *, clean: bool = False) -> Path:
    prefix = CLEAN_PREFIX if clean else BASE_PREFIX
    return (EXP / f"{prefix}envelope_test" /
            f"{BASE_STRUCTURE[family]}_{family}_{tier}" /
            f"results_seed{seed}.jsonl")


def arm_path(arm: str, family: str, tier: str, seed: int, *, clean: bool = False) -> Path:
    tag = CLEAN_CAUSAL if clean else CAUSAL_TAG
    return EXP / tag / f"{arm}_{family}_{tier}" / f"results_seed{seed}.jsonl"


def freeze() -> None:
    REC.mkdir(parents=True, exist_ok=True)
    targets: dict[str, list[str]] = {}
    input_hashes = {}
    for family, tier in CELLS:
        for seed in SEEDS:
            path = base_path(family, tier, seed)
            input_hashes[str(path.relative_to(ROOT))] = digest(path)
            bad = sorted(r["task_id"] for r in read_rows(path)
                         if r.get("status") not in GOOD)
            targets[f"{family}/{tier}/seed{seed}"] = bad
    sources = [Path(__file__), ROOT / "hbws/dsl.py", ROOT / "hbws/runner.py",
               ROOT / "hbws/protocol.py", ROOT / "hbws/verify.py",
               ROOT / "scripts/run_envelope.py"]
    payload = {
        "status": "frozen before recovery calls",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "glm-4-flash-250414",
        "endpoint": "https://open.bigmodel.cn/api/paas/v4",
        "reason": "provider/runtime failure in a baseline row invalidates that baseline and its three provenance arms",
        "policy": "rerun only frozen failed task/seed pairs; baseline first, then all three arms; one worker; originals immutable",
        "targets": targets,
        "n_target_pairs": sum(map(len, targets.values())),
        "input_hashes": input_hashes,
        "source_hashes": {str(p.relative_to(ROOT)): digest(p) for p in sources},
        "clean_prefix": CLEAN_PREFIX,
        "clean_causal_tag": CLEAN_CAUSAL,
    }
    if MANIFEST.exists():
        old = json.loads(MANIFEST.read_text())
        payload["frozen_at_utc"] = old["frozen_at_utc"]
        if payload != old:
            raise RuntimeError("Refusing to alter the frozen recovery design")
    else:
        MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("frozen", payload["n_target_pairs"], "failed baseline task/seed pairs")


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


def recovery_rows(kind: str, family: str, tier: str, seed: int) -> dict[str, dict]:
    out = {}
    pattern = f"runs/{kind}_{family}_{tier}_seed{seed}_pass*/results_seed{seed}.jsonl"
    for path in sorted(REC.glob(pattern)):
        for row in read_rows(path):
            if row.get("status") in GOOD:
                out[row["task_id"]] = row
    return out


def next_pass(kind: str, family: str, tier: str, seed: int) -> int:
    pattern = f"{kind}_{family}_{tier}_seed{seed}_pass*"
    return len(list((REC / "runs").glob(pattern)))


def run_batch(kind: str, family: str, tier: str, seed: int,
              tasks: list[dict], wf: dict, workers: int) -> None:
    if not tasks:
        return
    number = next_pass(kind, family, tier, seed)
    name = (f"glm4flash_repaired_matrix_recovery_20260910/runs/"
            f"{kind}_{family}_{tier}_seed{seed}_pass{number:02d}")
    print(kind, family, tier, "seed", seed, "retry", len(tasks), flush=True)
    evaluate(wf, tasks, BUDGET_TIERS[tier], run_name=name, seed=seed,
             use_cache=False, workers=workers)


def run(workers: int) -> None:
    cfg = config()
    provider_env()
    (REC / "runs").mkdir(parents=True, exist_ok=True)
    for family, tier in CELLS:
        task_map = {t["id"]: t for t in load_split(family, "test")[:150]}
        for seed in SEEDS:
            key = f"{family}/{tier}/seed{seed}"
            targets = cfg["targets"][key]
            if not targets:
                continue
            recovered_base = recovery_rows("baseline", family, tier, seed)
            remaining = [tid for tid in targets if tid not in recovered_base]
            baseline_wf = clamp_to_tier(
                ENVELOPE_LIB[BASE_STRUCTURE[family]](), BUDGET_TIERS[tier])
            run_batch("baseline", family, tier, seed,
                      [task_map[t] for t in remaining], baseline_wf, workers)
            recovered_base = recovery_rows("baseline", family, tier, seed)
            missing = [tid for tid in targets if tid not in recovered_base]
            if missing:
                print("baseline still incomplete", key, len(missing), flush=True)
                continue
            for arm, builders in ARM_BUILDERS.items():
                recovered_arm = recovery_rows(arm, family, tier, seed)
                remaining = [tid for tid in targets if tid not in recovered_arm]
                if arm == "arm1_assign":
                    arm_tasks = [{**task_map[t],
                                  "_assign_solution": recovered_base[t]["solution"]}
                                 for t in remaining]
                    wf = wf_assign_refine()
                else:
                    arm_tasks = [task_map[t] for t in remaining]
                    wf = builders[family]()
                run_batch(arm, family, tier, seed, arm_tasks, wf, workers)
    print("recovery pass complete; rerun this stage if any rows remain", flush=True)


def materialize() -> None:
    cfg = config()
    missing = []
    for family, tier in CELLS:
        for seed in SEEDS:
            key = f"{family}/{tier}/seed{seed}"
            targets = set(cfg["targets"][key])
            base_recovery = recovery_rows("baseline", family, tier, seed)
            arm_recovery = {arm: recovery_rows(arm, family, tier, seed)
                            for arm in ARM_BUILDERS}
            for tid in targets:
                if tid not in base_recovery:
                    missing.append((key, "baseline", tid))
                for arm in ARM_BUILDERS:
                    if tid not in arm_recovery[arm]:
                        missing.append((key, arm, tid))
            if missing:
                continue
            base = {r["task_id"]: r for r in read_rows(base_path(family, tier, seed))}
            base.update(base_recovery)
            out = base_path(family, tier, seed, clean=True)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("".join(json.dumps(base[t], sort_keys=True) + "\n"
                                   for t in sorted(base)))
            for arm in ARM_BUILDERS:
                rows = {r["task_id"]: r for r in read_rows(arm_path(arm, family, tier, seed))}
                rows.update(arm_recovery[arm])
                out = arm_path(arm, family, tier, seed, clean=True)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text("".join(json.dumps(rows[t], sort_keys=True) + "\n"
                                       for t in sorted(rows)))
    if missing:
        raise RuntimeError(f"recovery incomplete: {len(missing)} rows, first={missing[:5]}")
    print("materialized clean matrix", CLEAN_PREFIX, CLEAN_CAUSAL)


def analyze() -> None:
    materialize()
    subprocess.run([
        sys.executable, "scripts/analyze_repaired_matrix.py",
        "--prefix", CLEAN_PREFIX, "--tag", CLEAN_CAUSAL,
        "--output", "glm4flash_repaired_matrix_clean_20260910_analysis.json",
    ], cwd=ROOT, check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("freeze", "run", "materialize", "analyze"))
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    {"freeze": freeze, "run": lambda: run(args.workers),
     "materialize": materialize, "analyze": analyze}[args.stage]()
