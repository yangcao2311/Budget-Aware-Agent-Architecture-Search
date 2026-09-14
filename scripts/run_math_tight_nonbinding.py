#!/usr/bin/env python3
"""Frozen nonbinding-cap rerun for the confounded math/tight provenance cell.

This is a diagnostic companion to, not a replacement for, the original tight
budget cell.  Prompts, tasks, references, seeds, verifier, suffix, per-call
output limits, and all three arms are unchanged.  Only the task-level ledger
caps are widened enough that the complete graph is feasible in every arm.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hbws.data import load_split
from hbws.dsl import _wf_verify_refine, wf_assign_refine, wf_incumbent_refine_cot
from hbws.ledger import BudgetCaps
from hbws.protocol import evaluate
from scripts.run_glm4flash_replication import load_key
from scripts.run_provenance_causal import load_baseline_by_seed


EXP = ROOT / "experiments"
TAG = "glm4flash_math_tight_nonbinding_20260911"
BASELINE_PREFIX = "glm4flash_repaired_code_20260909_"
MANIFEST = EXP / f"{TAG}_manifest.json"

# The longest unchanged path is generate + four checks + four refinements.
# These task-level caps make that path feasible without changing any node's
# own max-output limit.  They are deliberately not called another budget tier.
CAPS = BudgetCaps(
    max_llm_calls=9,
    max_in_tokens=128000,
    max_out_tokens=9 * 1536,
    max_tool_calls=8,
    max_wall_sec=900,
    max_usd=1.0,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def baseline_files() -> list[Path]:
    base = EXP / f"{BASELINE_PREFIX}envelope_test/cot_math_tight"
    return [base / f"results_seed{s}.jsonl" for s in (0, 1, 2)]


def freeze(model: str, endpoint: str) -> None:
    source_files = [
        ROOT / "scripts/run_math_tight_nonbinding.py",
        ROOT / "scripts/run_provenance_causal.py",
        ROOT / "hbws/runner.py",
        ROOT / "hbws/llm.py",
        ROOT / "hbws/dsl.py",
        ROOT / "hbws/prompts.py",
        ROOT / "hbws/verify.py",
        ROOT / "hbws/ledger.py",
        ROOT / "hbws/protocol.py",
    ]
    payload = {
        "status": "frozen before any nonbinding-cap calls",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "diagnose active budget gating in original math/tight cell",
        "model": model,
        "endpoint": endpoint,
        "logical_price_usd": 0,
        "design": {
            "family": "math",
            "original_tier": "tight",
            "condition": "same cell with nonbinding task-level ledger caps",
            "tasks": 150,
            "seeds": [0, 1, 2],
            "arms": ["assign_stored", "same_policy_regenerate",
                     "different_policy_regenerate"],
            "cache": False,
            "caps": CAPS.as_vec(),
            "unchanged": ["tasks", "stored references", "prompts", "verifier",
                          "suffix graph", "per-call output limits", "seeds"],
        },
        "source_hashes": {str(p.relative_to(ROOT)): sha(p) for p in source_files},
        "baseline_hashes": {
            str(p.relative_to(ROOT)): sha(p) for p in baseline_files()
        },
    }
    if MANIFEST.exists():
        old = json.loads(MANIFEST.read_text())
        payload["frozen_at_utc"] = old["frozen_at_utc"]
        if payload != old:
            raise RuntimeError("Refusing to alter frozen nonbinding rerun")
    else:
        MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("frozen:", MANIFEST)


def provider_env() -> None:
    os.environ.update({
        "LLM_PROVIDER": "glm",
        "GLM_API_KEY": load_key(),
        "GLM_BASE_URL": os.environ.get(
            "GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
        "GLM_MODEL": "glm-4-flash-250414",
        "LLM_PRICE_IN_PER_M": "0",
        "LLM_PRICE_OUT_PER_M": "0",
        "LLM_BACKOFF_SCHEDULE": os.environ.get("LLM_BACKOFF_SCHEDULE", "3,10,30"),
        "LLM_MAX_RETRIES": os.environ.get("LLM_MAX_RETRIES", "4"),
        "LLM_ATTEMPT_LOG": str(EXP / f"{TAG}_attempts.jsonl"),
        "LLM_DISABLE_SEED": "0",
    })
    os.environ.pop("LLM_REASONING_EFFORT", None)


def run(workers: int) -> None:
    provider_env()
    freeze(os.environ["GLM_MODEL"], os.environ["GLM_BASE_URL"])
    tasks = load_split("math", "test")[:150]
    rows = []
    for seed in (0, 1, 2):
        reference = load_baseline_by_seed(
            "math", "tight", BASELINE_PREFIX, (seed,))[seed]
        arms = [
            ("arm1_assign", wf_assign_refine(),
             [{**t, "_assign_solution": reference[t["id"]][0]} for t in tasks]),
            ("arm2_samepolicy", wf_incumbent_refine_cot(), tasks),
            ("arm3_diffpolicy", _wf_verify_refine(3), tasks),
        ]
        for arm, workflow, arm_tasks in arms:
            name = f"{TAG}/{arm}_math_tight_nonbinding"
            result_path = EXP / name / f"results_seed{seed}.jsonl"
            if result_path.exists() and sum(1 for _ in result_path.open()) == 150:
                print("complete; skip:", arm, "seed", seed)
                continue
            summary = evaluate(workflow, arm_tasks, CAPS, run_name=name,
                               seed=seed, use_cache=False, workers=workers)
            rows.append({**summary, "family": "math",
                         "original_tier": "tight",
                         "condition": "nonbinding_caps", "arm": arm})
            print(arm, "seed", seed, summary, flush=True)
    if rows:
        with (EXP / f"{TAG}_summary.jsonl").open("a") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "run"))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    provider_env()
    if args.stage == "freeze":
        freeze(os.environ["GLM_MODEL"], os.environ["GLM_BASE_URL"])
    else:
        run(args.workers)


if __name__ == "__main__":
    main()
