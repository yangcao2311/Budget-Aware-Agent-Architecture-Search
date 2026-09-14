#!/usr/bin/env python3
"""Frozen GLM-4.7-Flash within-family math/tight replication.

This is explicitly a same-provider, same-family replication.  It cannot
separate a model effect from Zhipu's serving stack and is not described as an
independent endpoint.  The causal suffix uses nonbinding task-level caps so it
does not repeat the original math/tight censoring problem.
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


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hbws.data import load_split
from hbws.dsl import _wf_verify_refine, wf_assign_refine, wf_incumbent_refine_cot
from hbws.protocol import evaluate
from scripts.run_glm4flash_replication import load_key
from scripts.run_math_tight_nonbinding import CAPS
from scripts.run_provenance_causal import load_baseline_by_seed


PY = sys.executable
MODEL = "glm-4.7-flash"
ENDPOINT = "https://open.bigmodel.cn/api/paas/v4"
PREFIX = "glm47flash_20260911_"
TAG = "glm47flash_20260911_math_tight_nonbinding"
MANIFEST = ROOT / "experiments/glm47flash_20260911_manifest.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provider_env() -> dict[str, str]:
    value = os.environ.copy()
    value.update({
        "LLM_PROVIDER": "glm",
        "GLM_API_KEY": load_key(),
        "GLM_BASE_URL": ENDPOINT,
        "GLM_MODEL": MODEL,
        "LLM_PRICE_IN_PER_M": "0",
        "LLM_PRICE_OUT_PER_M": "0",
        "LLM_BACKOFF_SCHEDULE": "3,10,30",
        "LLM_MAX_RETRIES": "4",
        "LLM_ATTEMPT_LOG": str(ROOT / "experiments/glm47flash_20260911_attempts.jsonl"),
        "LLM_DISABLE_SEED": "0",
    })
    value.pop("LLM_REASONING_EFFORT", None)
    return value


def freeze() -> None:
    sources = [
        Path(__file__),
        ROOT / "scripts/run_envelope.py",
        ROOT / "scripts/run_provenance_causal.py",
        ROOT / "scripts/run_math_tight_nonbinding.py",
        ROOT / "hbws/runner.py",
        ROOT / "hbws/llm.py",
        ROOT / "hbws/dsl.py",
        ROOT / "hbws/prompts.py",
        ROOT / "hbws/verify.py",
        ROOT / "hbws/ledger.py",
        ROOT / "hbws/protocol.py",
    ]
    payload = {
        "status": "frozen before any GLM-4.7-Flash calls",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "positioning": "within-family replication; same provider and serving stack",
        "model": MODEL,
        "endpoint": ENDPOINT,
        "logical_price_usd": 0,
        "design": {
            "family": "math",
            "profile": "tight prompts and per-call limits",
            "causal_task_caps": CAPS.as_vec(),
            "tasks": 150,
            "seeds": [0, 1, 2],
            "arms": ["reference", "assign_stored", "same_policy_regenerate",
                     "different_policy_regenerate"],
            "cache": False,
            "workers": 1,
            "selection": "all tasks; no outcome-based exclusions",
        },
        "source_hashes": {
            str(path.relative_to(ROOT)): sha(path) for path in sources
        },
    }
    if MANIFEST.exists():
        old = json.loads(MANIFEST.read_text())
        payload["frozen_at_utc"] = old["frozen_at_utc"]
        if payload != old:
            raise RuntimeError("Refusing to alter frozen GLM-4.7 protocol")
    else:
        MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("frozen:", MANIFEST)


def baseline() -> None:
    for seed in (0, 1, 2):
        command = [
            PY, "scripts/run_envelope.py", "--split", "test", "--families", "math",
            "--tiers", "tight", "--structures", "cot", "--n", "150", "--seed",
            str(seed), "--workers", "1", "--tag-prefix", PREFIX, "--no-cache",
        ]
        subprocess.run(command, cwd=ROOT, env=provider_env(), check=True)


def causal() -> None:
    os.environ.update(provider_env())
    tasks = load_split("math", "test")[:150]
    for seed in (0, 1, 2):
        reference = load_baseline_by_seed("math", "tight", PREFIX, (seed,))[seed]
        arms = (
            ("arm1_assign", wf_assign_refine(),
             [{**task, "_assign_solution": reference[task["id"]][0]}
              for task in tasks]),
            ("arm2_samepolicy", wf_incumbent_refine_cot(), tasks),
            ("arm3_diffpolicy", _wf_verify_refine(3), tasks),
        )
        for arm, workflow, arm_tasks in arms:
            evaluate(workflow, arm_tasks, CAPS,
                     run_name=f"{TAG}/{arm}_math_tight_nonbinding",
                     seed=seed, use_cache=False, workers=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "baseline", "causal"))
    args = parser.parse_args()
    freeze()
    if args.stage == "baseline":
        baseline()
    elif args.stage == "causal":
        causal()


if __name__ == "__main__":
    main()
