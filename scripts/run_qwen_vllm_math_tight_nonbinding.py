#!/usr/bin/env python3
"""Frozen nonbinding-cap rerun for math/tight, served locally by vLLM.

Linux/vLLM-specific driver -- mirrors scripts/run_math_tight_nonbinding.py
(the GLM driver) exactly in design (150 frozen tasks, seeds 0/1/2, three
provenance arms, identical nonbinding task-level caps) but talks to a local
vLLM OpenAI-compatible server instead of a paid remote API. Do not modify the
frozen GLM or Mac/MLX drivers; this file is new and Linux/vLLM only.
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
from scripts.run_provenance_causal import load_baseline_by_seed


EXP = ROOT / "experiments"
TAG = "qwen25coder7b_server_math_tight_20260914_causal"
BASELINE_PREFIX = "qwen25coder7b_server_math_tight_20260914_"
MANIFEST = EXP / "qwen25coder7b_server_math_tight_20260914_manifest.json"

# Identical nonbinding task-level caps to the GLM driver's diagnostic rerun:
# the longest unchanged path is generate + up to 3 verify-checks +
# up to 3 refinements, so 9 LLM calls and their token budget must not gate
# the suffix. Per-call output limits, prompts, verifier, and seeds are
# unchanged; only these task-level ledger caps are widened.
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


def freeze(model: str, endpoint: str, extra: dict | None = None) -> None:
    source_files = [
        ROOT / "scripts/run_qwen_vllm_math_tight_nonbinding.py",
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
        "purpose": "local vLLM/Qwen2.5-Coder-7B-Instruct cross-provider "
                   "deterministic-serving control for math/tight, nonbinding "
                   "task-level caps",
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
    if extra:
        payload.update(extra)
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
        "LLM_PROVIDER": "vllm",
        "VLLM_API_KEY": os.environ.get("VLLM_API_KEY", "local"),
        "VLLM_BASE_URL": os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        "VLLM_MODEL": os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct"),
        "LLM_PRICE_IN_PER_M": "0",
        "LLM_PRICE_OUT_PER_M": "0",
        "LLM_BACKOFF_SCHEDULE": os.environ.get("LLM_BACKOFF_SCHEDULE", "3,10,30"),
        "LLM_MAX_RETRIES": os.environ.get("LLM_MAX_RETRIES", "4"),
        "LLM_ATTEMPT_LOG": str(
            EXP / "qwen25coder7b_server_math_tight_20260914_causal_attempts.jsonl"),
        "LLM_DISABLE_SEED": "0",
    })
    os.environ.pop("LLM_REASONING_EFFORT", None)


def arm_targets(tasks, seed, arm_name):
    reference = load_baseline_by_seed(
        "math", "tight", BASELINE_PREFIX, (seed,))[seed]
    if arm_name == "arm1_assign":
        return wf_assign_refine(), [
            {**t, "_assign_solution": reference[t["id"]][0]} for t in tasks]
    if arm_name == "arm2_samepolicy":
        return wf_incumbent_refine_cot(), tasks
    if arm_name == "arm3_diffpolicy":
        return _wf_verify_refine(3), tasks
    raise ValueError(arm_name)


def _load_rows(path: Path) -> dict[str, dict]:
    rows = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line:
                r = json.loads(line)
                rows[r["task_id"]] = r
    return rows


def run(workers: int, only_seeds: list[int] | None, only_arms: list[str] | None,
        recover: bool) -> None:
    provider_env()
    freeze(os.environ["VLLM_MODEL"], os.environ["VLLM_BASE_URL"])
    tasks = load_split("math", "test")[:150]
    task_ids = [t["id"] for t in tasks]
    rows = []
    seeds = only_seeds or [0, 1, 2]
    arms = only_arms or ["arm1_assign", "arm2_samepolicy", "arm3_diffpolicy"]
    for seed in seeds:
        for arm in arms:
            workflow, arm_tasks = arm_targets(tasks, seed, arm)
            name = f"{TAG}/{arm}_math_tight_nonbinding"
            result_path = EXP / name / f"results_seed{seed}.jsonl"

            existing = _load_rows(result_path)
            good_ids = {tid for tid, r in existing.items()
                        if not str(r.get("status", "")).startswith("error")}

            if recover and result_path.exists():
                missing_or_bad = [t for t in arm_tasks if t["id"] not in good_ids]
                if not missing_or_bad:
                    print("complete; skip:", arm, "seed", seed)
                    continue
                print(f"recovering {len(missing_or_bad)} rows:", arm, "seed", seed)
                summary = evaluate(workflow, missing_or_bad, CAPS, run_name=name,
                                   seed=seed, use_cache=False, workers=workers)
                # overlay: merge freshly completed rows onto the kept-good rows,
                # then rewrite the full 150-row file (evaluate() only wrote the
                # recovered subset to disk).
                fresh = _load_rows(result_path)
                merged = {tid: existing[tid] for tid in good_ids}
                merged.update(fresh)
                assert set(merged) == set(task_ids), (
                    f"post-recovery row set mismatch for {arm} seed {seed}: "
                    f"have {len(merged)}, want {len(task_ids)}")
                with result_path.open("w") as f:
                    for tid in sorted(merged):
                        f.write(json.dumps(merged[tid]) + "\n")
                rows.append({"run": name, "seed": seed, "n": len(merged),
                             "recovered": len(missing_or_bad)})
                print("recovered:", arm, "seed", seed, "n=", len(merged))
                continue

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
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--arms", nargs="*", default=None,
                        choices=["arm1_assign", "arm2_samepolicy", "arm3_diffpolicy"])
    parser.add_argument("--recover", action="store_true",
                        help="Only (re)run rows whose prior status was an "
                             "error, overlaying them onto kept-good rows "
                             "instead of resampling the whole cell.")
    args = parser.parse_args()
    provider_env()
    if args.stage == "freeze":
        freeze(os.environ["VLLM_MODEL"], os.environ["VLLM_BASE_URL"])
    else:
        run(args.workers, args.seeds, args.arms, args.recover)


if __name__ == "__main__":
    main()
