#!/usr/bin/env python3
"""Top-level Linux/vLLM driver for the local Qwen2.5-Coder-7B-Instruct
math/tight nonbinding cross-provider control.

No paid or remote API is ever called: this only talks to a local vLLM
OpenAI-compatible server the operator started separately (see
SERVER_QWEN_MATH_TIGHT_TASK.md). Does not modify the frozen Mac/MLX or GLM
drivers.

Stages:
  freeze   -- write the manifest + environment record; no LLM calls.
  smoke    -- 1 baseline task (seed 0) + 1 task through each causal arm
              (seed 0), all against a throwaway `_smoke/` run directory
              excluded from the final package. Checks the protocol runs and
              parses end to end and that the nonbinding caps are not
              gating -- never used to tune anything.
  baseline -- the 150-task, 3-seed cot reference (writes
              experiments/qwen25coder7b_server_math_tight_20260914_envelope_test/
              cot_math_tight/results_seed{0,1,2}.jsonl via run_envelope.py).
  causal   -- the three provenance arms, 150 tasks x 3 seeds each, via
              run_qwen_vllm_math_tight_nonbinding.py.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
PREFIX = "qwen25coder7b_server_math_tight_20260914_"


def env() -> dict[str, str]:
    value = os.environ.copy()
    value.update({
        "LLM_PROVIDER": "vllm",
        "VLLM_API_KEY": value.get("VLLM_API_KEY", "local"),
        "VLLM_BASE_URL": value.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        "VLLM_MODEL": value.get("VLLM_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct"),
        "LLM_PRICE_IN_PER_M": "0",
        "LLM_PRICE_OUT_PER_M": "0",
        "LLM_MAX_RETRIES": value.get("LLM_MAX_RETRIES", "4"),
        "LLM_BACKOFF_SCHEDULE": value.get("LLM_BACKOFF_SCHEDULE", "3,10,30"),
        "LLM_ATTEMPT_LOG": str(ROOT / f"experiments/{PREFIX}attempts.jsonl"),
        "LLM_DISABLE_SEED": "0",
    })
    value.pop("LLM_REASONING_EFFORT", None)
    return value


def run(command: list[str]) -> None:
    print("running:", " ".join(command), flush=True)
    subprocess.run([PY, *command], cwd=ROOT, env=env(), check=True)


def freeze() -> None:
    run(["scripts/run_qwen_vllm_math_tight_nonbinding.py", "freeze"])
    run(["scripts/write_qwen_vllm_environment_record.py"])


def smoke() -> None:
    run(["scripts/run_envelope.py", "--split", "test", "--families", "math",
         "--tiers", "tight", "--structures", "cot", "--n", "1", "--seed", "0",
         "--workers", "1", "--tag-prefix", "_smoke_", "--no-cache"])


def baseline(seeds: list[int]) -> None:
    for seed in seeds:
        run(["scripts/run_envelope.py", "--split", "test", "--families", "math",
             "--tiers", "tight", "--structures", "cot", "--n", "150",
             "--seed", str(seed), "--workers", "1", "--tag-prefix", PREFIX,
             "--no-cache"])


def causal(seeds: list[int], arms: list[str] | None, recover: bool) -> None:
    cmd = ["scripts/run_qwen_vllm_math_tight_nonbinding.py", "run",
           "--workers", "1", "--seeds", *map(str, seeds)]
    if arms:
        cmd += ["--arms", *arms]
    if recover:
        cmd += ["--recover"]
    run(cmd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "smoke", "baseline", "causal"))
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--arms", nargs="*", default=None,
                        choices=["arm1_assign", "arm2_samepolicy", "arm3_diffpolicy"])
    parser.add_argument("--recover", action="store_true")
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze()
    elif args.stage == "smoke":
        smoke()
    elif args.stage == "baseline":
        baseline(args.seeds)
    else:
        causal(args.seeds, args.arms, args.recover)


if __name__ == "__main__":
    main()
