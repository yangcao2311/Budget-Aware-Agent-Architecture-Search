#!/usr/bin/env python3
"""Frozen non-Zhipu local control on the code/loose provenance cell.

This is a local deterministic-serving control, not a second full primary
matrix and not evidence that model quality transfers.  It uses a quantized
Qwen2.5-Coder-7B-Instruct checkpoint served by MLX-LM on the author's Mac.
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
PY = sys.executable
MODEL_DIR = ROOT / ".models/Qwen2.5-Coder-7B-Instruct-4bit"
MODEL_ID = str(MODEL_DIR)
PREFIX = "qwen25coder7b_local_20260911_"
CAUSAL_TAG = "qwen25coder7b_local_20260911_causal"
MANIFEST = ROOT / "experiments/qwen25coder7b_local_20260911_manifest.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def env() -> dict[str, str]:
    value = os.environ.copy()
    value.update({
        "LLM_PROVIDER": "mlx",
        "MLX_API_KEY": "local",
        "MLX_BASE_URL": value.get("MLX_BASE_URL", "http://127.0.0.1:8081/v1"),
        "MLX_MODEL": MODEL_ID,
        "LLM_PRICE_IN_PER_M": "0",
        "LLM_PRICE_OUT_PER_M": "0",
        "LLM_MAX_RETRIES": "2",
        "LLM_BACKOFF_SCHEDULE": "0.2,1",
        "LLM_ATTEMPT_LOG": str(
            ROOT / "experiments/qwen25coder7b_local_20260911_attempts.jsonl"),
        "LLM_DISABLE_SEED": "0",
    })
    value.pop("LLM_REASONING_EFFORT", None)
    return value


def source_files() -> list[Path]:
    return [
        Path(__file__),
        ROOT / "scripts/openai_stop_proxy.py",
        ROOT / "scripts/run_envelope.py",
        ROOT / "scripts/run_provenance_causal.py",
        ROOT / "hbws/runner.py",
        ROOT / "hbws/llm.py",
        ROOT / "hbws/dsl.py",
        ROOT / "hbws/prompts.py",
        ROOT / "hbws/verify.py",
        ROOT / "hbws/ledger.py",
        ROOT / "hbws/protocol.py",
    ]


def freeze() -> None:
    payload = {
        "status": "frozen before local benchmark calls",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "positioning": "local deterministic-serving control; not a second full matrix",
        "checkpoint": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
        "model_path": MODEL_ID,
        "model_sha256": sha(MODEL_DIR / "model.safetensors"),
        "tokenizer_config_sha256": sha(MODEL_DIR / "tokenizer_config.json"),
        "serving": {
            "runtime": "mlx-lm 0.31.3",
            "base_url": env()["MLX_BASE_URL"],
            "temperature_zero_seed_forwarded": True,
            "transport_eos_stop": "<|im_end|>",
            "workers": 1,
        },
        "design": {
            "family": "code",
            "profile": "loose",
            "tasks": 150,
            "seeds": [0, 1, 2],
            "arms": ["reference", "assign_stored", "same_policy_regenerate",
                     "different_policy_regenerate"],
            "cache": False,
            "logical_price_usd": 0,
            "selection": "all tasks; no outcome-based exclusions",
        },
        "source_hashes": {
            str(path.relative_to(ROOT)): sha(path) for path in source_files()
        },
    }
    if MANIFEST.exists():
        old = json.loads(MANIFEST.read_text())
        payload["frozen_at_utc"] = old["frozen_at_utc"]
        if payload != old:
            raise RuntimeError("Refusing to alter frozen local-control protocol")
    else:
        MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("frozen:", MANIFEST)


def run(command: list[str]) -> None:
    print("running:", " ".join(command), flush=True)
    subprocess.run([PY, *command], cwd=ROOT, env=env(), check=True)


def baseline(n: int, seeds: list[int]) -> None:
    for seed in seeds:
        run(["scripts/run_envelope.py", "--split", "test", "--families", "code",
             "--tiers", "loose", "--structures", "direct", "--n", str(n),
             "--seed", str(seed), "--workers", "1", "--tag-prefix", PREFIX,
             "--no-cache"])


def causal(n: int, seeds: list[int]) -> None:
    run(["scripts/run_provenance_causal.py", "--families", "code", "--tiers",
         "loose", "--n", str(n), "--seeds", *map(str, seeds), "--tag",
         CAUSAL_TAG, "--baseline-tag-prefix", PREFIX, "--workers", "1"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "smoke", "baseline", "causal"))
    args = parser.parse_args()
    freeze()
    if args.stage == "freeze":
        return
    if args.stage == "smoke":
        baseline(1, [0])
        causal(1, [0])
    elif args.stage == "baseline":
        baseline(150, [0, 1, 2])
    else:
        causal(150, [0, 1, 2])


if __name__ == "__main__":
    main()
