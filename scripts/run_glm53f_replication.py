#!/usr/bin/env python
"""Run the frozen GLM-5.3-Flash provenance replication safely.

The key is read from GLM_API_KEY, glm_api_key.txt, or the legacy api_key.txt
`glm:` entry and is never printed or passed on a command line.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def load_key() -> str:
    if os.environ.get("GLM_API_KEY"):
        return os.environ["GLM_API_KEY"].strip()
    dedicated = ROOT / "glm_api_key.txt"
    if dedicated.exists():
        key = dedicated.read_text().strip()
        if key:
            return key
    legacy = ROOT / "api_key.txt"
    if legacy.exists():
        for raw in legacy.read_text().splitlines():
            line = raw.strip()
            for sep in (":", "："):
                if line.lower().startswith(f"glm{sep}"):
                    key = line.split(sep, 1)[1].strip()
                    if key:
                        return key
    raise SystemExit("No GLM key found. Set GLM_API_KEY or create glm_api_key.txt.")


def provider_env(disable_seed: bool) -> dict:
    env = os.environ.copy()
    env.update({
        "LLM_PROVIDER": "glm",
        "GLM_API_KEY": load_key(),
        "GLM_BASE_URL": env.get("GLM_BASE_URL", "https://api.z.ai/api/paas/v4/"),
        "GLM_MODEL": env.get("GLM_MODEL", "glm-5.3-flash"),
        "LLM_REASONING_EFFORT": "low",
        "LLM_PRICE_IN_PER_M": env.get("LLM_PRICE_IN_PER_M", "0.075"),
        "LLM_PRICE_OUT_PER_M": env.get("LLM_PRICE_OUT_PER_M", "0.25"),
        "LLM_BACKOFF_SCHEDULE": "10,30,90",
        "LLM_MAX_RETRIES": "3",
        "LLM_ATTEMPT_LOG": str(ROOT / "experiments" / "glm53f_attempts.jsonl"),
        "LLM_DISABLE_SEED": "1" if disable_seed else "0",
    })
    return env


def run(args: list[str], env: dict) -> None:
    shown = " ".join(args)
    print("running:", shown, flush=True)
    subprocess.run([PY, *args], cwd=ROOT, env=env, check=True)


def smoke(env: dict) -> None:
    for seed in (0,):
        run(["scripts/run_envelope.py", "--structures", "direct", "--tiers", "loose",
             "--families", "code", "--n", "1", "--split", "test", "--seed", str(seed),
             "--workers", "1", "--tag-prefix", "_glm53f_smoke_", "--no-cache"], env)
        run(["scripts/run_envelope.py", "--structures", "cot", "--tiers", "loose",
             "--families", "math", "--n", "1", "--split", "test", "--seed", str(seed),
             "--workers", "1", "--tag-prefix", "_glm53f_smoke_", "--no-cache"], env)
    run(["scripts/run_provenance_causal.py", "--families", "code", "math", "--n", "1",
         "--seeds", "0", "--tag", "_glm53f_smoke/provenance_causal",
         "--baseline-tag-prefix", "_glm53f_smoke_", "--workers", "1"], env)


def baseline(env: dict) -> None:
    for seed in (0, 1, 2):
        run(["scripts/run_envelope.py", "--structures", "direct", "--tiers", "loose",
             "--families", "code", "--n", "150", "--split", "test", "--seed", str(seed),
             "--workers", "2", "--tag-prefix", "glm53f_", "--no-cache"], env)
        run(["scripts/run_envelope.py", "--structures", "cot", "--tiers", "loose",
             "--families", "math", "--n", "150", "--split", "test", "--seed", str(seed),
             "--workers", "2", "--tag-prefix", "glm53f_", "--no-cache"], env)


def causal(env: dict) -> None:
    run(["scripts/run_provenance_causal.py", "--families", "code", "math", "--n", "150",
         "--seeds", "0", "1", "2", "--tag", "glm53f_provenance_causal",
         "--baseline-tag-prefix", "glm53f_", "--workers", "2"], env)


def analyze(env: dict) -> None:
    run(["scripts/analyze_provenance_replication.py",
         "--baseline-tag-prefix", "glm53f_",
         "--causal-tag", "glm53f_provenance_causal",
         "--output", "glm53f_provenance_analysis.json"], env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("smoke", "baseline", "causal", "analyze", "all"),
                    required=True)
    ap.add_argument("--disable-seed", action="store_true")
    ap.add_argument("--confirm-exposed-key-risk", action="store_true",
                    help="Required when continuing with a key exposed in local diagnostic output.")
    args = ap.parse_args()
    if not args.confirm_exposed_key_risk:
        raise SystemExit("Refusing to call provider until --confirm-exposed-key-risk is supplied.")
    env = provider_env(args.disable_seed)
    stages = ("smoke", "baseline", "causal", "analyze") if args.stage == "all" else (args.stage,)
    for stage in stages:
        globals()[stage](env)


if __name__ == "__main__":
    main()
