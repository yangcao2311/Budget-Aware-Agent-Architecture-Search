#!/usr/bin/env python3
"""Run a GLM-4-Flash replication through Zhipu's official API.

The key is read from GLM_API_KEY, glm_api_key.txt, or the ``glm:`` entry in
api_key.txt.  It is never printed or placed on a command line.  The default
endpoint and model identifier follow the official BigModel documentation.

Stages:
  smoke     one task per family plus all three provenance arms
  baseline  model-specific code/math reference outputs for three seeds
  backfill  retry only failed math-baseline rows at low concurrency
  causal    full two-family, three-arm provenance intervention
  causal_math  math-only three-arm provenance intervention
  analyze   bootstrap/path analysis only; no inference
  table10   ten verifier/domain conditions used by the sensitivity table
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
PREFIX = "glm4flash_"
CAUSAL_TAG = "glm4flash_provenance_causal"
FINAL_PREFIX = "glm4flash_final_"
FINAL_CODE_CAUSAL_TAG = "glm4flash_final_provenance_causal"
REPAIRED_CODE_PREFIX = "glm4flash_repaired_code_20260909_"
REPAIRED_CODE_CAUSAL_TAG = "glm4flash_repaired_code_20260909_causal"
REPAIRED_CODE_SMOKE_PREFIX = "_glm4flash_repaired_code_20260909_smoke_"
REPAIRED_CODE_SMOKE_TAG = "_glm4flash_repaired_code_20260909_smoke/causal"
REPAIRED_MATRIX_MANIFEST = ROOT / "experiments/glm4flash_repaired_matrix_20260909_manifest.json"


def load_key() -> str:
    if os.environ.get("GLM_API_KEY"):
        return os.environ["GLM_API_KEY"].strip()
    dedicated = ROOT / "glm_api_key.txt"
    if dedicated.exists() and dedicated.read_text().strip():
        return dedicated.read_text().strip()
    legacy = ROOT / "api_key.txt"
    if legacy.exists():
        for raw in legacy.read_text().splitlines():
            line = raw.strip()
            for sep in (":", "："):
                if line.lower().startswith(f"glm{sep}"):
                    key = line.split(sep, 1)[1].strip()
                    if key:
                        return key
    raise SystemExit("No GLM key found in GLM_API_KEY, glm_api_key.txt, or api_key.txt")


def provider_env(disable_seed: bool) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "LLM_PROVIDER": "glm",
        "GLM_API_KEY": load_key(),
        "GLM_BASE_URL": env.get(
            "GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"
        ),
        "GLM_MODEL": env.get("GLM_MODEL", "glm-4-flash-250414"),
        "LLM_PRICE_IN_PER_M": "0",
        "LLM_PRICE_OUT_PER_M": "0",
        "LLM_BACKOFF_SCHEDULE": env.get("LLM_BACKOFF_SCHEDULE", "3,10,30"),
        "LLM_MAX_RETRIES": env.get("LLM_MAX_RETRIES", "4"),
        "LLM_ATTEMPT_LOG": str(ROOT / "experiments" / "glm4flash_attempts.jsonl"),
        "LLM_DISABLE_SEED": "1" if disable_seed else "0",
    })
    # GLM-4-Flash predates the provider's reasoning-effort extension.
    env.pop("LLM_REASONING_EFFORT", None)
    return env


def run(args: list[str], env: dict[str, str]) -> None:
    print("running:", " ".join(args), flush=True)
    subprocess.run([PY, *args], cwd=ROOT, env=env, check=True)


def envelope(env: dict[str, str], *, split: str, families: list[str],
             tiers: list[str], structures: list[str], n: int, seed: int,
             workers: int, mask_tests: float | None = None,
             tag_prefix: str = PREFIX) -> None:
    args = ["scripts/run_envelope.py", "--split", split, "--families", *families,
            "--tiers", *tiers, "--structures", *structures, "--n", str(n),
            "--seed", str(seed), "--workers", str(workers),
            "--tag-prefix", tag_prefix, "--no-cache"]
    if mask_tests is not None:
        args.extend(["--mask-tests", str(mask_tests)])
    run(args, env)


def smoke(env: dict[str, str]) -> None:
    smoke_prefix = "_glm4flash_smoke_"
    envelope(env, split="test", families=["code"], tiers=["loose"],
             structures=["direct"], n=1, seed=0, workers=1,
             tag_prefix=smoke_prefix)
    envelope(env, split="test", families=["math"], tiers=["loose"],
             structures=["cot"], n=1, seed=0, workers=1,
             tag_prefix=smoke_prefix)
    run(["scripts/run_provenance_causal.py", "--families", "code", "math",
         "--n", "1", "--seeds", "0", "--tag",
         "_glm4flash_smoke/provenance_causal", "--baseline-tag-prefix",
         smoke_prefix, "--workers", "1"], env)


def baseline(env: dict[str, str], workers: int) -> None:
    for seed in (0, 1, 2):
        envelope(env, split="test", families=["code"], tiers=["loose"],
                 structures=["direct"], n=150, seed=seed, workers=workers)
        envelope(env, split="test", families=["math"], tiers=["loose"],
                 structures=["cot"], n=150, seed=seed, workers=workers)


def causal(env: dict[str, str], workers: int) -> None:
    run(["scripts/run_provenance_causal.py", "--families", "code", "math",
         "--n", "150", "--seeds", "0", "1", "2", "--tag", CAUSAL_TAG,
         "--baseline-tag-prefix", PREFIX, "--workers", str(workers)], env)


def causal_math(env: dict[str, str], workers: int, seeds: list[int]) -> None:
    run(["scripts/run_provenance_causal.py", "--families", "math",
         "--n", "150", "--seeds", *map(str, seeds), "--tag", CAUSAL_TAG,
         "--baseline-tag-prefix", PREFIX, "--workers", str(workers),
         "--stable-correct-only"], env)


def regrade_code(env: dict[str, str], workers: int) -> None:
    run(["scripts/regrade_glm4flash_code.py", "--workers", str(max(1, workers))], env)


def causal_code(env: dict[str, str], workers: int) -> None:
    run(["scripts/run_provenance_causal.py", "--families", "code",
         "--n", "150", "--seeds", "0", "1", "2",
         "--tag", FINAL_CODE_CAUSAL_TAG,
         "--baseline-tag-prefix", FINAL_PREFIX,
         "--workers", str(workers), "--stable-correct-only"], env)


def analyze_code(env: dict[str, str]) -> None:
    run(["scripts/analyze_provenance_replication.py",
         "--baseline-tag-prefix", FINAL_PREFIX,
         "--causal-tag", FINAL_CODE_CAUSAL_TAG,
         "--families", "code",
         "--output", "glm4flash_final_code_provenance_analysis.json",
         "--primary-only"], env)


def repaired_code_smoke(env: dict[str, str]) -> None:
    """One-task fail-fast check for the repaired-executor code rerun."""
    envelope(env, split="test", families=["code"], tiers=["loose"],
             structures=["direct"], n=1, seed=0, workers=1,
             tag_prefix=REPAIRED_CODE_SMOKE_PREFIX)
    run(["scripts/run_provenance_causal.py", "--families", "code",
         "--n", "1", "--seeds", "0", "--tag", REPAIRED_CODE_SMOKE_TAG,
         "--baseline-tag-prefix", REPAIRED_CODE_SMOKE_PREFIX,
         "--workers", "1"], env)


def repaired_code_baseline(env: dict[str, str], workers: int) -> None:
    """Fresh direct baseline on all 150 frozen code tasks and three seeds."""
    for seed in (0, 1, 2):
        envelope(env, split="test", families=["code"], tiers=["loose"],
                 structures=["direct"], n=150, seed=seed, workers=workers,
                 tag_prefix=REPAIRED_CODE_PREFIX)


def repaired_code_causal(env: dict[str, str], workers: int) -> None:
    """Full-population assign/same-policy/vanilla rerun, without caching."""
    run(["scripts/run_provenance_causal.py", "--families", "code",
         "--n", "150", "--seeds", "0", "1", "2",
         "--tag", REPAIRED_CODE_CAUSAL_TAG,
         "--baseline-tag-prefix", REPAIRED_CODE_PREFIX,
         "--workers", str(workers)], env)


def repaired_code_analyze(env: dict[str, str]) -> None:
    run(["scripts/analyze_provenance_replication.py",
         "--baseline-tag-prefix", REPAIRED_CODE_PREFIX,
         "--causal-tag", REPAIRED_CODE_CAUSAL_TAG,
         "--families", "code",
         "--output", "glm4flash_repaired_code_20260909_analysis.json"], env)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repaired_matrix_freeze(env: dict[str, str]) -> None:
    """Freeze the GLM 2x2 expansion before any missing-cell API calls."""
    source_files = [
        ROOT / "scripts/run_glm4flash_replication.py",
        ROOT / "scripts/run_provenance_causal.py",
        ROOT / "scripts/run_envelope.py",
        ROOT / "scripts/analyze_repaired_matrix.py",
        ROOT / "hbws/runner.py", ROOT / "hbws/ledger.py",
        ROOT / "hbws/protocol.py", ROOT / "hbws/dsl.py",
    ]
    existing = sorted((ROOT / "experiments").glob(
        "glm4flash_repaired_code_20260909_*code_loose/results_seed*.jsonl"))
    payload = {
        "status": "frozen before missing-cell calls",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": env["GLM_MODEL"], "endpoint": env["GLM_BASE_URL"],
        "logical_price_usd": 0,
        "design": {
            "families": ["code", "math"], "tiers": ["tight", "loose"],
            "arms": ["reference", "assign_stored", "same_policy_regenerate",
                     "different_policy_verify_refine"],
            "tasks_per_family": 150, "seeds": [0, 1, 2],
            "cache": False, "analysis_unit": "task",
            "bootstrap_draws": 10000,
            "valid_statuses": ["completed", "reserve_rejected"],
            "executor_defects": "all other statuses; surfaced, never recoded",
        },
        "provenance": {
            "code_loose": "completed cleanly before this expansion and reused",
            "missing_cells": ["code/tight", "math/tight", "math/loose"],
            "selection": "all four cells reported; no outcome-based exclusions",
        },
        "source_hashes": {str(p.relative_to(ROOT)): _sha(p) for p in source_files},
        "preexisting_code_loose_hashes": {
            str(p.relative_to(ROOT)): _sha(p) for p in existing
        },
    }
    if REPAIRED_MATRIX_MANIFEST.exists():
        old = json.loads(REPAIRED_MATRIX_MANIFEST.read_text())
        payload["frozen_at_utc"] = old["frozen_at_utc"]
        if payload != old:
            raise RuntimeError("Refusing to alter frozen repaired-matrix protocol")
    else:
        REPAIRED_MATRIX_MANIFEST.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("frozen:", REPAIRED_MATRIX_MANIFEST)


def repaired_matrix_baseline(env: dict[str, str], workers: int) -> None:
    """Run only the three cells missing from the already-clean code/loose cell."""
    repaired_matrix_freeze(env)
    for seed in (0, 1, 2):
        envelope(env, split="test", families=["code"], tiers=["tight"],
                 structures=["direct"], n=150, seed=seed, workers=workers,
                 tag_prefix=REPAIRED_CODE_PREFIX)
        envelope(env, split="test", families=["math"], tiers=["tight", "loose"],
                 structures=["cot"], n=150, seed=seed, workers=workers,
                 tag_prefix=REPAIRED_CODE_PREFIX)


def repaired_matrix_causal(env: dict[str, str], workers: int) -> None:
    repaired_matrix_freeze(env)
    run(["scripts/run_provenance_causal.py", "--families", "code",
         "--tiers", "tight", "--n", "150", "--seeds", "0", "1", "2",
         "--tag", REPAIRED_CODE_CAUSAL_TAG, "--baseline-tag-prefix",
         REPAIRED_CODE_PREFIX, "--workers", str(workers)], env)
    run(["scripts/run_provenance_causal.py", "--families", "math",
         "--tiers", "tight", "loose", "--n", "150", "--seeds", "0", "1", "2",
         "--tag", REPAIRED_CODE_CAUSAL_TAG, "--baseline-tag-prefix",
         REPAIRED_CODE_PREFIX, "--workers", str(workers)], env)


def repaired_matrix_analyze(env: dict[str, str]) -> None:
    repaired_matrix_freeze(env)
    run(["scripts/analyze_repaired_matrix.py", "--prefix", REPAIRED_CODE_PREFIX,
         "--tag", REPAIRED_CODE_CAUSAL_TAG,
         "--output", "glm4flash_repaired_matrix_20260909_analysis.json"], env)


def backfill(env: dict[str, str], workers: int) -> None:
    env = dict(env)
    env["LLM_BACKOFF_SCHEDULE"] = "10,30,90"
    env["LLM_MAX_RETRIES"] = "5"
    run(["scripts/glm4flash_retry_failed.py", "--workers", str(workers)], env)


def analyze(env: dict[str, str]) -> None:
    run(["scripts/analyze_provenance_replication.py",
         "--baseline-tag-prefix", PREFIX, "--causal-tag", CAUSAL_TAG,
         "--output", "glm4flash_provenance_analysis.json",
         "--primary-only"], env)


def table10(env: dict[str, str], workers: int) -> None:
    for seed in (0, 1, 2):
        envelope(env, split="test", families=["code"], tiers=["loose", "tight"],
                 structures=["incumbent_refine", "direct"], n=150,
                 seed=seed, workers=workers)
        envelope(env, split="test", families=["math"], tiers=["loose", "tight"],
                 structures=["incumbent_refine_cot", "cot"], n=150,
                 seed=seed, workers=workers)
        envelope(env, split="logic_prospective", families=["logic"],
                 tiers=["loose"], structures=["incumbent_refine", "direct"],
                 n=120, seed=seed, workers=workers)
        envelope(env, split="ood", families=["code"], tiers=["loose"],
                 structures=["incumbent_refine", "direct"], n=100,
                 seed=seed, workers=workers)
        envelope(env, split="ood", families=["math"], tiers=["loose"],
                 structures=["incumbent_refine_cot", "cot"], n=100,
                 seed=seed, workers=workers)
        envelope(env, split="ood_visible", families=["code"], tiers=["loose"],
                 structures=["incumbent_refine", "direct"], n=100,
                 seed=seed, workers=workers)
        for frac in (0.5, 0.0):
            envelope(env, split="test", families=["code"], tiers=["loose"],
                     structures=["verify_refine_3", "direct"], n=150,
                     seed=seed, workers=workers, mask_tests=frac)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("smoke", "baseline", "backfill", "causal",
                             "causal_math", "regrade_code", "causal_code",
                             "analyze", "analyze_code", "table10", "all",
                             "repaired_code_smoke", "repaired_code_baseline",
                             "repaired_code_causal", "repaired_code_analyze",
                             "repaired_matrix_freeze", "repaired_matrix_baseline",
                             "repaired_matrix_causal", "repaired_matrix_analyze"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                    help="Seeds for the causal_math stage.")
    ap.add_argument("--disable-seed", action="store_true")
    args = ap.parse_args()
    env = provider_env(args.disable_seed)
    stages = ("smoke", "baseline", "causal", "analyze", "table10") \
        if args.stage == "all" else (args.stage,)
    for stage in stages:
        if stage == "causal_math":
            causal_math(env, args.workers, args.seeds)
        elif stage in ("baseline", "backfill", "causal", "regrade_code",
                       "causal_code", "table10", "repaired_code_baseline",
                       "repaired_code_causal", "repaired_matrix_baseline",
                       "repaired_matrix_causal"):
            globals()[stage](env, args.workers)
        else:
            globals()[stage](env)


if __name__ == "__main__":
    main()
