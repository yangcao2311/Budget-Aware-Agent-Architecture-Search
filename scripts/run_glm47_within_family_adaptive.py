#!/usr/bin/env python3
"""Run the frozen GLM-4.7-Flash replication with a pre-call concurrency rule.

The 2026-09-11 protocol remains the scientific design of record.  This driver
adds only an operational amendment: the first complete baseline seed is run
with three workers, after which a decision based solely on provider/transport
metadata either retains three workers or lowers subsequent runs to two.
Accuracy, repair, breakage, verifier decisions, and model text are never read
when making that decision.
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
from hbws.dsl import _wf_verify_refine, wf_assign_refine, wf_incumbent_refine_cot
from hbws.protocol import evaluate
import scripts.run_glm47_within_family as frozen
from scripts.run_provenance_causal import load_baseline_by_seed


PY = sys.executable
EXP = ROOT / "experiments"
AMENDMENT = EXP / "glm47flash_20260913_concurrency_amendment.json"
DECISION = EXP / "glm47flash_20260913_concurrency_decision.json"
ATTEMPTS = EXP / "glm47flash_20260911_attempts.jsonl"
PILOT_RUN = "glm47flash_20260911_envelope_test/cot_math_tight"
PILOT_RESULT = EXP / PILOT_RUN / "results_seed0.jsonl"
PILOT_SUMMARY = EXP / PILOT_RUN / "summary_seed0.json"
PILOT_WORKERS = 3
FALLBACK_WORKERS = 2
RETRYABLE_RATE_LIMIT = 0.05


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def complete(path: Path, n: int = 150) -> bool:
    return path.exists() and sum(1 for line in path.open() if line.strip()) == n


def amendment_payload(timestamp: str) -> dict:
    sources = [
        Path(__file__),
        ROOT / "scripts/run_glm47_within_family.py",
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
    return {
        "status": "frozen before any GLM-4.7-Flash calls",
        "frozen_at_utc": timestamp,
        "original_manifest": str(frozen.MANIFEST.relative_to(ROOT)),
        "original_manifest_sha256": sha(frozen.MANIFEST),
        "scope": "operational concurrency only; all scientific design fields unchanged",
        "pilot": {
            "run_name": PILOT_RUN,
            "seed": 0,
            "tasks": 150,
            "workers": PILOT_WORKERS,
        },
        "decision_rule": {
            "inputs": [
                "provider attempt status",
                "provider exception type",
                "final task execution status",
            ],
            "excluded_inputs": [
                "model response text",
                "accuracy",
                "repair",
                "breakage",
                "verifier acceptance or rejection",
            ],
            "retain_workers_3_if": (
                "zero RateLimitError attempts, zero final task errors, and "
                "retryable provider-attempt rate <= 0.05"
            ),
            "otherwise": "use workers=2 for every remaining baseline and causal run",
            "retryable_rate_definition": (
                "retryable_error attempts divided by completed plus retryable_error attempts"
            ),
        },
        "source_hashes": {
            str(path.relative_to(ROOT)): sha(path) for path in sources
        },
    }


def freeze_amendment() -> None:
    # Revalidate the original frozen design; this does not make provider calls.
    frozen.freeze()
    if AMENDMENT.exists():
        old = json.loads(AMENDMENT.read_text())
        payload = amendment_payload(old["frozen_at_utc"])
        if payload != old:
            raise RuntimeError("Refusing to alter frozen concurrency amendment")
        print("frozen:", AMENDMENT)
        return

    # The amendment is legitimate only before the first endpoint call.
    evidence = [ATTEMPTS, DECISION, PILOT_RESULT, PILOT_SUMMARY]
    causal_root = EXP / frozen.TAG
    if any(path.exists() for path in evidence) or causal_root.exists():
        raise RuntimeError(
            "Cannot create pre-call amendment: GLM-4.7 output already exists")
    payload = amendment_payload(datetime.now(timezone.utc).isoformat())
    AMENDMENT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("frozen:", AMENDMENT)


def run_baseline_seed(seed: int, workers: int) -> None:
    result = EXP / "glm47flash_20260911_envelope_test/cot_math_tight" / f"results_seed{seed}.jsonl"
    if complete(result):
        print("complete; skip baseline seed", seed)
        return
    command = [
        PY, "scripts/run_envelope.py", "--split", "test", "--families", "math",
        "--tiers", "tight", "--structures", "cot", "--n", "150", "--seed",
        str(seed), "--workers", str(workers), "--tag-prefix", frozen.PREFIX,
        "--no-cache",
    ]
    subprocess.run(command, cwd=ROOT, env=frozen.provider_env(), check=True)


def make_decision() -> dict:
    if DECISION.exists():
        return json.loads(DECISION.read_text())
    if not complete(PILOT_RESULT) or not PILOT_SUMMARY.exists():
        raise RuntimeError("Pilot seed 0 is not complete")

    attempts = [row for row in read_jsonl(ATTEMPTS)
                if row.get("run_name") == PILOT_RUN and row.get("seed") == 0]
    completed_attempts = sum(row.get("status") == "completed" for row in attempts)
    retryable_attempts = sum(row.get("status") == "retryable_error" for row in attempts)
    rate_limit_attempts = sum(
        row.get("status") == "retryable_error"
        and row.get("error_type") == "RateLimitError"
        for row in attempts
    )
    denominator = completed_attempts + retryable_attempts
    retryable_rate = retryable_attempts / denominator if denominator else 1.0
    rows = read_jsonl(PILOT_RESULT)
    final_errors = sum(str(row.get("status", "")).startswith("error") for row in rows)
    retain_three = (
        rate_limit_attempts == 0
        and final_errors == 0
        and retryable_rate <= RETRYABLE_RATE_LIMIT
    )
    payload = {
        "status": "decided from frozen operational rule",
        "decided_at_utc": datetime.now(timezone.utc).isoformat(),
        "amendment_sha256": sha(AMENDMENT),
        "pilot_result_sha256": sha(PILOT_RESULT),
        "pilot_summary_sha256": sha(PILOT_SUMMARY),
        "operational_counts": {
            "completed_attempts": completed_attempts,
            "retryable_attempts": retryable_attempts,
            "retryable_rate": retryable_rate,
            "rate_limit_attempts": rate_limit_attempts,
            "final_task_errors": final_errors,
        },
        "workers_for_remaining_runs": (
            PILOT_WORKERS if retain_three else FALLBACK_WORKERS
        ),
        "note": "No response text or scientific outcome was inspected.",
    }
    DECISION.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def baseline() -> None:
    run_baseline_seed(0, PILOT_WORKERS)
    decision = make_decision()
    workers = int(decision["workers_for_remaining_runs"])
    for seed in (1, 2):
        run_baseline_seed(seed, workers)


def causal() -> None:
    decision = make_decision()
    workers = int(decision["workers_for_remaining_runs"])
    os.environ.update(frozen.provider_env())
    tasks = load_split("math", "test")[:150]
    for seed in (0, 1, 2):
        reference = load_baseline_by_seed("math", "tight", frozen.PREFIX, (seed,))[seed]
        arms = (
            ("arm1_assign", wf_assign_refine(),
             [{**task, "_assign_solution": reference[task["id"]][0]}
              for task in tasks]),
            ("arm2_samepolicy", wf_incumbent_refine_cot(), tasks),
            ("arm3_diffpolicy", _wf_verify_refine(3), tasks),
        )
        for arm, workflow, arm_tasks in arms:
            name = f"{frozen.TAG}/{arm}_math_tight_nonbinding"
            result = EXP / name / f"results_seed{seed}.jsonl"
            if complete(result):
                print("complete; skip causal", arm, "seed", seed)
                continue
            evaluate(workflow, arm_tasks, frozen.CAPS, run_name=name,
                     seed=seed, use_cache=False, workers=workers)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "baseline", "causal", "decision"))
    args = parser.parse_args()
    freeze_amendment()
    if args.stage == "baseline":
        baseline()
    elif args.stage == "causal":
        causal()
    elif args.stage == "decision":
        print(json.dumps(make_decision(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
