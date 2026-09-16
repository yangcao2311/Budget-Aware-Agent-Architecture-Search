#!/usr/bin/env python3
"""Serving-configuration intervention on the local Qwen2.5-Coder-7B-Instruct
math/tight same-policy suffix (reviewer B1 follow-up).

Same model weights/revision, same tokenizer, same BF16 precision, same
vLLM/torch/CUDA versions, same physical GPU as the archived
qwen25coder7b_server_math_tight_20260914 campaign (see
experiments/qwen_serving_regime_20260915_environment_diff.json for the
explicit comparison). Only the vLLM serving flags change across conditions
(S / B / P / S2); the base_url is the only thing that may differ in a
request's construction, and even that only points at a different port of
the SAME local machine/GPU.

Runs ONLY the same-policy suffix (hbws.dsl.wf_incumbent_refine_cot) -- not
the full assign/different-policy matrix -- against the same 150 frozen
math/test tasks x seeds 0/1/2, temperature 0, seed forwarded, cache
disabled, under the SAME nonbinding task-level caps as the archived
provenance campaign (imported directly from
scripts/run_qwen_vllm_math_tight_nonbinding.py, not re-typed).

Installs, for every condition uniformly:
  - qwen_exact_reservation: exact Qwen-tokenizer input-token reservation
    (fixes the archived campaign's 14 tiktoken-estimate settlement
    overruns; changes ONLY the ledger reservation, never the request text,
    sampling parameters, or per-call output cap -- see
    tests/test_qwen_exact_reservation_noninvasive.py).
  - qwen_call_logger: full per-call request/response logging, tagged with
    the condition name, so `first_draft` (the node "g" row) is captured
    directly rather than inferred from the final `solution`.

Does not modify hbws/*.py, the archived qwen25coder7b_server_math_tight_
20260914_* files, or scripts/run_qwen_vllm_math_tight_nonbinding.py.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from hbws.data import load_split
from hbws.dsl import wf_incumbent_refine_cot
from hbws.protocol import evaluate
from run_qwen_vllm_math_tight_nonbinding import CAPS  # unmodified import

import qwen_call_logger
import qwen_exact_reservation

EXP = ROOT / "experiments"
TAG = "qwen_serving_regime_20260915"
MANIFEST = EXP / f"{TAG}_manifest.json"
ROSTER = EXP / f"{TAG}_roster.json"

# Fixed run order: S must run (and be archived) before B/P; S2 repeats S.
CONDITIONS = {
    "S":  {"max_num_seqs": 1,  "enable_prefix_caching": False, "workers": 1,
           "note": "serial control: one in-flight request at a time"},
    "B":  {"max_num_seqs": 16, "enable_prefix_caching": False, "workers": 8,
           "note": "batched/concurrent regime: up to 16 concurrent "
                   "sequences, 8 client workers"},
    "P":  {"max_num_seqs": 1,  "enable_prefix_caching": True,  "workers": 1,
           "note": "prefix-cache condition: serial concurrency, prefix "
                   "caching enabled"},
    "S2": {"max_num_seqs": 1,  "enable_prefix_caching": False, "workers": 1,
           "note": "repeat of S: time-drift / reversibility check"},
}
SEEDS = (0, 1, 2)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> None:
    if not ROSTER.exists():
        raise RuntimeError(
            "run scripts/build_serving_regime_roster.py first")
    source_files = [
        ROOT / "scripts/run_qwen_serving_regime.py",
        ROOT / "scripts/build_serving_regime_roster.py",
        ROOT / "scripts/qwen_exact_reservation.py",
        ROOT / "scripts/qwen_call_logger.py",
        ROOT / "scripts/run_qwen_vllm_math_tight_nonbinding.py",
        ROOT / "hbws/runner.py",
        ROOT / "hbws/llm.py",
        ROOT / "hbws/dsl.py",
        ROOT / "hbws/prompts.py",
        ROOT / "hbws/verify.py",
        ROOT / "hbws/ledger.py",
        ROOT / "hbws/protocol.py",
    ]
    payload = {
        "status": "frozen before any serving-regime benchmark call",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "reviewer B1: does changing vLLM concurrency/batching or "
                   "prefix caching -- same model weights, same physical "
                   "GPU, same request -- change output bytes",
        "conditions": CONDITIONS,
        "condition_order": ["S", "B", "P", "S2"],
        "seeds": list(SEEDS),
        "n_tasks": 150,
        "workflow": "wf_incumbent_refine_cot (same-policy suffix only; "
                    "assign/different-policy arms are not rerun here)",
        "caps": CAPS.as_vec(),
        "cache": False,
        "temperature": 0,
        "seed_forwarded": True,
        "roster_sha256": sha(ROSTER),
        "reservation_fix": "qwen_exact_reservation installed for every "
                           "condition uniformly (see module docstring); "
                           "changes only ledger admission reservation, "
                           "never request text/sampling params/output cap. "
                           "The archived campaign's 14 "
                           "budget_exceeded:settle_overrun:in_tokens rows "
                           "are left untouched.",
        "source_hashes": {str(p.relative_to(ROOT)): sha(p) for p in source_files},
    }
    if MANIFEST.exists():
        old = json.loads(MANIFEST.read_text())
        payload["frozen_at_utc"] = old["frozen_at_utc"]
        if payload != old:
            raise RuntimeError(
                "Refusing to alter frozen serving-regime manifest")
        print("already frozen, unchanged:", MANIFEST)
        return
    MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("frozen:", MANIFEST)


def provider_env(base_url: str) -> None:
    os.environ.update({
        "LLM_PROVIDER": "vllm",
        "VLLM_API_KEY": "local",
        "VLLM_BASE_URL": base_url,
        "VLLM_MODEL": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "LLM_PRICE_IN_PER_M": "0",
        "LLM_PRICE_OUT_PER_M": "0",
        "LLM_MAX_RETRIES": "4",
        "LLM_BACKOFF_SCHEDULE": "3,10,30",
        "LLM_ATTEMPT_LOG": str(EXP / f"{TAG}_attempts.jsonl"),
        "LLM_DISABLE_SEED": "0",
    })
    os.environ.pop("LLM_REASONING_EFFORT", None)


def _load_rows(path: Path) -> dict[str, dict]:
    rows = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line:
                r = json.loads(line)
                rows[r["task_id"]] = r
    return rows


def run_condition(condition: str, base_url: str, seeds: list[int],
                   n_tasks: int | None = None, label_suffix: str = "",
                   recover: bool = False, recover_workers: int | None = None) -> None:
    if condition not in CONDITIONS:
        raise ValueError(condition)
    cfg = CONDITIONS[condition]
    provider_env(base_url)
    tasks = load_split("math", "test")[: (n_tasks or 150)]
    task_ids = [t["id"] for t in tasks]

    call_log = EXP / f"{TAG}_{condition}{label_suffix}_calls.jsonl"
    original_estimate = qwen_exact_reservation.exact_in_tokens
    qwen_exact_reservation.install()
    original_chat = qwen_call_logger.install(call_log, condition=condition + label_suffix)
    try:
        for seed in seeds:
            name = f"{TAG}/{condition}{label_suffix}_samepolicy"
            result_path = EXP / name / f"results_seed{seed}.jsonl"

            existing = _load_rows(result_path)
            good_ids = {tid for tid, r in existing.items()
                        if not str(r.get("status", "")).startswith("error")}

            if recover and result_path.exists():
                missing_or_bad = [t for t in tasks if t["id"] not in good_ids]
                if not missing_or_bad:
                    print("complete; skip:", condition, "seed", seed)
                    continue
                print(f"recovering {len(missing_or_bad)} rows:", condition, "seed", seed,
                      [t["id"] for t in missing_or_bad])
                # Recover with a lower worker count than the condition's own
                # (default: serial) -- this repairs execution-error rows
                # without resampling anything that already succeeded, and
                # avoids re-triggering a concurrency-only infra race; it
                # does not change the request text, sampling parameters, or
                # nonbinding caps for the recovered rows.
                evaluate(wf_incumbent_refine_cot(), missing_or_bad, CAPS,
                        run_name=name, seed=seed, use_cache=False,
                        workers=recover_workers or 1)
                fresh = _load_rows(result_path)
                merged = {tid: existing[tid] for tid in good_ids}
                merged.update(fresh)
                assert set(merged) == set(task_ids), (
                    f"post-recovery row set mismatch for {condition} seed {seed}: "
                    f"have {len(merged)}, want {len(task_ids)}")
                with result_path.open("w") as f:
                    for tid in sorted(merged):
                        f.write(json.dumps(merged[tid]) + "\n")
                print("recovered:", condition, "seed", seed, "n=", len(merged))
                continue

            if result_path.exists() and sum(1 for _ in result_path.open()) == len(tasks):
                print("complete; skip:", condition, "seed", seed)
                continue
            t0 = time.time()
            summary = evaluate(wf_incumbent_refine_cot(), tasks, CAPS,
                               run_name=name, seed=seed, use_cache=False,
                               workers=cfg["workers"])
            print(condition, "seed", seed, summary,
                  "wall_sec=", round(time.time() - t0, 1), flush=True)
    finally:
        qwen_call_logger.uninstall(original_chat)
        import hbws.llm as _llm
        _llm.estimate_in_tokens = original_estimate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("freeze", "preflight", "run"))
    ap.add_argument("--condition", choices=list(CONDITIONS))
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    ap.add_argument("--n-tasks", type=int, default=None,
                    help="preflight-only: limit task count")
    ap.add_argument("--label-suffix", default="",
                    help="e.g. use '_preflight' to keep preflight runs "
                         "out of the frozen roster directories")
    ap.add_argument("--recover", action="store_true",
                    help="Only (re)run rows whose prior status was an "
                         "execution error, overlaying them onto kept-good "
                         "rows instead of resampling the whole condition.")
    ap.add_argument("--recover-workers", type=int, default=None)
    args = ap.parse_args()

    if args.stage == "freeze":
        freeze()
        return
    if not args.condition:
        ap.error("--condition is required for preflight/run")
    if args.stage == "preflight":
        run_condition(args.condition, args.base_url, args.seeds,
                     n_tasks=args.n_tasks or 10, label_suffix="_preflight")
    else:
        run_condition(args.condition, args.base_url, args.seeds,
                     label_suffix=args.label_suffix, recover=args.recover,
                     recover_workers=args.recover_workers)


if __name__ == "__main__":
    main()
