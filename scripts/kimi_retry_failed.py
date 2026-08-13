#!/usr/bin/env python
"""Backfill protocol for the Kimi K3 Table-1 replication, frozen after the
2026-08-13 run exposed provider-side (429/5xx) failures unevenly across
cells. Logged as a deviation (see PREREGISTRATION.md §9 / HANDOFF.md): this
script only ever touches task x seed x arm combinations that previously got a
provider error, never a completed one; runs at low concurrency; and appends
every attempt to a log that is never overwritten.

Required invocation (sets the provider, the fixed backoff schedule, and the
retry cap -- see hbws/llm.py's _env_backoff_schedule / LLM_MAX_RETRIES):

    LLM_PROVIDER=kimi LLM_BACKOFF_SCHEDULE=10,30,90 LLM_MAX_RETRIES=3 \\
        .venv/bin/python scripts/kimi_retry_failed.py --workers 2

Before any real backfill, run a smoke test on a handful of tasks at realistic
prompt/output length, e.g.:

    LLM_PROVIDER=kimi LLM_BACKOFF_SCHEDULE=10,30,90 LLM_MAX_RETRIES=3 \\
        .venv/bin/python scripts/kimi_retry_failed.py --smoke-test 20

Target: near-100% completion per cell, or at minimum back under the <5%
provider-failure bar Table~app:kimi already uses. If gaps remain after
backfill, the *comparison* (not this script) must restrict to the matched
task x seed intersection across arms and report both the completion rate and
n -- this script does not decide that; the analysis script does.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hbws.data import load_split, load_ood_visible, load_logic
from hbws.dsl import ENVELOPE_LIB
from hbws.ledger import BUDGET_TIERS
from hbws.protocol import evaluate

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
ATTEMPT_LOG = ROOT / "experiments" / "kimi_backfill_attempts.jsonl"

# (run_dir, structure_key, family, tier, split, n, mask)
CELLS = [
    ("kimi_envelope_test/incumbent_refine_code_loose", "incumbent_refine", "code", "loose", "test", 150, 1.0),
    ("kimi_envelope_test/direct_code_loose", "direct", "code", "loose", "test", 150, 1.0),
    ("kimi_envelope_test/incumbent_refine_cot_math_loose", "incumbent_refine_cot", "math", "loose", "test", 150, 1.0),
    ("kimi_envelope_test/cot_math_loose", "cot", "math", "loose", "test", 150, 1.0),
    ("kimi_envelope_logic_prospective/incumbent_refine_logic_loose", "incumbent_refine", "logic", "loose", "logic_prospective", 120, 1.0),
    ("kimi_envelope_logic_prospective/direct_logic_loose", "direct", "logic", "loose", "logic_prospective", 120, 1.0),
    ("kimi_envelope_ood/incumbent_refine_code_loose", "incumbent_refine", "code", "loose", "ood", 100, 1.0),
    ("kimi_envelope_ood/direct_code_loose", "direct", "code", "loose", "ood", 100, 1.0),
    ("kimi_envelope_ood/incumbent_refine_cot_math_loose", "incumbent_refine_cot", "math", "loose", "ood", 100, 1.0),
    ("kimi_envelope_ood/cot_math_loose", "cot", "math", "loose", "ood", 100, 1.0),
    ("kimi_envelope_test_mask0.5_k1/verify_refine_3_code_loose", "verify_refine_3", "code", "loose", "test", 150, 0.5),
    ("kimi_envelope_test_mask0.0_k1/verify_refine_3_code_loose", "verify_refine_3", "code", "loose", "test", 150, 0.0),
]


def mask_tests(task, frac):
    import math
    t = dict(task)
    lines = [l for l in task["feedback_tests"].splitlines() if l.strip()]
    asserts = [l for l in lines if l.lstrip().startswith("assert")]
    other = [l for l in lines if not l.lstrip().startswith("assert")]
    keep = math.ceil(frac * len(asserts))
    t["feedback_tests"] = "\n".join(other + asserts[:keep]) if keep else ""
    return t


def load_tasks(fam, split, n, mask):
    if split == "ood_visible":
        tasks = load_ood_visible()[:n]
    elif split == "logic_prospective":
        tasks = load_logic()[:n]
    else:
        tasks = load_split(fam, split)[:n]
    if fam == "code" and mask < 1.0:
        tasks = [mask_tests(t, mask) for t in tasks]
    return {t["id"]: t for t in tasks}


def clamp_to_tier(wf, caps):
    import copy
    wf = copy.deepcopy(wf)
    for n in wf["nodes"]:
        p = n.get("params")
        if p and "max_output_tokens" in p:
            share = caps.max_out_tokens // n["k"] if n["type"] == "vote" else caps.max_out_tokens
            p["max_output_tokens"] = max(64, min(p["max_output_tokens"], share))
    return wf


def log_attempt(record: dict) -> None:
    """Append-only: never truncate or overwrite ATTEMPT_LOG."""
    with open(ATTEMPT_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")


def smoke_test(n: int, workers: int) -> None:
    """Cheap, realistic-length probe before any real backfill: a code and a
    math task at the actual node config (solve_direct/solve_cot, real token
    caps), not a trivial 'ping'. Logs every attempt; makes no other writes."""
    tasks_code = load_split("code", "test")[:n]
    tasks_math = load_split("math", "test")[:n]
    wf_code = clamp_to_tier(ENVELOPE_LIB["incumbent_refine"](), BUDGET_TIERS["loose"])
    wf_math = clamp_to_tier(ENVELOPE_LIB["incumbent_refine_cot"](), BUDGET_TIERS["loose"])
    for fam, wf, tasks in [("code", wf_code, tasks_code), ("math", wf_math, tasks_math)]:
        s = evaluate(wf, tasks, BUDGET_TIERS["loose"], run_name=f"_kimi_smoke_test/{fam}",
                     seed=0, use_cache=False, workers=workers)
        log_attempt({"phase": "smoke_test", "family": fam, "n": s["n"],
                     "success_rate": s["success_rate"], "errors": s.get("errors", 0),
                     "reserve_rejected": s.get("reserve_rejected", 0),
                     "ts": time.time()})
        print(f"smoke test {fam}: n={s['n']} succ={s['success_rate']:.3f} "
              f"errors={s.get('errors', 0)}")
    print("Inspect experiments/_kimi_smoke_test/ and kimi_backfill_attempts.jsonl "
          "before proceeding to a real backfill.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--smoke-test", type=int, default=0,
                    help="Run an N-task smoke test instead of the real backfill.")
    args = ap.parse_args()

    if args.smoke_test:
        smoke_test(args.smoke_test, args.workers)
        return

    log_attempt({"phase": "backfill_start", "ts": time.time(), "cells": len(CELLS)})
    for run_dir, sname, fam, tier, split, n, mask in CELLS:
        out_dir = EXP / run_dir
        by_id = load_tasks(fam, split, n, mask)
        wf = clamp_to_tier(ENVELOPE_LIB[sname](), BUDGET_TIERS[tier])
        for seed in (0, 1, 2):
            f = out_dir / f"results_seed{seed}.jsonl"
            if not f.exists():
                print(f"SKIP (missing) {run_dir} seed{seed}")
                continue
            rows = [json.loads(l) for l in open(f)]
            bad_ids = [r["task_id"] for r in rows if r["status"] not in ("completed", "reserve_rejected")]
            if not bad_ids:
                continue
            retry_tasks = [by_id[i] for i in bad_ids if i in by_id]
            print(f"{run_dir} seed{seed}: retrying {len(retry_tasks)} tasks")
            log_attempt({"phase": "cell_start", "run_dir": run_dir, "seed": seed,
                         "n_retry": len(retry_tasks), "ts": time.time()})
            s = evaluate(wf, retry_tasks, BUDGET_TIERS[tier],
                         run_name=f"_kimi_retry/{run_dir}", seed=seed,
                         use_cache=False, workers=args.workers)
            new_rows = {json.loads(l)["task_id"]: json.loads(l)
                        for l in open(EXP / f"_kimi_retry/{run_dir}/results_seed{seed}.jsonl")}
            merged = [new_rows.get(r["task_id"], r) for r in rows]
            still_bad_ids = [r["task_id"] for r in merged
                             if r["status"] not in ("completed", "reserve_rejected")]
            with open(f, "w") as fh:
                for r in merged:
                    fh.write(json.dumps(r) + "\n")
            log_attempt({"phase": "cell_done", "run_dir": run_dir, "seed": seed,
                         "success_rate": s["success_rate"], "n_retried": len(retry_tasks),
                         "still_bad": len(still_bad_ids), "still_bad_ids": still_bad_ids,
                         "ts": time.time()})
            print(f"  succ={s['success_rate']:.3f} still_bad={len(still_bad_ids)}")

    log_attempt({"phase": "backfill_done", "ts": time.time()})
    print("=== RETRY PASS DONE ===")


if __name__ == "__main__":
    main()
