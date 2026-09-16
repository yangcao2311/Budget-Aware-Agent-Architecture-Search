#!/usr/bin/env python3
"""GLM-4-Flash budget-parity follow-up: assign vs. same-policy regeneration
on math/tight and math/loose, with drafting cost and suffix (verify+
refine) budget accounted separately so the same-policy arm's extra draft
call cannot shrink its own suffix reachability relative to assign's.

Reuses each tier's existing frozen 150-task math roster, seeds 0/1/2, and
COMPLETE stored-reference outputs (experiments/glm4flash_repaired_matrix_
clean_20260910_envelope_test/cot_math_{tight,loose}/results_seed{seed}.jsonl,
transferred and checksum-verified against server_transfer/
glm_budget_parity_references_20260916/SHA256SUMS -- never regenerated).
Keeps the existing, unmodified verifier, refinement graph, prompts,
temperature, and per-call output caps (hbws.dsl.wf_assign_refine /
wf_incumbent_refine_cot). Does NOT use the old total-task caps from
BUDGET_TIERS: those are exactly the confound this follow-up isolates.

Budget-parity mechanism (see module-level SUFFIX_CAPS / DRAFT_CALL_COST):
a single hbws.ledger.TaskLedger still meters the whole task (this is not a
change to hbws/ledger.py or hbws/runner.py), but the same_policy arm's
total cap is SUFFIX_CAPS plus exactly one extra call's worst-case cost
(DRAFT_CALL_COST) -- the assign arm's cap is SUFFIX_CAPS unchanged (assign
itself costs nothing). After the same_policy arm's draft call settles,
both arms have identical remaining headroom for the verify+refine suffix.
SUFFIX_CAPS is sized generously (matching the sizing already validated
nonbinding in the Qwen math/tight causal campaign) so the prescribed
verify + up to 3 refine iterations is never budget-gated in either arm;
this is verified in the smoke/preflight stage before any full roster run,
not tuned afterward.

Total calls/tokens/cost are always reported in full (see the per-call
glm_call_logger.py log and each row's own `budget` field) -- the
separate-accounting mechanism changes ONLY ledger admission, never what is
reported.
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
from hbws.dsl import wf_assign_refine, wf_incumbent_refine_cot
from hbws.ledger import BudgetCaps
from hbws.protocol import evaluate

import glm_call_logger
import glm_exact_reservation
from run_glm4flash_replication import load_key

EXP = ROOT / "experiments"
TAG = "glm_budget_parity_20260916"
MANIFEST = EXP / f"{TAG}_manifest.json"
ROSTER = EXP / f"{TAG}_roster.json"
REF_PREFIX = "glm4flash_repaired_matrix_clean_20260910_envelope_test"

SEEDS = (0, 1, 2)
TIERS = ("tight", "loose")
ARMS = ("assign", "same_policy")

# The prescribed suffix: 1 verify + up to 3x(refine, verify) = at most 4
# verifies + 3 refines = 7 LLM calls, sized with headroom to 8 (matching
# the Qwen math/tight nonbinding campaign's validated sizing, scaled to
# this suffix-only budget). Never task/answer-dependent.
SUFFIX_CAPS = BudgetCaps(
    max_llm_calls=8, max_in_tokens=128000, max_out_tokens=8 * 1536,
    max_tool_calls=8, max_wall_sec=900, max_usd=1.0,
)
# Worst-case cost of exactly one extra draft call (wf_incumbent_refine_cot's
# "g" node: temperature 0.0, max_output_tokens 1536). in_tokens headroom is
# already generous in SUFFIX_CAPS (128000), so no separate addition needed
# there -- only llm_calls and out_tokens must be enlarged by the draft's
# own worst case for the same_policy arm to have identical *suffix*
# headroom to assign after the draft settles.
DRAFT_EXTRA_CALLS = 1
DRAFT_EXTRA_OUT_TOKENS = 1536

ASSIGN_CAPS = SUFFIX_CAPS
SAME_POLICY_CAPS = BudgetCaps(
    max_llm_calls=SUFFIX_CAPS.max_llm_calls + DRAFT_EXTRA_CALLS,
    max_in_tokens=SUFFIX_CAPS.max_in_tokens,
    max_out_tokens=SUFFIX_CAPS.max_out_tokens + DRAFT_EXTRA_OUT_TOKENS,
    max_tool_calls=SUFFIX_CAPS.max_tool_calls,
    max_wall_sec=SUFFIX_CAPS.max_wall_sec,
    max_usd=SUFFIX_CAPS.max_usd,
)
CAPS_BY_ARM = {"assign": ASSIGN_CAPS, "same_policy": SAME_POLICY_CAPS}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> None:
    if not ROSTER.exists():
        raise RuntimeError("run scripts/build_glm_budget_parity_roster.py first")
    source_files = [
        ROOT / "scripts/run_glm_budget_parity.py",
        ROOT / "scripts/build_glm_budget_parity_roster.py",
        ROOT / "scripts/glm_exact_reservation.py",
        ROOT / "scripts/glm_call_logger.py",
        ROOT / "hbws/runner.py",
        ROOT / "hbws/llm.py",
        ROOT / "hbws/dsl.py",
        ROOT / "hbws/prompts.py",
        ROOT / "hbws/verify.py",
        ROOT / "hbws/ledger.py",
        ROOT / "hbws/protocol.py",
    ]
    reference_hashes = {}
    for tier in TIERS:
        for seed in SEEDS:
            p = EXP / REF_PREFIX / f"cot_math_{tier}" / f"results_seed{seed}.jsonl"
            reference_hashes[str(p.relative_to(ROOT))] = sha(p)
    payload = {
        "status": "frozen before any GLM budget-parity call",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "post-hoc controlled follow-up: separate drafting cost "
                   "from verify/refine suffix budget so same-policy "
                   "regeneration's extra draft call cannot reduce its own "
                   "suffix reachability relative to assign, on the "
                   "existing frozen math/tight and math/loose GLM-4-Flash "
                   "reference cells",
        "not_a_replacement_for": "the original clean matrix / any "
                                 "preregistered primary result -- new, "
                                 "independently-named result directories only",
        "provider": "glm", "model": "glm-4-flash-250414",
        "endpoint": "https://open.bigmodel.cn/api/paas/v4",
        "cells": TIERS, "arms": list(ARMS), "seeds": list(SEEDS), "n_tasks": 150,
        "cache": False, "temperature_first_draft": 0.0, "seed_forwarded": True,
        "suffix_caps": SUFFIX_CAPS.as_vec(),
        "draft_extra_calls": DRAFT_EXTRA_CALLS,
        "draft_extra_out_tokens": DRAFT_EXTRA_OUT_TOKENS,
        "assign_caps": ASSIGN_CAPS.as_vec(),
        "same_policy_caps": SAME_POLICY_CAPS.as_vec(),
        "reservation": "glm_exact_reservation (GLM-4-family matching "
                       "tokenizer proxy + explicit safety margin; see "
                       "module docstring) installed uniformly for both "
                       "arms and both tiers -- never the old tiered "
                       "BUDGET_TIERS total-task caps",
        "roster_sha256": sha(ROSTER),
        "reference_sha256": reference_hashes,
        "source_hashes": {str(p.relative_to(ROOT)): sha(p) for p in source_files},
        "not_filtered_by_correctness": True,
        "gate": "Zhipu calls only after the operator confirms no concurrent "
                "Mac GLM-4.7 task",
    }
    if MANIFEST.exists():
        old = json.loads(MANIFEST.read_text())
        payload["frozen_at_utc"] = old["frozen_at_utc"]
        if payload != old:
            raise RuntimeError("Refusing to alter frozen budget-parity manifest")
        print("already frozen, unchanged:", MANIFEST)
        return
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
        "LLM_BACKOFF_SCHEDULE": "3,10,30",
        "LLM_MAX_RETRIES": "4",
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


def run_cell(tier: str, arm: str, seeds: list[int], n_tasks: int | None = None,
             label_suffix: str = "", recover: bool = False,
             recover_workers: int | None = None, workers: int = 1) -> None:
    provider_env()
    tasks = load_split("math", "test")[: (n_tasks or 150)]
    task_ids = [t["id"] for t in tasks]

    ref_path_by_seed = {
        seed: EXP / REF_PREFIX / f"cot_math_{tier}" / f"results_seed{seed}.jsonl"
        for seed in seeds
    }

    call_log = EXP / f"{TAG}_{tier}_{arm}{label_suffix}_calls.jsonl"
    glm_exact_reservation.install()
    original_chat = glm_call_logger.install(call_log, condition=f"{tier}_{arm}{label_suffix}")
    try:
        for seed in seeds:
            reference = {}
            for line in ref_path_by_seed[seed].read_text().splitlines():
                if line:
                    row = json.loads(line)
                    reference[row["task_id"]] = row["solution"] or ""

            if arm == "assign":
                arm_tasks = [{**t, "_assign_solution": reference[t["id"]]}
                            for t in tasks]
                wf_factory = wf_assign_refine
            else:
                arm_tasks = tasks
                wf_factory = wf_incumbent_refine_cot

            name = f"{TAG}/{tier}_{arm}{label_suffix}"
            result_path = EXP / name / f"results_seed{seed}.jsonl"

            existing = _load_rows(result_path)
            good_ids = {tid for tid, r in existing.items()
                        if not str(r.get("status", "")).startswith("error")}

            if recover and result_path.exists():
                missing_or_bad = [t for t in arm_tasks if t["id"] not in good_ids]
                if not missing_or_bad:
                    print("complete; skip:", tier, arm, "seed", seed)
                    continue
                print(f"recovering {len(missing_or_bad)} rows:", tier, arm, "seed", seed)
                evaluate(wf_factory(), missing_or_bad, CAPS_BY_ARM[arm],
                        run_name=name, seed=seed, use_cache=False,
                        workers=recover_workers or 1)
                fresh = _load_rows(result_path)
                merged = {tid: existing[tid] for tid in good_ids}
                merged.update(fresh)
                assert set(merged) == set(task_ids)
                with result_path.open("w") as f:
                    for tid in sorted(merged):
                        f.write(json.dumps(merged[tid]) + "\n")
                print("recovered:", tier, arm, "seed", seed, "n=", len(merged))
                continue

            if result_path.exists() and sum(1 for _ in result_path.open()) == len(arm_tasks):
                print("complete; skip:", tier, arm, "seed", seed)
                continue
            t0 = time.time()
            summary = evaluate(wf_factory(), arm_tasks, CAPS_BY_ARM[arm],
                               run_name=name, seed=seed, use_cache=False,
                               workers=workers)
            print(tier, arm, "seed", seed, summary,
                  "wall_sec=", round(time.time() - t0, 1), flush=True)
    finally:
        glm_call_logger.uninstall(original_chat)
        import hbws.llm as _llm
        original_estimate = glm_exact_reservation.exact_in_tokens
        _llm.estimate_in_tokens = original_estimate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("freeze", "preflight", "run"))
    ap.add_argument("--tier", choices=TIERS)
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    ap.add_argument("--n-tasks", type=int, default=None)
    ap.add_argument("--label-suffix", default="")
    ap.add_argument("--recover", action="store_true")
    ap.add_argument("--recover-workers", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    if args.stage == "freeze":
        freeze()
        return
    if not args.tier or not args.arm:
        ap.error("--tier and --arm are required for preflight/run")
    if args.stage == "preflight":
        run_cell(args.tier, args.arm, args.seeds, n_tasks=args.n_tasks or 10,
                 label_suffix=args.label_suffix or "_preflight",
                 workers=args.workers)
    else:
        run_cell(args.tier, args.arm, args.seeds, label_suffix=args.label_suffix,
                 recover=args.recover, recover_workers=args.recover_workers,
                 workers=args.workers)


if __name__ == "__main__":
    main()
