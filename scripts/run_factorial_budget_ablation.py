#!/usr/bin/env python3
"""Factorial budget ablation: LLM-call cap x per-task output-token cap,
decomposing the paper's composite tight/loose budget profiles.

The tight/loose profiles move BOTH the call cap (max_llm_calls) and the
token cap (max_out_tokens) together, along with three other dimensions
(max_in_tokens, max_tool_calls, max_wall_sec, max_usd) that happen to
scale in lockstep in hbws.ledger.BUDGET_TIERS. This run separates the two
NAMED axes (call cap, token cap) into a 2x2 grid while bundling the three
auxiliary dimensions with whichever named axis they most naturally track
(count-of-actions dimensions with the call cap; token-volume dimensions
with the token cap) -- so that mixing an axis from one tier with an axis
from the other still produces a single well-formed BudgetCaps object, and
so that the two corners where both axes come from the SAME tier are
IDENTICAL, field-for-field, to that tier's own BUDGET_TIERS entry (see
CELLS below and its corner-equality assertions -- this is the
"corner-reproduction" check, verified by construction, not by re-running
a separately-stored historical result under a different verifier: the
existing direct_code_tight/loose reference runs used the CLEAN verifier,
which this factorial deliberately does not, per the sizing constraint
below, so no numeric accuracy/breakage comparison against those older
runs would be meaningful here).

Sizing rationale (why NOT the clean verifier): a power simulation over
the real eligible denominators (150 tasks x 3 seeds, task-clustered,
rho_arm=0.3) shows the clean/normal verifier's baseline breakage (~0.02)
gives essentially no power (9-27%) to detect even a 50% relative
reduction -- reproducing the uninformative null already seen in the code
budget-parity replication. The fully-masked verifier (mask fraction 0.0)
has baseline breakage ~0.235, giving 84-100% power. See
scripts/build_factorial_masked_code_roster.py for the masking mechanism
(existing, unmodified scripts/run_envelope.py::mask_tests).

Single self-drafting arm only (hbws.dsl.wf_incumbent_refine) -- this is
NOT the assign-vs-same_policy budget-parity design; there is no separate
drafting/suffix budget split here, by design (the point is to decompose
the EXISTING composite tight/loose ledger semantics as-is, not to
introduce a new accounting scheme on top of them).

Per-iteration logging (scripts/glm_iteration_logger.py) records, for
every LLM call (draft or refine): the exact reservation vector computed
before the call, whether it was refused, the full request, the response
text, and usage -- so any tighter/looser cap combination can be replayed
entirely offline afterward (scripts/reconstruct_iterations.py) without
another GLM call.
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
from hbws.dsl import wf_incumbent_refine
from hbws.ledger import BUDGET_TIERS, BudgetCaps
from hbws.protocol import evaluate
from run_envelope import mask_tests
from run_glm4flash_replication import load_key

import glm_exact_reservation
import glm_iteration_logger

EXP = ROOT / "experiments"
TAG = "factorial_budget_ablation_20260917"
MANIFEST = EXP / f"{TAG}_manifest.json"
ROSTER = EXP / f"{TAG}_roster.json"
MASK_FRAC = 0.0
SEEDS = (0, 1, 2)
CELLS_ORDER = ("A", "B", "C", "D")

TIGHT = BUDGET_TIERS["tight"]
LOOSE = BUDGET_TIERS["loose"]


def _call_bundle(tier: BudgetCaps) -> dict:
    return {"max_llm_calls": tier.max_llm_calls, "max_tool_calls": tier.max_tool_calls,
            "max_wall_sec": tier.max_wall_sec, "max_usd": tier.max_usd}


def _token_bundle(tier: BudgetCaps) -> dict:
    return {"max_in_tokens": tier.max_in_tokens, "max_out_tokens": tier.max_out_tokens}


def make_caps(call_tier: str, token_tier: str) -> BudgetCaps:
    call_src = TIGHT if call_tier == "tight" else LOOSE
    token_src = TIGHT if token_tier == "tight" else LOOSE
    return BudgetCaps(**_call_bundle(call_src), **_token_bundle(token_src))


CELLS = {
    "A": make_caps("tight", "tight"),
    "B": make_caps("tight", "loose"),
    "C": make_caps("loose", "tight"),
    "D": make_caps("loose", "loose"),
}
# Corner-reproduction check, verified by construction: A and D must be
# field-identical to the existing tight/loose tier objects.
assert CELLS["A"] == TIGHT, "cell A must reproduce BUDGET_TIERS['tight'] exactly"
assert CELLS["D"] == LOOSE, "cell D must reproduce BUDGET_TIERS['loose'] exactly"

CELL_CALL_TOKEN_TIER = {
    "A": ("tight", "tight"), "B": ("tight", "loose"),
    "C": ("loose", "tight"), "D": ("loose", "loose"),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> None:
    if not ROSTER.exists():
        raise RuntimeError("run scripts/build_factorial_masked_code_roster.py first")
    source_files = [
        ROOT / "scripts/run_factorial_budget_ablation.py",
        ROOT / "scripts/build_factorial_masked_code_roster.py",
        ROOT / "scripts/glm_exact_reservation.py",
        ROOT / "scripts/glm_iteration_logger.py",
        ROOT / "scripts/run_envelope.py",
        ROOT / "hbws/runner.py", ROOT / "hbws/llm.py", ROOT / "hbws/dsl.py",
        ROOT / "hbws/prompts.py", ROOT / "hbws/verify.py",
        ROOT / "hbws/ledger.py", ROOT / "hbws/protocol.py",
    ]
    payload = {
        "status": "frozen before any factorial-ablation GLM call",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "decompose the composite tight/loose budget profile "
                   "into its LLM-call-cap and output-token-cap axes; "
                   "report main effects and interaction on breakage",
        "provider": "glm", "model": "glm-4-flash-250414",
        "endpoint": "https://open.bigmodel.cn/api/paas/v4",
        "family": "code", "mask_frac": MASK_FRAC,
        "workflow": "wf_incumbent_refine (self-drafting arm only)",
        "cells": {c: CELLS[c].as_vec() for c in CELLS_ORDER},
        "cell_call_token_tier": CELL_CALL_TOKEN_TIER,
        "corner_check": {
            "cell_A_equals_BUDGET_TIERS_tight": CELLS["A"] == TIGHT,
            "cell_D_equals_BUDGET_TIERS_loose": CELLS["D"] == LOOSE,
        },
        "seeds": list(SEEDS), "n_tasks": 150,
        "cache": False, "temperature_first_draft": 0.0, "seed_forwarded": True,
        "reservation": "glm_exact_reservation (GLM-4-family matching "
                       "tokenizer proxy + explicit safety margin), "
                       "installed uniformly for all four cells",
        "roster_sha256": sha(ROSTER),
        "source_hashes": {str(p.relative_to(ROOT)): sha(p) for p in source_files},
        "primary_metrics_frozen": [
            "main effect of call cap on breakage (B+D)/2 - (A+C)/2, "
            "conditional on reference-correct pairs",
            "main effect of token cap on breakage (C+D)/2 - (A+B)/2",
            "interaction: (D-C) - (B-A)",
            "reference here means the OTHER three cells' own first-draft "
            "outcome is not the reference; breakage/repair are computed "
            "against each row's own held-out grader correctness, "
            "conditioned on whether the row's FIRST draft (pre-refine) "
            "was correct -- see analyze_factorial_budget_ablation.py",
        ],
        "ci_method": "task-clustered paired bootstrap, 10000 draws, "
                     "labeled post-hoc descriptive",
        "gate": "Zhipu calls only after the operator confirms no "
                "concurrent Mac GLM-4.7 task",
    }
    if MANIFEST.exists():
        old = json.loads(MANIFEST.read_text())
        payload["frozen_at_utc"] = old["frozen_at_utc"]
        if payload != old:
            raise RuntimeError("Refusing to alter frozen factorial-ablation manifest")
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
        "LLM_PRICE_IN_PER_M": "0", "LLM_PRICE_OUT_PER_M": "0",
        "LLM_BACKOFF_SCHEDULE": "3,10,30", "LLM_MAX_RETRIES": "4",
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


def run_cell(cell: str, seeds: list[int], n_tasks: int | None = None,
             label_suffix: str = "", recover: bool = False,
             recover_workers: int | None = None, workers: int = 1) -> None:
    provider_env()
    tasks = [mask_tests(t, MASK_FRAC)
            for t in load_split("code", "test")[: (n_tasks or 150)]]
    task_ids = [t["id"] for t in tasks]

    call_log = EXP / f"{TAG}_{cell}{label_suffix}_calls.jsonl"
    import hbws.llm as llm_mod
    original_estimate = llm_mod.estimate_in_tokens
    glm_exact_reservation.install()
    original_chat = glm_iteration_logger.install(call_log, cell=cell + label_suffix)
    try:
        for seed in seeds:
            name = f"{TAG}/cell_{cell}{label_suffix}"
            result_path = EXP / name / f"results_seed{seed}.jsonl"

            existing = _load_rows(result_path)
            good_ids = {tid for tid, r in existing.items()
                        if not str(r.get("status", "")).startswith("error")}

            if recover and result_path.exists():
                missing_or_bad = [t for t in tasks if t["id"] not in good_ids]
                if not missing_or_bad:
                    print("complete; skip:", cell, "seed", seed)
                    continue
                print(f"recovering {len(missing_or_bad)} rows:", cell, "seed", seed)
                evaluate(wf_incumbent_refine(), missing_or_bad, CELLS[cell],
                        run_name=name, seed=seed, use_cache=False,
                        workers=recover_workers or 1)
                fresh = _load_rows(result_path)
                merged = {tid: existing[tid] for tid in good_ids}
                merged.update(fresh)
                assert set(merged) == set(task_ids)
                with result_path.open("w") as f:
                    for tid in sorted(merged):
                        f.write(json.dumps(merged[tid]) + "\n")
                print("recovered:", cell, "seed", seed, "n=", len(merged))
                continue

            if result_path.exists() and sum(1 for _ in result_path.open()) == len(tasks):
                print("complete; skip:", cell, "seed", seed)
                continue
            t0 = time.time()
            summary = evaluate(wf_incumbent_refine(), tasks, CELLS[cell],
                               run_name=name, seed=seed, use_cache=False,
                               workers=workers)
            print(cell, "seed", seed, summary,
                  "wall_sec=", round(time.time() - t0, 1), flush=True)
    finally:
        glm_iteration_logger.uninstall(original_chat)
        llm_mod.estimate_in_tokens = original_estimate
        glm_exact_reservation.uninstall(original_estimate)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("freeze", "preflight", "run"))
    ap.add_argument("--cell", choices=CELLS_ORDER)
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
    if not args.cell:
        ap.error("--cell is required for preflight/run")
    if args.stage == "preflight":
        run_cell(args.cell, args.seeds, n_tasks=args.n_tasks or 10,
                 label_suffix=args.label_suffix or "_preflight", workers=args.workers)
    else:
        run_cell(args.cell, args.seeds, label_suffix=args.label_suffix,
                 recover=args.recover, recover_workers=args.recover_workers,
                 workers=args.workers)


if __name__ == "__main__":
    main()
