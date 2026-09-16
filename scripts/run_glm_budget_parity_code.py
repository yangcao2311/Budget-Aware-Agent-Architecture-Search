#!/usr/bin/env python3
"""GLM-4-Flash CODE-DOMAIN budget-parity follow-up: assign vs. same-policy
regeneration on the prior-tight and prior-loose code cohorts.

This is a code-domain REPLICATION of the already-completed math
budget-parity follow-up (scripts/run_glm_budget_parity.py) -- identical
design, only the domain (and therefore the workflow's first-node prompt
and the verifier) changes. It is NOT a cross-domain replication claim and
must not be reported as one.

Reuses the frozen 150-task code roster and COMPLETE stored-reference
outputs (experiments/glm4flash_repaired_matrix_clean_20260910_
envelope_test/direct_code_{tight,loose}/results_seed{seed}.jsonl,
transferred and checksum-verified against server_transfer/
glm_budget_parity_code_references_20260916/SHA256SUMS -- never
regenerated). Keeps the existing, unmodified verifier (the task's own
visible feedback_tests, run in the sandbox -- held-out grading_tests are
never seen by the runner, only by external scoring), refinement graph,
prompts, temperature, and per-call output caps (hbws.dsl.wf_assign_refine /
wf_incumbent_refine). Does NOT use the old BUDGET_TIERS total-task caps
for the real run: those are exactly the confound this follow-up isolates
(their would-have-gated positions are instead audited counterfactually,
see analyze_glm_budget_parity_code.py).

Budget-parity mechanism (identical structure to the math version, only
DRAFT_EXTRA_OUT_TOKENS changed to match code's own first-draft node):
a single hbws.ledger.TaskLedger still meters the whole task (no change to
hbws/ledger.py or hbws/runner.py); same_policy's total cap is SUFFIX_CAPS
plus exactly one extra call's worst-case cost (wf_incumbent_refine's "g"
node: max_output_tokens=1024, NOT 1536 -- code's own direct first-draft
cap, unlike math's cot first-draft). assign's cap is SUFFIX_CAPS
unchanged. After same_policy's draft settles, both arms have identical
remaining headroom for the verify+refine suffix -- proven for this module
in tests/test_glm_budget_parity_code_noninvasive.py, mirroring the math
version's arithmetic proof.

Note on code's own verify cost: for family=="code", hbws.runner's verify
node reserves a TOOL call (tool_call_vec()), not an LLM call -- it runs
the task's own feedback_tests in the sandbox, at zero LLM token cost. So
SUFFIX_CAPS's generous max_llm_calls=8 headroom is used only by up to 3
refine calls in this domain (never by verify itself); max_tool_calls=8
covers the up-to-4 verify tool calls (1 initial + up to 3 after refines).
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
from hbws.dsl import wf_assign_refine, wf_incumbent_refine
from hbws.ledger import BudgetCaps
from hbws.protocol import evaluate

import glm_call_logger
import glm_exact_reservation
from run_glm4flash_replication import load_key

EXP = ROOT / "experiments"
TAG = "glm_budget_parity_code_20260917"
MANIFEST = EXP / f"{TAG}_manifest.json"
ROSTER = EXP / f"{TAG}_roster.json"
REF_PREFIX = "glm4flash_repaired_matrix_clean_20260910_envelope_test"

SEEDS = (0, 1, 2)
COHORTS = ("tight", "loose")
ARMS = ("assign", "same_policy")

# Identical structure/sizing to the math follow-up's SUFFIX_CAPS (see
# scripts/run_glm_budget_parity.py) -- reused verbatim, per the operator's
# "identical to the math budget-parity design, only the domain changes."
SUFFIX_CAPS = BudgetCaps(
    max_llm_calls=8, max_in_tokens=128000, max_out_tokens=8 * 1536,
    max_tool_calls=8, max_wall_sec=900, max_usd=1.0,
)
DRAFT_EXTRA_CALLS = 1
# Code's own wf_incumbent_refine "g" node cap (1024), NOT math's 1536 --
# the draft-extra allowance must match the actual worst-case cost of the
# one extra call this domain's same_policy arm makes.
DRAFT_EXTRA_OUT_TOKENS = 1024

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

# The directional prediction, frozen before any real GLM call and
# reported regardless of outcome (per operator instruction: do not add
# conditions, swap metrics, or report only one cohort if this does not
# hold).
DIRECTIONAL_PREDICTION = (
    "Expect the code-domain assignment advantage to be smaller than "
    "math's, possibly indistinguishable. Basis: (a) code's verifier runs "
    "the task's own tests, so an incorrect regenerated draft is usually "
    "rejected before acceptance, whereas math's gold-free "
    "self-consistency check can agree with an incorrect draft; (b) the "
    "first-accepted-draft byte-identical rate is 94.9%/93.0% for code "
    "cohorts vs. 32.6%/23.8% for math -- the triggering event itself is "
    "about four times rarer. Reported regardless of whether the result "
    "matches, is reversed, or is indistinguishable; no cohort is omitted "
    "and no metric is swapped after seeing outcomes."
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> None:
    if not ROSTER.exists():
        raise RuntimeError("run scripts/build_glm_budget_parity_code_roster.py first")
    source_files = [
        ROOT / "scripts/run_glm_budget_parity_code.py",
        ROOT / "scripts/build_glm_budget_parity_code_roster.py",
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
    for cohort in COHORTS:
        for seed in SEEDS:
            p = EXP / REF_PREFIX / f"direct_code_{cohort}" / f"results_seed{seed}.jsonl"
            reference_hashes[str(p.relative_to(ROOT))] = sha(p)
    payload = {
        "status": "frozen before any GLM code-domain budget-parity call",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "code-domain REPLICATION (not cross-domain claim) of "
                   "the completed math budget-parity follow-up: separate "
                   "drafting cost from the verify/refine suffix budget so "
                   "same-policy's extra draft call cannot reduce its own "
                   "suffix reachability relative to assign",
        "directional_prediction_frozen_before_run": DIRECTIONAL_PREDICTION,
        "provider": "glm", "model": "glm-4-flash-250414",
        "endpoint": "https://open.bigmodel.cn/api/paas/v4",
        "family": "code", "cohorts": COHORTS, "arms": list(ARMS),
        "seeds": list(SEEDS), "n_tasks": 150,
        "cache": False, "temperature_first_draft": 0.0, "seed_forwarded": True,
        "suffix_caps": SUFFIX_CAPS.as_vec(),
        "draft_extra_calls": DRAFT_EXTRA_CALLS,
        "draft_extra_out_tokens": DRAFT_EXTRA_OUT_TOKENS,
        "assign_caps": ASSIGN_CAPS.as_vec(),
        "same_policy_caps": SAME_POLICY_CAPS.as_vec(),
        "original_tier_caps_for_counterfactual_audit_only": {
            "tight": {"max_llm_calls": 4, "max_in_tokens": 8000,
                      "max_out_tokens": 2000, "max_tool_calls": 4,
                      "max_wall_sec": 90, "max_usd": 0.10},
            "loose": {"max_llm_calls": 8, "max_in_tokens": 16000,
                      "max_out_tokens": 4000, "max_tool_calls": 6,
                      "max_wall_sec": 180, "max_usd": 0.25},
        },
        "reservation": "glm_exact_reservation (GLM-4-family matching "
                       "tokenizer proxy + explicit safety margin) "
                       "installed uniformly for both arms and both "
                       "cohorts -- never the old BUDGET_TIERS total-task "
                       "caps for the real run",
        "roster_sha256": sha(ROSTER),
        "reference_sha256": reference_hashes,
        "source_hashes": {str(p.relative_to(ROOT)): sha(p) for p in source_files},
        "primary_metrics_frozen": [
            "assign_minus_same_policy conditional breakage (denominator: "
            "reference-correct pairs)",
            "assign_minus_same_policy conditional repair (denominator: "
            "reference-wrong pairs)",
            "assign_minus_same_policy accuracy (denominator: all pairs)",
            "reported separately per cohort",
        ],
        "secondary_metrics_frozen": [
            "accepted-path vs rejected-then-executed path decomposition",
            "stratified by whether same_policy's first draft was correct",
            "drafting vs suffix calls/tokens/cost, reported separately",
        ],
        "ci_method": "task-clustered bootstrap, 10000 draws, labeled "
                     "post-hoc descriptive; zero-event arms must not use "
                     "percentile-bootstrap endpoints to claim extra risk "
                     "is ruled out",
        "not_filtered_by_correctness": True,
        "gate": "Zhipu calls only after the operator confirms no "
                "concurrent Mac GLM-4.7 task",
    }
    if MANIFEST.exists():
        old = json.loads(MANIFEST.read_text())
        payload["frozen_at_utc"] = old["frozen_at_utc"]
        if payload != old:
            raise RuntimeError("Refusing to alter frozen code budget-parity manifest")
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


def run_cell(cohort: str, arm: str, seeds: list[int], n_tasks: int | None = None,
             label_suffix: str = "", recover: bool = False,
             recover_workers: int | None = None, workers: int = 1) -> None:
    provider_env()
    tasks = load_split("code", "test")[: (n_tasks or 150)]
    task_ids = [t["id"] for t in tasks]

    ref_path_by_seed = {
        seed: EXP / REF_PREFIX / f"direct_code_{cohort}" / f"results_seed{seed}.jsonl"
        for seed in seeds
    }

    call_log = EXP / f"{TAG}_{cohort}_{arm}{label_suffix}_calls.jsonl"
    glm_exact_reservation.install()
    original_chat = glm_call_logger.install(call_log, condition=f"{cohort}_{arm}{label_suffix}")
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
                wf_factory = wf_incumbent_refine

            name = f"{TAG}/{cohort}_{arm}{label_suffix}"
            result_path = EXP / name / f"results_seed{seed}.jsonl"

            existing = _load_rows(result_path)
            good_ids = {tid for tid, r in existing.items()
                        if not str(r.get("status", "")).startswith("error")}

            if recover and result_path.exists():
                missing_or_bad = [t for t in arm_tasks if t["id"] not in good_ids]
                if not missing_or_bad:
                    print("complete; skip:", cohort, arm, "seed", seed)
                    continue
                print(f"recovering {len(missing_or_bad)} rows:", cohort, arm, "seed", seed)
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
                print("recovered:", cohort, arm, "seed", seed, "n=", len(merged))
                continue

            if result_path.exists() and sum(1 for _ in result_path.open()) == len(arm_tasks):
                print("complete; skip:", cohort, arm, "seed", seed)
                continue
            t0 = time.time()
            summary = evaluate(wf_factory(), arm_tasks, CAPS_BY_ARM[arm],
                               run_name=name, seed=seed, use_cache=False,
                               workers=workers)
            print(cohort, arm, "seed", seed, summary,
                  "wall_sec=", round(time.time() - t0, 1), flush=True)
    finally:
        glm_call_logger.uninstall(original_chat)
        import hbws.llm as _llm
        original_estimate = glm_exact_reservation.exact_in_tokens
        _llm.estimate_in_tokens = original_estimate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("freeze", "preflight", "run"))
    ap.add_argument("--cohort", choices=COHORTS)
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
    if not args.cohort or not args.arm:
        ap.error("--cohort and --arm are required for preflight/run")
    if args.stage == "preflight":
        run_cell(args.cohort, args.arm, args.seeds, n_tasks=args.n_tasks or 10,
                 label_suffix=args.label_suffix or "_preflight", workers=args.workers)
    else:
        run_cell(args.cohort, args.arm, args.seeds, label_suffix=args.label_suffix,
                 recover=args.recover, recover_workers=args.recover_workers,
                 workers=args.workers)


if __name__ == "__main__":
    main()
