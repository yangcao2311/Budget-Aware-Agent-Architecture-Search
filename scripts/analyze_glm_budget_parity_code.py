#!/usr/bin/env python3
"""Analysis for the GLM-4-Flash code-domain budget-parity follow-up:
assign vs. same-policy regeneration on the prior-tight and prior-loose
code cohorts, with drafting cost excluded from the shared verify/refine
suffix budget.

This is a code-domain REPLICATION of the completed math budget-parity
analysis (scripts/analyze_glm_budget_parity.py) -- not a cross-domain
claim. tight and loose (inherited cohort names from the frozen reference
directories) are reported completely separately.

In addition to the primary/secondary metrics also computed for math, this
script runs a COUNTERFACTUAL audit: for every actually-executed LLM/tool
call (reconstructed from each row's own `trace` plus the raw call log,
which preserve exact execution order), replay it against a fresh shadow
ledger using the ORIGINAL BUDGET_TIERS[cohort] caps -- the tiered caps
this follow-up deliberately does NOT use for the real run -- and record
the first position (if any) at which that shadow ledger would have
refused admission. This never affects the real run; it only reports how
much of the real, budget-parity-enabled suffix would have been gated
under the original design.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hbws.ledger import BudgetCaps, TaskLedger, llm_call_vec, tool_call_vec

EXP = ROOT / "experiments"
TAG = "glm_budget_parity_code_20260917"
REF_PREFIX = "glm4flash_repaired_matrix_clean_20260910_envelope_test"
SEEDS = (0, 1, 2)
COHORTS = ("tight", "loose")
ARMS = ("assign", "same_policy")

ORIGINAL_TIER_CAPS = {
    "tight": BudgetCaps(4, 8000, 2000, 4, 90, 0.10),
    "loose": BudgetCaps(8, 16000, 4000, 6, 180, 0.25),
}

sys.path.insert(0, str(ROOT / "scripts"))
import glm_exact_reservation  # noqa: E402


def load(path: Path) -> dict[str, dict]:
    out = {}
    for line in path.read_text().splitlines():
        if line:
            row = json.loads(line)
            out[row["task_id"]] = row
    return out


def load_calls(cohort: str, arm: str) -> dict[tuple[str, int], list[dict]]:
    """(task_id, seed) -> ordered list of LLM call-log rows (execution
    order == file order, since each task/seed runs single-threaded)."""
    path = EXP / f"{TAG}_{cohort}_{arm}_calls.jsonl"
    out: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for line in path.read_text().splitlines():
        if line:
            row = json.loads(line)
            out[(row["task_id"], row["seed"])].append(row)
    return out


def success(row: dict) -> bool:
    return bool(row.get("success_symbolic", row.get("success", False)))


def first_gate(row: dict) -> tuple[str, dict]:
    trace = row.get("trace") or []
    for i, item in enumerate(trace):
        if item.get("type") != "verify":
            continue
        if "reserve_rejected" in item:
            return "verify_gated", {}
        following = trace[i + 1] if i + 1 < len(trace) else {}
        if following.get("type") == "refine":
            return "reject", following
        return "pass", {}
    return "no_verify", {}


def ci(values: list[float], seed: int, draws: int = 10000) -> dict:
    if not values:
        return {"point": None, "lo": None, "hi": None, "ci_is_degenerate": True}
    point = sum(values) / len(values)
    n_events = sum(1 for v in values if v > 0)
    rng = random.Random(seed)
    n = len(values)
    samples = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n
                     for _ in range(draws))
    lo, hi = samples[int(.025 * draws)], samples[int(.975 * draws)]
    degenerate = n_events <= 2 or (hi - lo) < (1.0 / n)
    return {"point": point, "lo": lo, "hi": hi, "ci_is_degenerate": degenerate,
            "n_events": n_events, "n_clusters": n, "note": "post-hoc descriptive"}


def counterfactual_blocked(row: dict, llm_calls: list[dict], cohort: str) -> dict:
    """Replay this row's actual LLM/tool calls, in execution order, against
    a fresh shadow ledger using the ORIGINAL (pre-budget-parity) tier
    caps. Returns the first blocked position, if any -- informational
    only; never affects the real (already-completed) run."""
    caps = ORIGINAL_TIER_CAPS[cohort]
    ledger = TaskLedger(caps)
    trace = row.get("trace") or []
    llm_iter = iter(llm_calls)
    for idx, item in enumerate(trace):
        node_type = item.get("type")
        if node_type in ("generate", "refine"):
            call = next(llm_iter, None)
            if call is None or call.get("status") != "completed":
                # A real execution error here means no reservation was
                # even attempted under the real (parity) caps; nothing to
                # replay past this point.
                return {"blocked": False, "blocked_at_index": None,
                       "blocked_at_type": None, "note": "real row errored "
                       "before this point; counterfactual not applicable"}
            req = call["request"]
            in_tok = glm_exact_reservation.exact_in_tokens(req["messages"])
            vec = llm_call_vec(in_tok, req["max_tokens"])
            lease = ledger.reserve(vec)
            if lease is None:
                return {"blocked": True, "blocked_at_index": idx,
                       "blocked_at_type": node_type}
            ledger.settle(lease, {"llm_calls": 1,
                                  "in_tokens": call.get("in_tokens") or in_tok,
                                  "out_tokens": call.get("out_tokens") or 0})
        elif node_type == "verify":
            # code family: verify is a tool call, zero LLM cost.
            lease = ledger.reserve(tool_call_vec())
            if lease is None:
                return {"blocked": True, "blocked_at_index": idx,
                       "blocked_at_type": node_type}
            ledger.settle(lease, {"tool_calls": 1})
        # "assign" nodes are free; nothing to reserve.
    return {"blocked": False, "blocked_at_index": None, "blocked_at_type": None}


def analyze_cohort(cohort: str) -> dict:
    reference = {s: load(EXP / REF_PREFIX / f"direct_code_{cohort}" / f"results_seed{s}.jsonl")
                for s in SEEDS}
    task_ids = sorted(reference[0])
    assert all(sorted(reference[s]) == task_ids for s in SEEDS), (
        f"{cohort}: reference task ids differ across seeds")

    arms_out = {}
    for arm in ARMS:
        rows = {s: load(EXP / TAG / f"{cohort}_{arm}" / f"results_seed{s}.jsonl")
               for s in SEEDS}
        calls_by_seed = load_calls(cohort, arm)

        counts = Counter()
        task_delta, task_repair, task_break = [], [], []
        total_calls = 0
        total_tokens = 0
        blocked_positions = []
        # secondary: stratify same_policy by whether its first draft was
        # correct (byte/behaviorally correct == passed first verify)
        strat = {"first_draft_passed": Counter(), "first_draft_rejected": Counter()}

        for tid in task_ids:
            deltas, repairs, breaks = [], [], []
            for s in SEEDS:
                ref_row = reference[s][tid]
                row = rows[s][tid]
                ref_ok, arm_ok = success(ref_row), success(row)
                deltas.append(float(arm_ok) - float(ref_ok))
                repairs.append(float((not ref_ok) and arm_ok))
                breaks.append(float(ref_ok and not arm_ok))
                counts["reference_correct"] += int(ref_ok)
                counts["reference_wrong"] += int(not ref_ok)
                counts["repair"] += int((not ref_ok) and arm_ok)
                counts["breakage"] += int(ref_ok and not arm_ok)
                gate, _ = first_gate(row)
                counts[gate] += 1
                if gate == "reject":
                    counts["reject_reference_correct"] += int(ref_ok)
                total_calls += row.get("budget", {}).get("llm_calls", 0)
                total_tokens += (row.get("budget", {}).get("in_tokens", 0)
                                + row.get("budget", {}).get("out_tokens", 0))

                if arm == "same_policy":
                    strat_key = "first_draft_passed" if gate == "pass" else "first_draft_rejected"
                    strat[strat_key]["n"] += 1
                    strat[strat_key]["final_success"] += int(arm_ok)

                cf = counterfactual_blocked(row, calls_by_seed.get((tid, s), []), cohort)
                if cf["blocked"]:
                    blocked_positions.append({"task_id": tid, "seed": s,
                                              **cf})
            task_delta.append(sum(deltas) / len(SEEDS))
            task_repair.append(sum(repairs) / len(SEEDS))
            task_break.append(sum(breaks) / len(SEEDS))

        n = len(task_ids) * len(SEEDS)
        arms_out[arm] = {
            "rows": n,
            "accuracy": sum(success(rows[s][tid]) for tid in task_ids for s in SEEDS) / n,
            "delta_vs_reference_ci95": ci(task_delta, 20260917),
            "repair_events": counts["repair"],
            "repair_denominator": counts["reference_wrong"],
            "repair_rate": (counts["repair"] / counts["reference_wrong"]
                           if counts["reference_wrong"] else None),
            "repair_rate_ci95": ci(task_repair, 20260918),
            "breakage_events": counts["breakage"],
            "breakage_denominator": counts["reference_correct"],
            "breakage_rate": (counts["breakage"] / counts["reference_correct"]
                              if counts["reference_correct"] else None),
            "breakage_rate_ci95": ci(task_break, 20260919),
            "control_flow": {k: counts[k] for k in
                            ("pass", "reject", "reject_reference_correct",
                             "verify_gated", "no_verify")},
            "total_llm_calls": total_calls,
            "total_tokens": total_tokens,
            "counterfactual_original_caps_blocked": {
                "n_blocked_positions": len(blocked_positions),
                "denominator": n,
                "positions": blocked_positions,
            },
        }
        if arm == "same_policy":
            arms_out[arm]["stratified_by_first_draft_outcome"] = {
                k: {"n": v["n"], "final_accuracy": v["final_success"] / v["n"] if v["n"] else None}
                for k, v in strat.items()
            }

    assign, same_policy = arms_out["assign"], arms_out["same_policy"]
    pairwise = {
        "assign_minus_same_policy_breakages": assign["breakage_events"] - same_policy["breakage_events"],
        "assign_minus_same_policy_repairs": assign["repair_events"] - same_policy["repair_events"],
        "assign_minus_same_policy_accuracy": assign["accuracy"] - same_policy["accuracy"],
        "same_policy_extra_llm_calls_from_drafting": (
            same_policy["total_llm_calls"] - assign["total_llm_calls"]),
        "same_policy_extra_tokens_from_drafting": (
            same_policy["total_tokens"] - assign["total_tokens"]),
    }
    return {"cohort": cohort, "n_tasks": len(task_ids), "arms": arms_out,
            "pairwise": pairwise}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    report = {
        "condition": "glm4flash_budget_parity_code_20260917",
        "positioning": "code-domain REPLICATION of the completed math "
                       "budget-parity follow-up; not a cross-domain "
                       "replication claim",
        "directional_prediction": (
            "Expected the code-domain assignment advantage to be smaller "
            "than math's, possibly indistinguishable, because code's "
            "verifier runs the task's own tests (rejecting incorrect "
            "regenerated drafts before acceptance far more reliably than "
            "math's gold-free self-consistency check) and because the "
            "triggering event (a non-byte-identical first-accepted draft) "
            "is itself ~4x rarer in code (94.9%/93.0% byte-identical) than "
            "math (32.6%/23.8%). Reported regardless of direction; no "
            "cohort omitted, no metric swapped after seeing outcomes."),
        "bootstrap_unit": "task", "bootstrap_draws": 10000,
        "cohorts": {cohort: analyze_cohort(cohort) for cohort in COHORTS},
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
