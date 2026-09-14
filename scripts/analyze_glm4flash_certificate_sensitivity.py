#!/usr/bin/env python3
"""Post-hoc statistical and cost sensitivity for the frozen GLM certificate.

This script reads only completed Dcert traces.  It does not alter the frozen
candidate, the prespecified Hoeffding decision, or any model output.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hbws.stats import binomial_exact_upper

EXP = ROOT / "experiments"
DATA = ROOT / "data" / "math_dcert_glm4flash.jsonl"
BASELINE = EXP / "glm4flash_dcert" / "baseline" / "results_seed0.jsonl"
CANDIDATE = EXP / "glm4flash_dcert" / "candidate" / "results_seed0.jsonl"
OUTPUT = EXP / "glm4flash_dcert_posthoc_sensitivity.json"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def tokens(budget: dict) -> int:
    return budget.get("in_tokens", 0) + budget.get("out_tokens", 0)


def add_cost(target: Counter, budget: dict, *, wall_sec: float | None = None) -> None:
    target["calls"] += budget.get("llm_calls", 0)
    target["tokens"] += tokens(budget)
    target["wall_sec"] += budget.get("wall_sec", 0.0) if wall_sec is None else wall_sec


def main() -> None:
    tasks = {row["id"]: row for row in load_jsonl(DATA)}
    baseline = {row["task_id"]: row for row in load_jsonl(BASELINE)}
    candidate = {row["task_id"]: row for row in load_jsonl(CANDIDATE)}
    if set(tasks) != set(baseline) or set(tasks) != set(candidate):
        raise RuntimeError("Dcert task, baseline, and candidate IDs do not align")

    counts = Counter()
    by_subject: dict[str, Counter] = defaultdict(Counter)
    reference_cost = Counter()
    verifier_cost = Counter()
    refinement_cost = Counter()
    for task_id, task in tasks.items():
        b_row = baseline[task_id]
        c_row = candidate[task_id]
        b_ok = bool(b_row["success"])
        c_ok = bool(c_row["success"])
        trace = c_row.get("trace", [])
        verifier = next(node for node in trace if node.get("type") == "verify")
        refined = any(node.get("type") == "refine" for node in trace)

        counts["tasks"] += 1
        counts["reference_correct"] += int(b_ok)
        counts["false_rejections"] += int(b_ok and refined)
        counts["breakages"] += int(b_ok and not c_ok)
        subject = by_subject[task["subject"]]
        subject["tasks"] += 1
        subject["reference_correct"] += int(b_ok)
        subject["false_rejections"] += int(b_ok and refined)
        subject["breakages"] += int(b_ok and not c_ok)

        add_cost(reference_cost, b_row["budget"])
        add_cost(verifier_cost, verifier["budget"], wall_sec=verifier["sec"])
        final_budget = c_row["budget"]
        verifier_budget = verifier["budget"]
        refinement_cost["calls"] += final_budget["llm_calls"] - verifier_budget["llm_calls"]
        refinement_cost["tokens"] += tokens(final_budget) - tokens(verifier_budget)
        refinement_cost["wall_sec"] += final_budget["wall_sec"] - verifier_budget["wall_sec"]

    n_correct = counts["reference_correct"]
    frr_events = counts["false_rejections"]
    break_events = counts["breakages"]
    audit_cost = reference_cost + verifier_cost
    full_cost = audit_cost + refinement_cost

    projections = {}
    for n_refiners in (1, 5, 10, 20):
        direct = audit_cost + Counter({
            key: refinement_cost[key] * n_refiners
            for key in ("calls", "tokens", "wall_sec")
        })
        projections[str(n_refiners)] = {
            "assumption": "each refiner has the observed one-call refinement load on the same rejected tasks",
            "direct_candidate_specific": dict(direct),
            "shared_frr_audit": dict(audit_cost),
            "fraction_avoided": {
                key: (direct[key] - audit_cost[key]) / direct[key]
                for key in ("calls", "tokens", "wall_sec")
            },
        }

    output = {
        "status": "post_hoc_sensitivity; does not replace the prespecified Hoeffding decision",
        "counts": dict(counts),
        "posthoc_iid_mixture_model": {
            "assumption": "task/execution pairs are independent draws from one fixed deployment mixture, making outcomes among reference-correct draws binomial",
            "frr_exact_one_sided_ucb95": binomial_exact_upper(frr_events, n_correct),
            "breakage_exact_one_sided_ucb95": binomial_exact_upper(break_events, n_correct),
        },
        "sampling_design_caveat": "Dcert is the complete residual eligible benchmark pool after fixed exclusions, not a new random sample or a fixed-quota draw designed for this comparison; extrapolation to deployment requires an explicit representative-population argument.",
        "subject_counts": {name: dict(values) for name, values in by_subject.items()},
        "measured_cost": {
            "reference": dict(reference_cost),
            "initial_verifier": dict(verifier_cost),
            "refinement": dict(refinement_cost),
            "reference_plus_verifier": dict(audit_cost),
            "full_single_candidate": dict(full_cost),
            "single_candidate_fraction_avoided_by_frr_only": {
                key: refinement_cost[key] / full_cost[key]
                for key in ("calls", "tokens", "wall_sec")
            },
            "wall_sec_definition": "sum of per-call/task recorded latency, not concurrent end-to-end elapsed time",
        },
        "projected_shared_certificate_cost": projections,
    }
    OUTPUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
