#!/usr/bin/env python3
"""Independent mathematical audit of the five-refiner temporal study.

The frozen execution script is preserved unchanged.  This audit corrects the
closed-form tie threshold and asserts the dominance and screening implications
used in the paper before emitting the report consumed by the claim audit.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "experiments/shared_refiner5_temporal_20260909"
BASE = ROOT / "experiments/glm4flash_dcert/baseline/results_seed0.jsonl"
GATE = ROOT / "experiments/glm4flash_dcert/candidate/results_seed0.jsonl"


def rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x]


def sum_cost(items: list[dict], *, trace_type: str | None = None) -> dict:
    out = {"calls": 0, "tokens": 0, "seconds": 0.0}
    for item in items:
        budget = item["budget"] if trace_type is None else next(
            x for x in item["trace"] if x["type"] == trace_type)["budget"]
        out["calls"] += budget.get("llm_calls", 0)
        out["tokens"] += budget.get("in_tokens", 0) + budget.get("out_tokens", 0)
        out["seconds"] += budget.get("wall_sec", 0)
    return out


def main() -> None:
    cfg = json.loads((DIR / "manifest.json").read_text())
    report = json.loads((DIR / "analysis.json").read_text())
    alpha = cfg["alpha"]
    q = report["false_rejections"]
    common = report["shared_exact_ceiling"]
    for arm, row in report["refiners"].items():
        k = row["breakage_events"]
        direct = row["direct_stored_simultaneous_ucb"]
        assert direct <= common + 1e-12, (arm, direct, common)
        # k + sqrt(q log(m/alpha)/2) >= q iff
        # m >= alpha * exp(2(q-k)^2/q).  The multiplication by alpha is
        # essential; using 1/alpha is an algebraic error.
        threshold = (alpha * math.exp(2 * (q - k) ** 2 / q)
                     if q and k < q else 0.0)
        row["m_for_direct_to_tie_common_ceiling"] = threshold
        row["direct_weakly_dominates_shared"] = True
    for item in report["screen_decisions"]:
        # A shared bound is no smaller than each direct bound.  Passing the
        # shared screen therefore cannot create a direct-bound false positive.
        assert item["false_positive_count"] == 0, item
    reference_cost = sum_cost(rows(BASE))
    gate_cost = sum_cost(rows(GATE), trace_type="verify")
    report["cost"]["audited_decomposition"] = {
        "reference_acquisition_sunk_if_outputs_already_stored": reference_cost,
        "shared_gate_pre_suffix": gate_cost,
        "stored_reference_plus_shared_gate": {
            key: reference_cost[key] + gate_cost[key] for key in reference_cost
        },
        "marginal_suffix_by_refiner": {
            arm: row["cost"] for arm, row in report["refiners"].items()
        },
        "interpretation": (
            "FRR screening pays the shared gate before suffix generation; "
            "stored-reference acquisition is a sunk cost when auditing an incumbent. "
            "Direct final certification additionally requires each candidate suffix."
        ),
    }
    report["math_audit"] = {
        "status": "passed",
        "tie_condition": "m >= alpha * exp(2*(Q-K)^2/Q)",
        "implication": "shared is a pre-suffix class screen; direct is the post-suffix final certificate",
    }
    out = DIR / "analysis_audited.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
