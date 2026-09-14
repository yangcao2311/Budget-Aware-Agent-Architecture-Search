#!/usr/bin/env python3
"""Audit first-verifier reachability and refinement gating in the clean matrix."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
BASE = EXP / "glm4flash_repaired_matrix_clean_20260910_envelope_test"
CAUSAL = EXP / "glm4flash_repaired_matrix_final_20260910_causal"
OUT = EXP / "glm4flash_matrix_control_flow_audit_20260911.json"
TEX = ROOT / "paper/generated_control_flow_rows.tex"
ARMS = (
    ("assign", "arm1_assign"),
    ("same policy", "arm2_samepolicy"),
    ("different policy", "arm3_diffpolicy"),
)


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def correct(row: dict) -> bool:
    return bool(row.get("success_symbolic", row.get("success", False)))


def first_gate(row: dict) -> tuple[str, dict]:
    trace = row.get("trace") or []
    for index, item in enumerate(trace):
        if item.get("type") != "verify":
            continue
        if "reserve_rejected" in item:
            return "verify_gated", {}
        next_item = trace[index + 1] if index + 1 < len(trace) else {}
        if next_item.get("type") == "refine":
            return "reject", next_item
        return "pass", {}
    return "no_verify", {}


def main() -> None:
    audit = []
    for family in ("code", "math"):
        structure = "direct" if family == "code" else "cot"
        for tier in ("tight", "loose"):
            baseline = {}
            for seed in (0, 1, 2):
                path = BASE / f"{structure}_{family}_{tier}" / f"results_seed{seed}.jsonl"
                for row in rows(path):
                    baseline[(row["task_id"], seed)] = row
            assert len(baseline) == 450
            for arm_label, arm_dir in ARMS:
                workflows = {}
                for seed in (0, 1, 2):
                    path = CAUSAL / f"{arm_dir}_{family}_{tier}" / f"results_seed{seed}.jsonl"
                    for row in rows(path):
                        workflows[(row["task_id"], seed)] = row
                assert len(workflows) == 450
                counts = Counter()
                for key, workflow in workflows.items():
                    base_ok = correct(baseline[key])
                    workflow_ok = correct(workflow)
                    gate, initial_refine = first_gate(workflow)
                    counts[gate] += 1
                    trace = workflow.get("trace") or []
                    if any(x.get("type") == "refine" and
                           "reserve_rejected" not in x for x in trace):
                        counts["any_refinement_executed"] += 1
                    if any(x.get("type") == "refine" and
                           "reserve_rejected" in x for x in trace):
                        counts["any_refinement_gated"] += 1
                    if gate == "reject":
                        key_name = ("initial_refinement_gated"
                                    if "reserve_rejected" in initial_refine
                                    else "initial_refinement_executed")
                        counts[key_name] += 1
                    if base_ok:
                        counts["reference_correct"] += 1
                        if gate == "reject":
                            counts["reject_reference_correct"] += 1
                        elif gate == "verify_gated":
                            counts["verify_gated_reference_correct"] += 1
                        if not workflow_ok:
                            counts["breakage"] += 1
                            if gate in ("verify_gated", "no_verify"):
                                counts["preverification_breakage"] += 1
                            elif gate == "pass":
                                counts["acceptance_path_breakage"] += 1
                            else:
                                counts["rejection_path_breakage"] += 1
                assert sum(counts[x] for x in ("pass", "reject", "verify_gated",
                                                "no_verify")) == 450
                audit.append({
                    "family": family,
                    "tier": tier,
                    "arm": arm_label,
                    "n": 450,
                    **dict(counts),
                })

    assert all(row.get("acceptance_path_breakage", 0) == 0
               for row in audit if row["arm"] == "assign")
    OUT.write_text(json.dumps({
        "scope": "first verifier and task-level refinement reachability",
        "unit": "matched task-seed pair",
        "rows": audit,
    }, indent=2, sort_keys=True) + "\n")

    labels = {"assign": "assign", "same policy": "same-policy",
              "different policy": "different-policy"}
    lines = []
    for row in audit:
        get = lambda key: row.get(key, 0)
        lines.append(
            f"{row['family']}/{row['tier']} & {labels[row['arm']]} & "
            f"{get('verify_gated')} & {get('pass')} & {get('reject')} & "
            f"{get('reject_reference_correct')} & {get('initial_refinement_executed')} & "
            f"{get('initial_refinement_gated')} & {get('preverification_breakage')} & "
            f"{get('acceptance_path_breakage')} & {get('rejection_path_breakage')} \\\\"
        )
    lines = [line.rstrip("\\") + r"\\" for line in lines]
    lines[-1] = lines[-1].rstrip("\\").rstrip()
    TEX.write_text("\n".join(lines) + "\n")
    print("wrote", OUT)
    print("wrote", TEX)
    for row in audit:
        print(row)


if __name__ == "__main__":
    main()
