#!/usr/bin/env python3
"""Audit byte identity in the frozen clean GLM provenance matrix.

The stored reference is the final solution in the matching baseline row.  A
same-policy row exposes its *first draft* in the frozen JSONL only when no
refinement node ran: in that case the final ``solution`` is still the generated
draft.  Rows that entered refinement do not retain the original draft text, so
the all-first-draft identity rate is deliberately reported as unavailable
rather than inferred from the final solution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
BASE = EXP / "glm4flash_repaired_matrix_clean_20260910_envelope_test"
CAUSAL = EXP / "glm4flash_repaired_matrix_final_20260910_causal"

CELLS = {
    "code/tight": (
        "direct_code_tight",
        "arm1_assign_code_tight",
        "arm2_samepolicy_code_tight",
    ),
    "code/loose": (
        "direct_code_loose",
        "arm1_assign_code_loose",
        "arm2_samepolicy_code_loose",
    ),
    "math/tight": (
        "cot_math_tight",
        "arm1_assign_math_tight",
        "arm2_samepolicy_math_tight",
    ),
    "math/loose": (
        "cot_math_loose",
        "arm1_assign_math_loose",
        "arm2_samepolicy_math_loose",
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_unique(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            task_id = row["task_id"]
            if task_id in rows:
                raise ValueError(f"duplicate task_id {task_id} in {path}")
            rows[task_id] = row
    if len(rows) != 150:
        raise ValueError(f"expected 150 rows in {path}, found {len(rows)}")
    return rows


def trace_types(row: dict) -> list[str]:
    return [str(step.get("type")) for step in row.get("trace", [])]


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def signed(value: float, digits: int = 3) -> str:
    return f"{value:+.{digits}f}"


def cell_tex(cell: str) -> str:
    family, tier = cell.split("/", 1)
    return f"{family}/\\texttt{{{tier}}}"


def pairwise_metrics(task_rows: list[list[tuple[dict, dict, dict]]]) -> dict:
    """Task-clustered assign-minus-same-policy paired intervals."""

    def estimate(sample: list[list[tuple[dict, dict, dict]]]) -> dict[str, float]:
        all_rows = [row for task in sample for row in task]
        baseline_wrong = [row for row in all_rows if not bool(row[0].get("success"))]
        baseline_correct = [row for row in all_rows if bool(row[0].get("success"))]
        repair_diff = (
            sum(bool(a.get("success")) - bool(s.get("success"))
                for _, a, s in baseline_wrong)
            / len(baseline_wrong)
        )
        # Positive means assignment causes more breakage than same-policy.
        breakage_diff = (
            sum((not bool(a.get("success"))) - (not bool(s.get("success")))
                for _, a, s in baseline_correct)
            / len(baseline_correct)
        )
        accuracy_diff = (
            sum(bool(a.get("success")) - bool(s.get("success"))
                for _, a, s in all_rows)
            / len(all_rows)
        )
        return {
            "repair_rate_difference": repair_diff,
            "breakage_rate_difference": breakage_diff,
            "accuracy_difference": accuracy_diff,
        }

    point = estimate(task_rows)
    rng = random.Random(20260915)
    draws: dict[str, list[float]] = {key: [] for key in point}
    n = len(task_rows)
    for _ in range(10_000):
        sampled = [task_rows[rng.randrange(n)] for _ in range(n)]
        result = estimate(sampled)
        for key, value in result.items():
            draws[key].append(value)
    return {
        key: {
            "point": value,
            "ci95": [percentile(draws[key], 0.025), percentile(draws[key], 0.975)],
        }
        for key, value in point.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=EXP / "glm4flash_repaired_matrix_final_20260910_byte_identity.json",
    )
    parser.add_argument(
        "--identity-tex",
        type=Path,
        default=ROOT / "paper/generated_clean_identity_rows.tex",
    )
    parser.add_argument(
        "--pairwise-tex",
        type=Path,
        default=ROOT / "paper/generated_clean_pairwise_rows.tex",
    )
    args = parser.parse_args()

    report: dict = {
        "definition": (
            "UTF-8 solution strings are compared for exact equality; no "
            "normalization, whitespace stripping, or parser is applied."
        ),
        "all_regenerated_first_drafts": {
            "status": "not_recoverable_from_frozen_result_rows",
            "reason": (
                "The runner stores only the final solution.  Once a refine "
                "node executes, the original generated draft is overwritten."
            ),
        },
        "cells": {},
        "source_files": [],
    }

    aggregate = {
        "paired_rows": 0,
        "first_draft_observable": 0,
        "entered_refinement": 0,
        "first_accepted": 0,
        "first_accepted_byte_identical": 0,
    }

    for cell, (baseline_run, assign_run, same_run) in CELLS.items():
        counts = dict.fromkeys(aggregate, 0)
        cell_task_rows: dict[str, list[tuple[dict, dict, dict]]] = {}
        for seed in range(3):
            baseline_path = BASE / baseline_run / f"results_seed{seed}.jsonl"
            assign_path = CAUSAL / assign_run / f"results_seed{seed}.jsonl"
            same_path = CAUSAL / same_run / f"results_seed{seed}.jsonl"
            baseline = load_unique(baseline_path)
            assign = load_unique(assign_path)
            same = load_unique(same_path)
            if baseline.keys() != assign.keys() or baseline.keys() != same.keys():
                raise ValueError(f"task mismatch for {cell}, seed {seed}")
            report["source_files"].extend(
                [
                    {"path": str(baseline_path.relative_to(ROOT)), "sha256": sha256(baseline_path)},
                    {"path": str(assign_path.relative_to(ROOT)), "sha256": sha256(assign_path)},
                    {"path": str(same_path.relative_to(ROOT)), "sha256": sha256(same_path)},
                ]
            )

            for task_id, regenerated in same.items():
                counts["paired_rows"] += 1
                types = trace_types(regenerated)
                entered_refinement = "refine" in types
                counts["entered_refinement"] += int(entered_refinement)
                counts["first_draft_observable"] += int(not entered_refinement)

                # In this workflow, a completed row with no refine node ended
                # immediately after the first verifier accepted the draft.
                first_accepted = (
                    regenerated.get("status") == "completed"
                    and not entered_refinement
                    and types[:2] == ["generate", "verify"]
                )
                if not first_accepted:
                    continue
                counts["first_accepted"] += 1
                counts["first_accepted_byte_identical"] += int(
                    regenerated.get("solution", "")
                    == baseline[task_id].get("solution", "")
                )
                cell_task_rows.setdefault(task_id, []).append(
                    (baseline[task_id], assign[task_id], regenerated)
                )

            # Pairwise outcomes do not require the first draft to remain
            # observable, so include every frozen row for the direct contrast.
            for task_id in baseline:
                if not any(
                    existing[0] is baseline[task_id]
                    for existing in cell_task_rows.get(task_id, [])
                ):
                    cell_task_rows.setdefault(task_id, []).append(
                        (baseline[task_id], assign[task_id], same[task_id])
                    )

        counts["first_accepted_byte_identity_rate"] = (
            counts["first_accepted_byte_identical"] / counts["first_accepted"]
            if counts["first_accepted"]
            else None
        )
        counts["assign_minus_samepolicy"] = pairwise_metrics(
            [cell_task_rows[task_id] for task_id in sorted(cell_task_rows)]
        )
        report["cells"][cell] = counts
        for key in aggregate:
            aggregate[key] += counts[key]

    aggregate["first_accepted_byte_identity_rate"] = (
        aggregate["first_accepted_byte_identical"] / aggregate["first_accepted"]
        if aggregate["first_accepted"]
        else None
    )
    report["aggregate"] = aggregate

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    identity_lines = []
    pairwise_lines = []
    for cell in CELLS:
        item = report["cells"][cell]
        identity_lines.append(
            f"{cell_tex(cell)} & "
            f"{item['first_accepted_byte_identical']}/{item['first_accepted']} & "
            f"{item['first_accepted_byte_identity_rate']:.3f} \\\\"
        )
        pair = item["assign_minus_samepolicy"]
        fields = []
        for key in (
            "repair_rate_difference",
            "breakage_rate_difference",
            "accuracy_difference",
        ):
            point = pair[key]["point"]
            lo, hi = pair[key]["ci95"]
            fields.append(f"${signed(point)}$ [${signed(lo)}, {signed(hi)}$]")
        pairwise_lines.append(
            f"{cell_tex(cell)} & "
            + " & ".join(fields)
            + " \\\\"
        )
    args.identity_tex.write_text("\n".join(identity_lines) + "\n\\bottomrule\n")
    args.pairwise_tex.write_text("\n".join(pairwise_lines) + "\n\\bottomrule\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
