#!/usr/bin/env python3
"""Audit current submission evidence from raw rows; never call a provider.

The nonbinding companion has invalid frozen references. Selection must depend
only on pretreatment reference status, never on a workflow's later outcome.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXP = ROOT / "experiments"
ARMS = ("arm1_assign", "arm2_samepolicy", "arm3_diffpolicy")
LABELS = ("assign", "same-policy", "different-policy")


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def keyed(directory: Path) -> dict[tuple[str, int], dict]:
    result = {}
    for seed in range(3):
        rows = read(directory / f"results_seed{seed}.jsonl")
        assert len(rows) == 150, directory
        for row in rows:
            key = row["task_id"], seed
            assert key not in result, key
            result[key] = row
    assert len({key[0] for key in result}) == 150
    return result


def transitions(base: dict, arm: dict, eligible: set) -> dict:
    assert set(base) == set(arm)
    counts = Counter()
    for key in sorted(eligible):
        b, w = base[key], arm[key]
        assert b["status"] == "completed"
        assert w["status"] in ("completed", "reserve_rejected")
        correct, final = bool(b["success"]), bool(w["success"])
        counts["pairs"] += 1
        counts["reference_correct"] += correct
        counts["reference_wrong"] += not correct
        counts["repairs"] += not correct and final
        counts["breakages"] += correct and not final
        trace = w.get("trace") or []
        vi = next((i for i, node in enumerate(trace)
                   if node["type"] == "verify"), None)
        if vi is None or "reserve_rejected" in trace[vi]:
            counts["verification_gated"] += vi is not None
            counts["preverification_breakages"] += correct and not final
            continue
        rejection = vi + 1 < len(trace) and trace[vi + 1]["type"] == "refine"
        counts["rejections"] += rejection
        counts["correct_rejections"] += rejection and correct
        counts["initial_executed"] += rejection and "reserve_rejected" not in trace[vi + 1]
        counts["initial_gated"] += rejection and "reserve_rejected" in trace[vi + 1]
        counts["acceptance_breakages"] += correct and not final and not rejection
        counts["rejection_breakages"] += correct and not final and rejection
    assert counts["breakages"] == sum(counts[name] for name in
            ("preverification_breakages", "acceptance_breakages", "rejection_breakages"))
    return dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    analysis = json.loads((EXP / "glm4flash_repaired_matrix_final_20260910_analysis.json").read_text())
    report = {"clean": {}, "nonbinding": {}, "five_suffixes": {}}
    main_rows, cost_rows, utility_rows = [], [], []
    for cell in ("code/tight", "code/loose", "math/tight", "math/loose"):
        family, profile = cell.split("/")
        structure = "direct" if family == "code" else "cot"
        base = keyed(EXP / "glm4flash_repaired_matrix_clean_20260910_envelope_test"
                     / f"{structure}_{family}_{profile}")
        eligible = set(base)
        report["clean"][cell] = {}
        cost_rows.append(f"{cell} & reference & {sum(r['budget']['llm_calls'] for r in base.values()):,} & "
                         f"{sum(r['budget']['in_tokens'] + r['budget']['out_tokens'] for r in base.values()):,} \\\\")
        raw_arms = {}
        for index, (arm, label) in enumerate(zip(ARMS, LABELS)):
            rows = keyed(EXP / "glm4flash_repaired_matrix_final_20260910_causal" / f"{arm}_{family}_{profile}")
            raw_arms[arm] = rows
            counts = transitions(base, rows, eligible)
            a = analysis["cells"][cell]["arms"][arm]
            for actual, stored in (("repairs", "repair_events"), ("breakages", "breakage_events"),
                                   ("reference_correct", "breakage_denominator"),
                                   ("reference_wrong", "repair_denominator"),
                                   ("acceptance_breakages", "acceptance_path_breakage_events")):
                assert counts.get(actual, 0) == a[stored], (cell, arm, actual)
            calls = sum(r["budget"]["llm_calls"] for r in rows.values())
            tokens = sum(r["budget"]["in_tokens"] + r["budget"]["out_tokens"] for r in rows.values())
            assert calls == a["calls"] and tokens == a["tokens"]
            point, low, high = a["accuracy_delta_ci95"]
            assert abs(point - (counts["repairs"] - counts["breakages"]) / 450) < 1e-12
            def signed(value):
                return f"{value:+.3f}".replace("+0.", "+.").replace("-0.", "-.")
            main_rows.append(f"{cell if index == 0 else ''} & {label} & "
                             f"${signed(point)}\\,[{signed(low)},{signed(high)}]$ & "
                             f"{counts['repairs']}/{counts['reference_wrong']} & "
                             f"{counts['breakages']}/{counts['reference_correct']} & "
                             f"{counts.get('acceptance_breakages', 0)} & "
                             f"{counts.get('rejections', 0)}/{counts.get('correct_rejections', 0)} & "
                             f"{counts.get('initial_executed', 0)}/{counts.get('initial_gated', 0)} \\\\")
            cost_rows.append(f" & {label} & {calls:,} & {tokens:,} \\\\")
            report["clean"][cell][arm] = counts
        main_rows.append(r"\addlinespace[2pt]")
        if cell != "math/tight":
            import numpy as np
            task_ids = sorted({key[0] for key in eligible})
            rng = random.Random(20260915)
            samples = np.fromiter((rng.randrange(len(task_ids))
                                  for _ in range(10000 * len(task_ids))), dtype=int).reshape(10000, len(task_ids))
            task_differences = []
            for task in task_ids:
                repair_gap = break_gap = 0
                for seed in range(3):
                    key = task, seed
                    b = bool(base[key]["success"])
                    a = bool(raw_arms[ARMS[0]][key]["success"])
                    s = bool(raw_arms[ARMS[1]][key]["success"])
                    repair_gap += (not b) * (int(a) - int(s))
                    break_gap += b * (int(not a) - int(not s))
                task_differences.append((repair_gap / 3, break_gap / 3))
            differences = np.array(task_differences)
            utility = {}
            columns = []
            for loss in (1, 3, 5):
                values = differences[:, 0] - loss * differences[:, 1]
                point = float(values.mean())
                boot = np.sort(values[samples].mean(axis=1))
                low, high = boot[250], boot[9750]
                utility[loss] = {"point": point, "ci95": [float(low), float(high)]}
                columns.append(f"${point:+.3f}$ [$ {low:+.3f},{high:+.3f}$]")
            report["clean"][cell]["paired_gross_utility_difference"] = utility
            utility_rows.append(cell + " & " + " & ".join(columns) + r" \\")

    base = keyed(EXP / "glm4flash_repaired_code_20260909_envelope_test/cot_math_tight")
    eligible = {key for key, row in base.items() if row["status"] == "completed"}
    assert len(eligible) == 383 and sum(base[key]["success"] for key in eligible) == 294
    report["nonbinding"]["invalid_references_excluded"] = len(base) - len(eligible)
    for arm in ARMS:
        rows = keyed(EXP / "glm4flash_math_tight_nonbinding_20260911" / f"{arm}_math_tight_nonbinding")
        assert all(row["status"] == "completed" for row in rows.values())
        counts = transitions(base, rows, eligible)
        assert counts.get("initial_gated", 0) == 0
        report["nonbinding"][arm] = counts

    # Regrade the five complete suffix contracts, including all non-rejected
    # tasks by unchanged-reference fallback, and audit truncation association.
    from hbws.verify import grade_math
    from scripts.shared_refiner5_temporal_study import inputs
    tasks, base, gate, rejected = inputs()
    trajectory_rows = read(EXP / "shared_refiner5_temporal_20260909/trajectories.jsonl")
    trajectories = {(row["task_id"], row["arm"]): row for row in trajectory_rows}
    assert len(trajectories) == len(trajectory_rows) == 500
    old = json.loads((EXP / "shared_refiner5_temporal_20260909/analysis_audited.json").read_text())
    for arm, previous in old["refiners"].items():
        counts = Counter()
        for task in rejected:
            row = trajectories[task, arm]
            assert row["status"] == "completed"
            correct = bool(base[task]["success"])
            final = bool(grade_math(row["solution"], tasks[task]["gold_answer"]))
            length = "length" in row["finish_reasons"]
            counts["repairs"] += not correct and final
            counts["breakages"] += correct and not final
            counts["length_terminated"] += length
            counts["length_breakages"] += correct and not final and length
        assert counts["repairs"] == previous["repair_events"]
        assert counts["breakages"] == previous["breakage_events"]
        assert counts["length_terminated"] == previous["length_terminated"]
        report["five_suffixes"][arm] = dict(counts)
    if args.render:
        (ROOT / "paper/generated_clean_matrix_rows.tex").write_text(
            "\n".join(main_rows[:-1] + [r"\bottomrule"]) + "\n")
        (ROOT / "paper/generated_clean_cost_rows.tex").write_text(
            "\n".join(cost_rows + [r"\bottomrule"]) + "\n")
        (ROOT / "paper/generated_clean_utility_rows.tex").write_text(
            "\n".join(utility_rows + [r"\bottomrule"]) + "\n")
        (EXP / "submission_auxiliary_independent_audit_20260915.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
