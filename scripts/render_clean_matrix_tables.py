#!/usr/bin/env python3
"""Render LaTeX row fragments directly from the clean-matrix analysis JSON."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "experiments/glm4flash_repaired_matrix_final_20260910_analysis.json"
MATRIX_OUT = ROOT / "paper/generated_clean_matrix_rows.tex"
LAMBDA_OUT = ROOT / "paper/generated_lambda_rows.tex"
ARMS = (("arm1_assign", "assign"),
        ("arm2_samepolicy", "same-policy"),
        ("arm3_diffpolicy", "different-policy"))


def dec(x: float) -> str:
    value = f"{abs(x):.3f}"
    if value.startswith("0"):
        value = value[1:]
    return value


def signed(x: float) -> str:
    return ("+" if x >= 0 else "-") + dec(x)


def main() -> None:
    report = json.loads(ANALYSIS.read_text())
    matrix = []
    lambdas = []
    for key in ("code/tight", "code/loose", "math/tight", "math/loose"):
        cell = report["cells"][key]
        family, tier = key.split("/")
        p = dec(cell["reference"]["accuracy"])
        for index, (arm_key, arm_label) in enumerate(ARMS):
            arm = cell["arms"][arm_key]
            point, lo, hi = arm["accuracy_delta_ci95"]
            cell_label = f"{family}/{tier}" if index == 0 else ""
            p_label = p if index == 0 else ""
            rr = arm["status_counts"].get("reserve_rejected", 0)
            matrix.append(
                f"{cell_label} & {p_label} & {arm_label} & "
                f"${signed(point)}\\,[{signed(lo)},{signed(hi)}]$ & "
                f"{arm['repair_events']}/{arm['repair_denominator']} & "
                f"{arm['breakage_events']}/{arm['breakage_denominator']} & "
                f"{arm['acceptance_path_breakage_events']} & {rr} \\\\"
            )
        matrix.append("\\addlinespace[2pt]")
        assign = cell["arms"]["arm1_assign"]
        diff = cell["arms"]["arm3_diffpolicy"]
        value = cell["lambda_star_assign_vs_diffpolicy"]
        label = "undefined" if value is None else dec(value)
        lambdas.append(
            f"{family}/{tier} & {assign['repair_events']}/{assign['breakage_events']} & "
            f"{diff['repair_events']}/{diff['breakage_events']} & {label} \\\\"
        )
    MATRIX_OUT.write_text("\n".join(matrix[:-1] + [r"\bottomrule"]) + "\n")
    LAMBDA_OUT.write_text("\n".join(lambdas + [r"\bottomrule"]) + "\n")
    print("wrote", MATRIX_OUT, "and", LAMBDA_OUT)


if __name__ == "__main__":
    main()
