#!/usr/bin/env python3
"""Render LaTeX fragments from the design-based precision analysis.

Detectability figures are conditional on an assumed paired structure
(``rho_arm``), which the executed design does not identify, so task counts are
rendered as scenarios across that structure rather than as single numbers.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "experiments/effect_precision_20260916.json"
PAPER = ROOT / "paper"

CELL_TEX = {"code/tight": r"code/\texttt{tight}",
            "code/loose": r"code/\texttt{loose}",
            "math/loose": r"math/\texttt{loose}"}
RHO_KEYS = ("rho_arm=0", "rho_arm=0.5", "rho_arm=0.9")


def signed(value: float, digits: int = 3) -> str:
    out = f"{abs(value):.{digits}f}"
    out = out[1:] if out.startswith("0") else out
    return ("+" if value >= 0 else "-") + out


def power(value: float) -> str:
    """Two decimals, never rounding a probability below one up to 1.00."""
    if value >= 1.0:
        return "1.00"
    return f"{min(value, 0.99):.2f}".replace("0.", ".")


def interval(entry: dict) -> str:
    lo, hi = entry["ci95"]
    return f"${signed(entry['point'])}$ [{signed(lo)}, {signed(hi)}]"


def span(values: list) -> str:
    """Render a range across paired structures; '--' where unreachable."""
    if any(v is None for v in values):
        finite = [v for v in values if v is not None]
        if not finite:
            return "--"
        return f"$\\geq${min(finite):,}"
    lo, hi = min(values), max(values)
    return f"{lo:,}" if lo == hi else f"{lo:,}--{hi:,}"


def main() -> None:
    report = json.loads(REPORT.read_text())
    detect = {e["cell"]: e for e in report["executed_cell_detectability"]}

    rows = []
    for cell, payload in report["cells"].items():
        d = payload["assignment_minus_same_policy"]
        e = detect[cell]
        icc = f"{e['observed_within_task_icc']:.2f}".replace("0.", ".")
        rate = f"{e['observed_breakage_rate']:.4f}".replace("0.", ".")
        pw = [e["power_half_reduction_as_executed"][k] for k in RHO_KEYS]
        pw_span = (power(min(pw)) if min(pw) == max(pw)
                   else f"{power(min(pw))}--{power(max(pw))}")
        scen = [e["tasks_scenario_80pct_half_reduction"][k] for k in RHO_KEYS]
        rows.append(
            f"{CELL_TEX[cell]} & {interval(d['repair_difference'])} & "
            f"{interval(d['breakage_difference'])} & "
            f"{interval(d['accuracy_difference'])} & "
            f"{e['tasks_with_an_eligible_seed']}/{e['eligible_positions']} & "
            f"{rate}\\,/\\,{icc} & {pw_span} & {span(scen)} \\\\")
    (PAPER / "generated_precision_rows.tex").write_text(
        "\n".join(rows + [r"\bottomrule"]) + "\n")

    rows = []
    for entry in report["detectability_surface"]:
        if entry["breakage_rate"] not in (0.005, 0.02, 0.08):
            continue
        # rows vary the within-task ICC; the three rho_arm column groups are a
        # different correlation and are always all shown.
        if entry["within_task_icc"] not in (0.0, 0.5):
            continue
        cells = []
        for k in RHO_KEYS:
            mde = entry["mde_relative_at_80pct"][k]
            cells.append(power(entry["power_half_reduction"][k]))
            cells.append("--" if mde is None
                         else f"{mde:.2f}".replace("0.", "."))
        rows.append(
            f"{entry['breakage_rate']:.3f}".replace("0.", ".")
            + f" & {entry['within_task_icc']:.1f}".replace("0.", ".")
            + f" & {entry['tasks_screened']:,} & "
            + " & ".join(cells) + r" \\")
    (PAPER / "generated_design_surface_rows.tex").write_text(
        "\n".join(rows + [r"\bottomrule"]) + "\n")

    fe = report["certification_feasibility"]
    prof = report["surface_eligibility_profile"]
    sizes = {c: report["cells"][c]["tasks_with_an_eligible_seed"]
             for c in report["cells"]}
    elig = {c: report["cells"][c]["eligible_positions"] for c in report["cells"]}
    macros = [
        rf"\newcommand{{\CertZeroFloor}}{{{fe['zero_rejection_floor_at_this_n']:.5f}}}",
        rf"\newcommand{{\CertTasksNeeded}}{{{fe['tasks_needed_if_zero_false_rejections']:,}}}",
        rf"\newcommand{{\CertObservedBound}}{{{fe['observed_bound']:.5f}}}",
        rf"\newcommand{{\NoEligibleShare}}{{{100 * prof['share_of_tasks_with_no_eligible_seed']:.0f}\%}}",
        rf"\newcommand{{\ClusterRange}}{{{min(sizes.values())}--{max(sizes.values())}}}",
        rf"\newcommand{{\EligibleRange}}{{{min(elig.values())}--{max(elig.values())}}}",
    ]
    (PAPER / "generated_precision_macros.tex").write_text("\n".join(macros) + "\n")
    print("wrote precision fragments")


if __name__ == "__main__":
    main()
