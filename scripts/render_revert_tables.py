#!/usr/bin/env python3
"""Render the revert-control fragment and macros from the analysis JSON."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "experiments/revert_control_20260917.json"
PAPER = ROOT / "paper"

ORDER = ["code/tight", "code/loose", "math/tight", "math/loose"]


def f3(x: float) -> str:
    """Format as a signed three-decimal value with a leading-zero strip."""
    s = f"{x:+.3f}"
    return s.replace("0.", ".", 1) if abs(x) < 1 else s


def main() -> None:
    r = json.load(open(SRC))["cells"]
    rows = []
    for c in ORDER:
        d = r[c]
        a, v = d["actual"], d["revert"]
        ci = d["revert_minus_actual_ci95"]["d_accuracy"]
        cell = c.replace("/", "/\\texttt{") + "}"
        rows.append(
            f"{cell} & {d['pairs']} & {d['accepted']} & {d['no_acceptance']} & "
            f"{d['returns_changed_by_revert']} & "
            f"{a['repair_count']}/{d['reference_wrong']} & "
            f"{v['repair_count']}/{d['reference_wrong']} & "
            f"{a['breakage_count']}/{d['reference_correct']} & "
            f"{v['breakage_count']}/{d['reference_correct']} & "
            f"${f3(v['delta'] - a['delta'])}$ [{f3(ci[0])}, {f3(ci[1])}] \\\\")
    (PAPER / "generated_revert_rows.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")

    tot_pairs = sum(r[c]["pairs"] for c in ORDER)
    tot_chg = sum(r[c]["returns_changed_by_revert"] for c in ORDER)
    ml = r["math/loose"]
    ci = ml["revert_minus_actual_ci95"]["d_accuracy"]
    macros = [
        f"\\newcommand{{\\RevertPairs}}{{{tot_pairs:,}}}",
        f"\\newcommand{{\\RevertChanged}}{{{tot_chg}}}",
        f"\\newcommand{{\\RevertMathLoose}}{{${f3(ml['revert']['delta'] - ml['actual']['delta'])}$}}",
        f"\\newcommand{{\\RevertMathLooseCI}}{{[{f3(ci[0])}, {f3(ci[1])}]}}",
    ]
    (PAPER / "generated_revert_macros.tex").write_text("\n".join(macros) + "\n")
    print("\n".join(macros))
    print("---")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
