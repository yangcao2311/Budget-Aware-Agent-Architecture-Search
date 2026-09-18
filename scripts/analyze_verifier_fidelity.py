#!/usr/bin/env python3
"""Candidate-level verifier fidelity on the assignment arm of the clean matrix.

Where the incumbent is assigned, its correctness is the stored reference's
correctness, so the first verifier's decision can be scored against a known
label without retaining the rejected candidates. Treats "the incumbent is
wrong" as the positive class and "the verifier rejects" as the positive
prediction, so precision is the share of rejections that were warranted and
the false-positive rate is the share of correct incumbents rejected.

Reads the control-flow audit; issues no provider request.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
PAPER = ROOT / "paper"
SRC = EXP / "glm4flash_matrix_control_flow_audit_20260911.json"
OUT = EXP / "verifier_fidelity_assign_arm.json"

VERIFIER = {"code": "visible tests", "math": "gold-free self-check"}


def mcc(tp: int, fp: int, fn: int, tn: int) -> float:
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / den if den else float("nan")


def main() -> None:
    rows = json.loads(SRC.read_text())["rows"]
    report, tex = {"generator": "scripts/analyze_verifier_fidelity.py",
                   "source": SRC.name, "cells": {}}, []
    for r in sorted(rows, key=lambda x: (x["family"] != "code", x["tier"] != "tight")):
        if r["arm"] != "assign":
            continue
        n, correct = r["n"], r["reference_correct"]
        wrong = n - correct
        fp = r["reject_reference_correct"]          # rejected a correct incumbent
        tp = r["reject"] - fp                       # rejected a wrong incumbent
        tn = correct - fp                           # accepted a correct incumbent
        fn = wrong - tp                             # accepted a wrong incumbent
        assert tp + fp + tn + fn == n, (r["family"], r["tier"])
        m = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
             "reject_precision": tp / (tp + fp) if tp + fp else float("nan"),
             "fpr": fp / correct, "fnr": fn / wrong, "mcc": mcc(tp, fp, fn, tn)}
        cell = f"{r['family']}/{r['tier']}"
        report["cells"][cell] = m
        tex.append(
            f"{r['family']}/\\texttt{{{r['tier']}}} & {VERIFIER[r['family']]} & "
            f"{tp} & {fp} & {fn} & {tn} & "
            + " & ".join(f"{m[k]:.3f}".replace("0.", ".")
                         for k in ("reject_precision", "fpr", "fnr", "mcc"))
            + r" \\")
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    (PAPER / "generated_verifier_fidelity_rows.tex").write_text(
        "\n".join(tex + [r"\bottomrule"]) + "\n")
    c = report["cells"]
    (PAPER / "generated_verifier_fidelity_macros.tex").write_text("\n".join([
        rf"\newcommand{{\VerFprCodeTight}}{{{c['code/tight']['fpr']:.3f}".replace("0.", ".") + "}",
        rf"\newcommand{{\VerFprMathTight}}{{{c['math/tight']['fpr']:.3f}".replace("0.", ".") + "}",
        rf"\newcommand{{\VerFprMathLoose}}{{{c['math/loose']['fpr']:.3f}".replace("0.", ".") + "}",
    ]) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
