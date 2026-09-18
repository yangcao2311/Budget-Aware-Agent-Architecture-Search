#!/usr/bin/env python3
"""Observed terminal-path decomposition of the GPT-4o budget result.

For each paired baseline--workflow execution we record whether the workflow
terminated early because the reservation ledger refused a planned call, how
many refinements had already executed at that point, whether the baseline was
correct, and whether the state the workflow returned was correct.

This is descriptive. It cannot say what a refused call would have produced had
it run, and it cannot reconstruct intermediate answers that refinement already
overwrote. Breakage under early termination is reported as "the execution
terminated before a planned call and returned an incorrect available state",
never as an error the budget caused.

No model calls.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/envelope_test"
OUT = ROOT / "experiments/blocked_paths_20260917.json"
PAPER = ROOT / "paper"
CELLS = {"code/tight": ("direct_code_tight", "verify_refine_3_code_tight"),
         "code/loose": ("direct_code_loose", "verify_refine_3_code_loose")}


def load(run: str, seed: int) -> dict:
    return {json.loads(l)["task_id"]: json.loads(l)
            for l in open(EXP / run / f"results_seed{seed}.jsonl")}


def main() -> None:
    report = {"generator": "scripts/analyze_blocked_paths.py",
              "status_note": "early termination == reservation refused a planned call",
              "cells": {}}
    for cell, (base, wf) in CELLS.items():
        t, b = Counter(), Counter()
        for seed in range(3):
            bb, ww = load(base, seed), load(wf, seed)
            assert bb.keys() == ww.keys()
            for task in bb:
                r = ww[task]
                early = r["status"] != "completed"
                types = [s.get("type") for s in r.get("trace", [])]
                n_ref = types.count("refine")
                ok_b, ok_w = bool(bb[task]["success"]), bool(r["success"])
                t["pairs"] += 1
                t["early" if early else "completed"] += 1
                if early:
                    t[f"early_refinements_{min(n_ref, 3)}"] += 1
                    if ok_b:
                        t["early_baseline_correct"] += 1
                        t["early_bc_returned_correct" if ok_w
                          else "early_bc_returned_wrong"] += 1
                if ok_b and not ok_w:
                    b["total"] += 1
                    if early:
                        b["early_before_any_refinement" if n_ref == 0
                          else "early_after_refinement"] += 1
                    else:
                        b["completed_with_refinement" if n_ref
                          else "completed_without_refinement"] += 1
        report["cells"][cell] = {"terminal_states": dict(t), "breakage": dict(b)}

    OUT.write_text(json.dumps(report, indent=2) + "\n")
    ct = report["cells"]["code/tight"]
    m = [
        rf"\newcommand{{\BlockedEarly}}{{{ct['terminal_states']['early']}}}",
        rf"\newcommand{{\BlockedBaseCorrect}}{{{ct['terminal_states']['early_baseline_correct']}}}",
        rf"\newcommand{{\BlockedBCWrong}}{{{ct['terminal_states']['early_bc_returned_wrong']}}}",
        rf"\newcommand{{\BlockedBreakTotal}}{{{ct['breakage']['total']}}}",
        rf"\newcommand{{\BlockedBreakEarly}}{{{ct['breakage']['early_after_refinement']}}}",
        rf"\newcommand{{\BlockedZeroRefine}}{{{ct['terminal_states'].get('early_refinements_0', 0)}}}",
    ]
    (PAPER / "generated_blocked_macros.tex").write_text("\n".join(m) + "\n")
    rows = []
    for cell, v in report["cells"].items():
        ts, bk = v["terminal_states"], v["breakage"]
        fam, tier = cell.split("/")
        rows.append(
            f"{fam}/\\texttt{{{tier}}}"
            + f" & {ts['pairs']} & {ts['early']} & "
            f"{ts.get('early_refinements_0', 0)} & {ts.get('early_baseline_correct', 0)} & "
            f"{ts.get('early_bc_returned_wrong', 0)} & {bk['total']} & "
            f"{bk.get('early_after_refinement', 0)} & "
            f"{bk.get('completed_with_refinement', 0)} & "
            f"{bk.get('completed_without_refinement', 0)} \\\\")
    (PAPER / "generated_blocked_rows.tex").write_text(
        "\n".join(rows + [r"\bottomrule"]) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
