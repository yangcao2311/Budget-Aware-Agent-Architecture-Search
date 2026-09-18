#!/usr/bin/env python3
"""Path decomposition of the verifier-signal ablation.

For each masking level we split breakage on reference-correct pairs into the
three terminal paths the audit distinguishes: the verifier never executed, the
verifier accepted the incumbent outright, or it rejected and the subsequent
refinement produced the final answer.  This connects the ablation's total
effect to the terms of the decomposition rather than to aggregate r, b and
Delta alone.  No model calls.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
OUT = ROOT / "experiments/signal_paths_20260917.json"
PAPER = ROOT / "paper"
CONDS = [("1.0", "envelope_test"), ("0.5", "envelope_test_mask0.5_k1"),
         ("0.0", "envelope_test_mask0.0_k1")]


def load(run: str, seed: int) -> dict:
    return {json.loads(l)["task_id"]: json.loads(l)
            for l in open(EXP / run / f"results_seed{seed}.jsonl")}


def main() -> None:
    report = {"generator": "scripts/analyze_signal_paths.py",
              "family": "code/loose, GPT-4o frozen test split",
              "levels": {}}
    rows = []
    for label, tag in CONDS:
        c = Counter()
        for seed in range(3):
            base = load(f"{tag}/direct_code_loose", seed)
            wf = load(f"{tag}/verify_refine_3_code_loose", seed)
            assert base.keys() == wf.keys()
            for task in base:
                if not base[task]["success"]:
                    continue
                r = wf[task]
                ok = bool(r["success"])
                types = [s.get("type") for s in r.get("trace", [])]
                c["reference_correct"] += 1
                if "verify" not in types:
                    c["no_verification"] += 1
                    c["breakage_no_verification"] += (not ok)
                elif "refine" not in types:
                    c["accepted"] += 1
                    c["breakage_acceptance_path"] += (not ok)
                else:
                    c["rejected"] += 1
                    c["breakage_rejection_path"] += (not ok)
        total = (c["breakage_no_verification"] + c["breakage_acceptance_path"]
                 + c["breakage_rejection_path"])
        c["breakage_total"] = total
        report["levels"][label] = dict(c)
        rate = total / c["reference_correct"]
        rows.append(
            f"${label}$ & {c['reference_correct']} & "
            f"{c['rejected']} & {c['no_verification']} & "
            f"{c['breakage_acceptance_path']}/{c['accepted']} & "
            f"{c['breakage_rejection_path']}/{c['rejected']} & "
            f"{total} & " + f"{rate:.3f}".replace("0.", ".") + r" \\")
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    (PAPER / "generated_signal_path_rows.tex").write_text(
        "\n".join(rows + [r"\bottomrule"]) + "\n")
    full, none = report["levels"]["1.0"], report["levels"]["0.0"]
    m = [
        rf"\newcommand{{\SigFullAccBrk}}{{{full['breakage_acceptance_path']}/{full['accepted']}}}",
        rf"\newcommand{{\SigFullRej}}{{{full['rejected']}/{full['reference_correct']}}}",
        rf"\newcommand{{\SigNoneRej}}{{{none['rejected']}/{none['reference_correct']}}}",
        rf"\newcommand{{\SigNoneRejBrk}}{{{none['breakage_rejection_path']}/{none['rejected']}}}",
    ]
    (PAPER / "generated_signal_macros.tex").write_text("\n".join(m) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
