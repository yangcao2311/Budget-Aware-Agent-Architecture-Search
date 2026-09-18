#!/usr/bin/env python3
"""Independent recomputation of the code-domain budget-parity replication.

Mirrors scripts/analyze_provenance_followup.py Part B on the two frozen code
reference cohorts. Verifies the frozen reference files by SHA-256 before
recomputing, so the pairing base is provably the one the original matrix used.
Intervals are post-hoc descriptive and task-clustered; the counterfactual
column counts positions that the ORIGINAL caps would have blocked, which is a
ledger replay, not an executed arm. No model calls.
"""
from __future__ import annotations

import hashlib
import json
import re as _re  # _tighten
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "experiments/glm4flash_repaired_matrix_clean_20260910_envelope_test"
OUT = ROOT / "experiments/code_budget_parity_20260917_analysis.json"
PAPER = ROOT / "paper"
BOOT, SEED = 10_000, 20260918


def rows(path: Path) -> dict:
    return {json.loads(l)["task_id"]: json.loads(l) for l in open(path)}


def estimates(tasks):
    flat = [x for t in tasks for x in t]
    right = [x for x in flat if x[0]]
    wrong = [x for x in flat if not x[0]]
    return ((sum(1 for x in right if not x[1]) - sum(1 for x in right if not x[2]))
            / len(right),
            (sum(1 for x in wrong if x[1]) - sum(1 for x in wrong if x[2]))
            / len(wrong),
            sum(x[1] - x[2] for x in flat) / len(flat))


def main() -> None:
    pkg = Path(sys.argv[1])
    exp = pkg / "experiments"
    base = exp / "glm_budget_parity_code_20260917"
    manifest = json.loads(
        (exp / "glm_budget_parity_code_20260917_manifest.json").read_text())
    server = json.loads(
        (exp / "glm_budget_parity_code_20260917_analysis.json").read_text())

    integrity = {}
    for rel, want in manifest["reference_sha256"].items():
        got = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        integrity[rel] = (got == want)
    assert all(integrity.values()), "frozen reference file mismatch"

    report = {"generator": "scripts/analyze_code_parity.py",
              "reference_files_verified": len(integrity),
              "interval_status": "post-hoc descriptive, task-clustered",
              "cohorts": {}}
    tex = []
    for tier in ("tight", "loose"):
        per_task, status = {}, set()
        for seed in range(3):
            ref = rows(REF / f"direct_code_{tier}" / f"results_seed{seed}.jsonl")
            a = rows(base / f"{tier}_assign" / f"results_seed{seed}.jsonl")
            s = rows(base / f"{tier}_same_policy" / f"results_seed{seed}.jsonl")
            assert ref.keys() == a.keys() == s.keys()
            for t in ref:
                status.update({a[t]["status"], s[t]["status"]})
                # the code verifier is a local sandbox test run and records no
                # explicit decision; with zero gating a trailing `refine` is
                # exactly a first rejection, so its absence is acceptance.
                acc = {k: "refine" not in [x.get("type") for x in r.get("trace", [])]
                       for k, r in (("a", a[t]), ("s", s[t]))}
                per_task.setdefault(t, []).append(
                    (bool(ref[t]["success"]), bool(a[t]["success"]),
                     bool(s[t]["success"]), acc["a"], acc["s"]))
        assert status == {"completed"}, status
        tasks = list(per_task.values())
        flat = [x for t in tasks for x in t]
        right = [x for x in flat if x[0]]
        wrong = [x for x in flat if not x[0]]
        point = estimates(tasks)
        rng = random.Random(SEED)
        draws = [[], [], []]
        for _ in range(BOOT):
            smp = [tasks[rng.randrange(len(tasks))] for _ in range(len(tasks))]
            for i, v in enumerate(estimates(smp)):
                draws[i].append(v)
        ci = [[sorted(d)[int(.025 * BOOT)], sorted(d)[int(.975 * BOOT)]]
              for d in draws]
        arms = server["cohorts"][tier]["arms"]
        cf = {k: arms[k]["counterfactual_original_caps_blocked"]["n_blocked_positions"]
              for k in ("assign", "same_policy")}
        report["cohorts"][tier] = {
            "pairs": len(flat), "reference_correct": len(right),
            "reference_wrong": len(wrong),
            "breakage": {"assign": sum(1 for x in right if not x[1]),
                         "same_policy": sum(1 for x in right if not x[2])},
            "acceptance_path_breakage": {
                "assign": sum(1 for x in right if not x[1] and x[3]),
                "same_policy": sum(1 for x in right if not x[2] and x[4])},
            "repair": {"assign": sum(1 for x in wrong if x[1]),
                       "same_policy": sum(1 for x in wrong if x[2])},
            "accuracy": {"assign": sum(x[1] for x in flat) / len(flat),
                         "same_policy": sum(x[2] for x in flat) / len(flat)},
            "counterfactual_original_caps_blocked": cf,
            "paired_differences_assign_minus_same_policy": {
                "breakage_rate": {"point": point[0], "ci95": ci[0]},
                "repair_rate": {"point": point[1], "ci95": ci[1]},
                "accuracy": {"point": point[2], "ci95": ci[2]}},
        }
        b = report["cohorts"][tier]

        def sgn(v):
            out = f"{abs(v):.3f}"
            return ("+" if v >= 0 else "-") + (out[1:] if out.startswith("0") else out)

        def ival(e):
            return f"${sgn(e['point'])}$ [{sgn(e['ci95'][0])}, {sgn(e['ci95'][1])}]"

        d = b["paired_differences_assign_minus_same_policy"]
        tex.append(
            f"code/\\texttt{{{tier}}} & {b['reference_correct']} & "
            f"{b['breakage']['assign']} & {b['breakage']['same_policy']} & "
            f"{b['acceptance_path_breakage']['assign']} & "
            f"{b['acceptance_path_breakage']['same_policy']} & "
            f"{ival(d['breakage_rate'])} & {ival(d['accuracy'])} \\\\")

    OUT.write_text(json.dumps(report, indent=2) + "\n")
    (PAPER / "generated_code_parity_rows.tex").write_text(
        _re.sub(r"\[(\S+?), (\S+?)\]", r"[\1,\2]",
                "\n".join(tex + [r"\bottomrule"])) + "\n")
    t, l = report["cohorts"]["tight"], report["cohorts"]["loose"]
    m = [
        rf"\newcommand{{\CodeParityBrkTight}}{{{t['breakage']['assign']}/{t['reference_correct']}}}",
        rf"\newcommand{{\CodeParityBrkTightSame}}{{{t['breakage']['same_policy']}/{t['reference_correct']}}}",
        rf"\newcommand{{\CodeParityBrkLoose}}{{{l['breakage']['assign']}/{l['reference_correct']}}}",
        rf"\newcommand{{\CodeParityBrkLooseSame}}{{{l['breakage']['same_policy']}/{l['reference_correct']}}}",
        rf"\newcommand{{\CodeCfBlockedTight}}{{{t['counterfactual_original_caps_blocked']['assign']}}}",
        rf"\newcommand{{\CodeCfBlockedTightSame}}{{{t['counterfactual_original_caps_blocked']['same_policy']}}}",
    ]
    (PAPER / "generated_code_parity_macros.tex").write_text("\n".join(m) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
