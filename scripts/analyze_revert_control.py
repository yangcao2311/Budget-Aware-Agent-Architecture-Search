#!/usr/bin/env python3
"""Best-so-far retention (revert-on-non-acceptance) control, computed offline.

In this workflow the verifier is the only selection signal and the loop
terminates the moment it accepts, so at most one candidate is ever accepted.
"Best verified so far" therefore has an exact meaning: if the execution
terminated without an acceptance, return the incumbent instead of the
refinement left in hand. On the assignment arms the incumbent is the stored
baseline output, whose correctness is already known, so the counterfactual
return is determined -- no model call and no intermediate answer is needed.

This is exactly the revert rule of QualityFlow-style protective acceptance.
It is NOT the unrestricted "best of all candidates" rule, which would need a
scalar score the workflow never computes and intermediate answers the logs do
not retain.

No model calls.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/envelope_test"
OUT = ROOT / "experiments/revert_control_20260917.json"

CELLS = {
    "code/tight":  ("direct_code_tight",  "incumbent_refine_code_tight"),
    "code/loose":  ("direct_code_loose",  "incumbent_refine_code_loose"),
    "math/tight":  ("cot_math_tight",     "incumbent_refine_cot_math_tight"),
    "math/loose":  ("cot_math_loose",     "incumbent_refine_cot_math_loose"),
}
B = 10_000
RNG = np.random.default_rng(20260917)


def load(run: str, seed: int) -> dict:
    return {json.loads(l)["task_id"]: json.loads(l)
            for l in open(EXP / run / f"results_seed{seed}.jsonl")}


def accepted(rec: dict) -> bool:
    """Terminated because the verifier accepted the candidate in hand."""
    if rec["status"] != "completed":
        return False
    types = [s.get("type") for s in rec.get("trace", [])]
    return bool(types) and types[-1] == "verify"


def rates(ok_b, ok_x):
    ok_b, ok_x = np.asarray(ok_b), np.asarray(ok_x)
    wrong, right = ~ok_b, ok_b
    rep = ok_x[wrong].mean() if wrong.any() else np.nan
    brk = (~ok_x[right]).mean() if right.any() else np.nan
    return rep, brk, ok_x.mean() - ok_b.mean()


def boot(tasks, ok_b, ok_w, ok_r):
    """Task-clustered paired bootstrap of the revert-minus-actual deltas."""
    tasks = np.asarray(tasks)
    uniq = np.unique(tasks)
    idx = {t: np.flatnonzero(tasks == t) for t in uniq}
    out = {"d_repair": [], "d_breakage": [], "d_accuracy": []}
    for _ in range(B):
        pick = RNG.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx[t] for t in pick])
        b, w, r = ok_b[sel], ok_w[sel], ok_r[sel]
        rw = rates(b, w)
        rr = rates(b, r)
        out["d_repair"].append(rr[0] - rw[0])
        out["d_breakage"].append(rr[1] - rw[1])
        out["d_accuracy"].append(rr[2] - rw[2])
    return {k: [round(float(np.nanpercentile(v, 2.5)), 4),
                round(float(np.nanpercentile(v, 97.5)), 4)]
            for k, v in out.items()}


def main() -> None:
    report = {"generator": "scripts/analyze_revert_control.py",
              "rule": "on termination without verifier acceptance, return the "
                      "stored incumbent instead of the refinement in hand",
              "arm": "assignment (incumbent == stored baseline output)",
              "cells": {}}
    for cell, (base, wf) in CELLS.items():
        tasks, ok_b, ok_w, ok_r = [], [], [], []
        n_acc = n_noacc = n_changed = 0
        empty_ok = True
        for seed in range(3):
            bb, ww = load(base, seed), load(wf, seed)
            assert bb.keys() == ww.keys(), cell
            for t in sorted(bb):
                b_ok = bool(bb[t]["success"])
                w_ok = bool(ww[t]["success"])
                acc = accepted(ww[t])
                r_ok = w_ok if acc else b_ok
                # sanity: an execution that ran nothing must already equal the incumbent
                if not ww[t].get("trace"):
                    empty_ok &= (w_ok == b_ok)
                n_acc += acc
                n_noacc += (not acc)
                n_changed += (r_ok != w_ok)
                tasks.append(t); ok_b.append(b_ok); ok_w.append(w_ok); ok_r.append(r_ok)
        ok_b, ok_w, ok_r = map(np.asarray, (ok_b, ok_w, ok_r))
        rw, rr = rates(ok_b, ok_w), rates(ok_b, ok_r)
        report["cells"][cell] = {
            "pairs": len(ok_b),
            "reference_correct": int(ok_b.sum()),
            "reference_wrong": int((~ok_b).sum()),
            "accepted": n_acc, "no_acceptance": n_noacc,
            "returns_changed_by_revert": n_changed,
            "empty_trace_matches_incumbent": bool(empty_ok),
            "actual":  {"repair": round(float(rw[0]), 4),
                        "breakage": round(float(rw[1]), 4),
                        "delta": round(float(rw[2]), 4),
                        "breakage_count": int((~ok_w[ok_b]).sum()),
                        "repair_count": int(ok_w[~ok_b].sum())},
            "revert":  {"repair": round(float(rr[0]), 4),
                        "breakage": round(float(rr[1]), 4),
                        "delta": round(float(rr[2]), 4),
                        "breakage_count": int((~ok_r[ok_b]).sum()),
                        "repair_count": int(ok_r[~ok_b].sum())},
            "revert_minus_actual_ci95": boot(tasks, ok_b, ok_w, ok_r),
        }
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
