#!/usr/bin/env python3
"""Freeze the exact request roster for the factorial budget ablation
(call cap x output-token cap), before any real GLM call.

Domain: code. Verifier: visible tests fully masked (mask fraction 0.0,
via the existing, unmodified scripts/run_envelope.py::mask_tests -- "keep
0 asserts" -> feedback_tests = "", the same mask0.0 semantics already
used by the paper's Fig.3 envelope study). Workflow: the standard
verify-refine suffix, self-drafting arm (hbws.dsl.wf_incumbent_refine:
prompt_id "solve_direct", temperature 0.0, then verify+refine, max_iter
3). Held-out grading_tests are never touched by masking or by the
runner -- only used later, offline, to compute each iteration's real
correctness.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from hbws.data import load_split
from hbws.dsl import wf_incumbent_refine
from hbws.prompts import render
from run_envelope import mask_tests  # existing, unmodified

MASK_FRAC = 0.0


def build_roster() -> dict:
    tasks = [mask_tests(t, MASK_FRAC) for t in load_split("code", "test")[:150]]
    assert all(t["feedback_tests"] == "" for t in tasks), (
        "mask fraction 0.0 must yield fully empty feedback_tests")

    wf = wf_incumbent_refine()
    g_node = wf["nodes"][0]
    assert g_node["id"] == "g" and g_node["prompt_id"] == "solve_direct"

    roster = {}
    for t in tasks:
        content = render(g_node["prompt_id"], t["family"], t["prompt"])
        roster[t["id"]] = {
            "first_messages": [{"role": "user", "content": content}],
            "temperature": g_node["params"]["temperature"],
            "max_tokens": g_node["params"]["max_output_tokens"],
            "feedback_tests_masked": t["feedback_tests"],
            "grading_tests": t["grading_tests"],
        }
    return {"mask_frac": MASK_FRAC, "n_tasks": len(roster), "seeds": [0, 1, 2],
            "family": "code", "roster": roster}


def main() -> None:
    out_path = ROOT / "experiments/factorial_budget_ablation_20260917_roster.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_roster()
    if out_path.exists():
        old = json.loads(out_path.read_text())
        if old != payload:
            raise RuntimeError("Refusing to alter an already-frozen roster")
        print("roster already frozen and unchanged:", out_path)
        return
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("frozen roster:", out_path)
    print("roster sha256:", hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest())


if __name__ == "__main__":
    main()
