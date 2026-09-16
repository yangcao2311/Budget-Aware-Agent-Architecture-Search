#!/usr/bin/env python3
"""Freeze the exact request roster for the GLM-4-Flash code-domain
budget-parity follow-up (assign vs. same-policy, prior-tight and
prior-loose cohorts, both keyed to their existing frozen direct_code_*
references), before any real GLM call.

This is a code-domain replication of the already-completed math
budget-parity follow-up -- NOT a cross-domain replication claim; the
domain is what changed, run through the identical design.

Two arms only, same as the math version:
  - assign:       hbws.dsl.wf_assign_refine() -- the reference solution is
                  injected verbatim into task["_assign_solution"], zero
                  cost, then the existing code verifier (task's own
                  visible feedback_tests, run in the sandbox) + refine.
  - same_policy:  hbws.dsl.wf_incumbent_refine() -- an independent
                  first-generate call (prompt_id "solve_direct",
                  temperature 0.0, max_output_tokens 1024 -- the code
                  family's own "direct" first-draft policy, matching the
                  stored reference's own generation), then the SAME
                  verify+refine suffix.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hbws.data import load_split
from hbws.dsl import wf_assign_refine, wf_incumbent_refine
from hbws.prompts import render

SEEDS = (0, 1, 2)
COHORTS = ("tight", "loose")  # cohort names inherited from the frozen
                               # reference directories; NOT a budget-tier
                               # manipulation in this follow-up (see
                               # run_glm_budget_parity_code.py)
REF_PREFIX = "glm4flash_repaired_matrix_clean_20260910_envelope_test"


def load_reference(cohort: str, seed: int) -> dict[str, str]:
    path = ROOT / "experiments" / REF_PREFIX / f"direct_code_{cohort}" / f"results_seed{seed}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"missing frozen GLM code reference: {path} -- do not "
            "regenerate; obtain the original file")
    out = {}
    for line in path.read_text().splitlines():
        if not line:
            continue
        row = json.loads(line)
        out[row["task_id"]] = row["solution"] or ""
    return out


def build_roster() -> dict:
    tasks = load_split("code", "test")[:150]

    same_policy_wf = wf_incumbent_refine()
    g_node = same_policy_wf["nodes"][0]
    assert g_node["id"] == "g" and g_node["prompt_id"] == "solve_direct"

    assign_wf = wf_assign_refine()
    a_node = assign_wf["nodes"][0]
    assert a_node["id"] == "a" and a_node["type"] == "assign"

    roster = {"cohorts": {}}
    for cohort in COHORTS:
        by_seed = {}
        for seed in SEEDS:
            reference = load_reference(cohort, seed)
            missing = [t["id"] for t in tasks if t["id"] not in reference]
            if missing:
                raise RuntimeError(
                    f"cohort={cohort} seed={seed}: reference missing "
                    f"{len(missing)} task ids, e.g. {missing[:5]}")
            entries = {}
            for t in tasks:
                same_policy_content = render(
                    g_node["prompt_id"], t["family"], t["prompt"])
                entries[t["id"]] = {
                    "assign_solution_verbatim": reference[t["id"]],
                    "same_policy_first_messages": [
                        {"role": "user", "content": same_policy_content}],
                    "same_policy_temperature": g_node["params"]["temperature"],
                    "same_policy_max_tokens": g_node["params"]["max_output_tokens"],
                }
            by_seed[str(seed)] = entries
        roster["cohorts"][cohort] = by_seed
    roster["n_tasks"] = len(tasks)
    roster["seeds"] = list(SEEDS)
    roster["reference_prefix"] = REF_PREFIX
    roster["family"] = "code"
    return roster


def main() -> None:
    out_path = ROOT / "experiments/glm_budget_parity_code_20260917_roster.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_roster()
    if out_path.exists():
        old = json.loads(out_path.read_text())
        if old != payload:
            raise RuntimeError(
                "Refusing to alter an already-frozen code budget-parity roster")
        print("roster already frozen and unchanged:", out_path)
        return
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    roster_sha = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()
    print("frozen roster:", out_path)
    print("roster sha256:", roster_sha)


if __name__ == "__main__":
    main()
