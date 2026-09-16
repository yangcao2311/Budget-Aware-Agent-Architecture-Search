#!/usr/bin/env python3
"""Freeze the exact request roster for the GLM-4-Flash budget-parity
follow-up (assign vs. same-policy, math/tight and math/loose), before any
real GLM call.

Two arms only (no different-policy arm), two cells (tight, loose), each
150 tasks x 3 seeds:
  - assign:       hbws.dsl.wf_assign_refine() -- the reference solution is
                  injected verbatim into task["_assign_solution"] (no
                  reformatting), zero-cost first step, then verify+refine.
  - same_policy:  hbws.dsl.wf_incumbent_refine_cot() -- an independent
                  first-generate call (same prompt/temp/seed as the
                  reference), then the SAME verify+refine suffix.

Both workflows are the repo's existing, unmodified ones (same prompts,
verifier, refinement graph, temperature, per-call output caps as every
other campaign that has used them).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hbws.data import load_split
from hbws.dsl import wf_assign_refine, wf_incumbent_refine_cot
from hbws.prompts import render

SEEDS = (0, 1, 2)
TIERS = ("tight", "loose")
REF_PREFIX = "glm4flash_repaired_matrix_clean_20260910_envelope_test"


def load_reference(tier: str, seed: int) -> dict[str, str]:
    path = ROOT / "experiments" / REF_PREFIX / f"cot_math_{tier}" / f"results_seed{seed}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"missing frozen GLM reference: {path} -- do not regenerate; "
            "obtain the original file")
    out = {}
    for line in path.read_text().splitlines():
        if not line:
            continue
        row = json.loads(line)
        out[row["task_id"]] = row["solution"] or ""
    return out


def build_roster() -> dict:
    tasks = load_split("math", "test")[:150]

    same_policy_wf = wf_incumbent_refine_cot()
    g_node = same_policy_wf["nodes"][0]
    assert g_node["id"] == "g" and g_node["prompt_id"] == "solve_cot"

    assign_wf = wf_assign_refine()
    a_node = assign_wf["nodes"][0]
    assert a_node["id"] == "a" and a_node["type"] == "assign"

    roster = {"tiers": {}}
    for tier in TIERS:
        by_seed = {}
        for seed in SEEDS:
            reference = load_reference(tier, seed)
            missing = [t["id"] for t in tasks if t["id"] not in reference]
            if missing:
                raise RuntimeError(
                    f"tier={tier} seed={seed}: reference missing "
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
        roster["tiers"][tier] = by_seed
    roster["n_tasks"] = len(tasks)
    roster["seeds"] = list(SEEDS)
    roster["reference_prefix"] = REF_PREFIX
    return roster


def main() -> None:
    out_path = ROOT / "experiments/glm_budget_parity_20260916_roster.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_roster()
    if out_path.exists():
        old = json.loads(out_path.read_text())
        if old != payload:
            raise RuntimeError(
                "Refusing to alter an already-frozen budget-parity roster")
        print("roster already frozen and unchanged:", out_path)
        return
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    roster_sha = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()
    print("frozen roster:", out_path)
    print("roster sha256:", roster_sha)


if __name__ == "__main__":
    main()
