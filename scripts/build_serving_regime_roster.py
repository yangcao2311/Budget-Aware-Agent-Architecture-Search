#!/usr/bin/env python3
"""Freeze the exact first-generation request roster for the Qwen
serving-regime control (reviewer B1 follow-up), before any condition runs.

The roster is the literal {role, content} messages hbws.runner would send
for wf_incumbent_refine_cot's first node ("g", prompt_id "solve_cot",
temperature 0.0, max_output_tokens 1536) for each of the 150 frozen
math/test tasks -- built directly from the existing, unmodified
hbws.prompts.render and hbws.data.load_split, not hand-copied. Message
content does not depend on seed (only the request's `seed` field does), so
the roster is keyed by task_id; seeds 0/1/2 are recorded separately.

Every condition (S/B/P/S2) must send byte-identical request bodies for the
same (task_id, seed) -- only the server base_url may differ. This script's
output is the artifact that lets that be checked mechanically rather than
asserted.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hbws.data import load_split
from hbws.dsl import wf_incumbent_refine_cot
from hbws.prompts import render

SEEDS = (0, 1, 2)


def build_roster() -> dict:
    wf = wf_incumbent_refine_cot()
    first_node = wf["nodes"][0]
    assert first_node["id"] == "g" and first_node["prompt_id"] == "solve_cot"
    params = first_node["params"]
    tasks = load_split("math", "test")[:150]
    roster = {}
    for t in tasks:
        content = render(first_node["prompt_id"], t["family"], t["prompt"])
        roster[t["id"]] = {
            "messages": [{"role": "user", "content": content}],
            "temperature": params["temperature"],
            "max_tokens": params["max_output_tokens"],
        }
    return {
        "workflow_first_node": first_node,
        "seeds": list(SEEDS),
        "n_tasks": len(roster),
        "roster": roster,
    }


def main() -> None:
    out_path = ROOT / "experiments/qwen_serving_regime_20260915_roster.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_roster()
    if out_path.exists():
        old = json.loads(out_path.read_text())
        if old != payload:
            raise RuntimeError(
                "Refusing to alter an already-frozen serving-regime roster")
        print("roster already frozen and unchanged:", out_path)
        return
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    roster_sha = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()
    print("frozen roster:", out_path)
    print("roster sha256:", roster_sha)


if __name__ == "__main__":
    main()
