#!/usr/bin/env python
"""Provenance arm C: a genuinely different drafting policy.

The three-arm provenance comparison the paper says it did not run needs:
  A. explicit reuse of the reference's stored output      (I = B, structural)
  B. regeneration under an IDENTICAL policy                (recovered free from
     existing seeds -- scripts/provenance_from_logs.py)
  C. regeneration under a DIFFERENT drafting policy         (this script)

Arm C mirrors the actual first-draft node of the vanilla verify_refine_3
workflow (hbws/dsl.py: prompt_id="solve_cot", temperature=0.7,
max_output_tokens=1536) as a single generate call -- no verify, no refine --
on the same frozen test tasks already used for the code/loose and math/loose
baselines. Grading is the paper's own pipeline (hbws.protocol.grade), so the
result is directly comparable to the baseline-correct partition already used
everywhere else, and to arm B's leak numbers.

use_cache=False throughout: this must not share cache entries with any other
arm (protocol §5.3).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hbws.data import load_split
from hbws.ledger import BUDGET_TIERS
from hbws.protocol import evaluate

WF_COT_HOT = {
    "nodes": [{"id": "g", "type": "generate", "prompt_id": "solve_cot",
               "params": {"temperature": 0.7, "max_output_tokens": 1536}}],
    "edges": [{"from": "g", "to": "END"}],
}

FAMILIES = ["code", "math"]
N_TASKS = 150
SEEDS = [0, 1, 2]


def main():
    rows = []
    for fam in FAMILIES:
        tasks = load_split(fam, "test")[:N_TASKS]
        for seed in SEEDS:
            run = f"provenance_arm/cot_hot_{fam}_loose"
            s = evaluate(WF_COT_HOT, tasks, BUDGET_TIERS["loose"],
                         run_name=run, seed=seed, use_cache=False, workers=8)
            rows.append({**s, "family": fam})
            print(f"{fam:5s} seed={seed} succ={s['success_rate']:.3f} "
                  f"$/task={s['usd_per_task']:.5f} total=${sum(s['usd_per_task'] for _ in [0]) * s['n']:.3f} "
                  f"rr={s.get('reserve_rejected', 0)}")

    out = Path(__file__).resolve().parent.parent / "experiments" / "provenance_arm_summary.jsonl"
    with open(out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    total = sum(r["usd_per_task"] * r["n"] for r in rows)
    print(f"\ntotal spend this run: ${total:.3f}")
    print("summary appended to", out)


if __name__ == "__main__":
    main()
