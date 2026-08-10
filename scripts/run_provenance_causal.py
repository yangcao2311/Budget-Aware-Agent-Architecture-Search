#!/usr/bin/env python
"""The three-arm provenance causal test the Limitations section asks for.

Three arms, identical downstream (verify, then refine on rejection, up to 3
iterations), differing ONLY in where the first draft comes from:

  arm1  explicit reuse    -- wf_assign_refine(): an `assign` node injects the
                             reference's stored solution directly (zero cost,
                             no LLM call). I = B by construction.
  arm2  same-policy regen -- wf_incumbent_refine[_cot](), run fresh with
                             use_cache=False at the SAME seed as the
                             reference: same prompt, same temperature, same
                             seed, but a genuinely independent API call
                             instead of a cache hit.
  arm3  different-policy  -- verify_refine_3 (_wf_verify_refine(3)): the
                             vanilla workflow's actual first-draft policy
                             (solve_cot, temperature 0.7), fresh, cache off.

All three share the SAME frozen test tasks and the SAME baseline-correct
population (Table 1's convention: baseline correct in every one of its own 3
seeds). Because all three now run the full verify+refine loop, breakage is
observed directly for each arm -- no leak x accept-rate estimation needed.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hbws.data import load_split
from hbws.dsl import wf_assign_refine, wf_incumbent_refine, wf_incumbent_refine_cot
from hbws.dsl import _wf_verify_refine
from hbws.ledger import BUDGET_TIERS
from hbws.protocol import evaluate

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"

ARM2_WF = {"code": wf_incumbent_refine, "math": wf_incumbent_refine_cot}
BASELINE_DIR = {"code": "envelope_test/direct_code_loose",
                "math": "envelope_test/cot_math_loose"}


def load_baseline_by_seed(fam: str) -> dict:
    """seed -> {task_id: (solution_text, correct)}"""
    out = defaultdict(dict)
    for s in (0, 1, 2):
        p = EXP / BASELINE_DIR[fam] / f"results_seed{s}.jsonl"
        for r in map(json.loads, open(p)):
            ok = bool(r.get("success_symbolic", r["success"]))
            out[s][r["task_id"]] = (r.get("solution") or "", ok)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", default=["code", "math"])
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--tag", default="provenance_causal")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    caps = BUDGET_TIERS["loose"]
    rows = []
    for fam in args.families:
        tasks = load_split(fam, "test")[:args.n]
        by_id = {t["id"]: t for t in tasks}
        base = load_baseline_by_seed(fam)

        for seed in args.seeds:
            ref = base[seed]
            # arm 1: explicit reuse -- inject each task's reference solution
            tasks_assign = [
                {**t, "_assign_solution": ref[t["id"]][0]} for t in tasks
                if t["id"] in ref
            ]
            s1 = evaluate(wf_assign_refine(), tasks_assign, caps,
                          run_name=f"{args.tag}/arm1_assign_{fam}_loose",
                          seed=seed, use_cache=False, workers=args.workers)
            rows.append({**s1, "family": fam, "arm": "arm1_assign"})
            print(f"[arm1 assign  ] {fam:5s} seed={seed} n={s1['n']:3d} "
                  f"succ={s1['success_rate']:.3f} $/task={s1['usd_per_task']:.5f} "
                  f"errors={s1.get('errors', 0)}")

            # arm 2: same-policy regeneration, cache off, same seed
            s2 = evaluate(ARM2_WF[fam](), tasks, caps,
                          run_name=f"{args.tag}/arm2_samepolicy_{fam}_loose",
                          seed=seed, use_cache=False, workers=args.workers)
            rows.append({**s2, "family": fam, "arm": "arm2_samepolicy"})
            print(f"[arm2 same-pol] {fam:5s} seed={seed} n={s2['n']:3d} "
                  f"succ={s2['success_rate']:.3f} $/task={s2['usd_per_task']:.5f} "
                  f"errors={s2.get('errors', 0)}")

            # arm 3: different-policy regeneration (vanilla), cache off
            s3 = evaluate(_wf_verify_refine(3), tasks, caps,
                          run_name=f"{args.tag}/arm3_diffpolicy_{fam}_loose",
                          seed=seed, use_cache=False, workers=args.workers)
            rows.append({**s3, "family": fam, "arm": "arm3_diffpolicy"})
            print(f"[arm3 diff-pol] {fam:5s} seed={seed} n={s3['n']:3d} "
                  f"succ={s3['success_rate']:.3f} $/task={s3['usd_per_task']:.5f} "
                  f"errors={s3.get('errors', 0)}")

    out = EXP / f"{args.tag}_summary.jsonl"
    with open(out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    total = sum(r["usd_per_task"] * r["n"] for r in rows)
    print(f"\ntotal spend this run: ${total:.3f}")
    print("summary appended to", out)


if __name__ == "__main__":
    main()
