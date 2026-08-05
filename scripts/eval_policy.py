#!/usr/bin/env python
"""Evaluate a discovered policy on frozen splits (Part II confirmation).

Selection rule (preregistered): the policy is chosen ONCE on the validation
split by j_mean, then run on test / ood / the unseen $0.15 tier. Search code
never sees these splits.

  python scripts/eval_policy.py --search-run A_hbws_code_s0 --family code \
      --select-on val --splits test ood --tiers tight unseen loose
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hbws.data import load_split
from hbws.hbws import EXP_DIR
from hbws.ledger import BUDGET_TIERS
from hbws.protocol import evaluate


def load_archive(run_name: str) -> list[dict]:
    p = EXP_DIR / "search" / run_name / "search_result.json"
    if not p.exists():
        sys.exit(f"no search result at {p}")
    return json.load(open(p))["archive"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-run", required=True)
    ap.add_argument("--family", required=True, choices=["code", "math"])
    ap.add_argument("--select-on", default="val", choices=["val", "none"])
    ap.add_argument("--top-k", type=int, default=3, help="archive members to screen")
    ap.add_argument("--splits", nargs="+", default=["test"])
    ap.add_argument("--tiers", nargs="+", default=["tight", "unseen", "loose"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    archive = load_archive(args.search_run)[:args.top_k]
    if not archive:
        sys.exit("empty archive")

    # -- selection on validation (one shot, preregistered) ------------------
    chosen = archive[0]
    if args.select_on == "val":
        val = load_split(args.family, "val")
        best, best_score = None, -1.0
        for cand in archive:
            scores = []
            for tier in ["tight", "loose"]:      # never 'unseen'
                s = evaluate(cand["wf"], val, BUDGET_TIERS[tier],
                             run_name=f"select/{args.search_run}/c{cand['cid']}_{tier}",
                             seed=0, use_cache=True, workers=args.workers)
                scores.append(s["success_rate"])
            score = 0.5 * (sum(scores) / len(scores)) + 0.5 * min(scores)
            print(f"  val screen cid={cand['cid']}: {[round(x,3) for x in scores]} "
                  f"score={score:.3f}")
            if score > best_score:
                best, best_score = cand, score
        chosen = best
    print(f"\nselected cid={chosen['cid']} (origin={chosen['origin']}, "
          f"budget_predicates={'budget_' in json.dumps(chosen['wf'])})")

    out_rows = []
    for split in args.splits:
        tasks = load_split(args.family, split)
        for tier in args.tiers:
            for seed in args.seeds:
                s = evaluate(chosen["wf"], tasks, BUDGET_TIERS[tier],
                             run_name=f"final/{args.search_run}/c{chosen['cid']}_{split}_{tier}",
                             seed=seed, use_cache=False, workers=args.workers)
                s.update({"split": split, "tier": tier, "cid": chosen["cid"],
                          "search_run": args.search_run})
                out_rows.append(s)
                print(f"  {split:5s} {tier:6s} seed{seed} succ={s['success_rate']:.3f} "
                      f"${s['usd_per_task']:.4f} rr={s.get('reserve_rejected',0)}")

    dest = EXP_DIR / "final" / f"{args.search_run}_policy_eval.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "a") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    json.dump(chosen, open(dest.with_name(f"{args.search_run}_chosen.json"), "w"), indent=2)
    print(f"\nwritten to {dest}")


if __name__ == "__main__":
    main()
