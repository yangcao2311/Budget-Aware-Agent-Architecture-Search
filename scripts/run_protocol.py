#!/usr/bin/env python
"""Protocol A / B drivers (方案 v4.0 §5.4).

Protocol A (deployment capability, deliberately biased toward static):
    HBWS gets cap S searching both tiers jointly;
    Static gets a FULL cap S per tier, i.e. 2S total.

Protocol B (design efficiency, matched total cost):
    every method gets total cap S. Static's B result is read off its A run's
    anytime curve at S/2 per tier — no extra spend.

    python scripts/run_protocol.py --family code --seed 0 --cap 100
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hbws.data import load_split
from hbws.hbws import EXP_DIR, hbws_search


def protocol_b_checkpoint(run_name: str, cap: float) -> dict | None:
    """Best archive J_CB reached by `run_name` once it had spent `cap`."""
    p = EXP_DIR / "search" / run_name / "search_result.json"
    if not p.exists():
        return None
    res = json.load(open(p))
    upto = [a for a in res["anytime"] if a["search_usd"] <= cap]
    if not upto:
        return None
    return {"run": run_name, "at_usd": cap,
            "best_j_cb": max(a["best_j_cb"] for a in upto),
            "candidates_evaluated": len(upto)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["code", "math"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cap", type=float, required=True, help="S, per-method cap")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-hbws", action="store_true")
    ap.add_argument("--skip-static", action="store_true")
    args = ap.parse_args()

    tasks = load_split(args.family, "dev")
    fam, s, S = args.family, args.seed, args.cap

    if not args.skip_hbws:
        print(f"\n### Protocol A/B — HBWS, cap ${S} (joint over tight+loose)")
        hbws_search(fam, tasks, cap_usd=S, budget_contingent=True, seed=s,
                    workers=args.workers, run_name=f"A_hbws_{fam}_s{s}")

    if not args.skip_static:
        # Static searches each tier separately with a full cap S (Protocol A).
        for tier in ["tight", "loose"]:
            print(f"\n### Protocol A — Static @ {tier}, cap ${S}")
            hbws_search(fam, tasks, cap_usd=S, budget_contingent=False, seed=s,
                        workers=args.workers, tiers=(tier,),
                        run_name=f"A_static_{tier}_{fam}_s{s}")

    print("\n### Protocol B checkpoints (matched total spend)")
    rows = [protocol_b_checkpoint(f"A_hbws_{fam}_s{s}", S)]
    for tier in ["tight", "loose"]:
        rows.append(protocol_b_checkpoint(f"A_static_{tier}_{fam}_s{s}", S / 2))
    for r in rows:
        if r:
            print(f"  {r['run']:28s} @${r['at_usd']:6.1f}  "
                  f"best_J_CB={r['best_j_cb']:.3f}  n={r['candidates_evaluated']}")
    out = EXP_DIR / "search" / f"protocolB_{fam}_s{s}.json"
    json.dump([r for r in rows if r], open(out, "w"), indent=2)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
