#!/usr/bin/env python
"""Part II driver: HBWS (M8) vs Static Evolution Search (M7) vs Random (M6).

  python scripts/run_search.py --family code --mode hbws --cap 100 --seed 0

Protocol A: static gets a full cap per tier (run twice, once per tier).
Protocol B: every method gets the same total cap (checkpointed from A).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hbws.data import load_split
from hbws.hbws import hbws_search


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["code", "math"])
    ap.add_argument("--mode", required=True, choices=["hbws", "static"])
    ap.add_argument("--cap", type=float, required=True, help="search $ cap")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tiers", nargs="+", default=["tight", "loose"])
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    tasks = load_split(args.family, "dev")
    res = hbws_search(args.family, tasks, cap_usd=args.cap,
                      budget_contingent=(args.mode == "hbws"),
                      seed=args.seed, workers=args.workers,
                      tiers=tuple(args.tiers), run_name=args.run_name)
    print(f"\ntop archive J_CB: "
          f"{[round(a['j_cb'], 3) for a in res['archive'][:5]]}")


if __name__ == "__main__":
    main()
