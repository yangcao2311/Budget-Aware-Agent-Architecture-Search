#!/usr/bin/env python
"""Run a template workflow (baseline) on a data split.

Usage:
  python scripts/run_baseline.py --template direct --family code --split dev --n 10
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hbws.data import load_split
from hbws.dsl import TEMPLATES
from hbws.ledger import BudgetCaps
from hbws.protocol import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True, choices=sorted(TEMPLATES))
    ap.add_argument("--family", required=True, choices=["code", "math"])
    ap.add_argument("--split", default="dev", choices=["dev", "val", "test", "ood"])
    ap.add_argument("--n", type=int, default=None, help="limit #tasks (smoke tests)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget-usd", type=float, default=0.10)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    tasks = load_split(args.family, args.split)
    if args.n:
        tasks = tasks[:args.n]
    caps = BudgetCaps(max_usd=args.budget_usd)
    run_name = f"{args.template}_{args.family}_{args.split}" + (f"_n{args.n}" if args.n else "")
    summary = evaluate(TEMPLATES[args.template](), tasks, caps, run_name=run_name,
                       seed=args.seed, use_cache=not args.no_cache, workers=args.workers)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
