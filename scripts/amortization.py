#!/usr/bin/env python
"""Amortization analysis (方案 v4.0 §6, H3c).

Separates the three costs the paper insists on keeping apart:
  deployment cost  — $ per task of the final workflow
  search cost      — $ spent to discover it
  break-even N*    — tasks that must be deployed before the deployment
                     saving repays the search cost

N* = ceil((C_search_method - C_design_baseline) / delta_c), and is +inf when
the method is not cheaper per task or fails its quality bar.
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"

# Human design cost is reported in hours, never converted to dollars
# (PREREGISTRATION §6.5); the baseline's search cost in dollars is 0.
C_DESIGN_BASELINE_USD = 0.0


def break_even(search_usd: float, cost_method: float, cost_baseline: float,
               quality_ok: bool) -> float:
    delta = cost_baseline - cost_method
    if not quality_ok or delta <= 0:
        return math.inf
    return math.ceil((search_usd - C_DESIGN_BASELINE_USD) / delta)


def report(rows, mixtures=((0.1, 0.9), (0.5, 0.5), (0.9, 0.1))):
    """rows: list of dicts with keys method, tier, cost, baseline_cost,
    search_usd, quality_ok."""
    print(f"{'method':22s}{'tier':8s}{'$/task':>10s}{'base $/task':>13s}"
          f"{'search $':>10s}{'N*':>10s}")
    for r in rows:
        n = break_even(r["search_usd"], r["cost"], r["baseline_cost"],
                       r["quality_ok"])
        print(f"{r['method']:22s}{r['tier']:8s}{r['cost']:>10.4f}"
              f"{r['baseline_cost']:>13.4f}{r['search_usd']:>10.2f}"
              f"{('inf' if n == math.inf else f'{n:,.0f}'):>10s}")

    print("\nDeployment-mixture sensitivity (tight, loose weights):")
    by_method = {}
    for r in rows:
        by_method.setdefault(r["method"], {})[r["tier"]] = r
    for m, tiers in by_method.items():
        if not {"tight", "loose"} <= set(tiers):
            continue
        for wt, wl in mixtures:
            cost = wt * tiers["tight"]["cost"] + wl * tiers["loose"]["cost"]
            base = wt * tiers["tight"]["baseline_cost"] + wl * tiers["loose"]["baseline_cost"]
            ok = all(tiers[t]["quality_ok"] for t in ("tight", "loose"))
            n = break_even(tiers["tight"]["search_usd"], cost, base, ok)
            print(f"  {m:22s} {int(wt*100):>3d}% tight / {int(wl*100):>3d}% loose"
                  f"  ->  N* = {('inf' if n == math.inf else f'{n:,.0f}')}")


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else EXP / "amortization_input.json"
    if not Path(src).exists():
        print(f"No input at {src}.\nExpected a JSON list of rows:\n"
              '  {"method","tier","cost","baseline_cost","search_usd","quality_ok"}\n'
              "Produced after the Part-II searches and final test runs complete.")
        return
    report(json.load(open(src)))


if __name__ == "__main__":
    main()
