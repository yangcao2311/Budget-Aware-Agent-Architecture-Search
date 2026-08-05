#!/usr/bin/env python
"""Part II analysis: HBWS (one cross-budget policy) vs Static Evolution
Search (a separate policy per tier, given a full search cap each).

Protocol A is a PER-TIER comparison: HBWS's single policy at tier t against
the static policy that was searched specifically for tier t. Comparing
HBWS's cross-tier J_CB against a static single-tier J_CB is meaningless —
the former takes a min over a harder tier the latter never saw.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEARCH = ROOT / "experiments" / "search"


def load(run: str) -> dict | None:
    p = SEARCH / run / "search_result.json"
    return json.load(open(p)) if p.exists() else None


def tier_stats(cand: dict, tier: str) -> dict | None:
    if not cand.get("stats"):
        return None
    lvl = max(int(k) for k in cand["stats"])
    pt = cand["stats"][str(lvl)]["per_tier"]
    if tier not in pt:
        return None
    return {**pt[tier], "fidelity": lvl}


def best_at(res: dict, tier: str) -> tuple[dict, dict] | None:
    """Best archive member at `tier`, ranked by success at the highest
    fidelity it reached (ties broken by fidelity, then by lower cost)."""
    scored = []
    for c in res["archive"]:
        s = tier_stats(c, tier)
        if s:
            scored.append((s["success_rate"], s["fidelity"], -s["usd_per_task"], c, s))
    if not scored:
        return None
    scored.sort(key=lambda x: (-x[0], -x[1], x[2]))
    return scored[0][3], scored[0][4]


def describe(wf: dict) -> str:
    kinds = [n["type"] for n in wf["nodes"]]
    thr = sorted({float(m) for m in re.findall(
        r"budget_(?:above|below):([0-9.]+)", json.dumps(wf))})
    return (f"{'->'.join(kinds)}"
            + (f"  budget_thresholds={thr}" if thr else "  (no budget predicate)"))


def main(family="code", seed=0):
    hb = load(f"A_hbws_{family}_s{seed}")
    if not hb:
        sys.exit(f"missing HBWS run for {family} s{seed}")

    print("=" * 74)
    print(f"PROTOCOL A — per-tier comparison, {family}, search seed {seed}")
    print("HBWS: ONE policy, cap $S over both tiers.  "
          "Static: a policy PER tier, cap $S each (2x total).")
    print("=" * 74)

    for tier in ["tight", "loose"]:
        st = load(f"A_static_{tier}_{family}_s{seed}")
        hb_best = best_at(hb, tier)
        st_best = best_at(st, tier) if st else None
        print(f"\n--- tier: {tier} ---")
        if hb_best:
            c, s = hb_best
            print(f"  HBWS   cid={c['cid']:<4d} succ={s['success_rate']:.3f} "
                  f"${s['usd_per_task']:.4f}  F{s['fidelity']}  {describe(c['wf'])}")
        if st_best:
            c, s = st_best
            print(f"  Static cid={c['cid']:<4d} succ={s['success_rate']:.3f} "
                  f"${s['usd_per_task']:.4f}  F{s['fidelity']}  {describe(c['wf'])}")
        if hb_best and st_best:
            d = hb_best[1]["success_rate"] - st_best[1]["success_rate"]
            same_fid = hb_best[1]["fidelity"] == st_best[1]["fidelity"]
            print(f"  delta(HBWS - Static) = {d:+.3f}"
                  f"{'' if same_fid else '   [different fidelity: dev-only, not confirmatory]'}")

    # -- budget-predicate usage in the HBWS archive -------------------------
    used = [c for c in hb["archive"] if "budget_" in json.dumps(c["wf"])]
    print(f"\n--- budget predicates in HBWS archive: {len(used)}/{len(hb['archive'])}")
    thr_all = sorted({float(m) for c in used for m in re.findall(
        r"budget_(?:above|below):([0-9.]+)", json.dumps(c["wf"]))})
    print(f"    thresholds retained: {thr_all}")

    # -- Protocol B: matched total spend ------------------------------------
    print("\n" + "=" * 74)
    print("PROTOCOL B — matched TOTAL search spend")
    print("HBWS at $S vs static's two runs each read at $S/2 (checkpointed).")
    S = hb["cap_usd"]
    for tier in ["tight", "loose"]:
        st = load(f"A_static_{tier}_{family}_s{seed}")
        if not st:
            continue
        half = [a for a in st["anytime"] if a["search_usd"] <= S / 2]
        full = [a for a in hb["anytime"] if a["search_usd"] <= S]
        hb_best_mean = max((a.get("best_j_mean", -1) for a in full), default=-1)
        st_best_mean = max((a.get("best_j_mean", -1) for a in half), default=-1)
        print(f"  {tier:6s}: HBWS best j_mean@${S:.0f}={hb_best_mean:.3f} "
              f"(n={len(full)})  |  Static best j_mean@${S/2:.0f}={st_best_mean:.3f} "
              f"(n={len(half)})")
    print("\nNote: j_mean is cross-tier for HBWS and single-tier for static, so"
          "\nProtocol B is reported per-tier from the deployment evaluation, not"
          "\nfrom these search-internal scores. Use eval_policy.py for the"
          "\nconfirmatory numbers.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "code",
         int(sys.argv[2]) if len(sys.argv) > 2 else 0)
