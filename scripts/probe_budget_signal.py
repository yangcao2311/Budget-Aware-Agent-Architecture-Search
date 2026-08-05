#!/usr/bin/env python
"""Measure the budget signal a policy can actually condition on.

A budget-contingent policy reads normalized remaining budget, which is 1.0 at
every tier when a task starts, so tiers are distinguishable only through the
RATE at which a call consumes the budget. This probe reports that signal at
each decision point, which fixes the region the branch thresholds must cover.
Zero API cost.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hbws import llm
from hbws.data import load_split
from hbws.ledger import BUDGET_TIERS, TaskLedger, llm_call_vec, tool_call_vec
from hbws.search import BRANCH_THRESHOLDS

STEPS = ["start", "after generate", "+verify", "+refine", "+refine2"]


def trace(tier, prompt):
    caps = BUDGET_TIERS[tier]
    led = TaskLedger(caps)
    est = llm.estimate_in_tokens([{"role": "user", "content": prompt}])
    out = min(1536, caps.max_out_tokens)
    fr = [led.remaining_frac]
    for vec, actual in [
        (llm_call_vec(est, out), {"llm_calls": 1, "in_tokens": est, "out_tokens": out // 2}),
        (tool_call_vec(), {"tool_calls": 1}),
        (llm_call_vec(est * 3, out), {"llm_calls": 1, "in_tokens": est * 3, "out_tokens": out // 2}),
        (llm_call_vec(est * 4, out), {"llm_calls": 1, "in_tokens": est * 4, "out_tokens": out // 2}),
    ]:
        lease = led.reserve(vec)
        if lease is None:
            fr.append(float("nan"))   # reserve rejected: node cannot run
            continue
        led.settle(lease, actual)
        fr.append(led.remaining_frac)
    return fr


def main():
    prompt = load_split("code", "dev")[0]["prompt"]
    tiers = ["tight", "unseen", "loose"]
    rows = {t: trace(t, prompt) for t in tiers}
    print(f"{'step':16s}" + "".join(f"{t:>10s}" for t in tiers) + "   band")
    for i, step in enumerate(STEPS):
        vals = [rows[t][i] for t in tiers]
        ok = [v for v in vals if v == v]
        lo, hi = (min(ok), max(ok)) if ok else (float("nan"), float("nan"))
        mark = "  <-- usable" if (ok and hi - lo > 0.05) else ""
        print(f"{step:16s}" + "".join(f"{v:>10.3f}" for v in vals)
              + f"   {lo:.2f}-{hi:.2f}{mark}")
    print(f"\nthresholds available to mutation: {BRANCH_THRESHOLDS}")
    usable = sorted({t for t in BRANCH_THRESHOLDS
                     for a in tiers for b in tiers
                     if rows[a][1] == rows[a][1] and rows[b][1] == rows[b][1]
                     and rows[a][1] < t <= rows[b][1]})
    print(f"thresholds separating at least two tiers after generate: {usable}")
    if not usable:
        print("WARNING: no threshold can distinguish tiers - the policy class "
              "cannot express tier-dependent behaviour.")


if __name__ == "__main__":
    main()
