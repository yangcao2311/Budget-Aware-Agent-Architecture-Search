#!/usr/bin/env python
"""Part I envelope study driver (方案 v4.0 §5.6, Fig.2/Fig.3).

Fig.2:  python scripts/run_envelope.py --tiers E1 E2 tight unseen loose --n 120
Fig.3:  python scripts/run_envelope.py --tiers tight loose --mask-tests 0.5 \
            --critic-k 2 --structures verify_refine_1 verify_refine_3 direct vote3

Verifier-reliability degradation (gold-free by construction):
  --mask-tests f : code family keeps only ceil(f * #visible asserts)
  --critic-k k   : math heuristic_critic uses k independent re-derivations
"""
import argparse
import copy
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hbws.data import load_split
from hbws.dsl import ENVELOPE_LIB
from hbws.ledger import BUDGET_TIERS
from hbws.protocol import evaluate


def mask_tests(task: dict, frac: float) -> dict:
    t = dict(task)
    lines = [l for l in task["feedback_tests"].splitlines() if l.strip()]
    asserts = [l for l in lines if l.lstrip().startswith("assert")]
    other = [l for l in lines if not l.lstrip().startswith("assert")]
    keep = math.ceil(frac * len(asserts))
    t["feedback_tests"] = "\n".join(other + asserts[:keep])
    return t


def clamp_to_tier(wf: dict, caps) -> dict:
    """Preregistered rule (方案 §5.2): every node's per-call output cap is
    clamped to the tier's total output cap so the first call is always
    reservable; structure degradation beyond that is the phenomenon we
    measure, not an artifact."""
    wf = copy.deepcopy(wf)
    for n in wf["nodes"]:
        p = n.get("params")
        if p and "max_output_tokens" in p:
            p["max_output_tokens"] = min(p["max_output_tokens"], caps.max_out_tokens)
    return wf


def set_critic_k(wf: dict, k: int) -> dict:
    wf = copy.deepcopy(wf)
    for n in wf["nodes"]:
        if n["type"] == "verify":
            n["k"] = k
    return wf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--structures", nargs="+", default=list(ENVELOPE_LIB))
    ap.add_argument("--tiers", nargs="+", default=["E1", "E2", "tight", "unseen", "loose"])
    ap.add_argument("--families", nargs="+", default=["code", "math"])
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mask-tests", type=float, default=1.0)
    ap.add_argument("--critic-k", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    deg = f"_mask{args.mask_tests}_k{args.critic_k}" \
        if (args.mask_tests < 1.0 or args.critic_k != 1) else ""
    rows = []
    for fam in args.families:
        tasks = load_split(fam, "dev")[:args.n]
        if fam == "code" and args.mask_tests < 1.0:
            tasks = [mask_tests(t, args.mask_tests) for t in tasks]
        for tier in args.tiers:
            for sname in args.structures:
                wf = clamp_to_tier(ENVELOPE_LIB[sname](), BUDGET_TIERS[tier])
                if fam == "math" and args.critic_k != 1:
                    wf = set_critic_k(wf, args.critic_k)
                run = f"envelope{deg}/{sname}_{fam}_{tier}"
                s = evaluate(wf, tasks, BUDGET_TIERS[tier], run_name=run,
                             seed=args.seed, use_cache=True, workers=args.workers)
                s.update({"structure": sname, "family": fam, "tier": tier,
                          "mask_tests": args.mask_tests, "critic_k": args.critic_k})
                rows.append(s)
                print(f"{fam:5s} {tier:6s} {sname:16s} "
                      f"succ={s['success_rate']:.3f} $/task={s['usd_per_task']:.4f} "
                      f"rr={s.get('reserve_rejected', 0)}")

    out = Path(__file__).resolve().parent.parent / "experiments" / f"envelope{deg}_summary.jsonl"
    with open(out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("summary appended to", out)


if __name__ == "__main__":
    main()
