#!/usr/bin/env python
"""Lock in out-of-domain predictions BEFORE the OOD deltas are computed.

Why this exists. The decomposition Delta = (1-p)*r - p*b is an ALGEBRAIC
IDENTITY for binary outcomes, so "validating" it on the same data used to
estimate r and b is tautological and proves nothing. The only content-bearing
test is out-of-sample transfer: estimate r and b in one domain, then predict
Delta in a domain the parameters never saw.

This script estimates r_eff and b_eff on the frozen in-domain TEST split,
reads only the OOD baseline success rate p_ood (needed as an input, and not
itself a prediction target), and writes the predicted Delta for each OOD
condition. It is committed before the OOD deltas are looked at.

Preregistered predictions (transfer form):
  Delta_pred(OOD) = (1 - p_ood) * r_test - p_ood * b_test
and, structurally:
  breakage_pred(incumbent-protecting structure, OOD) <= 0.02
the latter following from the construction (the incumbent is only replaced
when a verifier rejects it), and therefore expected to be domain-invariant.
"""
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
SEEDS = [0, 1, 2]
BASE = {"code": "direct", "math": "cot"}
INCUMBENT = {"code": "incumbent_refine", "math": "incumbent_refine_cot"}


def per_task(dirname):
    acc = defaultdict(list)
    for s in SEEDS:
        p = EXP / dirname / f"results_seed{s}.jsonl"
        if p.exists():
            for r in map(json.loads, open(p)):
                acc[r["task_id"]].append(bool(r["success"]))
    return {t: sum(v) / len(v) for t, v in acc.items()}


def params(struct_dir, base_dir):
    S, B = per_task(struct_dir), per_task(base_dir)
    ids = sorted(set(S) & set(B))
    if not ids:
        return None
    hard = [S[t] for t in ids if B[t] == 0.0]
    easy = [1 - S[t] for t in ids if B[t] == 1.0]
    return {"p": sum(B[t] for t in ids) / len(ids),
            "r_eff": sum(hard) / len(hard) if hard else 0.0,
            "b_eff": sum(easy) / len(easy) if easy else 0.0}


def main():
    preds = {"written_at": datetime.now().isoformat(timespec="seconds"),
             "rule": "Delta_pred = (1 - p_ood) * r_test - p_ood * b_test",
             "note": "r_test, b_test estimated on the in-domain frozen TEST "
                     "split; p_ood is the OOD baseline success (an input, not "
                     "a prediction target). Deltas on OOD were not inspected "
                     "before writing this file.",
             "predictions": []}

    for fam in ("code", "math"):
        for tier in ("tight", "loose"):
            for struct in ("verify_refine_3", INCUMBENT[fam]):
                src = params(f"envelope_test/{struct}_{fam}_{tier}",
                             f"envelope_test/{BASE[fam]}_{fam}_{tier}")
                base_ood = per_task(f"envelope_ood/{BASE[fam]}_{fam}_{tier}")
                if not src or not base_ood:
                    continue
                p_ood = sum(base_ood.values()) / len(base_ood)
                delta_pred = (1 - p_ood) * src["r_eff"] - p_ood * src["b_eff"]
                preds["predictions"].append({
                    "family": fam, "tier": tier, "structure": struct,
                    "r_test": round(src["r_eff"], 4),
                    "b_test": round(src["b_eff"], 4),
                    "p_test": round(src["p"], 4),
                    "p_ood": round(p_ood, 4),
                    "delta_pred_ood": round(delta_pred, 4),
                    "breakage_pred_ood": (0.02 if struct == INCUMBENT[fam]
                                          else round(src["b_eff"], 4)),
                    "breakage_pred_is_bound": struct == INCUMBENT[fam],
                })

    out = EXP / "ood_predictions_locked.json"
    json.dump(preds, open(out, "w"), indent=2)
    print(f"locked {len(preds['predictions'])} predictions -> {out}\n")
    print(f"{'family':7s}{'tier':7s}{'structure':22s}{'r_test':>8s}{'b_test':>8s}"
          f"{'p_ood':>8s}{'PRED delta':>11s}{'PRED breakage':>15s}")
    for p in preds["predictions"]:
        bp = f"<= {p['breakage_pred_ood']}" if p["breakage_pred_is_bound"] else f"{p['breakage_pred_ood']:.3f}"
        print(f"{p['family']:7s}{p['tier']:7s}{p['structure']:22s}"
              f"{p['r_test']:>8.3f}{p['b_test']:>8.3f}{p['p_ood']:>8.3f}"
              f"{p['delta_pred_ood']:>+11.3f}{bp:>15s}")


if __name__ == "__main__":
    main()
