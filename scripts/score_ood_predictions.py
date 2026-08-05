#!/usr/bin/env python
"""Score the locked OOD predictions against the OOD outcomes.

Predictions were written and committed (tag `ood-predictions-locked`) before
any OOD delta was computed. This script only reads them back and compares.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
SEEDS = [0, 1, 2]
BASE = {"code": "direct", "math": "cot"}
N_BOOT = 10000


def per_task(dirname):
    acc = defaultdict(list)
    for s in SEEDS:
        p = EXP / dirname / f"results_seed{s}.jsonl"
        if p.exists():
            for r in map(json.loads, open(p)):
                acc[r["task_id"]].append(bool(r["success"]))
    return {t: sum(v) / len(v) for t, v in acc.items()}


def measure(struct_dir, base_dir, seed=0):
    S, B = per_task(struct_dir), per_task(base_dir)
    ids = sorted(set(S) & set(B))
    if not ids:
        return None
    d = [S[t] - B[t] for t in ids]
    rng = random.Random(seed)
    n = len(d)
    boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(N_BOOT))
    easy = [1 - S[t] for t in ids if B[t] == 1.0]
    hard = [S[t] for t in ids if B[t] == 0.0]
    return {"delta": sum(d) / n, "lo": boots[int(0.025 * N_BOOT)],
            "hi": boots[int(0.975 * N_BOOT)],
            "breakage": sum(easy) / len(easy) if easy else float("nan"),
            "repair": sum(hard) / len(hard) if hard else float("nan"),
            "n": n, "n_easy": len(easy), "n_hard": len(hard)}


def main():
    locked = json.load(open(EXP / "ood_predictions_locked.json"))
    print("=" * 86)
    print("OOD TRANSFER TEST — predictions locked at "
          f"{locked['written_at']} (tag ood-predictions-locked)")
    print("Rule:", locked["rule"])
    print("=" * 86)
    print(f"\n{'family':7s}{'tier':7s}{'structure':22s}{'PRED d':>8s}{'OBS d':>8s}"
          f"{'err':>8s}{'PRED brk':>10s}{'OBS brk':>9s}{'  brk ok'}")

    errs, sign_ok, brk_ok, brk_n = [], 0, 0, 0
    rows = []
    for p in locked["predictions"]:
        fam, tier, struct = p["family"], p["tier"], p["structure"]
        m = measure(f"envelope_ood/{struct}_{fam}_{tier}",
                    f"envelope_ood/{BASE[fam]}_{fam}_{tier}")
        if not m:
            print(f"{fam:7s}{tier:7s}{struct:22s}  (missing OOD data)")
            continue
        err = p["delta_pred_ood"] - m["delta"]
        errs.append(abs(err))
        agree = (p["delta_pred_ood"] > 0) == (m["delta"] > 0) or abs(m["delta"]) < 0.01
        sign_ok += agree
        if p["breakage_pred_is_bound"]:
            brk_n += 1
            ok = m["breakage"] <= 0.02
            brk_ok += ok
            btxt, flag = f"<= {p['breakage_pred_ood']}", ("OK" if ok else "VIOLATED")
        else:
            btxt, flag = f"{p['breakage_pred_ood']:.3f}", ""
        rows.append((p, m, err))
        print(f"{fam:7s}{tier:7s}{struct:22s}{p['delta_pred_ood']:>+8.3f}"
              f"{m['delta']:>+8.3f}{err:>+8.3f}{btxt:>10s}{m['breakage']:>9.3f}  {flag}")

    if errs:
        print(f"\nmean |error| = {sum(errs)/len(errs):.4f}   "
              f"sign agreement = {sign_ok}/{len(errs)}   "
              f"breakage bound held = {brk_ok}/{brk_n}")

    print("\n--- full OOD intervals (95% paired bootstrap) ---")
    for p, m, _ in rows:
        print(f"  {p['family']:5s} {p['tier']:6s} {p['structure']:22s} "
              f"delta={m['delta']:+.3f} [{m['lo']:+.3f},{m['hi']:+.3f}] "
              f"repair={m['repair']:.3f} breakage={m['breakage']:.3f} "
              f"(n={m['n']}, easy={m['n_easy']}, hard={m['n_hard']})")

    json.dump([{"pred": p, "obs": m} for p, m, _ in rows],
              open(EXP / "ood_transfer_scored.json", "w"), indent=2)


if __name__ == "__main__":
    main()
