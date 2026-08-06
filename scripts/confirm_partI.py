#!/usr/bin/env python
"""Confirmatory Part-I evaluation of C1-C5 exactly as preregistered
(PREREGISTRATION §2A, frozen 2026-08-05) on the frozen test split.

Every number here is the paper's confirmatory evidence; dev results are
never mixed in. Run after the frozen-test-set execution completes.
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
SEEDS = [0, 1, 2]
N_BOOT = 10000
TIERS = ["tight", "unseen", "loose"]
BASELINE = {"code": "direct", "math": "cot"}
INCUMBENT = {"code": "incumbent_refine", "math": "incumbent_refine_cot"}


def per_task(dirname: str) -> dict[str, float]:
    """Seed-averaged per-task success for one (structure, family, tier) cell."""
    acc = defaultdict(list)
    for s in SEEDS:
        p = EXP / dirname / f"results_seed{s}.jsonl"
        if p.exists():
            for r in map(json.loads, open(p)):
                ok = bool(r.get("success_symbolic", r["success"]))
                acc[r["task_id"]].append(ok)
    return {t: sum(v) / len(v) for t, v in acc.items()}


def cost_of(dirname: str) -> float:
    tot, n = 0.0, 0
    for s in SEEDS:
        p = EXP / dirname / f"summary_seed{s}.json"
        if p.exists():
            d = json.load(open(p))
            tot += d["usd_per_task"]
            n += 1
    return tot / n if n else float("nan")


def pboot(A, B, seed=0):
    ids = sorted(set(A) & set(B))
    if not ids:
        return None
    rng = random.Random(seed)
    d = [A[i] - B[i] for i in ids]
    n = len(d)
    boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(N_BOOT))
    return {"diff": sum(d) / n, "lo": boots[int(0.025 * N_BOOT)],
            "hi": boots[int(0.975 * N_BOOT)], "n": n,
            "lcb90": boots[int(0.05 * N_BOOT)]}


def rate_boot(vals, seed=0):
    """Bootstrap CI for a mean over tasks (repair / breakage rates)."""
    if not vals:
        return None
    rng = random.Random(seed)
    n = len(vals)
    boots = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(N_BOOT))
    return {"rate": sum(vals) / n, "lo": boots[int(0.025 * N_BOOT)],
            "hi": boots[int(0.975 * N_BOOT)], "n": n}


def d(fam, struct, tier, tag="envelope_test"):
    return f"{tag}/{struct}_{fam}_{tier}"


def repair_breakage(struct_dir, base_dir):
    S, B = per_task(struct_dir), per_task(base_dir)
    ids = set(S) & set(B)
    hard = [S[t] for t in ids if B[t] == 0.0]
    easy = [1 - S[t] for t in ids if B[t] == 1.0]
    return rate_boot(hard), rate_boot(easy)


def main():
    verdicts = {}
    print("=" * 78)
    print("CONFIRMATORY PART I — frozen test split (150 tasks/family, 3 seeds)")
    print("=" * 78)

    # ---------------- C1: net-zero of vanilla structure at loose ----------
    print("\n[C1] vanilla verify_refine_3 - baseline @ loose (CI should contain 0,"
          " |diff|<=0.05, and cost >= 2x)")
    c1_ok = True
    for fam in ["code", "math"]:
        b, v = d(fam, BASELINE[fam], "loose"), d(fam, "verify_refine_3", "loose")
        r = pboot(per_task(v), per_task(b))
        cb, cv = cost_of(b), cost_of(v)
        if not r:
            print(f"  {fam}: MISSING DATA")
            c1_ok = False
            continue
        contains0 = r["lo"] <= 0 <= r["hi"]
        small = abs(r["diff"]) <= 0.05
        costly = cv >= 2 * cb
        ok = contains0 and small and costly
        c1_ok &= ok
        print(f"  {fam:5s} diff={r['diff']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}] "
              f"${cb:.4f}->${cv:.4f} ({cv/cb:.1f}x)  "
              f"contains0={contains0} small={small} costly={costly} -> {'OK' if ok else 'FAIL'}")
    verdicts["C1"] = c1_ok

    # ---------------- C2: budget floor (code @ tight) ---------------------
    print("\n[C2] vanilla - direct @ tight, code (CI upper bound < 0)")
    r = pboot(per_task(d("code", "verify_refine_3", "tight")),
              per_task(d("code", "direct", "tight")))
    if r:
        ok = r["hi"] < 0
        print(f"  code  diff={r['diff']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}] -> "
              f"{'OK' if ok else 'FAIL'}")
        verdicts["C2"] = ok
    else:
        print("  MISSING DATA")
        verdicts["C2"] = False

    # ---------------- C3: dose-response (code @ loose) --------------------
    print("\n[C3] vanilla - direct @ loose across masking 1.0 -> 0.5 -> 0.0 "
          "(monotone non-increasing; significant at 0.0)")
    seq = []
    for mask, tag in [(1.0, "envelope_test"),
                      (0.5, "envelope_test_mask0.5_k1"),
                      (0.0, "envelope_test_mask0.0_k1")]:
        r = pboot(per_task(d("code", "verify_refine_3", "loose", tag)),
                  per_task(d("code", "direct", "loose", tag)))
        if r:
            seq.append((mask, r))
            print(f"  mask={mask:.1f}: {r['diff']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}]")
    if len(seq) == 3:
        mono = seq[0][1]["diff"] >= seq[1][1]["diff"] >= seq[2][1]["diff"]
        sig = seq[2][1]["hi"] < 0
        print(f"  monotone={mono} significant_at_0={sig} -> "
              f"{'OK' if (mono and sig) else 'FAIL'}")
        verdicts["C3"] = mono and sig
    else:
        print("  MISSING DATA")
        verdicts["C3"] = False

    # ---------------- C4: incumbent protection ----------------------------
    print("\n[C4] incumbent-protecting structure: breakage<=0.02 and "
          "non-inferior (LCB>=-0.02) at all tiers; code significant at >=2 tiers")
    c4_ok, code_sig = True, 0
    for fam in ["code", "math"]:
        for tier in TIERS:
            inc, base = d(fam, INCUMBENT[fam], tier), d(fam, BASELINE[fam], tier)
            r = pboot(per_task(inc), per_task(base))
            rep, brk = repair_breakage(inc, base)
            if not r or not brk:
                print(f"  {fam:5s} @{tier:6s} MISSING DATA")
                c4_ok = False
                continue
            low_break = brk["rate"] <= 0.02
            noninf = r["lo"] >= -0.02
            sig = r["lo"] > 0
            if fam == "code" and sig:
                code_sig += 1
            c4_ok &= (low_break and noninf)
            print(f"  {fam:5s} @{tier:6s} diff={r['diff']:+.3f} "
                  f"[{r['lo']:+.3f},{r['hi']:+.3f}] breakage={brk['rate']:.3f} "
                  f"repair={rep['rate'] if rep else float('nan'):.3f} "
                  f"lowbreak={low_break} noninf={noninf} sig={sig}")
    print(f"  code significant tiers = {code_sig}/3 (need >=2) -> "
          f"{'OK' if (c4_ok and code_sig >= 2) else 'FAIL'}")
    verdicts["C4"] = c4_ok and code_sig >= 2

    # ---------------- C5: repair is verifier-bounded ----------------------
    print("\n[C5] repair rate: code (oracle) > math (non-oracle); math CI "
          "contains 0")
    reps = {}
    for fam in ["code", "math"]:
        rep, _ = repair_breakage(d(fam, INCUMBENT[fam], "loose"),
                                 d(fam, BASELINE[fam], "loose"))
        reps[fam] = rep
        if rep:
            print(f"  {fam:5s} repair={rep['rate']:.3f} "
                  f"[{rep['lo']:.3f},{rep['hi']:.3f}] n={rep['n']}")
    if reps.get("code") and reps.get("math"):
        separated = reps["code"]["lo"] > reps["math"]["hi"]
        math0 = reps["math"]["lo"] <= 0 <= reps["math"]["hi"]
        print(f"  CIs separated={separated} math_contains_0={math0} -> "
              f"{'OK' if (separated and math0) else 'FAIL'}")
        verdicts["C5"] = separated and math0
    else:
        verdicts["C5"] = False

    # Never persist verdicts from an incomplete run: a partial "NOT
    # SUPPORTED" is indistinguishable from a real refutation on disk.
    required = []
    for fam in ["code", "math"]:
        for tier in TIERS:
            required += [d(fam, BASELINE[fam], tier), d(fam, "verify_refine_3", tier),
                         d(fam, INCUMBENT[fam], tier)]
    for tag in ["envelope_test_mask0.5_k1", "envelope_test_mask0.0_k1"]:
        required += [d("code", "verify_refine_3", "loose", tag),
                     d("code", "direct", "loose", tag)]
    missing = [c for c in required
               if not (EXP / c / f"results_seed{SEEDS[-1]}.jsonl").exists()]

    print("\n" + "=" * 78)
    if missing:
        print(f"INCOMPLETE RUN: {len(missing)}/{len(required)} cells missing "
              f"(e.g. {missing[0]}). Verdicts NOT written.")
        print("=" * 78)
        return
    print("VERDICTS: " + "  ".join(
        f"{k}={'SUPPORTED' if v else 'NOT SUPPORTED'}" for k, v in verdicts.items()))
    print("=" * 78)
    out = EXP / "partI_confirmatory_verdicts.json"
    json.dump(verdicts, open(out, "w"), indent=2)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
