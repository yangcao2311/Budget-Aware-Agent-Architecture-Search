#!/usr/bin/env python
"""Confirmatory Part-II evaluation of P1-P5 (PREREGISTRATION §3A, frozen
2026-08-05) on the frozen test split.

Policies were selected once on validation; this script only reads the
resulting frozen-test runs.
"""
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
FINAL = EXP / "final"
SEEDS = [0, 1, 2]
N_BOOT = 10000
DELTA = 0.03


def cell(run: str, cid: int, split: str, tier: str) -> dict[str, float]:
    acc = defaultdict(list)
    for s in SEEDS:
        p = FINAL / run / f"c{cid}_{split}_{tier}" / f"results_seed{s}.jsonl"
        if p.exists():
            for r in map(json.loads, open(p)):
                acc[r["task_id"]].append(bool(r["success"]))
    return {t: sum(v) / len(v) for t, v in acc.items()}


def cost(run: str, cid: int, split: str, tier: str) -> float:
    tot, n = 0.0, 0
    for s in SEEDS:
        p = FINAL / run / f"c{cid}_{split}_{tier}" / f"summary_seed{s}.json"
        if p.exists():
            tot += json.load(open(p))["usd_per_task"]
            n += 1
    return tot / n if n else float("nan")


def chosen(run: str) -> dict | None:
    p = FINAL / f"{run}_chosen.json"
    return json.load(open(p)) if p.exists() else None


def pboot(A, B, seed=0):
    ids = sorted(set(A) & set(B))
    if not ids:
        return None
    rng = random.Random(seed)
    d = [A[i] - B[i] for i in ids]
    n = len(d)
    boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(N_BOOT))
    return {"diff": sum(d) / n, "lo": boots[int(0.025 * N_BOOT)],
            "hi": boots[int(0.975 * N_BOOT)],
            "lcb95_1sided": boots[int(0.05 * N_BOOT)], "n": n}


def pooled(runs_cids, split, tier):
    """Average per-task success across search seeds, then compare."""
    acc = defaultdict(list)
    for run, cid in runs_cids:
        for t, v in cell(run, cid, split, tier).items():
            acc[t].append(v)
    return {t: sum(v) / len(v) for t, v in acc.items() if v}


def main():
    fams = {"code": [0, 1, 2], "math": [0]}
    verdicts = {}

    hb = {f: [(f"A_hbws_{f}_s{s}", chosen(f"A_hbws_{f}_s{s}")) for s in ss]
          for f, ss in fams.items()}
    st = {(f, tier): [(f"A_static_{tier}_{f}_s{s}", chosen(f"A_static_{tier}_{f}_s{s}"))
                      for s in ss]
          for f, ss in fams.items() for tier in ("tight", "loose")}

    missing = [r for grp in list(hb.values()) + list(st.values()) for r, c in grp if not c]
    if missing:
        print(f"INCOMPLETE: missing chosen policies for {missing}")
        return

    print("=" * 78)
    print("CONFIRMATORY PART II — frozen test split (150 tasks/family, 3 exec seeds)")
    print("HBWS: ONE policy per search seed, search cap $60.")
    print("Static: a SEPARATE policy per tier, $60 each => 2x the search budget.")
    print("=" * 78)

    # ---------------- P1: non-inferiority at both seen tiers --------------
    print(f"\n[P1] HBWS - static(per-tier), one-sided 95% LCB >= -{DELTA}")
    p1 = True
    for fam in fams:
        for tier in ("tight", "loose"):
            A = pooled([(r, c["cid"]) for r, c in hb[fam]], "test", tier)
            B = pooled([(r, c["cid"]) for r, c in st[(fam, tier)]], "test", tier)
            r = pboot(A, B)
            if not r:
                print(f"  {fam:5s} {tier:6s} MISSING")
                p1 = False
                continue
            ok = r["lcb95_1sided"] >= -DELTA
            p1 &= ok
            print(f"  {fam:5s} {tier:6s} diff={r['diff']:+.3f} "
                  f"[{r['lo']:+.3f},{r['hi']:+.3f}] LCB={r['lcb95_1sided']:+.3f} "
                  f"-> {'OK' if ok else 'FAIL'}")
    verdicts["P1"] = p1

    # ---------------- P2: deployment cost at loose ------------------------
    print("\n[P2] deployment $/task at loose, HBWS vs static (code needs >=20% lower)")
    p2 = True
    for fam in fams:
        ch = sum(cost(r, c["cid"], "test", "loose") for r, c in hb[fam]) / len(hb[fam])
        cs = sum(cost(r, c["cid"], "test", "loose")
                 for r, c in st[(fam, "loose")]) / len(st[(fam, "loose")])
        red = (cs - ch) / cs if cs else float("nan")
        ok = red >= 0.20 if fam == "code" else red > 0
        if fam == "code":
            p2 = ok
        print(f"  {fam:5s} HBWS ${ch:.4f} vs static ${cs:.4f}  "
              f"reduction={red:+.1%} -> {'OK' if ok else 'FAIL'}")
    verdicts["P2"] = p2

    # ---------------- P3: unseen $0.15 tier -------------------------------
    print(f"\n[P3] unseen $0.15 tier: HBWS vs static transfer control, LCB >= -{DELTA}")
    p3 = True
    for fam in fams:
        A = pooled([(r, c["cid"]) for r, c in hb[fam]], "test", "unseen")
        # Transfer control: better of the two static policies at $0.15,
        # picked by whichever scores higher (reported both ways).
        opts = {}
        for tier in ("tight", "loose"):
            B = pooled([(r, c["cid"]) for r, c in st[(fam, tier)]], "test", "unseen")
            if B:
                opts[tier] = B
        if not A or not opts:
            print(f"  {fam:5s} MISSING")
            p3 = False
            continue
        best_tier = max(opts, key=lambda t: sum(opts[t].values()) / len(opts[t]))
        r = pboot(A, opts[best_tier])
        ok = r["lcb95_1sided"] >= -DELTA
        p3 &= ok
        for t, B in opts.items():
            mark = " <- control" if t == best_tier else ""
            print(f"    static-{t:6s} @unseen mean={sum(B.values())/len(B):.3f}{mark}")
        print(f"  {fam:5s} HBWS mean={sum(A.values())/len(A):.3f}  "
              f"diff={r['diff']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}] "
              f"LCB={r['lcb95_1sided']:+.3f} -> {'OK' if ok else 'FAIL'}")
    verdicts["P3"] = p3

    # ---------------- P4: principle convergence (descriptive) -------------
    print("\n[P4] selected policy shapes (descriptive)")
    n_budget, n_total, n_incumbent = 0, 0, 0
    for fam in fams:
        for label, grp in [("HBWS", hb[fam]),
                           ("static-tight", st[(fam, "tight")]),
                           ("static-loose", st[(fam, "loose")])]:
            for run, c in grp:
                wf = c["wf"]
                types = [n["type"] for n in wf["nodes"]]
                t0 = wf["nodes"][0].get("params", {}).get("temperature")
                incumbent = (types[0] in ("generate", "vote") and "verify" in types
                             and t0 == 0.0)
                thr = sorted({float(m) for m in re.findall(
                    r"budget_(?:above|below):([0-9.]+)", json.dumps(wf))})
                n_total += 1
                n_incumbent += incumbent
                if label == "HBWS" and thr:
                    n_budget += 1
                print(f"  {fam:5s} {label:13s} cid={c['cid']:<3d} "
                      f"{'->'.join(types):48s} t0={t0} "
                      f"incumbent={'Y' if incumbent else 'N'} thr={thr or '-'}")
    print(f"  => incumbent-protection shape in {n_incumbent}/{n_total} selected policies; "
          f"budget predicates retained in {n_budget}/{sum(len(v) for v in hb.values())} HBWS policies")
    verdicts["P4"] = f"{n_incumbent}/{n_total} incumbent, {n_budget} HBWS w/ budget pred"

    # ---------------- P5: amortization ------------------------------------
    print("\n[P5] design + deployment economics")
    for fam in fams:
        ch = sum(cost(r, c["cid"], "test", "loose") for r, c in hb[fam]) / len(hb[fam])
        cs = sum(cost(r, c["cid"], "test", "loose")
                 for r, c in st[(fam, "loose")]) / len(st[(fam, "loose")])
        search_hbws, search_static = 60.0, 120.0     # Protocol A caps
        dc = cs - ch
        extra = search_hbws - search_static           # negative => HBWS cheaper to design
        if dc <= 0:
            nstar = math.inf
        elif extra <= 0:
            nstar = 0
        else:
            nstar = math.ceil(extra / dc)
        print(f"  {fam:5s} search: HBWS ${search_hbws:.0f} vs static ${search_static:.0f} "
              f"(diff ${extra:+.0f});  deploy delta ${dc:+.5f}/task  =>  N* = "
              f"{'inf' if nstar == math.inf else int(nstar)}")
    verdicts["P5"] = "reported"

    print("\n" + "=" * 78)
    print("VERDICTS: " + "  ".join(f"{k}={v}" for k, v in verdicts.items()))
    print("=" * 78)
    json.dump(verdicts, open(EXP / "partII_confirmatory_verdicts.json", "w"),
              indent=2, default=str)


if __name__ == "__main__":
    main()
