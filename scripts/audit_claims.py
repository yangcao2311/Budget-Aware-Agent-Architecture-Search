#!/usr/bin/env python
"""Claim-evidence audit: recompute every quantitative claim in the paper
directly from raw per-task logs and diff against what the paper states.

Any mismatch is a paper bug. This is the last line of defence before a
reviewer recomputes a number and finds it wrong.
"""
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
TEX = ROOT / "paper" / "main.tex"
SEEDS = [0, 1, 2]
NB = 10000


def per_task(dirname):
    acc = defaultdict(list)
    for s in SEEDS:
        p = EXP / dirname / f"results_seed{s}.jsonl"
        if p.exists():
            for r in map(json.loads, open(p)):
                ok = bool(r.get("success_symbolic", r["success"]))
                acc[r["task_id"]].append(ok)
    return {t: sum(v) / len(v) for t, v in acc.items()}


def cost(dirname):
    tot = n = 0
    for s in SEEDS:
        p = EXP / dirname / f"summary_seed{s}.json"
        if p.exists():
            tot += json.load(open(p))["usd_per_task"]
            n += 1
    return tot / n if n else float("nan")


def stat(sd, bd, seed=0):
    S, B = per_task(sd), per_task(bd)
    ids = sorted(set(S) & set(B))
    if not ids:
        return None
    d = [S[t] - B[t] for t in ids]
    n = len(d)
    rng = random.Random(seed)
    bo = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(NB))
    easy = [1 - S[t] for t in ids if B[t] == 1.0]
    hard = [S[t] for t in ids if B[t] == 0.0]
    return {"delta": sum(d) / n, "lo": bo[int(.025 * NB)], "hi": bo[int(.975 * NB)],
            "brk": sum(easy) / len(easy) if easy else float("nan"),
            "rep": sum(hard) / len(hard) if hard else float("nan"),
            "cs": cost(sd), "cb": cost(bd), "n": n}


# (label, paper value, tolerance, recompute fn)
def check(label, claimed, tol, fn):
    try:
        actual = fn()
    except Exception as e:
        return (label, claimed, None, f"ERROR {type(e).__name__}: {e}")
    if actual is None:
        return (label, claimed, None, "NO DATA")
    ok = abs(actual - claimed) <= tol
    return (label, claimed, actual, "OK" if ok else "MISMATCH")


def main():
    T, TO, TV, TP = ("envelope_test", "envelope_test_mask0.0_k1",
                     "envelope_ood_visible", "envelope_logic_prospective")
    OD = "envelope_ood"
    checks = []

    # --- abstract / intro figures ---
    r = stat(f"{T}/verify_refine_3_code_loose", f"{T}/direct_code_loose")
    checks.append(check("abstract: repair 20.5%", 0.205, 0.001, lambda: r["rep"]))
    checks.append(check("abstract: breakage 14.0%", 0.140, 0.001, lambda: r["brk"]))
    checks.append(check("abstract: net -3.3pt", -0.033, 0.001, lambda: r["delta"]))
    checks.append(check("abstract: 5.5x cost", 5.5, 0.06, lambda: r["cs"] / r["cb"]))
    rm = stat(f"{TO}/verify_refine_3_code_loose", f"{TO}/direct_code_loose")
    checks.append(check("abstract/C3: no-signal -11.8pt", -0.118, 0.001,
                        lambda: rm["delta"]))
    rmath = stat(f"{T}/verify_refine_3_math_loose", f"{T}/cot_math_loose")
    checks.append(check("C1 math delta +0.020", 0.020, 0.001, lambda: rmath["delta"]))
    checks.append(check("C1 math 2.2x cost", 2.2, 0.06, lambda: rmath["cs"] / rmath["cb"]))

    # --- Table 1: every cell ---
    BASE = {"code": "direct", "math": "cot"}
    INC = {"code": "incumbent_refine", "math": "incumbent_refine_cot"}
    tab1 = {
        ("code", "tight", "verify_refine_3"): (-0.104, -0.162, -0.047, 0.162, 0.224, 0.0062),
        ("code", "tight", "inc"): (0.013, 0.002, 0.031, 0.034, 0.000, 0.0022),
        ("code", "unseen", "verify_refine_3"): (-0.038, -0.093, 0.018, 0.214, 0.150, 0.0085),
        ("code", "unseen", "inc"): (0.020, 0.002, 0.042, 0.060, 0.000, 0.0025),
        ("code", "loose", "verify_refine_3"): (-0.033, -0.087, 0.020, 0.205, 0.140, 0.0090),
        ("code", "loose", "inc"): (0.020, 0.002, 0.042, 0.060, 0.000, 0.0025),
        ("math", "tight", "verify_refine_3"): (0.007, -0.027, 0.040, 0.140, 0.050, 0.0076),
        ("math", "tight", "inc"): (0.000, 0.000, 0.000, 0.000, 0.000, 0.0075),
        ("math", "unseen", "verify_refine_3"): (0.013, -0.020, 0.044, 0.129, 0.040, 0.0141),
        ("math", "unseen", "inc"): (0.000, -0.009, 0.009, 0.022, 0.000, 0.0141),
        ("math", "loose", "verify_refine_3"): (0.020, -0.011, 0.051, 0.129, 0.037, 0.0153),
        ("math", "loose", "inc"): (0.007, -0.002, 0.018, 0.032, 0.000, 0.0153),
    }
    for (fam, tier, w), (d, lo, hi, rep, brk, c) in tab1.items():
        wf = INC[fam] if w == "inc" else w
        st = stat(f"{T}/{wf}_{fam}_{tier}", f"{T}/{BASE[fam]}_{fam}_{tier}")
        tag = f"Tab1 {fam}/{tier}/{w}"
        checks.append(check(f"{tag} delta", d, 0.001, lambda st=st: st["delta"]))
        checks.append(check(f"{tag} CIlo", lo, 0.002, lambda st=st: st["lo"]))
        checks.append(check(f"{tag} CIhi", hi, 0.002, lambda st=st: st["hi"]))
        checks.append(check(f"{tag} repair", rep, 0.001, lambda st=st: st["rep"]))
        checks.append(check(f"{tag} breakage", brk, 0.001, lambda st=st: st["brk"]))
        checks.append(check(f"{tag} $/task", c, 0.0002, lambda st=st: st["cs"]))

    # --- C2 mechanism numbers quoted in prose ---
    rt = stat(f"{T}/verify_refine_3_code_tight", f"{T}/direct_code_tight")
    checks.append(check("C2 prose repair 0.205->0.162", 0.162, 0.001, lambda: rt["rep"]))
    checks.append(check("C2 prose breakage 0.140->0.224", 0.224, 0.001, lambda: rt["brk"]))

    def reserve_rejects():
        tot = 0
        p = EXP / f"{T}/verify_refine_3_code_tight" / "summary_seed0.json"
        return json.load(open(p))["reserve_rejected"]
    checks.append(check("C2 prose: 35/150 reserve rejections", 35, 0, reserve_rejects))

    # --- C3 dose-response ---
    r5 = stat("envelope_test_mask0.5_k1/verify_refine_3_code_loose",
              "envelope_test_mask0.5_k1/direct_code_loose")
    checks.append(check("C3 mask0.5 delta", -0.033, 0.001, lambda: r5["delta"]))
    checks.append(check("C3 mask0.0 breakage 0.234", 0.234, 0.001, lambda: rm["brk"]))

    # --- C5 refutation intervals ---
    rc = stat(f"{T}/incumbent_refine_code_loose", f"{T}/direct_code_loose")
    rmi = stat(f"{T}/incumbent_refine_cot_math_loose", f"{T}/cot_math_loose")
    checks.append(check("C5 code repair 0.060", 0.060, 0.001, lambda: rc["rep"]))
    checks.append(check("C5 math repair 0.032", 0.032, 0.001, lambda: rmi["rep"]))

    # --- Table 2: verifier regimes ---
    tab2 = [("in-domain code", f"{T}/incumbent_refine_code_loose",
             f"{T}/direct_code_loose", 0.000, 0.020),
            ("in-domain math", f"{T}/incumbent_refine_cot_math_loose",
             f"{T}/cot_math_loose", 0.000, 0.007),
            ("OOD math", f"{OD}/incumbent_refine_cot_math_loose",
             f"{OD}/cot_math_loose", 0.025, -0.003),
            ("OOD code no-signal", f"{OD}/incumbent_refine_code_loose",
             f"{OD}/direct_code_loose", 0.114, -0.087),
            ("OOD code restored", f"{TV}/incumbent_refine_code_loose",
             f"{TV}/direct_code_loose", 0.000, 0.005),
            ("BBH new domain", f"{TP}/incumbent_refine_logic_loose",
             f"{TP}/direct_logic_loose", 0.006, 0.025)]
    for name, sd, bd, brk, d in tab2:
        st = stat(sd, bd)
        checks.append(check(f"Tab2 {name} breakage", brk, 0.001, lambda st=st: st["brk"]))
        checks.append(check(f"Tab2 {name} delta", d, 0.001, lambda st=st: st["delta"]))

    # --- Table 3: prospective ---
    tab3 = [("tight vanilla/direct", "verify_refine_3", "direct", "tight", 0.000, 0.026, 0.200),
            ("tight prot/direct", "incumbent_refine", "direct", "tight", 0.000, 0.000, 0.000),
            ("tight prot/cot", "incumbent_refine_cot", "cot", "tight", 0.000, 0.000, 0.000),
            ("loose vanilla/direct", "verify_refine_3", "direct", "loose", 0.025, 0.006, 0.200),
            ("loose prot/direct", "incumbent_refine", "direct", "loose", 0.025, 0.006, 0.200),
            ("loose prot/cot", "incumbent_refine_cot", "cot", "loose", 0.014, 0.003, 0.000)]
    for name, wf, base, tier, d, brk, rep in tab3:
        st = stat(f"{TP}/{wf}_logic_{tier}", f"{TP}/{base}_logic_{tier}")
        checks.append(check(f"Tab3 {name} delta", d, 0.001, lambda st=st: st["delta"]))
        checks.append(check(f"Tab3 {name} breakage", brk, 0.001, lambda st=st: st["brk"]))
        checks.append(check(f"Tab3 {name} repair", rep, 0.001, lambda st=st: st["rep"]))

    # --- prospective baseline p and threshold quoted in caption ---
    def p_tight():
        B = per_task(f"{TP}/direct_logic_tight")
        return sum(B.values()) / len(B)
    checks.append(check("Tab3 caption p=0.919", 0.919, 0.002, p_tight))

    # --- report ---
    bad = [c for c in checks if c[3] != "OK"]
    print(f"{'claim':44s}{'paper':>10s}{'recomputed':>12s}  verdict")
    print("-" * 82)
    for label, claimed, actual, verdict in checks:
        a = f"{actual:.4f}" if isinstance(actual, float) else str(actual)
        mark = "" if verdict == "OK" else "   <<<"
        print(f"{label:44s}{claimed:>10.4f}{a:>12s}  {verdict}{mark}")
    print("-" * 82)
    print(f"{len(checks) - len(bad)}/{len(checks)} claims verified against raw logs")
    if bad:
        print(f"\n{len(bad)} MISMATCHES — fix the paper before submission:")
        for b in bad:
            print(f"  {b[0]}: paper={b[1]} recomputed={b[2]} ({b[3]})")
        sys.exit(1)
    print("\nAll numeric claims in main.tex reproduce from raw per-task logs.")


if __name__ == "__main__":
    main()
