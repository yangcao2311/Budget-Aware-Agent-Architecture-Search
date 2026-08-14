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

    # --- Table 1 (tab:frr): every cell recomputed from false_rejection.py ---
    import false_rejection as FR

    TAB_FRR = {
        # name: (n tasks, FRR, FRR 95% UCB, breakage, slack)
        "math, self-check, tight":   (100, 0.000, 0.030, 0.000, 0.000),
        "code, oracle tests, loose": (107, 0.009, 0.044, 0.000, 0.009),
        "code, oracle tests, tight": (107, 0.009, 0.044, 0.000, 0.009),
        "math, self-check, loose":   (100, 0.013, 0.062, 0.000, 0.013),
        "BBH, self-check, loose":    (104, 0.019, 0.111, 0.006, 0.013),
        "code OOD, tests restored":  (62,  0.048, 0.120, 0.000, 0.048),
        "math OOD, self-check":      (27,  0.049, 0.263, 0.025, 0.025),
        "code, 50% tests, loose":    (107, 0.156, 0.370, 0.146, 0.009),
        "code, NO tests, loose":     (107, 1.000, 1.000, 0.234, 0.766),
        "code OOD, NO tests":        (88,  1.000, 1.000, 0.114, 0.886),
    }
    got = {name: FR.measure(st, ba, tag) for name, st, ba, tag in FR.CONDS}
    for name, (n_t, frr, frr_ub, brk, slack) in TAB_FRR.items():
        m = got.get(name)
        if m is None:
            checks.append((f"Tab1 {name}", 0.0, "absent", "MISSING"))
            continue
        checks.append(check(f"Tab1 {name} n", n_t, 0.5, lambda m=m: m["n_tasks"]))
        checks.append(check(f"Tab1 {name} FRR", frr, 0.001, lambda m=m: m["reject"]))
        checks.append(check(f"Tab1 {name} FRR UCB", frr_ub, 0.001, lambda m=m: m["reject_ub"]))
        checks.append(check(f"Tab1 {name} breakage", brk, 0.001, lambda m=m: m["breakage"]))
        checks.append(check(f"Tab1 {name} slack", slack, 0.001,
                            lambda m=m: m["reject"] - m["breakage"]))

    # the breakage UCB quoted in the mechanism paragraph (zero events, n=107)
    checks.append(check("code/loose breakage UCB", 0.028, 0.001,
                        lambda: got["code, oracle tests, loose"]["breakage_ub"]))


    # --- regeneration-leak quasi-experiment (Table regenleak, appendix) ---
    import regen_leak as RL

    TAB_REGEN = {
        "code":     (111, 0.447, 0.027, 0.613),
        "math":     (119, 0.111, 0.113, 0.417),
        "BBH":      (115, 0.061, 0.061, 0.600),
        "code OOD": (90,  0.560, 0.011, 0.000),
        "math OOD": (47,  0.000, 0.266, 0.138),
    }
    regen_rows = {}
    for label, tag, base, arm in RL.DOMAINS:
        runs = RL.load_solutions(tag, base)
        vals, n_pairs, identical = RL.per_task_leak(runs)
        point, lo, hi = RL.cluster_ci(vals)
        acc_rate, acc_n = RL.accept_given_incumbent_wrong(tag, arm, base)
        regen_rows[label] = dict(n=len(vals), identical=identical / n_pairs if n_pairs else float("nan"),
                                  leak=point, accept=acc_rate)
    for name, (n_t, ident, leak, acc) in TAB_REGEN.items():
        r = regen_rows.get(name)
        if r is None:
            checks.append((f"Regen {name}", 0.0, "absent", "MISSING"))
            continue
        checks.append(check(f"Regen {name} n", n_t, 0.5, lambda r=r: r["n"]))
        checks.append(check(f"Regen {name} identical", ident, 0.001, lambda r=r: r["identical"]))
        checks.append(check(f"Regen {name} leak", leak, 0.001, lambda r=r: r["leak"]))
        checks.append(check(f"Regen {name} accept|Iwrong", acc, 0.001, lambda r=r: r["accept"]))

    # combined accepting-path exposure quoted in main text and appendix
    checks.append(check("Regen code leak*accept", 0.017, 0.001,
                        lambda: regen_rows["code"]["leak"] * regen_rows["code"]["accept"]))
    checks.append(check("Regen math leak*accept", 0.047, 0.001,
                        lambda: regen_rows["math"]["leak"] * regen_rows["math"]["accept"]))

    # --- Table (tab:armc): arm C accepting-path exposure ---
    import subprocess, sys as _sys
    r = subprocess.run([_sys.executable, "scripts/provenance_arm_c_leak.py"],
                        cwd=str(ROOT), capture_output=True, text=True)
    out = r.stdout
    def _grab(pattern):
        import re
        m = re.search(pattern, out)
        return float(m.group(1)) if m else None

    code_leak = _grab(r"code\s*: baseline-correct n=\s*\d+\s*leak=([\d.]+)")
    math_leak = _grab(r"math\s*: baseline-correct n=\s*\d+\s*leak=([\d.]+)")
    code_accept = _grab(r"accept\|I wrong \(oracle feedback tests, zero-cost\) = ([\d.]+)")
    checks.append(check("TabArmC code leak", 0.224, 0.001, lambda: code_leak))
    checks.append(check("TabArmC math leak", 0.053, 0.001, lambda: math_leak))
    checks.append(check("TabArmC code accept|I wrong", 0.490, 0.001, lambda: code_accept))
    checks.append(check("TabArmC code leak*accept", 0.110, 0.001,
                        lambda: (code_leak or 0) * (code_accept or 0)))

    # --- Three-arm provenance causal test (§theory + Limitations) ---
    import subprocess as _sp, sys as _sys2
    r = _sp.run([_sys2.executable, "scripts/provenance_causal_analysis.py"],
                cwd=str(ROOT), capture_output=True, text=True)
    out = r.stdout
    import re as _re
    CAUSAL = {
        "code_arm1_assign": 0.000, "code_arm2_samepolicy": 0.028, "code_arm3_diffpolicy": 0.109,
        "math_arm1_assign": 0.003, "math_arm2_samepolicy": 0.027, "math_arm3_diffpolicy": 0.017,
    }
    # Parse the printed breakage values in order (fam blocks, 3 arms each)
    vals = [float(x) for x in _re.findall(r"breakage=([\d.]+)", out)]
    keys = list(CAUSAL)
    for key, v in zip(keys, vals):
        claimed = CAUSAL[key]
        checks.append(check(f"Causal {key}", claimed, 0.001, lambda v=v: v))

    # --- Table (tab:kimi): Kimi K3 cross-model check, low-error cells only ---
    import json as _json2
    from pathlib import Path as _P
    _EXP = ROOT / "experiments"

    def _kimi_repair_breakage(prot_dir, base_dir):
        def per_task(d):
            acc = {}
            for s in (0, 1, 2):
                fp = _EXP / d / f"results_seed{s}.jsonl"
                for r in map(_json2.loads, open(fp)):
                    if r["status"] not in ("completed", "reserve_rejected"):
                        continue
                    acc.setdefault(r["task_id"], []).append(
                        bool(r.get("success_symbolic", r["success"])))
            return acc
        P, B = per_task(prot_dir), per_task(base_dir)
        common = set(P) & set(B)
        p = sum(sum(v) / len(v) for v in B.values()) / len(B)
        rep_n = rep_d = brk_n = brk_d = 0
        for t in common:
            bv = sum(B[t]) / len(B[t])
            pv = sum(P[t]) / len(P[t])
            if bv < 1.0:
                rep_d += 1; rep_n += pv
            if bv > 0.0:
                brk_d += 1
                if bv == 1.0:
                    brk_n += (1 - pv)
        rep = rep_n / rep_d if rep_d else float("nan")
        brk = brk_n / brk_d if brk_d else float("nan")
        net = sum(sum(v) / len(v) for v in P.values()) / len(P) - p
        return p, rep, brk, net

    KIMI = {
        "code, oracle tests, loose": (
            "kimi_envelope_test/incumbent_refine_code_loose",
            "kimi_envelope_test/direct_code_loose", 0.771, 0.317, 0.000, 0.047),
        "code, oracle tests, tight": (
            "kimi_envelope_test/incumbent_refine_code_tight",
            "kimi_envelope_test/direct_code_tight", 0.773, 0.150, 0.000, 0.000),
        "math, self-check, loose": (
            "kimi_envelope_test/incumbent_refine_cot_math_loose",
            "kimi_envelope_test/cot_math_loose", 0.807, 0.252, 0.008, 0.002),
        "math, self-check, tight": (
            "kimi_envelope_test/incumbent_refine_cot_math_tight",
            "kimi_envelope_test/cot_math_tight", 0.804, 0.216, 0.000, 0.002),
        "BBH, self-check, loose": (
            "kimi_envelope_logic_prospective/incumbent_refine_logic_loose",
            "kimi_envelope_logic_prospective/direct_logic_loose", 0.950, 0.095, 0.014, -0.017),
        "code OOD, no tests": (
            "kimi_envelope_ood/incumbent_refine_code_loose",
            "kimi_envelope_ood/direct_code_loose", 0.867, 0.383, 0.104, -0.087),
        "math OOD, self-check": (
            "kimi_envelope_ood/incumbent_refine_cot_math_loose",
            "kimi_envelope_ood/cot_math_loose", 0.440, 0.186, 0.030, -0.010),
        "code OOD, tests restored": (
            "kimi_envelope_ood_visible/incumbent_refine_code_loose",
            "kimi_envelope_ood_visible/direct_code_loose", 0.926, 0.667, 0.010, 0.025),
        "code, 50% tests, loose": (
            "kimi_envelope_test_mask0.5_k1/verify_refine_3_code_loose",
            "kimi_envelope_test_mask0.5_k1/direct_code_loose", 0.773, 0.358, 0.046, 0.018),
        "code, no tests, loose": (
            "kimi_envelope_test_mask0.0_k1/verify_refine_3_code_loose",
            "kimi_envelope_test_mask0.0_k1/direct_code_loose", 0.773, 0.225, 0.152, -0.103),
    }
    for name, (prot, base, cp, crep, cbrk, cnet) in KIMI.items():
        p, rep, brk, net = _kimi_repair_breakage(prot, base)
        checks.append(check(f"Kimi {name} p", cp, 0.001, lambda p=p: p))
        checks.append(check(f"Kimi {name} repair", crep, 0.001, lambda rep=rep: rep))
        checks.append(check(f"Kimi {name} breakage", cbrk, 0.001, lambda brk=brk: brk))
        checks.append(check(f"Kimi {name} net", cnet, 0.001, lambda net=net: net))

    # --- Kimi effective n / CIs and failure-as-wrong sensitivity ---
    # These are deliberately recomputed by the standalone zero-cost analysis
    # script used to generate Table 7 and its sensitivity table.  The primary
    # view must remain complete-case; the second view scores residual provider
    # errors as wrong and is reported only as a sensitivity analysis.
    import kimi_sensitivity as KS

    KIMI_N_EFF = {
        "code, oracle, loose": 450,
        "code, oracle, tight": 446,
        "math, self-check, loose": 450,
        "math, self-check, tight": 439,
        "BBH, self-check, loose": 360,
        "code OOD, no tests": 300,
        "math OOD, self-check": 300,
        "code OOD, tests restored": 204,
        "code, 50% tests, loose": 450,
        "code, no tests, loose": 438,
    }
    # (repair point/lo/hi, breakage point/lo/hi, net point/lo/hi)
    KIMI_PRIMARY = {
        "code, oracle, loose": ((.317,.192,.450), (0,0,0), (.047,.024,.073)),
        "code, oracle, tight": ((.150,.075,.233), (0,0,0), (0,0,0)),
        "math, self-check, loose": ((.252,.153,.351), (.008,0,.018), (.002,-.011,.016)),
        "math, self-check, tight": ((.216,.135,.306), (0,0,0), (.002,0,.007)),
        "BBH, self-check, loose": ((.095,0,.190), (.014,.003,.029), (-.017,-.031,-.006)),
        "code OOD, no tests": ((.383,.233,.533), (.104,.068,.147), (-.087,-.133,-.040)),
        "math OOD, self-check": ((.186,.118,.255), (.030,.006,.067), (-.010,-.040,.020)),
        "code OOD, tests restored": ((.667,.375,.917), (.010,0,.026), (.025,-.010,.064)),
        "code, 50% tests, loose": ((.358,.233,.483), (.046,.019,.079), (.018,-.020,.056)),
        "code, no tests, loose": ((.225,.125,.333), (.153,.109,.200), (-.101,-.149,-.053)),
    }
    KIMI_FAILURE = {
        "code, oracle, loose": (450, (.317,.192,.450), (.000,.000,.000), (.047,.024,.073)),
        "code, oracle, tight": (450, (.150,.075,.233), (.008,.000,.019), (-.007,-.016,.000)),
        "math, self-check, loose": (450, (.252,.153,.351), (.008,.000,.018), (.002,-.011,.016)),
        "math, self-check, tight": (450, (.216,.135,.306), (.026,.010,.041), (-.020,-.036,-.007)),
        "BBH, self-check, loose": (360, (.095,.000,.190), (.014,.003,.029), (-.017,-.031,-.006)),
        "code OOD, no tests": (300, (.383,.233,.533), (.104,.068,.147), (-.087,-.133,-.040)),
        "math OOD, self-check": (300, (.186,.118,.255), (.030,.006,.067), (-.010,-.040,.020)),
        "code OOD, tests restored": (204, (.667,.375,.917), (.010,.000,.026), (.025,-.010,.064)),
        "code, 50% tests, loose": (450, (.358,.233,.483), (.046,.019,.079), (.018,-.020,.056)),
        "code, no tests, loose": (450, (.217,.125,.317), (.175,.131,.221), (-.124,-.171,-.080)),
    }
    for name, (prot, base) in KS.CONDITIONS.items():
        m = KS.metrics(prot, base, False)
        f = KS.metrics(prot, base, True)
        checks.append(check(f"Kimi effective n {name}", KIMI_N_EFF[name], 0, lambda m=m: m["n_eff"]))
        for label, got, exp in (("repair", m["repair"], KIMI_PRIMARY[name][0]),
                                ("breakage", m["breakage"], KIMI_PRIMARY[name][1]),
                                ("net", m["net"], KIMI_PRIMARY[name][2])):
            for j, suffix in enumerate(("point", "lo", "hi")):
                checks.append(check(f"Kimi primary {name} {label} {suffix}", exp[j], .001,
                                    lambda got=got, j=j: got[j]))
        exp_n, exp_r, exp_b, exp_d = KIMI_FAILURE[name]
        checks.append(check(f"Kimi failure n {name}", exp_n, 0, lambda f=f: f["n_eff"]))
        for label, got, exp in (("repair", f["repair"], exp_r),
                                ("breakage", f["breakage"], exp_b),
                                ("net", f["net"], exp_d)):
            for j, suffix in enumerate(("point", "lo", "hi")):
                checks.append(check(f"Kimi failure {name} {label} {suffix}", exp[j], .001,
                                    lambda got=got, j=j: got[j]))

    # --- Table (tab:bestof3): zero-cost three-sample selection vs reference-preserving ---
    import subprocess as _sp3, sys as _sys3
    r = _sp3.run([_sys3.executable, "scripts/best_of_3_zero_cost.py"],
                 cwd=str(ROOT), capture_output=True, text=True)
    out = r.stdout

    def _block(fam):
        pat = (fr"{fam} \([^)]+\): n=\d+\s+baseline p=([\d.]+)\s+"
               fr"three-sample baseline acc=([\d.]+)\s+\[95% CI [\d.]+, [\d.]+\]\s+"
               fr"reference-preserving acc=([\d.]+)\s+\[95% CI [\d.]+, [\d.]+\]\s+"
               fr"paired diff \(ref - three-sample\) = ([+-][\d.]+)\s+"
               fr"\[95% CI ([+-][\d.]+), ([+-][\d.]+)\]")
        m = re.search(pat, out)
        return [float(x) for x in m.groups()]

    math_p, math_acc, math_ref, math_diff, math_dlo, math_dhi = _block("math")
    code_p, code_acc, code_ref, code_diff, code_dlo, code_dhi = _block("code")

    checks.append(check("BestOf3 code p", 0.727, 0.001, lambda: code_p))
    checks.append(check("BestOf3 code accuracy", 0.733, 0.001, lambda: code_acc))
    checks.append(check("BestOf3 math p", 0.733, 0.001, lambda: math_p))
    checks.append(check("BestOf3 math accuracy", 0.753, 0.001, lambda: math_acc))
    checks.append(check("BestOf3 code reference-preserving accuracy", 0.747, 0.001, lambda: code_ref))
    checks.append(check("BestOf3 math reference-preserving accuracy", 0.740, 0.001, lambda: math_ref))
    checks.append(check("BestOf3 code paired diff", 0.013, 0.001, lambda: code_diff))
    checks.append(check("BestOf3 code paired diff CI lo", -0.007, 0.001, lambda: code_dlo))
    checks.append(check("BestOf3 code paired diff CI hi", 0.036, 0.001, lambda: code_dhi))
    checks.append(check("BestOf3 math paired diff", -0.013, 0.001, lambda: math_diff))
    checks.append(check("BestOf3 math paired diff CI lo", -0.036, 0.001, lambda: math_dlo))
    checks.append(check("BestOf3 math paired diff CI hi", 0.007, 0.001, lambda: math_dhi))

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
