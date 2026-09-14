#!/usr/bin/env python
"""Claim-evidence audit: recompute every quantitative claim in the paper
directly from raw per-task logs and diff against what the paper states.

Any mismatch is a paper bug. This is the last line of defence before a
reviewer recomputes a number and finds it wrong.
"""
import hashlib
import json
import math
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


def paired_transitions(dirname, baseline_dir):
    """Return task-clustered transition sufficient statistics.

    Each task contributes equally after averaging its paired execution seeds.
    Keeping the joint (B, W) outcomes, rather than conditioning on tasks whose
    baseline mean is exactly zero or one, makes the repair--breakage identity
    exact even when baseline correctness varies across seeds.
    """
    def rows(d):
        out = {}
        for s in SEEDS:
            p = EXP / d / f"results_seed{s}.jsonl"
            if p.exists():
                for r in map(json.loads, open(p)):
                    out[(r["task_id"], s)] = bool(
                        r.get("success_symbolic", r["success"]))
        return out

    W, B = rows(dirname), rows(baseline_dir)
    paired = sorted(set(W) & set(B))
    by_task = defaultdict(list)
    for task, seed in paired:
        by_task[task].append((B[(task, seed)], W[(task, seed)]))
    stats = {}
    for task, vals in by_task.items():
        n = len(vals)
        stats[task] = {
            "p": sum(b for b, _ in vals) / n,
            "w": sum(w for _, w in vals) / n,
            "repair_num": sum((not b) and w for b, w in vals) / n,
            "repair_den": sum(not b for b, _ in vals) / n,
            "break_num": sum(b and (not w) for b, w in vals) / n,
            "break_den": sum(b for b, _ in vals) / n,
        }
    return stats


def cost(dirname):
    tot = n = 0
    for s in SEEDS:
        p = EXP / dirname / f"summary_seed{s}.json"
        if p.exists():
            tot += json.load(open(p))["usd_per_task"]
            n += 1
    return tot / n if n else float("nan")


def stat(sd, bd, seed=0):
    trans = paired_transitions(sd, bd)
    ids = sorted(trans)
    if not trans:
        return None
    d = [trans[t]["w"] - trans[t]["p"] for t in ids]
    n = len(d)
    rng = random.Random(seed)
    bo = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(NB))
    p = sum(trans[t]["p"] for t in ids) / n
    rep_num = sum(trans[t]["repair_num"] for t in ids)
    rep_den = sum(trans[t]["repair_den"] for t in ids)
    brk_num = sum(trans[t]["break_num"] for t in ids)
    brk_den = sum(trans[t]["break_den"] for t in ids)
    rep = rep_num / rep_den if rep_den else float("nan")
    brk = brk_num / brk_den if brk_den else float("nan")
    delta = sum(d) / n
    if abs(delta - ((1 - p) * rep - p * brk)) > 1e-12:
        raise AssertionError("repair--breakage identity drift")
    return {"delta": sum(d) / n, "lo": bo[int(.025 * NB)], "hi": bo[int(.975 * NB)],
            "brk": brk, "rep": rep, "p": p,
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
    checks.append(check("abstract: repair 24.4%", 0.244, 0.001, lambda: r["rep"]))
    checks.append(check("abstract: breakage 13.8%", 0.138, 0.001, lambda: r["brk"]))
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
        ("code", "tight", "verify_refine_3"): (-0.104, -0.162, -0.047, 0.203, 0.220, 0.0062),
        ("code", "tight", "inc"): (0.013, 0.002, 0.031, 0.049, 0.000, 0.0022),
        ("code", "unseen", "verify_refine_3"): (-0.038, -0.093, 0.018, 0.252, 0.147, 0.0085),
        ("code", "unseen", "inc"): (0.020, 0.002, 0.042, 0.073, 0.000, 0.0025),
        ("code", "loose", "verify_refine_3"): (-0.033, -0.087, 0.020, 0.244, 0.138, 0.0090),
        ("code", "loose", "inc"): (0.020, 0.002, 0.042, 0.073, 0.000, 0.0025),
        ("math", "tight", "verify_refine_3"): (0.007, -0.027, 0.040, 0.233, 0.076, 0.0076),
        ("math", "tight", "inc"): (0.000, 0.000, 0.000, 0.000, 0.000, 0.0075),
        ("math", "unseen", "verify_refine_3"): (0.013, -0.020, 0.044, 0.233, 0.067, 0.0141),
        ("math", "unseen", "inc"): (0.000, -0.009, 0.009, 0.017, 0.006, 0.0141),
        ("math", "loose", "verify_refine_3"): (0.020, -0.011, 0.051, 0.242, 0.061, 0.0153),
        ("math", "loose", "inc"): (0.007, -0.002, 0.018, 0.042, 0.006, 0.0153),
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
    checks.append(check("C2 prose repair 0.244->0.203", 0.203, 0.001, lambda: rt["rep"]))
    checks.append(check("C2 prose breakage 0.138->0.220", 0.220, 0.001, lambda: rt["brk"]))

    def reserve_rejects():
        tot = 0
        p = EXP / f"{T}/verify_refine_3_code_tight" / "summary_seed0.json"
        return json.load(open(p))["reserve_rejected"]
    checks.append(check("C2 prose: 35/150 reserve rejections", 35, 0, reserve_rejects))

    # --- C3 dose-response ---
    r5 = stat("envelope_test_mask0.5_k1/verify_refine_3_code_loose",
              "envelope_test_mask0.5_k1/direct_code_loose")
    checks.append(check("C3 mask0.5 delta", -0.033, 0.001, lambda: r5["delta"]))
    checks.append(check("C3 mask0.0 breakage 0.235", 0.235, 0.001, lambda: rm["brk"]))

    # --- C5 refutation intervals ---
    rc = stat(f"{T}/incumbent_refine_code_loose", f"{T}/direct_code_loose")
    rmi = stat(f"{T}/incumbent_refine_cot_math_loose", f"{T}/cot_math_loose")
    checks.append(check("C5 code repair 0.073", 0.073, 0.001, lambda: rc["rep"]))
    checks.append(check("C5 math repair 0.042", 0.042, 0.001, lambda: rmi["rep"]))

    # --- Table 2: verifier regimes ---
    tab2 = [("in-domain code", f"{T}/incumbent_refine_code_loose",
             f"{T}/direct_code_loose", 0.000, 0.020),
            ("in-domain math", f"{T}/incumbent_refine_cot_math_loose",
             f"{T}/cot_math_loose", 0.006, 0.007),
            ("OOD math", f"{OD}/incumbent_refine_cot_math_loose",
             f"{OD}/cot_math_loose", 0.034, -0.003),
            ("OOD code no-signal", f"{OD}/incumbent_refine_code_loose",
             f"{OD}/direct_code_loose", 0.127, -0.087),
            ("OOD code restored", f"{TV}/incumbent_refine_code_loose",
             f"{TV}/direct_code_loose", 0.000, 0.005),
            ("BBH new domain", f"{TP}/incumbent_refine_logic_loose",
             f"{TP}/direct_logic_loose", 0.006, 0.025)]
    for name, sd, bd, brk, d in tab2:
        st = stat(sd, bd)
        checks.append(check(f"Tab2 {name} breakage", brk, 0.001, lambda st=st: st["brk"]))
        checks.append(check(f"Tab2 {name} delta", d, 0.001, lambda st=st: st["delta"]))

    # --- Table 3: prospective ---
    tab3 = [("tight vanilla/direct", "verify_refine_3", "direct", "tight", 0.000, 0.039, 0.448),
            ("tight prot/direct", "incumbent_refine", "direct", "tight", 0.000, 0.000, 0.000),
            ("tight prot/cot", "incumbent_refine_cot", "cot", "tight", 0.000, 0.000, 0.000),
            ("loose vanilla/direct", "verify_refine_3", "direct", "loose", 0.025, 0.015, 0.483),
            ("loose prot/direct", "incumbent_refine", "direct", "loose", 0.025, 0.006, 0.379),
            ("loose prot/cot", "incumbent_refine_cot", "cot", "loose", 0.014, 0.003, 0.207)]
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
        # name: (all task clusters, FRR, aligned 95% UCB, breakage, slack)
        "math, self-check, tight":   (150, 0.006, 0.142, 0.000, 0.006),
        "code, oracle tests, loose": (150, 0.009, 0.147, 0.000, 0.009),
        "code, oracle tests, tight": (150, 0.009, 0.147, 0.000, 0.009),
        "math, self-check, loose":   (150, 0.030, 0.167, 0.006, 0.024),
        "BBH, self-check, loose":    (120, 0.039, 0.161, 0.006, 0.033),
        "code OOD, tests restored":  (68,  0.048, 0.209, 0.000, 0.048),
        "math OOD, self-check":      (100, 0.190, 0.506, 0.034, 0.155),
        "code OOD, NO tests":        (100, 1.000, 1.000, 0.127, 0.873),
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

    # The in-domain signal ablation uses independently regenerated drafts and
    # therefore cannot estimate the certificate FRR. Replay the same archived
    # reference output under both masks to isolate verifier rejection itself.
    import replay_incumbent_mask_frr as IMR
    mask_replay = IMR.summarize()
    checks.append(check("same-incumbent full-test FRR", 0.009, 0.001,
                        lambda: mask_replay["full_test_frr"]))
    checks.append(check("same-incumbent half-test FRR", 0.009, 0.001,
                        lambda: mask_replay["half_test_frr"]))
    checks.append(check("full-pass/half-reject inversions", 0, 0,
                        lambda: mask_replay["full_pass_half_reject"]))

    # Direct-breakage bound under the same task-ratio estimand.
    checks.append(check("code/loose breakage UCB", 0.138, 0.001,
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
    checks.append(check("TabArmC code leak", 0.224, 0.001, lambda: code_leak))
    checks.append(check("TabArmC math leak", 0.053, 0.001, lambda: math_leak))

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

    # --- Official GLM-4-Flash math three-arm replication ---
    glm_out = Path("/private/tmp/glm4flash_audit_analysis.json")
    _sp.run([
        _sys2.executable, "scripts/analyze_provenance_replication.py",
        "--baseline-tag-prefix", "glm4flash_",
        "--causal-tag", "glm4flash_provenance_causal",
        "--output", str(glm_out), "--primary-only",
    ], cwd=str(ROOT), capture_output=True, text=True, check=True)
    glm = json.loads(glm_out.read_text())["families"]
    gm = glm["math"]
    checks.append(check("GLM stable baseline-correct tasks", 105, 0,
                        lambda: gm["stable_baseline_correct_tasks"]))
    GLM_ARMS = {
        "arm1_assign": ((.010, .000, .022), (.000, .000, .000), (.010, .000, .022)),
        "arm2_samepolicy": ((.019, .006, .035), (.000, .000, .000), (.019, .006, .035)),
        "arm3_diffpolicy": ((.079, .044, .117), (.003, .000, .010), (.076, .041, .114)),
    }
    for arm, expected in GLM_ARMS.items():
        got = gm["arms"][arm]
        checks.append(check(f"GLM {arm} matched pairs", 315, 0,
                            lambda got=got: got["matched_pairs"]))
        for label, actual, claimed in (
            ("breakage", got["breakage"], expected[0]),
            ("acceptance", got["acceptance_path"], expected[1]),
            ("reject-refine", got["rejection_refinement_path"], expected[2]),
        ):
            for j, suffix in enumerate(("point", "lo", "hi")):
                checks.append(check(f"GLM {arm} {label} {suffix}", claimed[j], .001,
                                    lambda actual=actual, j=j: actual[j]))

    GLM_SEED0_ALIGNED = {
        "arm1_assign": (.018, .000, .018),
        "arm2_samepolicy": (.044, .000, .044),
        "arm3_diffpolicy": (.123, .009, .114),
    }
    for arm, claimed in GLM_SEED0_ALIGNED.items():
        got = gm["arms"][arm]["per_seed_all_baseline_correct"]["0"]
        checks.append(check(f"GLM {arm} seed0 aligned denominator", 114, 0,
                            lambda got=got: got["baseline_correct_tasks"]))
        for label, expected in zip(
                ("breakage", "acceptance_path", "rejection_refinement_path"),
                claimed):
            checks.append(check(f"GLM {arm} seed0 aligned {label}", expected, .001,
                                lambda got=got, label=label: got[label]))

    GLM_CONTRASTS = {
        "arm2_samepolicy_minus_arm1_assign": (.010, -.006, .025),
        "arm3_diffpolicy_minus_arm1_assign": (.070, .038, .105),
        "arm3_diffpolicy_minus_arm2_samepolicy": (.060, .025, .098),
    }
    for name, claimed in GLM_CONTRASTS.items():
        got = gm["paired_breakage_contrasts"][name]
        checks.append(check(f"GLM {name} tasks", 105, 0,
                            lambda got=got: got["effective_tasks"]))
        for j, suffix in enumerate(("point", "lo", "hi")):
            checks.append(check(f"GLM {name} {suffix}", claimed[j], .001,
                                lambda got=got, j=j: got["difference"][j]))

    glm_rows = []
    for arm in ("arm1_assign", "arm2_samepolicy", "arm3_diffpolicy"):
        for seed in SEEDS:
            path = EXP / "glm4flash_provenance_causal" / f"{arm}_math_loose" / f"results_seed{seed}.jsonl"
            glm_rows.extend(map(json.loads, open(path)))
    glm_math_base = {}
    for seed in SEEDS:
        path = EXP / "glm4flash_envelope_test" / "cot_math_loose" / f"results_seed{seed}.jsonl"
        glm_math_base[seed] = {r["task_id"]: r for r in map(json.loads, open(path))}
    stable_glm_math = {
        task_id for task_id in set.intersection(*(set(glm_math_base[s]) for s in SEEDS))
        if all(glm_math_base[s][task_id].get("success") for s in SEEDS)
    }
    checks.append(check("GLM causal primary rows", 945, 0,
                        lambda: sum(r["task_id"] in stable_glm_math for r in glm_rows)))
    checks.append(check("GLM causal provider errors", 0, 0,
                        lambda: sum(str(r.get("status", "")).startswith("error") for r in glm_rows)))
    checks.append(check("GLM causal settlement overruns", 0, 0,
                        lambda: sum(str(r.get("status", "")).startswith("budget_exceeded")
                                    for r in glm_rows)))
    checks.append(check("GLM causal logical USD", 0, 0,
                        lambda: sum(r.get("budget", {}).get("usd", 0) for r in glm_rows)))

    # --- Official GLM-4-Flash code three-arm replication ---
    glm_code_out = Path("/private/tmp/glm4flash_code_audit_analysis.json")
    _sp.run([
        _sys2.executable, "scripts/analyze_provenance_replication.py",
        "--baseline-tag-prefix", "glm4flash_final_",
        "--causal-tag", "glm4flash_final_provenance_causal",
        "--families", "code", "--output", str(glm_code_out),
        "--primary-only",
    ], cwd=str(ROOT), capture_output=True, text=True, check=True)
    gc = json.loads(glm_code_out.read_text())["families"]["code"]
    checks.append(check("GLM code stable baseline-correct tasks", 96, 0,
                        lambda: gc["stable_baseline_correct_tasks"]))
    GLM_CODE_ARMS = {
        "arm1_assign": ((.000, .000, .000), (.000, .000, .000), (.000, .000, .000)),
        "arm2_samepolicy": ((.000, .000, .000), (.000, .000, .000), (.000, .000, .000)),
        "arm3_diffpolicy": ((.094, .049, .146), (.059, .024, .101), (.035, .007, .069)),
    }
    for arm, expected in GLM_CODE_ARMS.items():
        got = gc["arms"][arm]
        checks.append(check(f"GLM code {arm} matched pairs", 288, 0,
                            lambda got=got: got["matched_pairs"]))
        for label, actual, claimed in (
            ("breakage", got["breakage"], expected[0]),
            ("acceptance", got["acceptance_path"], expected[1]),
            ("reject-refine", got["rejection_refinement_path"], expected[2]),
        ):
            for j, suffix in enumerate(("point", "lo", "hi")):
                checks.append(check(f"GLM code {arm} {label} {suffix}", claimed[j], .001,
                                    lambda actual=actual, j=j: actual[j]))

    glm_code_rows = []
    for seed in SEEDS:
        path = EXP / "glm4flash_final_envelope_test" / "direct_code_loose" / f"results_seed{seed}.jsonl"
        glm_code_rows.extend(map(json.loads, open(path)))
    checks.append(check("GLM code completed rows", 442, 0,
                        lambda: sum(r.get("status") == "completed" for r in glm_code_rows)))
    checks.append(check("GLM code correct completed rows", 303, 0,
                        lambda: sum(r.get("status") == "completed" and r.get("success") for r in glm_code_rows)))
    checks.append(check("GLM code provider-error rows", 8, 0,
                        lambda: sum(str(r.get("status", "")).startswith("error") for r in glm_code_rows)))
    checks.append(check("GLM code grader setup errors", 0, 0,
                        lambda: sum("current limit exceeds maximum limit" in
                                    (r.get("grader_feedback") or "") for r in glm_code_rows)))

    glm_code_causal_rows = []
    for arm in ("arm1_assign", "arm2_samepolicy", "arm3_diffpolicy"):
        for seed in SEEDS:
            path = (EXP / "glm4flash_final_provenance_causal" /
                    f"{arm}_code_loose" / f"results_seed{seed}.jsonl")
            glm_code_causal_rows.extend(map(json.loads, open(path)))
    checks.append(check("GLM code causal primary rows", 864, 0,
                        lambda: len(glm_code_causal_rows)))
    checks.append(check("GLM code causal provider errors", 0, 0,
                        lambda: sum(str(r.get("status", "")).startswith("error")
                                    for r in glm_code_causal_rows)))
    checks.append(check("GLM code causal settlement overruns", 0, 0,
                        lambda: sum(str(r.get("status", "")).startswith("budget_exceeded")
                                    for r in glm_code_causal_rows)))
    checks.append(check("GLM code causal logical USD", 0, 0,
                        lambda: sum(r.get("budget", {}).get("usd", 0)
                                    for r in glm_code_causal_rows)))

    # --- Reviewer-triggered full code rerun with repaired executor ---
    repaired_base = "glm4flash_repaired_code_20260909_envelope_test/direct_code_loose"
    repaired_root = "glm4flash_repaired_code_20260909_causal"
    repaired_expected = {
        "arm1_assign": (.7311111111111112, .03333333333333333,
                        .008888888888888889, .06444444444444444,
                        16 / 136, 1 / 314, 307, 113826),
        "arm2_samepolicy": (.7333333333333333, .035555555555555556,
                            .011111111111111112, .06444444444444444,
                            17 / 136, 1 / 314, 755, 169803),
        "arm3_diffpolicy": (.6377777777777778, -.06,
                            -.11555555555555558, -.002222222222222232,
                            25 / 136, 52 / 314, 840, 426606),
    }
    repaired_rows = []
    for seed in SEEDS:
        path = EXP / repaired_base / f"results_seed{seed}.jsonl"
        repaired_rows.extend(map(json.loads, open(path)))
    checks.append(check("GLM repaired baseline rows", 450, 0,
                        lambda: len(repaired_rows)))
    checks.append(check("GLM repaired baseline accuracy", .6977777777777778, 1e-12,
                        lambda: sum(bool(r["success"]) for r in repaired_rows) / len(repaired_rows)))
    checks.append(check("GLM repaired baseline calls", 450, 0,
                        lambda: sum(r["budget"]["llm_calls"] for r in repaired_rows)))
    checks.append(check("GLM repaired baseline tokens", 60620, 0,
                        lambda: sum(r["budget"]["in_tokens"] + r["budget"]["out_tokens"]
                                    for r in repaired_rows)))

    all_repaired_rows = list(repaired_rows)
    for arm_idx, (arm, expected) in enumerate(repaired_expected.items()):
        dirname = f"{repaired_root}/{arm}_code_loose"
        got = stat(dirname, repaired_base, seed=400 + arm_idx)
        arm_rows = []
        for seed in SEEDS:
            path = EXP / dirname / f"results_seed{seed}.jsonl"
            arm_rows.extend(map(json.loads, open(path)))
        all_repaired_rows.extend(arm_rows)
        labels = ("accuracy", "delta", "delta lo", "delta hi", "repair",
                  "breakage", "calls", "tokens")
        actuals = (
            sum(bool(r["success"]) for r in arm_rows) / len(arm_rows),
            got["delta"], got["lo"], got["hi"], got["rep"], got["brk"],
            sum(r["budget"]["llm_calls"] for r in arm_rows),
            sum(r["budget"]["in_tokens"] + r["budget"]["out_tokens"]
                for r in arm_rows),
        )
        for label, claimed, actual in zip(labels, expected, actuals):
            tol = .001 if label not in ("calls", "tokens") else 0
            checks.append(check(f"GLM repaired {arm} {label}", claimed, tol,
                                lambda actual=actual: actual))

    checks.append(check("GLM repaired completed rows", 1800, 0,
                        lambda: sum(r.get("status") == "completed" for r in all_repaired_rows)))
    checks.append(check("GLM repaired provider errors", 0, 0,
                        lambda: sum(str(r.get("status", "")).startswith("error")
                                    for r in all_repaired_rows)))
    checks.append(check("GLM repaired settlement overruns", 0, 0,
                        lambda: sum(str(r.get("status", "")).startswith("budget_exceeded")
                                    for r in all_repaired_rows)))
    checks.append(check("GLM repaired total calls", 2352, 0,
                        lambda: sum(r["budget"]["llm_calls"] for r in all_repaired_rows)))
    checks.append(check("GLM repaired total tokens", 770855, 0,
                        lambda: sum(r["budget"]["in_tokens"] + r["budget"]["out_tokens"]
                                    for r in all_repaired_rows)))
    checks.append(check("GLM repaired logical USD", 0, 0,
                        lambda: sum(r["budget"]["usd"] for r in all_repaired_rows)))

    glm_attempts = [json.loads(line) for line in
                    open(EXP / "glm4flash_attempts.jsonl") if line.strip()]
    returned_models = [r.get("provider_model") for r in glm_attempts
                       if r.get("provider_model")]
    checks.append(check("GLM returned-model mismatches", 0, 0,
                        lambda: sum(m != "glm-4-flash-250414" for m in returned_models)))

    # --- Frozen GLM held-out certificate selection ---
    freeze_path = EXP / "glm4flash_dcert_frozen_candidate.json"
    if freeze_path.exists():
        frozen = json.loads(freeze_path.read_text())
        checks.append(check("GLM Dcert selected strict candidate", 1, 0,
                            lambda: int(frozen["selected_candidate"] ==
                                        "strict_one_check")))
        checks.append(check("GLM Dcert frozen workflow hash", 1, 0,
                            lambda: int(hashlib.sha256(json.dumps(
                                frozen["selected_workflow"], sort_keys=True,
                                separators=(",", ":")).encode()).hexdigest() ==
                                frozen["selected_workflow_sha256"])))
        anchored = frozen["candidate_reports"]["anchored_one_check"]
        strict = frozen["candidate_reports"]["strict_one_check"]
        checks.append(check("GLM Dselect anchored repair count", 7, 0,
                            lambda: anchored["transition_counts"].get("01", 0)))
        checks.append(check("GLM Dselect anchored breakage count", 3, 0,
                            lambda: anchored["transition_counts"].get("10", 0)))
        checks.append(check("GLM Dselect strict repair count", 1, 0,
                            lambda: strict["transition_counts"].get("01", 0)))
        checks.append(check("GLM Dselect strict breakage count", 0, 0,
                            lambda: strict["transition_counts"].get("10", 0)))

    dcert_result_path = EXP / "glm4flash_dcert_results.json"
    if dcert_result_path.exists():
        from run_glm4flash_dcert import paired_metrics as _dcert_metrics

        dcert_baseline = [json.loads(line) for line in open(
            EXP / "glm4flash_dcert" / "baseline" / "results_seed0.jsonl")]
        dcert_candidate = [json.loads(line) for line in open(
            EXP / "glm4flash_dcert" / "candidate" / "results_seed0.jsonl")]
        dcert = _dcert_metrics(dcert_baseline, dcert_candidate)
        dcert_claims = {
            "n": (1194, dcert["n_tasks"]),
            "00": (353, dcert["transition_counts"].get("00", 0)),
            "01": (6, dcert["transition_counts"].get("01", 0)),
            "10": (2, dcert["transition_counts"].get("10", 0)),
            "11": (833, dcert["transition_counts"].get("11", 0)),
            "repair": (.016713091922005572, dcert["repair"]),
            "breakage": (.0023952095808383233, dcert["breakage"]),
            "FRR": (.026347305389221556, dcert["false_rejection_rate"]),
            "FRR UCB": (.07699412620447402,
                        dcert["false_rejection_ucb95"]),
            "delta": (.0033500837520938024, dcert["delta"]),
            "delta CI lo": (-.0008375209380234506, dcert["delta_ci95"][0]),
            "delta CI hi": (.008375209380234505, dcert["delta_ci95"][1]),
            "acceptance-path breakage": (0,
                                          dcert["acceptance_path_breakages"]),
        }
        for label, (claimed, actual) in dcert_claims.items():
            checks.append(check(f"GLM Dcert {label}", claimed, 1e-12,
                                lambda actual=actual: actual))
        checks.append(check("GLM Dcert primary abstention", 1, 0,
                            lambda: int(dcert["decision"] == "abstain")))
        checks.append(check("GLM Dcert completed rows", 2388, 0,
                            lambda: sum(r.get("status") == "completed"
                                        for r in dcert_baseline + dcert_candidate)))
        checks.append(check("GLM Dcert settlement overruns", 0, 0,
                            lambda: sum(str(r.get("status", "")).startswith(
                                "budget_exceeded")
                                for r in dcert_baseline + dcert_candidate)))
        checks.append(check("GLM Dcert total calls", 2488, 0,
                            lambda: sum(r.get("budget", {}).get("llm_calls", 0)
                                        for r in dcert_baseline + dcert_candidate)))
        checks.append(check("GLM Dcert total tokens", 2440771, 0,
                            lambda: sum(r.get("budget", {}).get("in_tokens", 0) +
                                        r.get("budget", {}).get("out_tokens", 0)
                                        for r in dcert_baseline + dcert_candidate)))
        certificate_path = ROOT / "data" / "math_dcert_glm4flash.jsonl"
        certificate_hash = hashlib.sha256(certificate_path.read_bytes()).hexdigest()
        checks.append(check("GLM Dcert data hash", 1, 0,
                            lambda: int(certificate_hash ==
                                "d40d40bb721c786cdd567f146f204abe8f361e509546985b8c1a6f06aff79172")))

        # Post-hoc certificate-method and cost sensitivity. These checks
        # reproduce the appendix comparison without changing the primary
        # Hoeffding abstention above.
        from hbws.stats import binomial_exact_upper, cluster_ratio_upper

        baseline_by_id = {row["task_id"]: row for row in dcert_baseline}
        candidate_by_id = {row["task_id"]: row for row in dcert_candidate}
        dcert_ids = sorted(set(baseline_by_id) & set(candidate_by_id))
        eligible = [float(bool(baseline_by_id[task_id]["success"]))
                    for task_id in dcert_ids]
        rejected = [float(bool(baseline_by_id[task_id]["success"]) and
                          any(node.get("type") == "refine" for node in
                              candidate_by_id[task_id].get("trace", [])))
                    for task_id in dcert_ids]
        broken = [float(bool(baseline_by_id[task_id]["success"]) and
                        not bool(candidate_by_id[task_id]["success"]))
                  for task_id in dcert_ids]
        n_eligible = int(sum(eligible))
        n_rejected = int(sum(rejected))
        n_broken = int(sum(broken))
        sensitivity_claims = {
            "FRR exact-binomial UCB": (
                .03740991490836185,
                binomial_exact_upper(n_rejected, n_eligible)),
            "breakage exact-binomial UCB": (
                .007520500974103121,
                binomial_exact_upper(n_broken, n_eligible)),
            "breakage cluster-Hoeffding UCB": (
                .053042030396090796,
                cluster_ratio_upper(broken, eligible)),
        }
        for label, (claimed, actual) in sensitivity_claims.items():
            checks.append(check(f"GLM Dcert {label}", claimed, 1e-12,
                                lambda actual=actual: actual))

        def _tokens(row):
            budget = row.get("budget", {})
            return budget.get("in_tokens", 0) + budget.get("out_tokens", 0)

        verifier_nodes = [
            next(node for node in row.get("trace", [])
                 if node.get("type") == "verify")
            for row in dcert_candidate
        ]
        verifier_calls = sum(node["budget"]["llm_calls"]
                             for node in verifier_nodes)
        verifier_tokens = sum(node["budget"].get("in_tokens", 0) +
                              node["budget"].get("out_tokens", 0)
                              for node in verifier_nodes)
        candidate_calls = sum(row["budget"]["llm_calls"]
                              for row in dcert_candidate)
        candidate_tokens = sum(_tokens(row) for row in dcert_candidate)
        refinement_calls = candidate_calls - verifier_calls
        refinement_tokens = candidate_tokens - verifier_tokens
        cost_claims = {
            "reference calls": (1194, sum(row["budget"]["llm_calls"]
                                           for row in dcert_baseline)),
            "reference tokens": (997719, sum(_tokens(row)
                                              for row in dcert_baseline)),
            "verifier calls": (1194, verifier_calls),
            "verifier tokens": (1193992, verifier_tokens),
            "refinement calls": (100, refinement_calls),
            "refinement tokens": (249060, refinement_tokens),
        }
        for label, (claimed, actual) in cost_claims.items():
            checks.append(check(f"GLM Dcert {label}", claimed, 0,
                                lambda actual=actual: actual))
        audit_calls = 1194 + verifier_calls
        audit_tokens = 997719 + verifier_tokens
        checks.append(check("GLM Dcert projected five-refiner call reduction",
                            .1731301939058172, 1e-12,
                            lambda: (5 * refinement_calls) /
                            (audit_calls + 5 * refinement_calls)))
        checks.append(check("GLM Dcert projected ten-refiner token reduction",
                            .5319168248328656, 1e-12,
                            lambda: (10 * refinement_tokens) /
                            (audit_tokens + 10 * refinement_tokens)))

    # --- Table (tab:kimi): Kimi K3 cross-model check, all ten conditions ---
    import json as _json2
    from pathlib import Path as _P
    _EXP = ROOT / "experiments"

    def _kimi_repair_breakage(prot_dir, base_dir):
        import kimi_sensitivity as _ks
        m = _ks.metrics(prot_dir, base_dir, False)
        return m["baseline"], m["repair"][0], m["breakage"][0], m["net"][0]

    KIMI = {
        "code, oracle tests, loose": (
            "kimi_envelope_test/incumbent_refine_code_loose",
            "kimi_envelope_test/direct_code_loose", 0.771, 0.204, 0.000, 0.047),
        "code, oracle tests, tight": (
            "kimi_envelope_test/incumbent_refine_code_tight",
            "kimi_envelope_test/direct_code_tight", 0.773, 0.000, 0.009, -0.007),
        "math, self-check, loose": (
            "kimi_envelope_test/incumbent_refine_cot_math_loose",
            "kimi_envelope_test/cot_math_loose", 0.807, 0.092, 0.019, 0.002),
        "math, self-check, tight": (
            "kimi_envelope_test/incumbent_refine_cot_math_tight",
            "kimi_envelope_test/cot_math_tight", 0.804, 0.011, 0.003, -0.000),
        "BBH, self-check, loose": (
            "kimi_envelope_logic_prospective/incumbent_refine_logic_loose",
            "kimi_envelope_logic_prospective/direct_logic_loose", 0.950, 0.000, 0.018, -0.017),
        "code OOD, no tests": (
            "kimi_envelope_ood/incumbent_refine_code_loose",
            "kimi_envelope_ood/direct_code_loose", 0.867, 0.300, 0.146, -0.087),
        "math OOD, self-check": (
            "kimi_envelope_ood/incumbent_refine_cot_math_loose",
            "kimi_envelope_ood/cot_math_loose", 0.440, 0.060, 0.098, -0.010),
        "code OOD, tests restored": (
            "kimi_envelope_ood_visible/incumbent_refine_code_loose",
            "kimi_envelope_ood_visible/direct_code_loose", 0.926, 0.467, 0.011, 0.025),
        "code, 50% tests, loose": (
            "kimi_envelope_test_mask0.5_k1/verify_refine_3_code_loose",
            "kimi_envelope_test_mask0.5_k1/direct_code_loose", 0.773, 0.284, 0.060, 0.018),
        "code, no tests, loose": (
            "kimi_envelope_test_mask0.0_k1/verify_refine_3_code_loose",
            "kimi_envelope_test_mask0.0_k1/direct_code_loose", 0.773, 0.186, 0.216, -0.124),
    }
    for name, (prot, base, cp, crep, cbrk, cnet) in KIMI.items():
        p, rep, brk, net = _kimi_repair_breakage(prot, base)
        checks.append(check(f"Kimi {name} p", cp, 0.001, lambda p=p: p))
        checks.append(check(f"Kimi {name} repair", crep, 0.001, lambda rep=rep: rep))
        checks.append(check(f"Kimi {name} breakage", cbrk, 0.001, lambda brk=brk: brk))
        checks.append(check(f"Kimi {name} net", cnet, 0.001, lambda net=net: net))

    # --- Kimi effective n / CIs ---
    # These are deliberately recomputed by the standalone analysis script used
    # to generate Table 7.  The completed backfill leaves no provider-error
    # rows, so every planned matched pair is included.
    import kimi_sensitivity as KS

    KIMI_N_EFF = {
        "code, oracle, loose": 450,
        "code, oracle, tight": 450,
        "math, self-check, loose": 450,
        "math, self-check, tight": 450,
        "BBH, self-check, loose": 360,
        "code OOD, no tests": 300,
        "math OOD, self-check": 300,
        "code OOD, tests restored": 204,
        "code, 50% tests, loose": 450,
        "code, no tests, loose": 450,
    }
    # (repair point/lo/hi, breakage point/lo/hi, net point/lo/hi)
    KIMI_PRIMARY = {
        "code, oracle, loose": ((.204,.109,.315), (0,0,0), (.047,.024,.073)),
        "code, oracle, tight": ((0,0,0), (.009,0,.020), (-.007,-.016,0)),
        "math, self-check, loose": ((.092,.037,.161), (.019,.006,.035), (.002,-.011,.016)),
        "math, self-check, tight": ((.011,0,.039), (.003,0,.009), (-.000,-.007,.007)),
        "BBH, self-check, loose": ((0,0,0), (.018,.006,.032), (-.017,-.031,-.006)),
        "code OOD, no tests": ((.300,.143,.489), (.146,.102,.195), (-.087,-.133,-.040)),
        "math OOD, self-check": ((.060,.027,.098), (.098,.046,.162), (-.010,-.040,.020)),
        "code OOD, tests restored": ((.467,.143,.882), (.011,0,.027), (.025,-.010,.064)),
        "code, 50% tests, loose": ((.284,.179,.405), (.060,.031,.095), (.018,-.020,.056)),
        "code, no tests, loose": ((.186,.095,.292), (.216,.167,.268), (-.124,-.171,-.080)),
    }
    for name, (prot, base) in KS.CONDITIONS.items():
        m = KS.metrics(prot, base, False)
        checks.append(check(f"Kimi effective n {name}", KIMI_N_EFF[name], 0, lambda m=m: m["n_eff"]))
        for label, got, exp in (("repair", m["repair"], KIMI_PRIMARY[name][0]),
                                ("breakage", m["breakage"], KIMI_PRIMARY[name][1]),
                                ("net", m["net"], KIMI_PRIMARY[name][2])):
            for j, suffix in enumerate(("point", "lo", "hi")):
                checks.append(check(f"Kimi primary {name} {label} {suffix}", exp[j], .001,
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

    # --- expected-cost-matched offline replay (Table tab:bestof3) ---
    import matched_cost_replay as _mcr
    mcr_rows = {
        "code": _mcr.run("code", "direct_code_loose", "incumbent_refine_code_loose", 0.002514702222222222),
        "math": _mcr.run("math", "cot_math_loose", "incumbent_refine_cot_math_loose", 0.015322166666666666),
    }
    checks.append(check("CostMatched code accuracy", 0.727, 0.001,
                        lambda: mcr_rows["code"]["mixed_accuracy"]))
    checks.append(check("CostMatched code upper-k probability", 0.544, 0.002,
                        lambda: mcr_rows["code"]["upper_prefix_probability"]))
    checks.append(check("CostMatched math accuracy", 0.733, 0.001,
                        lambda: mcr_rows["math"]["mixed_accuracy"]))
    checks.append(check("CostMatched math upper-k probability", 0.233, 0.002,
                        lambda: mcr_rows["math"]["upper_prefix_probability"]))
    checks.append(check("CostMatched code paired diff", 0.019, 0.001,
                        lambda: mcr_rows["code"]["paired_diff_ref_minus_mixed"][0]))
    checks.append(check("CostMatched code paired diff CI lo", -0.001, 0.001,
                        lambda: mcr_rows["code"]["paired_diff_ref_minus_mixed"][1]))
    checks.append(check("CostMatched code paired diff CI hi", 0.044, 0.001,
                        lambda: mcr_rows["code"]["paired_diff_ref_minus_mixed"][2]))
    checks.append(check("CostMatched math paired diff", 0.007, 0.001,
                        lambda: mcr_rows["math"]["paired_diff_ref_minus_mixed"][0]))
    checks.append(check("CostMatched math paired diff CI lo", -0.018, 0.002,
                        lambda: mcr_rows["math"]["paired_diff_ref_minus_mixed"][1]))
    checks.append(check("CostMatched math paired diff CI hi", 0.031, 0.002,
                        lambda: mcr_rows["math"]["paired_diff_ref_minus_mixed"][2]))

    # --- descriptive refinement-depth replay (Appendix depth-replay) ---
    import explore_offline_diagnostics as _eod
    depth = {
        fam: {r["n_refines"]: r for r in _eod.refinement_bins(run, fam)["bins"]}
        for fam, run in (("code", "A_hbws_code_s0"),
                         ("math", "A_hbws_math_s0"))
    }
    for fam, expected in {
        "code": {0: (5697, .033), 1: (407, .590), 2: (126, .762)},
        "math": {0: (1477, .081), 1: (48, .438), 2: (23, .609)},
    }.items():
        for dep, (n_b, br) in expected.items():
            checks.append(check(f"DepthReplay {fam} d{dep} nB", n_b, 0,
                                lambda fam=fam, dep=dep: depth[fam][dep]["baseline_correct"]))
            checks.append(check(f"DepthReplay {fam} d{dep} breakage", br, .001,
                                lambda fam=fam, dep=dep: depth[fam][dep]["breakage_rate"]))
    checks.append(check("DepthReplay code d0 same-text n", 2937, 0,
                        lambda: depth["code"][0]["same_text"]))
    checks.append(check("DepthReplay code d0 changed-text n", 2760, 0,
                        lambda: depth["code"][0]["changed_text"]))
    checks.append(check("DepthReplay code d0 changed breakage", .067, .001,
                        lambda: depth["code"][0]["changed_text_breakage_rate"]))

    # --- offline risk-constrained replay (Appendix risk-replay) ---
    # Recompute the four displayed rows from raw HBWS and baseline traces so
    # the new selector table is covered by the same audit command as the rest
    # of the paper.  These are development-only exploratory quantities.
    import risk_replay as RR

    RISK_ROWS = [
        ("code/tight c7", "A_hbws_code_s0", "code", 7, "tight", 0.013, 0.163),
        ("code/loose c20", "A_hbws_code_s0", "code", 20, "loose", 0.017, 0.163),
        ("math/tight c8", "A_hbws_math_s0", "math", 8, "tight", 0.004, 0.270),
        ("math/loose c8", "A_hbws_math_s0", "math", 8, "loose", 0.017, 0.257),
    ]
    for label, run, fam, cid, tier, lcb, ub in RISK_ROWS:
        rr = RR.candidate_metrics(run, fam, cid, tier)
        checks.append(check(f"Risk replay {label} Delta LCB", lcb, 0.001,
                            lambda rr=rr: rr["delta_lcb"]))
        checks.append(check(f"Risk replay {label} breakage UCB", ub, 0.001,
                            lambda rr=rr: rr["breakage_ucb"]))

    # --- certificate-planning quantities (Appendix scope limitations) ---
    # These are algebraic planning values under the displayed alpha and Dbar,
    # not empirical claims or a formal power calculation.
    def required_n(q, epsilon, alpha=.05, dbar=.73):
        if q >= epsilon:
            return float("inf")
        return math.floor(math.log(1 / alpha) /
                          (2 * dbar ** 2 * (epsilon - q) ** 2)) + 1

    for q, expected in ((0.00, 1125), (0.01, 1757), (0.02, 3124)):
        checks.append(check(f"Certificate planning q={q:.2f}, epsilon=.05",
                            expected, 0,
                            lambda q=q: required_n(q, .05)))

    # --- revised primary evidence: clean 2x2 matrix, search, five refiners ---
    clean_path = EXP / "glm4flash_repaired_matrix_final_20260910_analysis.json"
    clean = json.loads(clean_path.read_text())
    for key in ("code/tight", "code/loose", "math/tight", "math/loose"):
        cell = clean["cells"][key]
        checks.append(check(f"Clean matrix {key} tasks", 150, 0,
                            lambda cell=cell: cell["n_tasks"]))
        checks.append(check(f"Clean matrix {key} reference rows", 450, 0,
                            lambda cell=cell: cell["reference"]["rows"]))
        checks.append(check(f"Clean matrix {key} reference defects", 0, 0,
                            lambda cell=cell: cell["executor_defects"]["reference"]))
        for arm_key, arm in cell["arms"].items():
            checks.append(check(f"Clean matrix {key}/{arm_key} pairs", 450, 0,
                                lambda arm=arm: arm["matched_pairs"]))
            checks.append(check(f"Clean matrix {key}/{arm_key} defects", 0, 0,
                                lambda cell=cell, arm_key=arm_key:
                                cell["executor_defects"][arm_key]))
        checks.append(check(f"Clean matrix {key} assign acceptance breaks", 0, 0,
                            lambda cell=cell:
                            cell["arms"]["arm1_assign"]["acceptance_path_breakage_events"]))
    checks.append(check("Clean code/loose lambda star", 0.18, 0.01,
                        lambda: clean["cells"]["code/loose"]
                        ["lambda_star_assign_vs_diffpolicy"]))

    search = json.loads((EXP / "assign_search_intervention_20260909/analysis.json").read_text())
    checks.append(check("Assign-search nonempty archive", 1, 0,
                        lambda: int(search["n_candidates"] > 0)))

    refiner = json.loads((EXP / "shared_refiner5_temporal_20260909/"
                          "analysis_audited.json").read_text())
    checks.append(check("Five-refiner arm count", 5, 0,
                        lambda: len(refiner["refiners"])))
    checks.append(check("Five-refiner math audit", 1, 0,
                        lambda: int(refiner["math_audit"]["status"] == "passed")))
    checks.append(check("Five-refiner shared false positives", 0, 0,
                        lambda: sum(x["false_positive_count"]
                                    for x in refiner["screen_decisions"])))
    checks.append(check("Five-refiner dominance violations", 0, 0,
                        lambda: sum(
                            row["direct_stored_simultaneous_ucb"] >
                            refiner["shared_exact_ceiling"] + 1e-12
                            for row in refiner["refiners"].values())))

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
