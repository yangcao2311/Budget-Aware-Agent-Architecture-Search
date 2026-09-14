#!/usr/bin/env python
"""Measure the verifier's false-rejection rate and audit the bound

    b <= Pr(reject | incumbent correct)                                   (3)

against observed breakage, across reference-preserving conditions already on
disk. No new inference is required: in these workflows the verifier sees the
stored reference output, so a trace that reaches a refine node records a false
rejection whenever the reference output is correct.

Do not add regenerated-incumbent conditions to CERTIFICATE_CONDS. When I != B,
``baseline correct and draft rejected'' is not the false-rejection estimand in
the certificate, because the independently generated draft may itself be
wrong. Such conditions belong in a verifier-signal ablation, not this table.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
sys.path.insert(0, str(ROOT))

from hbws.stats import cluster_ratio_upper
SEEDS = [0, 1, 2]
NB = 10000


def per_task(dirname):
    acc = defaultdict(list)
    for s in SEEDS:
        p = EXP / dirname / f"results_seed{s}.jsonl"
        if p.exists():
            for r in map(json.loads, open(p)):
                acc[r["task_id"]].append(
                    bool(r.get("success_symbolic", r["success"])))
    return {t: sum(v) / len(v) for t, v in acc.items()}


def measure(struct, base, tag="envelope_test"):
    """Measure transitions on baseline-correct paired executions.

    Point rates and upper bounds target the same ratio of task-cluster means.
    Seeds are averaged within task before the cluster-Hoeffding ratio bound is
    applied, so they are not treated as independent observations.
    """
    baseline_rows = {}
    for s in SEEDS:
        p = EXP / tag / base / f"results_seed{s}.jsonl"
        if p.exists():
            for r in map(json.loads, open(p)):
                baseline_rows[(r["task_id"], s)] = bool(
                    r.get("success_symbolic", r["success"]))
    per_task_reject = defaultdict(list)
    per_task_break = defaultdict(list)
    per_task_eligible = defaultdict(list)
    for s in SEEDS:
        p = EXP / tag / struct / f"results_seed{s}.jsonl"
        if not p.exists():
            continue
        for r in map(json.loads, open(p)):
            key = (r["task_id"], s)
            if key not in baseline_rows:
                continue
            eligible = baseline_rows[key]
            per_task_eligible[r["task_id"]].append(1.0 if eligible else 0.0)
            if not eligible:
                per_task_reject[r["task_id"]].append(0.0)
                per_task_break[r["task_id"]].append(0.0)
                continue
            types = [t["type"] for t in r.get("trace", [])]
            if "verify" not in types:
                per_task_reject[r["task_id"]].append(0.0)
                per_task_break[r["task_id"]].append(0.0)
                continue
            per_task_reject[r["task_id"]].append(1.0 if "refine" in types else 0.0)
            ok = bool(r.get("success_symbolic", r["success"]))
            per_task_break[r["task_id"]].append(0.0 if ok else 1.0)
    tasks = sorted(per_task_eligible)
    if not tasks:
        return None
    den = [sum(per_task_eligible[t]) / len(per_task_eligible[t]) for t in tasks]
    rej = [sum(per_task_reject[t]) / len(per_task_reject[t]) for t in tasks]
    brk = [sum(per_task_break[t]) / len(per_task_break[t]) for t in tasks]
    reject = sum(rej) / sum(den)
    breakage = sum(brk) / sum(den)
    return {
        "n_tasks": len(tasks),
        "n_tasks_eligible": sum(1 for x in den if x > 0),
        "n_eligible": sum(den),
        "reject": reject,
        "reject_ub": cluster_ratio_upper(rej, den),
        "breakage": breakage,
        "breakage_ub": cluster_ratio_upper(brk, den),
    }


CERTIFICATE_CONDS = [
    ("code, oracle tests, loose", "incumbent_refine_code_loose",
     "direct_code_loose", "envelope_test"),
    ("code, oracle tests, tight", "incumbent_refine_code_tight",
     "direct_code_tight", "envelope_test"),
    ("math, self-check, loose", "incumbent_refine_cot_math_loose",
     "cot_math_loose", "envelope_test"),
    ("math, self-check, tight", "incumbent_refine_cot_math_tight",
     "cot_math_tight", "envelope_test"),
    ("math OOD, self-check", "incumbent_refine_cot_math_loose",
     "cot_math_loose", "envelope_ood"),
    ("code OOD, NO tests", "incumbent_refine_code_loose",
     "direct_code_loose", "envelope_ood"),
    ("code OOD, tests restored", "incumbent_refine_code_loose",
     "direct_code_loose", "envelope_ood_visible"),
    ("BBH, self-check, loose", "incumbent_refine_logic_loose",
     "direct_logic_loose", "envelope_logic_prospective"),
]

# Backwards-compatible name used by the claim audit and figure generator.
CONDS = CERTIFICATE_CONDS


def main():
    print("=" * 88)
    print("Auditing the derived bound  breakage <= Pr(reject | reference correct)")
    print("Rates and one-sided 95% bounds target E[N_t]/E[D_t], with")
    print("execution seeds averaged inside each independent task cluster.")
    print("=" * 88)
    print(f"{'condition':30s}{'tasks':>7s}{'false-rej':>11s}{'[95% ub]':>10s}"
          f"{'breakage':>10s}{'[95% ub]':>10s}{'  bound':>8s}")
    rows, violations = [], []
    for name, st, ba, tag in CONDS:
        m = measure(st, ba, tag)
        if not m:
            print(f"{name:30s}  (no data)")
            continue
        holds = m["breakage"] <= m["reject"] + 1e-9
        if not holds:
            violations.append(name)
        rows.append((name, m))
        print(f"{name:30s}{m['n_tasks']:>7d}{m['reject']:>11.3f}"
              f"{m['reject_ub']:>10.3f}{m['breakage']:>10.3f}"
              f"{m['breakage_ub']:>10.3f}{'  holds' if holds else '  VIOLATED':>8s}")
    print("=" * 88)
    if violations:
        print(f"BOUND VIOLATED in {len(violations)}: " + "; ".join(violations))
    else:
        span = [m["reject"] for _, m in rows]
        print(f"Bound is respected in {len(rows)}/{len(rows)} eligible conditions, over the "
              f"full false-rejection range {min(span):.3f} to {max(span):.3f}.")
        tight = [(n, m) for n, m in rows if 0.05 < m["reject"] < 0.95]
        for n, m in tight:
            print(f"  near-tight: {n} -- breakage {m['breakage']:.3f} against "
                  f"bound {m['reject']:.3f}")
    json.dump([{"condition": n, **m} for n, m in rows],
              open(EXP / "false_rejection_table.json", "w"), indent=2)
    print(f"written to {EXP / 'false_rejection_table.json'}")


if __name__ == "__main__":
    main()


def latex_table():
    """The decision-relevant table: how many baseline-correct tasks each
    estimate rests on, the verifier's false-rejection rate, observed breakage,
    its upper confidence bound, and the slack the bound leaves."""
    rows = []
    for name, st, ba, tag in CONDS:
        m = measure(st, ba, tag)
        if m:
            rows.append((name, m))
    rows.sort(key=lambda r: r[1]["reject"])
    print("\n% --- LaTeX table ---")
    for name, m in rows:
        slack = m["reject"] - m["breakage"]
        print(f"{name} & {m['n_tasks']} & ${m['reject']:.3f}$ & ${m['reject_ub']:.3f}$ "
              f"& ${m['breakage']:.3f}$ & ${slack:.3f}$ \\\\")
