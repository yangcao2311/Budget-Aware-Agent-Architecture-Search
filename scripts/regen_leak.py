#!/usr/bin/env python
"""What the existing seeds already say about regenerating under the same
policy, at zero inference cost.

The provenance claim in \\S4 needs a contrast arm that regenerates its first
draft under an identical policy instead of holding the reference's actual
output. We did not run that arm on purpose, but the response-cache key
includes the execution seed (hbws/llm.py), so the SAME baseline prompt at
temperature 0 was already issued three times per task -- once per seed. Each
ordered pair of seeds on a task is an independent temperature-0 regeneration
of the same prompt: one seed stands in for "the reference's output", another
for "what a regenerating first-draft node would have produced instead".

This does not replace the three-arm causal experiment in Limitations -- the
baseline and the regenerated draft still share prompt and temperature by
construction, so this cannot separate "provenance" from "decoding noise" as a
mechanism. What it gives, at zero cost, is the exposure the accepting path
would actually carry if a workflow regenerated instead of reusing: the rate at
which a fresh temperature-0 draft is wrong on tasks where the reference draft
was right, and (combined with the measured acceptance rate on incumbent-wrong
tasks) an estimate of the accepting-path breakage term in eq. (3) that
reference preservation sets to exactly zero.
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
SEEDS = [0, 1, 2]
NB = 10000

DOMAINS = [
    ("code",     "envelope_test",              "direct_code_loose",     "incumbent_refine_code_loose"),
    ("math",     "envelope_test",              "cot_math_loose",        "incumbent_refine_cot_math_loose"),
    ("BBH",      "envelope_logic_prospective",  "direct_logic_loose",    "incumbent_refine_logic_loose"),
    ("code OOD", "envelope_ood",                "direct_code_loose",     "incumbent_refine_code_loose"),
    ("math OOD", "envelope_ood",                "cot_math_loose",        "incumbent_refine_cot_math_loose"),
]


def load_solutions(tag, name):
    out = defaultdict(dict)
    for s in SEEDS:
        p = EXP / tag / name / f"results_seed{s}.jsonl"
        if not p.exists():
            continue
        for r in map(json.loads, open(p)):
            ok = bool(r.get("success_symbolic", r["success"]))
            out[r["task_id"]][s] = (r.get("solution") or "", ok)
    return out


def per_task_leak(runs):
    """Per task: Pr(a differently-seeded regeneration is wrong | this seed's
    draft is correct), averaged over ordered seed pairs on that task."""
    vals, n_pairs_total, identical = [], 0, 0
    for _, by in runs.items():
        seeds = sorted(by)
        num = den = 0
        for s in seeds:
            for t in seeds:
                if s == t:
                    continue
                (txt_s, ok_s), (txt_t, ok_t) = by[s], by[t]
                n_pairs_total += 1
                identical += (txt_s == txt_t)
                if ok_s:
                    den += 1
                    num += (not ok_t)
        if den:
            vals.append(num / den)
    return vals, n_pairs_total, identical


def cluster_ci(vals, seed=0):
    if not vals:
        return float("nan"), float("nan"), float("nan")
    n = len(vals)
    rng = random.Random(seed)
    point = sum(vals) / n
    boots = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(NB))
    return point, boots[int(0.025 * NB)], boots[int(0.975 * NB)]


def accept_given_incumbent_wrong(tag, arm, base_name):
    """Pr(verifier accepts | incumbent B-wrong), on tasks where the baseline
    fails on every seed -- the exposure the accepting-path leak above would
    need to convert into actual breakage."""
    B = defaultdict(list)
    for s in SEEDS:
        p = EXP / tag / base_name / f"results_seed{s}.jsonl"
        if not p.exists():
            continue
        for r in map(json.loads, open(p)):
            B[r["task_id"]].append(bool(r.get("success_symbolic", r["success"])))
    acc = tot = 0
    for s in SEEDS:
        p = EXP / tag / arm / f"results_seed{s}.jsonl"
        if not p.exists():
            continue
        for r in map(json.loads, open(p)):
            vals = B.get(r["task_id"])
            if not vals or any(vals):
                continue
            types = [t["type"] for t in r.get("trace", [])]
            if "verify" not in types:
                continue
            tot += 1
            acc += ("refine" not in types)
    return (acc / tot, tot) if tot else (float("nan"), 0)


def main():
    print("=" * 92)
    print("Regeneration under an identical policy, recovered from stored seeds (no new inference)")
    print("=" * 92)
    print(f"{'domain':10s}{'tasks':>7s}{'identical text':>16s}{'n ref-correct':>15s}"
          f"{'leak':>9s}{'[95% CI]':>16s}{'accept|I wrong':>16s}{'leak*accept':>13s}")
    rows = {}
    for label, tag, base, arm in DOMAINS:
        runs = load_solutions(tag, base)
        vals, n_pairs, identical = per_task_leak(runs)
        point, lo, hi = cluster_ci(vals)
        acc_rate, acc_n = accept_given_incumbent_wrong(tag, arm, base)
        combo = point * acc_rate if acc_n else float("nan")
        rows[label] = dict(n_tasks=len(vals), identical=identical / n_pairs if n_pairs else float("nan"),
                            n_ref_correct=sum(1 for v in vals), leak=point, lo=lo, hi=hi,
                            accept_given_wrong=acc_rate, accept_n=acc_n, combo=combo)
        print(f"{label:10s}{len(vals):>7d}{identical/n_pairs if n_pairs else float('nan'):>16.3f}"
              f"{sum(1 for v in vals):>15d}{point:>9.3f}  [{lo:.3f},{hi:.3f}]"
              f"{acc_rate:>16.3f}{combo:>13.4f}")
    print("=" * 92)
    print("leak            = Pr(independent T=0 regeneration wrong | reference draft correct)")
    print("accept|I wrong  = Pr(verifier accepts | incumbent, i.e. baseline, wrong), on tasks")
    print("                  where the baseline fails every seed")
    print("leak*accept     = estimated accepting-path breakage a regenerating arm would carry:")
    print("                  Pr(accept, I wrong | B correct) in eq. (3), which reference")
    print("                  preservation sets to exactly zero by holding I = B instead")
    out = ROOT / "experiments" / "regen_leak.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
