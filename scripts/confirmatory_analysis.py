#!/usr/bin/env python
"""Confirmatory Part-I analysis (3 execution seeds × 120 dev tasks).

Outputs: per-tier tables (seed-averaged), per-seed crossover stability
(H-E1), paired-bootstrap CIs for the key structure-vs-direct contrasts and
the degradation trend (H-E2), and final Fig.2 / Fig.3.
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
TIER_ORDER = ["E1", "E2", "tight", "unseen", "loose"]
SEEDS = [0, 1, 2]


def load_cell(dirname: str, seed: int) -> dict[str, bool]:
    p = EXP / dirname / f"results_seed{seed}.jsonl"
    if not p.exists():
        return {}
    return {r["task_id"]: bool(r["success"])
            for r in map(json.loads, open(p))}


def per_task_avg(dirname: str) -> dict[str, float]:
    acc = defaultdict(list)
    for s in SEEDS:
        for tid, ok in load_cell(dirname, s).items():
            acc[tid].append(ok)
    return {tid: sum(v) / len(v) for tid, v in acc.items()}


def paired_bootstrap(a: dict, b: dict, n_boot=10000, seed=0):
    ids = sorted(set(a) & set(b))
    rng = random.Random(seed)
    da = [a[i] - b[i] for i in ids]
    n = len(da)
    point = sum(da) / n
    boots = sorted(sum(da[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(n_boot))
    return point, boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)], n


def main():
    structures = ["direct", "cot", "vote3", "vote5", "verify_refine_1",
                  "verify_refine_3", "decompose_agg", "vote_verify"]

    print("=" * 72)
    print("H-E1: per-seed argmax stability (envelope, seed-avg table)")
    for fam in ["code", "math"]:
        table = {}
        for tier in TIER_ORDER:
            for s in structures:
                d = per_task_avg(f"envelope/{s}_{fam}_{tier}")
                if d:
                    table[(tier, s)] = sum(d.values()) / len(d)
        print(f"\n--- {fam} (mean over {len(SEEDS)} seeds x 120 tasks) ---")
        print(f"{'structure':18s}" + "".join(f"{t:>9s}" for t in TIER_ORDER))
        for s in structures:
            print(f"{s:18s}" + "".join(
                f"{table.get((t, s), float('nan')):>9.3f}" for t in TIER_ORDER))
        # per-seed argmax sequences
        for seed in SEEDS:
            seq = []
            for tier in TIER_ORDER:
                best, bv = None, -1
                for s in structures:
                    d = load_cell(f"envelope/{s}_{fam}_{tier}", seed)
                    if d:
                        v = sum(d.values()) / len(d)
                        if v > bv:
                            best, bv = s, v
                seq.append(best)
            print(f"seed {seed} argmax: {' -> '.join(x or '-' for x in seq)}")

    print("\n" + "=" * 72)
    print("Key contrasts, paired bootstrap 95% CI (seed-averaged per task)")
    contrasts = [
        ("code", "verify_refine_3 - direct @ unseen",
         "envelope/verify_refine_3_code_unseen", "envelope/direct_code_unseen"),
        ("code", "verify_refine_3 - direct @ loose",
         "envelope/verify_refine_3_code_loose", "envelope/direct_code_loose"),
        ("code", "verify_refine_3 - direct @ tight",
         "envelope/verify_refine_3_code_tight", "envelope/direct_code_tight"),
        ("math", "verify_refine_3 - cot @ tight",
         "envelope/verify_refine_3_math_tight", "envelope/cot_math_tight"),
        ("math", "verify_refine_3 - cot @ loose",
         "envelope/verify_refine_3_math_loose", "envelope/cot_math_loose"),
    ]
    for fam, name, da, db in contrasts:
        A, B = per_task_avg(da), per_task_avg(db)
        if A and B:
            pt, lo, hi, n = paired_bootstrap(A, B)
            sig = "SIG" if (lo > 0 or hi < 0) else "ns"
            print(f"{fam:5s} {name:44s} {pt:+.3f} [{lo:+.3f},{hi:+.3f}] n={n} {sig}")

    print("\n" + "=" * 72)
    print("H-E2: degradation trend, verify_refine_3 - direct (code)")
    for tier in ["unseen", "loose"]:
        row = []
        for mask, tag in [(1.0, "envelope"), (0.5, "envelope_mask0.5_k1"),
                          (0.0, "envelope_mask0.0_k1")]:
            A = per_task_avg(f"{tag}/verify_refine_3_code_{tier}")
            B = per_task_avg(f"{tag}/direct_code_{tier}")
            if A and B:
                pt, lo, hi, n = paired_bootstrap(A, B)
                row.append((mask, pt, lo, hi))
        print(f"@{tier}: " + "  ".join(
            f"mask{m:.1f}: {p:+.3f}[{l:+.3f},{h:+.3f}]" for m, p, l, h in row))
        if len(row) == 3:
            mono = row[0][1] > row[1][1] > row[2][1]
            neg0 = row[2][3] < 0
            print(f"  monotone decrease: {mono}; no-signal harmful (CI<0): {neg0}")

    # -- final figures ------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    for ax, fam in zip(axes, ["code", "math"]):
        for s in structures:
            xs, ys = [], []
            for i, tier in enumerate(TIER_ORDER):
                d = per_task_avg(f"envelope/{s}_{fam}_{tier}")
                if d:
                    xs.append(i)
                    ys.append(sum(d.values()) / len(d))
            if xs:
                ax.plot(xs, ys, marker="o", label=s)
        ax.set_xticks(range(len(TIER_ORDER)), TIER_ORDER)
        ax.set_xlabel("budget tier")
        ax.set_ylabel("success rate (3-seed avg)")
        ax.set_title(f"{fam}: budget-structure envelope")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "paper" / "fig2_envelope_confirmatory.pdf")
    fig.savefig(ROOT / "paper" / "fig2_envelope_confirmatory.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for tier in ["unseen", "loose"]:
        xs, ys, es = [], [], []
        for mask, tag in [(1.0, "envelope"), (0.5, "envelope_mask0.5_k1"),
                          (0.0, "envelope_mask0.0_k1")]:
            A = per_task_avg(f"{tag}/verify_refine_3_code_{tier}")
            B = per_task_avg(f"{tag}/direct_code_{tier}")
            if A and B:
                pt, lo, hi, _ = paired_bootstrap(A, B)
                xs.append(mask)
                ys.append(pt)
                es.append((pt - lo, hi - pt))
        if xs:
            ax.errorbar(xs, ys, yerr=list(zip(*es)), marker="o", capsize=4,
                        label=f"@{tier}")
    ax.axhline(0, color="gray", lw=1, ls="--")
    ax.set_xlabel("visible-test fraction (verifier reliability)")
    ax.set_ylabel("verify_refine_3 - direct (success)")
    ax.set_title("code: structure advantage vs verifier reliability")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "paper" / "fig3_degradation_confirmatory.pdf")
    fig.savefig(ROOT / "paper" / "fig3_degradation_confirmatory.png", dpi=150)
    print("\nFig.2/Fig.3 confirmatory written to paper/")


if __name__ == "__main__":
    main()
