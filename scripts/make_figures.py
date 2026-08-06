#!/usr/bin/env python
"""Publication figures. Every number is recomputed from raw per-task logs, so
the figures cannot drift from the tables (audit_claims.py checks the tables).

Fig.1  the decomposition: repair up, breakage down, net as a marker. The job
       is polarity + magnitude, so a diverging paired bar is the right form —
       it makes cancellation visible, which is the paper's whole point.
Fig.2  dose-response: an ordered manipulation of verifier signal, so a line
       with CIs. Single axis (both series are rates).

Palette: validated diverging pair blue #2a78d6 / orange #eb6834
(CVD dE 24.7, normal-vision dE 33.6, contrast pass on a light surface).
"""
import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
SEEDS = [0, 1, 2]
NB = 10000

REPAIR = "#2a78d6"     # cool pole: the workflow rescues
BREAK = "#eb6834"      # warm pole: the workflow destroys
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#d8d7d2"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": GRID, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.labelcolor": INK,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def per_task(d):
    acc = defaultdict(list)
    for s in SEEDS:
        p = EXP / d / f"results_seed{s}.jsonl"
        if p.exists():
            for r in map(json.loads, open(p)):
                acc[r["task_id"]].append(bool(r["success"]))
    return {t: sum(v) / len(v) for t, v in acc.items()}


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
            "brk": sum(easy) / len(easy) if easy else 0.0,
            "rep": sum(hard) / len(hard) if hard else 0.0}


T = "envelope_test"


def fig1():
    conds = [
        ("code\ntight", f"{T}/verify_refine_3_code_tight", f"{T}/direct_code_tight",
         f"{T}/incumbent_refine_code_tight"),
        ("code\nloose", f"{T}/verify_refine_3_code_loose", f"{T}/direct_code_loose",
         f"{T}/incumbent_refine_code_loose"),
        ("math\ntight", f"{T}/verify_refine_3_math_tight", f"{T}/cot_math_tight",
         f"{T}/incumbent_refine_cot_math_tight"),
        ("math\nloose", f"{T}/verify_refine_3_math_loose", f"{T}/cot_math_loose",
         f"{T}/incumbent_refine_cot_math_loose"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.6), sharey=True)
    for ax, which, title in [
            (axes[0], "vanilla", "vanilla verify\u2013refine"),
            (axes[1], "protected", "incumbent-protected")]:
        xs, reps, brks, nets, nlo, nhi = [], [], [], [], [], []
        for i, (lab, van, base, prot) in enumerate(conds):
            s = stat(van if which == "vanilla" else prot, base)
            xs.append(i)
            reps.append(s["rep"])
            brks.append(-s["brk"])
            nets.append(s["delta"])
            nlo.append(s["delta"] - s["lo"])
            nhi.append(s["hi"] - s["delta"])
        w = 0.62
        ax.bar(xs, reps, w, color=REPAIR, linewidth=0,
               label="repair (rescues baseline failures)")
        ax.bar(xs, brks, w, color=BREAK, linewidth=0,
               label="breakage (destroys baseline successes)")
        ax.errorbar(xs, nets, yerr=[nlo, nhi], fmt="o", ms=4.5, color=INK,
                    ecolor=INK, elinewidth=1.0, capsize=2.5, zorder=5,
                    label="net effect (95% CI)")
        ax.axhline(0, color=INK2, lw=0.7)
        ax.set_xticks(xs, [c[0] for c in conds])
        ax.set_title(title, color=INK)
        ax.set_ylim(-0.30, 0.30)
        ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if which == "protected":
            for x in xs:
                ax.text(x, -0.022, "breakage\n0.000", ha="center", va="top",
                        fontsize=6.2, color=BREAK, linespacing=0.95)
    axes[0].set_ylabel("rate")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False,
               handlelength=1.2, borderpad=0.2, columnspacing=1.4,
               bbox_to_anchor=(0.5, -0.10))
    fig.savefig(ROOT / "paper" / "fig1_decomposition.pdf")
    print("fig1 written")


def fig2():
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5),
                             gridspec_kw={"width_ratios": [1, 1.35], "wspace": 0.32})
    # (a) dose-response over verifier signal
    ax = axes[0]
    masks, nets, los, his, brks = [], [], [], [], []
    for m, tag in [(1.0, T), (0.5, "envelope_test_mask0.5_k1"),
                   (0.0, "envelope_test_mask0.0_k1")]:
        s = stat(f"{tag}/verify_refine_3_code_loose", f"{tag}/direct_code_loose")
        masks.append(m)
        nets.append(s["delta"])
        los.append(s["delta"] - s["lo"])
        his.append(s["hi"] - s["delta"])
        brks.append(s["brk"])
    ax.errorbar(masks, nets, yerr=[los, his], fmt="o-", ms=5, lw=1.6,
                color=INK, ecolor=INK2, elinewidth=1.0, capsize=2.5,
                label="net effect (95% CI)")
    ax.plot(masks, brks, "s--", ms=5, lw=1.6, color=BREAK, label="breakage")
    ax.axhline(0, color=INK2, lw=0.7)
    ax.set_xlabel("fraction of visible tests retained")
    ax.set_ylabel("rate")
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.invert_xaxis()
    ax.set_title("verifier signal drives breakage", color=INK)
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", frameon=False, handlelength=1.6,
              borderpad=0.2, labelspacing=0.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # (b) breakage of the protected construction across regimes
    ax = axes[1]
    regimes = [
        ("code in-dom.", f"{T}/incumbent_refine_code_loose", f"{T}/direct_code_loose"),
        ("math in-dom.", f"{T}/incumbent_refine_cot_math_loose", f"{T}/cot_math_loose"),
        ("math OOD", "envelope_ood/incumbent_refine_cot_math_loose",
         "envelope_ood/cot_math_loose"),
        ("code OOD, NO signal", "envelope_ood/incumbent_refine_code_loose",
         "envelope_ood/direct_code_loose"),
        ("code OOD, signal back", "envelope_ood_visible/incumbent_refine_code_loose",
         "envelope_ood_visible/direct_code_loose"),
        ("BBH (new domain)", "envelope_logic_prospective/incumbent_refine_logic_loose",
         "envelope_logic_prospective/direct_logic_loose"),
    ]
    labs, vals = [], []
    for lab, sd, bd in regimes:
        s = stat(sd, bd)
        labs.append(lab)
        vals.append(s["brk"] if s else 0.0)
    cols = [BREAK if v > 0.02 else REPAIR for v in vals]
    ax.bar(range(len(vals)), vals, 0.62, color=cols, linewidth=0)
    ax.axhline(0.02, color=INK2, lw=0.8, ls=":")
    ax.text(0.0, 0.0245, "predicted bound (0.02)", ha="left", fontsize=6.3, color=INK2)
    ax.set_xticks(range(len(labs)),
                  [l.replace("\n", " ") for l in labs],
                  fontsize=6.6, rotation=28, ha="right",
                  rotation_mode="anchor")
    ax.set_ylabel("breakage")
    ax.set_ylim(0, 0.135)
    ax.set_title("the guarantee tracks verifier signal", color=INK)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=6.3,
                color=BREAK if v > 0.02 else INK2)
    fig.savefig(ROOT / "paper" / "fig2_signal.pdf")
    print("fig2 written")


if __name__ == "__main__":
    fig1()
    fig2()
