#!/usr/bin/env python
"""Publication figures. Every number is recomputed from raw per-task logs, so
the figures cannot drift from the tables (audit_claims.py checks the tables).

Fig.1  teaser: the decomposition. Repair up, breakage down, net as a marker.
       The job is polarity + magnitude, so a diverging paired bar is right --
       it makes cancellation visible, which is the paper's whole point.
Fig.3  the budget floor: how repair and breakage each move with budget.
Fig.4  dose-response: an ordered manipulation of verifier signal.
Fig.5  the guarantee across verifier regimes and domains.
Fig.6  search: what the optimiser converged to, and how the archive moved.

(Fig.2 is a TikZ mechanism diagram drawn in main.tex.)

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
                ok = bool(r.get("success_symbolic", r["success"]))
                acc[r["task_id"]].append(ok)
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
BASE = {"code": "direct", "math": "cot"}
INC = {"code": "incumbent_refine", "math": "incumbent_refine_cot"}


def _despine(ax, which=("top", "right")):
    for sp in which:
        ax.spines[sp].set_visible(False)


# --------------------------------------------------------------- Fig. 1 ----
def fig1():
    conds = [("code\n\\$0.10", "code", "tight"), ("code\n\\$0.25", "code", "loose"),
             ("math\n\\$0.10", "math", "tight"), ("math\n\\$0.25", "math", "loose")]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.8), sharey=True)
    for ax, which, title in [(axes[0], "vanilla", "vanilla verify–refine"),
                             (axes[1], "protected", "incumbent-protected")]:
        xs, reps, brks, nets, nlo, nhi = [], [], [], [], [], []
        for i, (lab, fam, tier) in enumerate(conds):
            wf = "verify_refine_3" if which == "vanilla" else INC[fam]
            s = stat(f"{T}/{wf}_{fam}_{tier}", f"{T}/{BASE[fam]}_{fam}_{tier}")
            xs.append(i)
            reps.append(s["rep"])
            brks.append(-s["brk"])
            nets.append(s["delta"])
            nlo.append(s["delta"] - s["lo"])
            nhi.append(s["hi"] - s["delta"])
        ax.bar(xs, reps, 0.62, color=REPAIR, linewidth=0,
               label="repair (rescues baseline failures)")
        ax.bar(xs, brks, 0.62, color=BREAK, linewidth=0,
               label="breakage (destroys baseline successes)")
        ax.errorbar(xs, nets, yerr=[nlo, nhi], fmt="o", ms=4, color=INK,
                    ecolor=INK, elinewidth=1.0, capsize=2.5, zorder=5,
                    label="net effect (95% CI)")
        ax.axhline(0, color=INK2, lw=0.7)
        ax.set_xticks(xs, [c[0] for c in conds], fontsize=7)
        ax.set_title(title, color=INK, pad=3)
        ax.set_ylim(-0.28, 0.28)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        _despine(ax)
        if which == "protected":
            for x in xs:
                ax.text(x, -0.02, "0.000", ha="center", va="top",
                        fontsize=6.2, color=BREAK)
    axes[0].set_ylabel("rate")
    h, l = axes[0].get_legend_handles_labels()
    # Legend sits clear of the tick labels; a one-line tick label plus this
    # offset leaves a visible gap at the figure's cropped bounding box.
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, handlelength=1.2,
               borderpad=0.15, columnspacing=1.3, bbox_to_anchor=(0.5, -0.30))
    fig.savefig(ROOT / "paper" / "fig1_decomposition.pdf")
    print("fig1 written")


# --------------------------------------------------------------- Fig. 3 ----
def fig3():
    """The budget floor: both terms move, and they move in opposite directions."""
    tiers = ["tight", "unseen", "loose"]
    xlab = ["\\$0.10", "\\$0.15", "\\$0.25"]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.85),
                             gridspec_kw={"wspace": 0.3})
    for ax, fam in zip(axes, ["code", "math"]):
        reps, brks, nets, lo, hi = [], [], [], [], []
        for t in tiers:
            s = stat(f"{T}/verify_refine_3_{fam}_{t}", f"{T}/{BASE[fam]}_{fam}_{t}")
            reps.append(s["rep"])
            brks.append(s["brk"])
            nets.append(s["delta"])
            lo.append(s["delta"] - s["lo"])
            hi.append(s["hi"] - s["delta"])
        x = list(range(len(tiers)))
        ax.plot(x, reps, "o-", ms=5, lw=1.7, color=REPAIR, label="repair")
        ax.plot(x, brks, "s-", ms=5, lw=1.7, color=BREAK, label="breakage")
        ax.errorbar(x, nets, yerr=[lo, hi], fmt="D-", ms=4, lw=1.4, color=INK,
                    ecolor=INK2, elinewidth=0.9, capsize=2.5, label="net effect")
        ax.axhline(0, color=INK2, lw=0.7)
        ax.set_xticks(x, xlab)
        ax.set_xlabel("per-task budget")
        ax.set_title(f"{fam}: vanilla verify–refine", color=INK, pad=3)
        ax.grid(color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        _despine(ax)
    axes[0].set_ylabel("rate")
    axes[0].legend(loc="center left", frameon=False, handlelength=1.5,
                   borderpad=0.2, labelspacing=0.25)
    axes[0].annotate("breakage overtakes repair\nbelow the entry fee",
                     xy=(0.06, 0.20), xytext=(0.55, 0.10), fontsize=6.4,
                     color=INK2, ha="left",
                     arrowprops=dict(arrowstyle="->", lw=0.7, color=INK2))
    fig.savefig(ROOT / "paper" / "fig3_budget_floor.pdf")
    print("fig3 written")


# --------------------------------------------------------------- Fig. 4 ----
def fig4():
    fig, ax = plt.subplots(figsize=(3.05, 1.95))
    masks, nets, los, his, brks, reps = [], [], [], [], [], []
    for m, tag in [(1.0, T), (0.5, "envelope_test_mask0.5_k1"),
                   (0.0, "envelope_test_mask0.0_k1")]:
        s = stat(f"{tag}/verify_refine_3_code_loose", f"{tag}/direct_code_loose")
        masks.append(m)
        nets.append(s["delta"])
        los.append(s["delta"] - s["lo"])
        his.append(s["hi"] - s["delta"])
        brks.append(s["brk"])
        reps.append(s["rep"])
    ax.plot(masks, reps, "o-", ms=5, lw=1.7, color=REPAIR, label="repair")
    ax.plot(masks, brks, "s-", ms=5, lw=1.7, color=BREAK, label="breakage")
    ax.errorbar(masks, nets, yerr=[los, his], fmt="D-", ms=4, lw=1.4, color=INK,
                ecolor=INK2, elinewidth=0.9, capsize=2.5, label="net effect")
    ax.axhline(0, color=INK2, lw=0.7)
    ax.set_xlabel("fraction of visible tests retained")
    ax.set_ylabel("rate")
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.invert_xaxis()
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="center left", frameon=False, handlelength=1.5,
              borderpad=0.2, labelspacing=0.25)
    _despine(ax)
    fig.savefig(ROOT / "paper" / "fig4_dose_response.pdf")
    print("fig4 written")


# --------------------------------------------------------------- Fig. 5 ----
def fig5():
    fig, ax = plt.subplots(figsize=(3.05, 1.95))
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
    # Park the annotation in the empty upper-left quadrant (those bars are
    # 0.000) and lead to the rule, so it cannot collide with a value label.
    ax.annotate("predicted bound (0.02)", xy=(1.45, 0.021), xytext=(-0.35, 0.072),
                ha="left", fontsize=6.3, color=INK2,
                arrowprops=dict(arrowstyle="->", lw=0.6, color=INK2,
                                shrinkA=1, shrinkB=1))
    ax.set_xticks(range(len(labs)), labs, fontsize=6.6, rotation=28,
                  ha="right", rotation_mode="anchor")
    ax.set_ylabel("breakage")
    ax.set_ylim(0, 0.135)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    _despine(ax)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=6.3,
                color=BREAK if v > 0.02 else INK2)
    fig.savefig(ROOT / "paper" / "fig5_regimes.pdf")
    print("fig5 written")


# --------------------------------------------------------------- Fig. 6 ----
def fig6():
    """Protocol A on the frozen test split: one budget-contingent policy against
    the static policy searched specifically for each budget.

    We deliberately do NOT plot search-internal scores against each other: the
    budget-contingent score is a cross-tier minimum and the static score is a
    single-tier value, so putting them on one axis would compare incomparable
    quantities. Deployment accuracy at a given tier is comparable, so that is
    what we show.
    """
    FINAL = EXP / "final"

    def policy_cells(run):
        p = FINAL / f"{run}_chosen.json"
        if not p.exists():
            return None
        return json.load(open(p))["cid"]

    def acc(run, cid, tier):
        acc_ = defaultdict(list)
        for s in SEEDS:
            f = FINAL / run / f"c{cid}_test_{tier}" / f"results_seed{s}.jsonl"
            if f.exists():
                for r in map(json.loads, open(f)):
                    acc_[r["task_id"]].append(
                        bool(r.get("success_symbolic", r["success"])))
        if not acc_:
            return None
        per = [sum(v) / len(v) for v in acc_.values()]
        return sum(per) / len(per), per

    fig, ax = plt.subplots(figsize=(5.5, 1.8))
    groups, hb, stv = [], [], []
    for fam, seeds in [("code", [0, 1, 2]), ("math", [0])]:
        for tier in ["tight", "loose"]:
            h, t = [], []
            for sd in seeds:
                hr = f"A_hbws_{fam}_s{sd}"
                sr = f"A_static_{tier}_{fam}_s{sd}"
                hc, sc = policy_cells(hr), policy_cells(sr)
                if hc is None or sc is None:
                    continue
                a, b = acc(hr, hc, tier), acc(sr, sc, tier)
                if a and b:
                    h.append(a[0])
                    t.append(b[0])
            if h:
                groups.append(f"{fam}\n{tier}")
                hb.append(sum(h) / len(h))
                stv.append(sum(t) / len(t))
    x = list(range(len(groups)))
    w = 0.36
    ax.bar([i - w / 2 for i in x], hb, w, color=REPAIR, linewidth=0,
           label="one budget-contingent policy (search cap \\$60)")
    ax.bar([i + w / 2 for i in x], stv, w, color=INK2, linewidth=0,
           label="per-budget static policy (search cap \\$60 each, \\$120 total)")
    for i, (a, b) in enumerate(zip(hb, stv)):
        ax.text(i - w / 2, a + 0.004, f"{a:.3f}", ha="center", fontsize=6.3,
                color=INK2)
        ax.text(i + w / 2, b + 0.004, f"{b:.3f}", ha="center", fontsize=6.3,
                color=INK2)
    ax.set_xticks(x, groups)
    ax.set_ylabel("accuracy (frozen test)")
    ax.set_ylim(0.60, 0.82)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", ncol=2, frameon=False, handlelength=1.2,
              borderpad=0.2, columnspacing=1.5, fontsize=6.8,
              bbox_to_anchor=(0.5, -0.34))
    _despine(ax)
    fig.savefig(ROOT / "paper" / "fig6_protocolA.pdf")
    print(f"fig6 written ({len(groups)} groups)")


if __name__ == "__main__":
    fig1()
    fig3()
    fig4()
    fig5()
    fig6()
