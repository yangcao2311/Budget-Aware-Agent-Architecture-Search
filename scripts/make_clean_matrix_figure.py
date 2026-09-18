#!/usr/bin/env python3
"""Generate the primary repair/breakage figure from the clean GLM matrix."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "experiments/glm4flash_repaired_matrix_final_20260910_analysis.json"
OUT = ROOT / "paper/fig1_decomposition.pdf"

REPAIR = "#2a78d6"
BREAK = "#eb6834"
INK = "#0b0b0b"
GRID = "#d8d7d2"
ARMS = ("arm1_assign", "arm2_samepolicy", "arm3_diffpolicy")
LABELS = ("assign", "same-policy", "different-policy\nstress test")


def main() -> None:
    report = json.loads(ANALYSIS.read_text())
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 7.5, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 6.8, "ytick.labelsize": 7,
        "axes.edgecolor": GRID, "axes.linewidth": .6,
        "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": .02,
    })
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 3.05), sharex=True, sharey=True)
    for ax, key in zip(axes.flat, ("code/tight", "code/loose",
                                  "math/tight", "math/loose")):
        cell = report["cells"][key]
        x = list(range(3))
        repairs = [cell["arms"][a]["repair_rate"] for a in ARMS]
        breaks = [-cell["arms"][a]["breakage_rate"] for a in ARMS]
        delta = [cell["arms"][a]["accuracy_delta_ci95"][0] for a in ARMS]
        lo = [d - cell["arms"][a]["accuracy_delta_ci95"][1]
              for d, a in zip(delta, ARMS)]
        hi = [cell["arms"][a]["accuracy_delta_ci95"][2] - d
              for d, a in zip(delta, ARMS)]
        ax.bar(x, repairs, .62, color=REPAIR, linewidth=0,
               label="repair rate")
        ax.bar(x, breaks, .62, color=BREAK, linewidth=0,
               label="breakage rate")
        ax.errorbar(x, delta, yerr=[lo, hi], fmt="o", ms=3.8, color=INK,
                    ecolor=INK, elinewidth=.9, capsize=2.2, zorder=5,
                    label="accuracy change (95% CI)")
        ax.axhline(0, color="#555555", lw=.65)
        ax.grid(axis="y", color=GRID, lw=.5)
        ax.set_axisbelow(True)
        if key == "math/tight":
            ax.set_facecolor("#f1f1f1")
            ax.set_title("math / tight (censored)")
        else:
            ax.set_title(key.replace("/", " / "))
        ax.set_xticks(x, LABELS, rotation=12, ha="right")
        ax.set_ylim(-.58, .58)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("rate / accuracy change")
    axes[1, 0].set_ylabel("rate / accuracy change")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, frameon=False, loc="lower center",
               bbox_to_anchor=(.5, -.045), columnspacing=1.2, handlelength=1.2)
    fig.subplots_adjust(wspace=.17, hspace=.28, bottom=.19)
    fig.savefig(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
