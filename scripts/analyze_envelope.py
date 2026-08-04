#!/usr/bin/env python
"""Analyze envelope results: per-tier best structures, crossovers (H-E1
signal), and the Fig.2 v0 plot (success vs actual $/task, per family).

Usage: python scripts/analyze_envelope.py [experiments/envelope_summary.jsonl]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TIER_ORDER = ["E1", "E2", "tight", "unseen", "loose"]


def load(path):
    rows = [json.loads(l) for l in open(path)]
    # keep the latest row per (family, tier, structure)
    latest = {}
    for r in rows:
        latest[(r["family"], r["tier"], r["structure"])] = r
    return list(latest.values())


def crossover_report(rows):
    by_fam = defaultdict(dict)
    for r in rows:
        by_fam[r["family"]][(r["tier"], r["structure"])] = r
    for fam, d in sorted(by_fam.items()):
        print(f"\n=== {fam} ===")
        tiers = [t for t in TIER_ORDER if any(k[0] == t for k in d)]
        structures = sorted({k[1] for k in d})
        header = f"{'structure':18s}" + "".join(f"{t:>10s}" for t in tiers)
        print(header)
        for s in structures:
            cells = []
            for t in tiers:
                r = d.get((t, s))
                cells.append(f"{r['success_rate']:.3f}" if r else "-")
            print(f"{s:18s}" + "".join(f"{c:>10s}" for c in cells))
        print("-" * len(header))
        best_seq = []
        for t in tiers:
            cand = [(d[(t, s)]["success_rate"], s) for s in structures if (t, s) in d]
            if cand:
                sr, s = max(cand)
                best_seq.append((t, s, sr))
        print("best per tier: " + "  ".join(f"{t}:{s}({sr:.3f})" for t, s, sr in best_seq))
        changes = sum(1 for a, b in zip(best_seq, best_seq[1:]) if a[1] != b[1])
        print(f"crossovers (argmax changes along budget axis): {changes}  "
              f"[H-E1 needs >=2 in >=1 family, stable over 3 seeds]")


def plot(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed — skipping Fig.2 v0 plot)")
        return
    fams = sorted({r["family"] for r in rows})
    fig, axes = plt.subplots(1, len(fams), figsize=(6.4 * len(fams), 4.8))
    axes = [axes] if len(fams) == 1 else list(axes)
    for ax, fam in zip(axes, fams):
        by_struct = defaultdict(list)
        for r in rows:
            if r["family"] == fam:
                by_struct[r["structure"]].append(r)
        for s, rs in sorted(by_struct.items()):
            rs.sort(key=lambda r: TIER_ORDER.index(r["tier"]))
            xs = [TIER_ORDER.index(r["tier"]) for r in rs]
            ys = [r["success_rate"] for r in rs]
            ax.plot(xs, ys, marker="o", label=s)
        ax.set_xticks(range(len(TIER_ORDER)), TIER_ORDER)
        ax.set_xlabel("budget tier")
        ax.set_ylabel("success rate")
        ax.set_title(f"{fam}: budget–structure envelope (dev, seed 0)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = ROOT / "paper" / "fig2_v0_envelope.pdf"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    print(f"\nFig.2 v0 written to {out}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else ROOT / "experiments" / "envelope_summary.jsonl"
    rows = load(path)
    print(f"{len(rows)} (family,tier,structure) cells loaded")
    crossover_report(rows)
    plot(rows)
