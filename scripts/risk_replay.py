#!/usr/bin/env python3
"""Offline risk-constrained re-ranking of searched workflows.

This script never calls a model.  It joins the per-task outputs written during
HBWS search with the matching direct baseline on the development split, builds
the four-cell (B,W) transition report, and applies a conservative breakage
constraint.  Search/test separation is preserved: only f2 development traces
are read, and no frozen confirmation result is used for selection.

Example:
  python scripts/risk_replay.py --run A_hbws_code_s0 A_hbws_math_s0
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
sys.path.insert(0, str(ROOT))

from hbws.stats import cluster_ratio_upper


def paired_delta_lcb(rows: dict[str, list[tuple[bool, bool]]], alpha: float = 0.05,
                     n_boot: int = 10000, seed: int = 20260902) -> tuple[float, float, float]:
    """Task-clustered percentile bootstrap for Delta = W - B."""
    tasks = sorted(rows)
    if not tasks:
        return float("nan"), float("nan"), float("nan")
    task_diff = [sum(w for _, w, _ in rows[t]) / len(rows[t]) -
                 sum(b for b, _, _ in rows[t]) / len(rows[t]) for t in tasks]
    point = sum(task_diff) / len(task_diff)
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        draws.append(sum(task_diff[rng.randrange(len(task_diff))]
                         for _ in task_diff) / len(task_diff))
    draws.sort()
    lo = draws[max(0, int(alpha / 2 * n_boot))]
    return point, lo, draws[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]


def load_success(path: Path) -> dict[tuple[str, int], bool]:
    out = {}
    if not path.exists():
        return out
    seed = int(path.stem.replace("results_seed", ""))
    for line in path.open():
        row = json.loads(line)
        out[(row["task_id"], seed)] = bool(row.get("success_symbolic", row["success"]))
    return out


def baseline_for(family: str) -> dict[tuple[str, int], bool]:
    base = EXP / "envelope" / f"direct_{family}_E2"
    out = {}
    for p in sorted(base.glob("results_seed*.jsonl")):
        out.update(load_success(p))
    return out


def candidate_metrics(run: str, family: str, cid: int, tier: str) -> dict | None:
    base = baseline_for(family)
    cand_dir = EXP / "search" / run
    files = sorted(cand_dir.glob(f"c{cid}_f2_{tier}_es*/results_seed*.jsonl"))
    if not files:
        return None
    pairs = defaultdict(list)
    costs = []
    for path in files:
        for line in path.open():
            row = json.loads(line)
            key = (row["task_id"], int(path.stem.replace("results_seed", "")))
            if key not in base:
                continue
            rejected = any(t.get("type") == "refine" for t in row.get("trace", []))
            pairs[row["task_id"]].append(
                (base[key], bool(row.get("success_symbolic", row["success"])), rejected))
            costs.append(float(row.get("budget", {}).get("usd", 0.0)))
    if not pairs:
        return None
    # Equal task weighting, with seeds averaged inside each task.
    t00 = sum(sum(1 for b, w, _ in ps if not b and not w) / len(ps) for ps in pairs.values())
    t01 = sum(sum(1 for b, w, _ in ps if not b and w) / len(ps) for ps in pairs.values())
    t10 = sum(sum(1 for b, w, _ in ps if b and not w) / len(ps) for ps in pairs.values())
    t11 = sum(sum(1 for b, w, _ in ps if b and w) / len(ps) for ps in pairs.values())
    n = len(pairs)
    p = (t10 + t11) / n
    r = t01 / (t00 + t01) if t00 + t01 else 0.0
    b = t10 / (t10 + t11) if t10 + t11 else 0.0
    delta, delta_lo, delta_hi = paired_delta_lcb(pairs)
    eligible = {t: sum(1.0 for bb, _, _ in ps if bb) / len(ps) for t, ps in pairs.items()}
    reject = {t: sum(1.0 for bb, _, rej in ps if bb and rej) / len(ps) for t, ps in pairs.items()}
    # A candidate's breakage event is W wrong on a B-correct pair.
    broke = {t: sum(1.0 for bb, ww, _ in ps if bb and not ww) / len(ps) for t, ps in pairs.items()}
    # Restrict numerators and denominator to B-correct seed mass.
    den = sum(eligible[t] for t in pairs)
    frr = sum(reject[t] for t in pairs) / den if den else float("nan")
    brk = sum(broke[t] for t in pairs) / den if den else float("nan")
    return {
        "run": run, "family": family, "cid": cid, "tier": tier,
        "n_tasks": n, "n_seeds": sum(len(v) for v in pairs.values()),
        "p": p, "repair": r, "breakage": b, "delta": delta,
        "delta_lcb": delta_lo, "delta_ucb": delta_hi,
        "frr": frr,
        "frr_ucb": cluster_ratio_upper(
            [reject[t] for t in pairs], [eligible[t] for t in pairs]),
        "breakage_ucb": cluster_ratio_upper(
            [broke[t] for t in pairs], [eligible[t] for t in pairs]),
        "usd_per_task_seed": sum(costs) / len(costs) if costs else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", required=True)
    ap.add_argument("--epsilon", type=float, default=0.05)
    ap.add_argument("--out", default=str(EXP / "risk_replay.json"))
    args = ap.parse_args()
    all_rows = []
    for run in args.run:
        family = "code" if "code" in run else "math"
        sr = json.load(open(EXP / "search" / run / "search_result.json"))
        for c in sr.get("archive", []):
            for tier in ("tight", "loose"):
                row = candidate_metrics(run, family, int(c["cid"]), tier)
                if row:
                    # General candidates do not necessarily satisfy the
                    # reference-preservation provenance premise.  Therefore
                    # the offline selector constrains the directly observed
                    # breakage UCB.  The cheaper FRR certificate is used only
                    # after a candidate has passed the provenance audit.
                    row.update({"j_mean": c.get("j_mean"), "origin": c.get("origin"),
                                "screen_pass": row["breakage_ucb"] <= args.epsilon,
                                "constraint": "empirical_breakage_ucb"})
                    all_rows.append(row)
    # Highest conservative Delta among risk-feasible candidates per family/tier.
    selected = {}
    for row in all_rows:
        key = (row["family"], row["tier"])
        if not row["screen_pass"]:
            continue
        old = selected.get(key)
        if old is None or (row["delta_lcb"], -row["usd_per_task_seed"]) > (old["delta_lcb"], -old["usd_per_task_seed"]):
            selected[key] = row
    payload = {"epsilon": args.epsilon,
               "status": "exploratory_same_data_screen_not_certificate",
               "rows": all_rows,
               "selected": {f"{f}:{t}": v for (f, t), v in selected.items()}}
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out} ({len(all_rows)} candidate-tier rows)")
    for key, row in sorted(selected.items()):
        print(f"selected {key[0]:4s}/{key[1]:5s}: c{row['cid']} "
              f"Delta_LCB={row['delta_lcb']:+.3f} UCB_b={row['breakage_ucb']:.3f} "
              f"b={row['breakage']:.3f} cost={row['usd_per_task_seed']:.5f}")


if __name__ == "__main__":
    main()
