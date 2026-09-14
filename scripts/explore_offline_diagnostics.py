#!/usr/bin/env python3
"""Exploratory offline diagnostics; does not alter the paper.

The two analyses deliberately remain descriptive.  Search candidates and
seeds are pooled to expose patterns in the stored trajectories, so the output
must not be read as an independent confirmatory estimate.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import risk_replay as RR


def risk_curve(run, family, epsilons):
    sr = json.load(open(EXP / "search" / run / "search_result.json"))
    rows = []
    for c in sr.get("archive", []):
        for tier in ("tight", "loose"):
            row = RR.candidate_metrics(run, family, int(c["cid"]), tier)
            if row:
                rows.append(row)
    out = []
    for eps in epsilons:
        selected = {}
        for row in rows:
            if row["breakage_ucb"] <= eps:
                key = row["tier"]
                old = selected.get(key)
                if old is None or (row["delta_lcb"], -row["usd_per_task_seed"]) > \
                        (old["delta_lcb"], -old["usd_per_task_seed"]):
                    selected[key] = row
        out.append({
            "epsilon": eps,
            "selected": {
                tier: {k: pick[k] for k in
                       ("cid", "delta_lcb", "breakage", "breakage_ucb", "usd_per_task_seed")}
                for tier, pick in selected.items()
            },
        })
    return {"run": run, "family": family, "n_candidate_tier_rows": len(rows),
            "curve": out}


def baseline_records(family):
    out = {}
    base = EXP / "envelope" / f"direct_{family}_E2"
    for path in sorted(base.glob("results_seed*.jsonl")):
        seed = int(path.stem.replace("results_seed", ""))
        for line in path.open():
            row = json.loads(line)
            out[(row["task_id"], seed)] = row
    return out


def refinement_bins(run, family):
    base = baseline_records(family)
    bins = defaultdict(Counter)
    root = EXP / "search" / run
    n_rows = 0
    for path in root.glob("c*_f2_*_es*/results_seed*.jsonl"):
        seed = int(path.stem.replace("results_seed", ""))
        for line in path.open():
            row = json.loads(line)
            key = (row["task_id"], seed)
            if key not in base:
                continue
            n_rows += 1
            b = base[key]
            baseline_ok = bool(b.get("success_symbolic", b.get("success", False)))
            if not baseline_ok:
                continue
            workflow_ok = bool(row.get("success_symbolic", row.get("success", False)))
            n_refines = sum(t.get("type") == "refine" for t in row.get("trace", []))
            same_text = (row.get("solution") or "") == (b.get("solution") or "")
            c = bins[n_refines]
            c["baseline_correct"] += 1
            c["breakage"] += int(not workflow_ok)
            c["same_text"] += int(same_text)
            c["changed_text"] += int(not same_text)
            c["same_text_breakage"] += int(same_text and not workflow_ok)
            c["changed_text_breakage"] += int((not same_text) and not workflow_ok)
    rows = []
    for n_refines, c in sorted(bins.items()):
        b = c["baseline_correct"]
        changed = c["changed_text"]
        rows.append({
            "n_refines": n_refines,
            **dict(c),
            "breakage_rate": c["breakage"] / b if b else None,
            "changed_text_breakage_rate": c["changed_text_breakage"] / changed if changed else None,
        })
    return {"run": run, "family": family, "n_trace_rows": n_rows, "bins": rows,
            "warning": "pooled candidate/seed rows; descriptive only"}


def policy_level_refinement_summary(run, family, min_count=5):
    """Aggregate breakage within each candidate before summarising depth bins."""
    base = baseline_records(family)
    policies = defaultdict(lambda: defaultdict(Counter))
    for path in (EXP / "search" / run).glob("c*_f2_*_es*/results_seed*.jsonl"):
        seed = int(path.stem.replace("results_seed", ""))
        policy = path.parent.name
        for line in path.open():
            row = json.loads(line)
            key = (row["task_id"], seed)
            if key not in base:
                continue
            b = base[key]
            if not bool(b.get("success_symbolic", b.get("success", False))):
                continue
            n_refines = sum(t.get("type") == "refine" for t in row.get("trace", []))
            w = bool(row.get("success_symbolic", row.get("success", False)))
            policies[policy][n_refines]["b"] += 1
            policies[policy][n_refines]["breakage"] += int(not w)
    by_depth = defaultdict(list)
    for policy, bins in policies.items():
        for n_refines, c in bins.items():
            if c["b"] >= min_count:
                by_depth[n_refines].append(c["breakage"] / c["b"])
    return {
        "min_policy_bin_count": min_count,
        "by_depth": {
            str(k): {
                "n_policies": len(v),
                "mean_breakage_rate": sum(v) / len(v),
                "median_breakage_rate": sorted(v)[len(v) // 2],
                "min_breakage_rate": min(v),
                "max_breakage_rate": max(v),
            } for k, v in sorted(by_depth.items())
        },
        "warning": "candidate-level descriptive summary; policies and seeds are dependent",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=EXP / "offline_diagnostics_exploratory.json")
    args = ap.parse_args()
    payload = {
        "risk_curves": [
            risk_curve("A_hbws_code_s0", "code", [.01, .02, .03, .05, .10, .20, .30]),
            risk_curve("A_hbws_math_s0", "math", [.01, .02, .03, .05, .10, .20, .30, .50]),
        ],
        "refinement_bins": [
            {**refinement_bins("A_hbws_code_s0", "code"),
             "policy_level": policy_level_refinement_summary("A_hbws_code_s0", "code")},
            {**refinement_bins("A_hbws_math_s0", "math"),
             "policy_level": policy_level_refinement_summary("A_hbws_math_s0", "math")},
        ],
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out}")
    for block in payload["refinement_bins"]:
        print(block["family"], block["bins"])


if __name__ == "__main__":
    main()
