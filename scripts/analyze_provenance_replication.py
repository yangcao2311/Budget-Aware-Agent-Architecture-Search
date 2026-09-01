#!/usr/bin/env python
"""Cross-model three-arm provenance analysis with path attribution.

The primary population is tasks whose model-specific reference baseline is
correct in all three completed replicates. Rates average replicates within a
task, then average tasks; bootstrap resamples tasks. Provider errors are not
silently scored as model failures in the primary view.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
SEEDS = (0, 1, 2)
ARM_NAMES = ("arm1_assign", "arm2_samepolicy", "arm3_diffpolicy")
BASE_SUFFIX = {"code": "envelope_test/direct_code_loose",
               "math": "envelope_test/cot_math_loose"}


def load(run_dir: str) -> dict:
    out = defaultdict(dict)
    for seed in SEEDS:
        path = EXP / run_dir / f"results_seed{seed}.jsonl"
        if not path.exists():
            continue
        for row in map(json.loads, open(path)):
            out[row["task_id"]][seed] = row
    return out


def valid(row: dict | None) -> bool:
    if not row:
        return False
    return row.get("status") in ("completed", "reserve_rejected")


def path_of(row: dict) -> str:
    """Classify a final wrong answer by its executed graph path."""
    if row.get("success"):
        return "correct"
    trace = row.get("trace") or []
    types = [x.get("type") for x in trace]
    completed_verify = any(x.get("type") == "verify" and
                           "reserve_rejected" not in x for x in trace)
    if "refine" in types:
        return "rejection_refinement"
    if completed_verify:
        return "acceptance"
    return "unclassified"


def mean_ci(values: list[float], seed: int = 0, n_boot: int = 10000):
    if not values:
        return [None, None, None]
    point = sum(values) / len(values)
    rng = random.Random(seed)
    n = len(values)
    boots = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(n_boot))
    return [point, boots[int(.025 * n_boot)], boots[int(.975 * n_boot)]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-tag-prefix", required=True)
    ap.add_argument("--causal-tag", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    result = {"baseline_tag_prefix": args.baseline_tag_prefix,
              "causal_tag": args.causal_tag, "families": {}}
    for fam in ("code", "math"):
        base = load(f"{args.baseline_tag_prefix}{BASE_SUFFIX[fam]}")
        arms = {arm: load(f"{args.causal_tag}/{arm}_{fam}_loose")
                for arm in ARM_NAMES}

        stable_correct = []
        for task_id, rows in base.items():
            if all(valid(rows.get(s)) and rows[s].get("success") for s in SEEDS):
                stable_correct.append(task_id)

        fam_out = {"stable_baseline_correct_tasks": len(stable_correct),
                   "arms": {}}
        for arm_idx, arm in enumerate(ARM_NAMES):
            task_breakage = []
            task_acceptance = []
            task_reject_refine = []
            matched_pairs = provider_errors = 0
            for task_id in stable_correct:
                vals = []
                acc = []
                rej = []
                for s in SEEDS:
                    row = arms[arm].get(task_id, {}).get(s)
                    if not valid(row):
                        if row is not None:
                            provider_errors += 1
                        continue
                    matched_pairs += 1
                    p = path_of(row)
                    vals.append(float(not row.get("success")))
                    acc.append(float(p == "acceptance"))
                    rej.append(float(p == "rejection_refinement"))
                if vals:
                    task_breakage.append(sum(vals) / len(vals))
                    task_acceptance.append(sum(acc) / len(acc))
                    task_reject_refine.append(sum(rej) / len(rej))

            fam_out["arms"][arm] = {
                "effective_tasks": len(task_breakage),
                "matched_pairs": matched_pairs,
                "nonvalid_rows_on_stable_tasks": provider_errors,
                "breakage": mean_ci(task_breakage, 100 + arm_idx),
                "acceptance_path": mean_ci(task_acceptance, 200 + arm_idx),
                "rejection_refinement_path": mean_ci(task_reject_refine, 300 + arm_idx),
            }

        # Pair-level audit over all tasks: p, repair, breakage, net. This is
        # secondary because it allows baseline correctness to vary by replicate.
        for arm_idx, arm in enumerate(ARM_NAMES):
            by_task = []
            for task_id, brows in base.items():
                pairs = []
                for s in SEEDS:
                    b = brows.get(s)
                    a = arms[arm].get(task_id, {}).get(s)
                    if valid(b) and valid(a):
                        pairs.append((float(b.get("success")), float(a.get("success"))))
                if pairs:
                    p = sum(x for x, _ in pairs) / len(pairs)
                    wa = sum(y for _, y in pairs) / len(pairs)
                    repair_den = sum(1 - x for x, _ in pairs)
                    break_den = sum(x for x, _ in pairs)
                    repair_num = sum((1 - x) * y for x, y in pairs)
                    break_num = sum(x * (1 - y) for x, y in pairs)
                    by_task.append({"p": p, "acc": wa, "net": wa - p,
                                    "repair_num": repair_num, "repair_den": repair_den,
                                    "break_num": break_num, "break_den": break_den})
            repair_num = sum(x["repair_num"] for x in by_task)
            repair_den = sum(x["repair_den"] for x in by_task)
            break_num = sum(x["break_num"] for x in by_task)
            break_den = sum(x["break_den"] for x in by_task)
            fam_out["arms"][arm]["pair_level_secondary"] = {
                "baseline_accuracy": sum(x["p"] for x in by_task) / len(by_task) if by_task else None,
                "workflow_accuracy": sum(x["acc"] for x in by_task) / len(by_task) if by_task else None,
                "net": mean_ci([x["net"] for x in by_task], 400 + arm_idx),
                "repair": repair_num / repair_den if repair_den else None,
                "breakage": break_num / break_den if break_den else None,
            }

        result["families"][fam] = fam_out

    out = EXP / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print("written to", out)


if __name__ == "__main__":
    main()

