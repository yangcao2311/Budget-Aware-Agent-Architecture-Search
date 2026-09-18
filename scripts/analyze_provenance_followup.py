#!/usr/bin/env python3
"""Independent recomputation of the 2026-09-15 provenance follow-ups.

Part A  Qwen serving-regime intervention (serial / batched / prefix-cache /
        repeat-serial), byte identity of first drafts and paired workflow
        outcomes, with exclusion sensitivities for rows that were not actually
        concurrent in the batched condition.

Part B  GLM budget-parity contrast on the two frozen math reference cohorts.
        Conditional repair/breakage denominators, path localisation, the
        first-incumbent stratification that separates "regeneration produced a
        wrong incumbent" from "refinement recovered less well", and a
        draft-versus-suffix cost split.

All intervals are post-hoc descriptive, task-clustered, and computed after the
outcomes were observed.  No model calls.
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hbws.data import load_split
from hbws.verify import grade_math

REF = ROOT / "experiments/glm4flash_repaired_matrix_clean_20260910_envelope_test"
REFDIR = {"tight": "cot_math_tight", "loose": "cot_math_loose"}
OUT = ROOT / "experiments/provenance_followup_20260917_analysis.json"
BOOT, SEED = 10_000, 20260917


def rows(path):
    return {json.loads(l)["task_id"]: json.loads(l) for l in open(path)}


def calls(path, node_type=None):
    for line in open(path):
        r = json.loads(line)
        if node_type is None or r["node_type"] == node_type:
            yield r


def accepted_first(row):
    """Did the first verifier decision accept the incumbent?"""
    for step in row.get("trace", []):
        if step.get("type") == "verify":
            v = step["verifier"]
            return v["agree_count"] >= (v["k"] + 1) // 2
    return None


def cluster_ci(tasks, stat):
    rng = random.Random(SEED)
    draws = []
    n = len(tasks)
    for _ in range(BOOT):
        draws.append(stat([tasks[rng.randrange(n)] for _ in range(n)]))
    draws.sort()
    return [draws[int(0.025 * BOOT)], draws[int(0.975 * BOOT)]]


# ---------------------------------------------------------------- Part A ----

def part_a(exp):
    def gen(cond):
        g = defaultdict(list)
        for r in calls(exp / f"qwen_serving_regime_20260915_{cond}_calls.jsonl",
                       "generate"):
            g[(r["task_id"], int(r["seed"]))].append(r)
        return g

    def draft(g):
        # the retained response is the last attempt for that (task, seed)
        return {k: max(v, key=lambda r: r["request_start_unix"])["response_text"]
                for k, v in g.items()}

    def outcome(cond):
        o = {}
        d = exp / "qwen_serving_regime_20260915" / f"{cond}_samepolicy"
        for seed in range(3):
            for line in open(d / f"results_seed{seed}.jsonl"):
                r = json.loads(line)
                o[(r["task_id"], seed)] = bool(r["success"])
        return o

    gS = gen("S")
    S, Sout = draft(gS), outcome("S")
    gB = gen("B")
    recovered = sorted(k for k, v in gB.items() if len(v) > 1)
    spans = sorted((r["request_start_unix"], r["request_end_unix"], k)
                   for k, v in gB.items() for r in v)
    isolated = sorted({k for s, e, k in spans if not any(
        not (e2 <= s or s2 >= e) for s2, e2, k2 in spans if k2 != k)})

    def one(cond, exclude):
        C, Cout = draft(gen(cond)), outcome(cond)
        keys = sorted(k for k in set(S) & set(C) if k not in set(exclude))
        ident = sum(1 for k in keys if S[k] == C[k])
        to_wrong = sum(1 for k in keys if Sout[k] and not Cout[k])
        to_right = sum(1 for k in keys if not Sout[k] and Cout[k])
        req_same = sum(1 for k in keys
                       if json.dumps(max(gen(cond)[k], key=lambda r: r["request_start_unix"])["request"],
                                     sort_keys=True) ==
                       json.dumps(max(gS[k], key=lambda r: r["request_start_unix"])["request"],
                                  sort_keys=True))
        by = defaultdict(list)
        for k in keys:
            by[k[0]].append(S[k] == C[k])
        tasks = list(by.values())
        ci = cluster_ci(tasks, lambda s: sum(x for t in s for x in t) /
                        sum(len(t) for t in s))
        return {
            "n_pairs": len(keys),
            "identical_request_bodies": req_same,
            "byte_identical_first_drafts": ident,
            "byte_identity_rate": ident / len(keys),
            "byte_identity_ci95": ci,
            "workflow_outcome_agreement": (len(keys) - to_wrong - to_right) / len(keys),
            "flips_correct_to_wrong": to_wrong,
            "flips_wrong_to_correct": to_right,
            "net_outcome_change": to_right - to_wrong,
        }

    return {
        "definition": ("first drafts compared as exact UTF-8 strings; outcome "
                       "flips are FINAL workflow correctness, not first-draft "
                       "correctness"),
        "batched_rows_rerun_serially": [list(k) for k in recovered],
        "batched_rows_never_concurrent": [list(k) for k in isolated],
        "conditions": {
            "S2_vs_S": one("S2", []),
            "B_vs_S_all": one("B", []),
            "B_vs_S_excl_recovered": one("B", recovered),
            "B_vs_S_excl_never_concurrent": one("B", isolated),
            "P_vs_S": one("P", []),
        },
    }


# ---------------------------------------------------------------- Part B ----

def part_b(exp):
    gold = {t["id"]: t["gold_answer"] for t in load_split("math", "test")[:150]}
    bp = exp / "glm_budget_parity_20260916"
    out = {}
    for tier in ("tight", "loose"):
        sp_draft = {(r["task_id"], int(r["seed"])): r["response_text"]
                    for r in calls(exp / f"glm_budget_parity_20260916_{tier}"
                                   f"_same_policy_calls.jsonl", "generate")}
        per_task = defaultdict(list)
        for seed in range(3):
            ref = rows(REF / REFDIR[tier] / f"results_seed{seed}.jsonl")
            a = rows(bp / f"{tier}_assign" / f"results_seed{seed}.jsonl")
            s = rows(bp / f"{tier}_same_policy" / f"results_seed{seed}.jsonl")
            assert ref.keys() == a.keys() == s.keys()
            for t in ref:
                d = sp_draft.get((t, seed))
                per_task[t].append({
                    "ref_ok": bool(ref[t]["success"]),
                    "a_ok": bool(a[t]["success"]), "a_acc": accepted_first(a[t]),
                    "s_ok": bool(s[t]["success"]), "s_acc": accepted_first(s[t]),
                    "s_draft_ok": grade_math(d, gold[t]) if d is not None else None,
                })
        tasks = list(per_task.values())

        def diffs(sample):
            fl = [x for t in sample for x in t]
            rt = [x for x in fl if x["ref_ok"]]
            wr = [x for x in fl if not x["ref_ok"]]
            return (
                (sum(1 for x in rt if not x["a_ok"]) -
                 sum(1 for x in rt if not x["s_ok"])) / len(rt),
                (sum(1 for x in wr if x["a_ok"]) -
                 sum(1 for x in wr if x["s_ok"])) / len(wr),
                sum(x["a_ok"] - x["s_ok"] for x in fl) / len(fl),
            )

        point = diffs(tasks)
        ci = [cluster_ci(tasks, lambda s, i=i: diffs(s)[i]) for i in range(3)]
        flat = [x for t in tasks for x in t]
        right = [x for x in flat if x["ref_ok"]]

        def arm(ok_k, acc_k, draft_ok=None):
            sub = right if draft_ok is None else [
                x for x in right if x["s_draft_ok"] is draft_ok]
            acc = [x for x in sub if x[acc_k]]
            rej = [x for x in sub if not x[acc_k]]
            return {
                "n": len(sub),
                "accepted": len(acc),
                "accepted_wrong": sum(1 for x in acc if not x[ok_k]),
                "rejected": len(rej),
                "rejected_wrong": sum(1 for x in rej if not x[ok_k]),
                "post_rejection_failure_rate":
                    (sum(1 for x in rej if not x[ok_k]) / len(rej)) if rej else None,
            }

        cost = {}
        for a_name, fn in (("assign", f"{tier}_assign"),
                           ("same_policy", f"{tier}_same_policy")):
            by_node = defaultdict(lambda: {"calls": 0, "in": 0, "out": 0})
            for r in calls(exp / f"glm_budget_parity_20260916_{fn}_calls.jsonl"):
                b = by_node[r["node_type"]]
                b["calls"] += 1
                b["in"] += r["in_tokens"]
                b["out"] += r["out_tokens"]
            drafting = dict(by_node.get("generate", {"calls": 0, "in": 0, "out": 0}))
            suffix = {"calls": 0, "in": 0, "out": 0}
            for k, v in by_node.items():
                if k == "generate":
                    continue
                for f in suffix:
                    suffix[f] += v[f]
            cost[a_name] = {"drafting": drafting, "suffix": suffix,
                            "by_node": {k: dict(v) for k, v in by_node.items()}}

        wrong_inc = [x for x in right if x["s_draft_ok"] is False]
        out[tier] = {
            "positions_with_wrong_regenerated_incumbent": {
                "n": len(wrong_inc),
                "same_policy_final_correct":
                    sum(1 for x in wrong_inc if x["s_ok"]),
                "assignment_final_correct_same_positions":
                    sum(1 for x in wrong_inc if x["a_ok"]),
                "note": ("paired on the same task-seed positions; assignment "
                         "held the stored reference there, so this is not a "
                         "counterfactual for assignment receiving a wrong "
                         "incumbent"),
            },
            "cohort_note": ("prior-%s reference cohort; per-call and task caps "
                            "are identical across the two cohorts in this "
                            "follow-up, so the tiers differ only in which "
                            "frozen references they reuse" % tier),
            "pairs": len(flat), "reference_correct": len(right),
            "reference_wrong": len(flat) - len(right),
            "assign": arm("a_ok", "a_acc"),
            "same_policy": arm("s_ok", "s_acc"),
            "same_policy_incumbent_correct": arm("s_ok", "s_acc", True),
            "same_policy_incumbent_wrong": arm("s_ok", "s_acc", False),
            "paired_differences_assign_minus_same_policy": {
                "breakage_rate": {"point": point[0], "ci95": ci[0],
                                  "denominator": "reference-correct pairs"},
                "repair_rate": {"point": point[1], "ci95": ci[1],
                                "denominator": "reference-wrong pairs"},
                "accuracy": {"point": point[2], "ci95": ci[2],
                             "denominator": "all pairs"},
            },
            "cost": cost,
        }
    return out


def main():
    pkg = Path(sys.argv[1])
    exp = pkg / "experiments"
    report = {
        "generator": "scripts/analyze_provenance_followup.py",
        "source_package_sha256":
            "93a4948a7b488302c8d7c1e3471d756e5d2ec587ed236cf3250657a8df13c6e4",
        "interval_status": ("post-hoc descriptive, task-clustered; computed "
                            "after the outcomes were observed, not a "
                            "preregistered confirmatory test"),
        "bootstrap_draws": BOOT, "bootstrap_seed": SEED,
        "part_a_serving_regime": part_a(exp),
        "part_b_budget_parity": part_b(exp),
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
