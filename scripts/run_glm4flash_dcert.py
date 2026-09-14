#!/usr/bin/env python3
"""Run the frozen GLM-4-Flash held-out certification experiment.

Every stage is resumable. Model responses are appended to a journal as they
finish; a canonical results file is materialized only from usable rows. The
script never reads Dcert outcomes when choosing the already-frozen candidate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from run_glm4flash_replication import provider_env

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
DCERT = ROOT / "data" / "math_dcert_glm4flash.jsonl"
GOOD = {"completed", "reserve_rejected"}
ALPHA = 0.05
EPSILON = 0.05
NB = 10000
FREEZE = EXP / "glm4flash_dcert_frozen_candidate.json"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def usable(row: dict) -> bool:
    return row.get("status") in GOOD


def run_resumable(workflow: dict, tasks: list[dict], *, run_name: str,
                  workers: int) -> list[dict]:
    from hbws.ledger import BUDGET_TIERS, BudgetCaps
    from hbws.protocol import grade_with_feedback
    from hbws.runner import run_workflow

    out_dir = EXP / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    journal = out_dir / "attempts_seed0.jsonl"
    canonical = out_dir / "results_seed0.jsonl"
    by_id = {row["task_id"]: row for row in load_jsonl(canonical) if usable(row)}
    for row in load_jsonl(journal):
        if usable(row):
            by_id[row["task_id"]] = row

    def one(task: dict) -> dict:
        try:
            result = run_workflow(
                workflow, task, BudgetCaps(**vars(BUDGET_TIERS["loose"])),
                use_cache=False, seed=0)
            result["success"], result["grader_feedback"] = grade_with_feedback(
                task, result["solution"])
            return result
        except Exception as exc:
            return {"task_id": task["id"],
                    "status": f"error:{type(exc).__name__}:{exc}",
                    "success": False, "solution": "", "budget": {},
                    "trace": []}

    task_by_id = {task["id"]: task for task in tasks}
    for pass_index in range(6):
        missing = [task_by_id[task_id] for task_id in task_by_id
                   if task_id not in by_id]
        if not missing:
            break
        print(f"[{run_name}] pass={pass_index + 1} remaining={len(missing)}",
              flush=True)
        completed_this_pass = 0
        with journal.open("a") as stream, ThreadPoolExecutor(
                max_workers=workers) as pool:
            futures = {pool.submit(one, task): task["id"] for task in missing}
            for index, future in enumerate(as_completed(futures), 1):
                row = future.result()
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                if usable(row):
                    by_id[row["task_id"]] = row
                    completed_this_pass += 1
                if index % 50 == 0 or index == len(missing):
                    print(f"[{run_name}] pass={pass_index + 1} "
                          f"finished={index}/{len(missing)} usable={completed_this_pass}",
                          flush=True)
        if completed_this_pass == 0:
            break
        if len(by_id) < len(task_by_id):
            time.sleep(min(30, 5 * (pass_index + 1)))

    ordered = [by_id[task["id"]] for task in tasks if task["id"] in by_id]
    canonical.write_text("".join(json.dumps(row) + "\n" for row in ordered))
    if len(ordered) != len(tasks):
        raise RuntimeError(
            f"{run_name}: only {len(ordered)}/{len(tasks)} usable rows; rerun to resume")
    summary = {
        "run": run_name,
        "n": len(ordered),
        "success_rate": sum(row["success"] for row in ordered) / len(ordered),
        "llm_calls_per_task": sum(row["budget"].get("llm_calls", 0)
                                  for row in ordered) / len(ordered),
        "tokens_per_task": sum(row["budget"].get("in_tokens", 0) +
                               row["budget"].get("out_tokens", 0)
                               for row in ordered) / len(ordered),
        "usd_per_task": sum(row["budget"].get("usd", 0)
                            for row in ordered) / len(ordered),
        "settlement_overruns": sum(str(row["status"]).startswith("budget_exceeded")
                                   for row in ordered),
    }
    (out_dir / "summary_seed0.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    return ordered


def baseline_tasks(tasks: list[dict]) -> list[dict]:
    return tasks


def assigned_tasks(tasks: list[dict], baseline_rows: list[dict]) -> list[dict]:
    reference = {row["task_id"]: row["solution"] for row in baseline_rows}
    return [{**task, "_assign_solution": reference[task["id"]]}
            for task in tasks]


def paired_metrics(baseline_rows: list[dict], candidate_rows: list[dict]) -> dict:
    from hbws.stats import cluster_ratio_upper

    baseline = {row["task_id"]: row for row in baseline_rows}
    candidate = {row["task_id"]: row for row in candidate_rows}
    ids = sorted(set(baseline) & set(candidate))
    transitions = Counter()
    rejects = []
    eligible = []
    acceptance_breaks = 0
    deltas = []
    for task_id in ids:
        b = bool(baseline[task_id]["success"])
        w = bool(candidate[task_id]["success"])
        transitions[f"{int(b)}{int(w)}"] += 1
        rejected = "refine" in [node["type"] for node in
                                candidate[task_id].get("trace", [])]
        eligible.append(1.0 if b else 0.0)
        rejects.append(1.0 if b and rejected else 0.0)
        acceptance_breaks += int(b and not w and not rejected)
        deltas.append(float(w) - float(b))
    n = len(ids)
    baseline_correct = transitions["10"] + transitions["11"]
    baseline_wrong = transitions["00"] + transitions["01"]
    frr = sum(rejects) / baseline_correct
    breakage = transitions["10"] / baseline_correct
    repair = transitions["01"] / baseline_wrong
    risk_ucb = cluster_ratio_upper(rejects, eligible, alpha=ALPHA)
    rng = random.Random(20260909)
    boots = sorted(sum(deltas[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(NB))
    return {
        "n_tasks": n,
        "transition_counts": dict(transitions),
        "baseline_accuracy": (transitions["10"] + transitions["11"]) / n,
        "candidate_accuracy": (transitions["01"] + transitions["11"]) / n,
        "delta": sum(deltas) / n,
        "delta_ci95": [boots[int(0.025 * NB)], boots[int(0.975 * NB)]],
        "repair": repair,
        "breakage": breakage,
        "false_rejection_rate": frr,
        "false_rejection_ucb95": risk_ucb,
        "acceptance_path_breakages": acceptance_breaks,
        "primary_epsilon": EPSILON,
        "decision": "adopt" if risk_ucb <= EPSILON else "abstain",
    }


def analyze(prefix: str, output: Path) -> dict:
    baseline = load_jsonl(EXP / prefix / "baseline" / "results_seed0.jsonl")
    candidate = load_jsonl(EXP / prefix / "candidate" / "results_seed0.jsonl")
    if not baseline or not candidate:
        raise RuntimeError(f"missing results under {prefix}")
    result = paired_metrics(baseline, candidate)
    result["baseline_calls_per_task"] = sum(
        row["budget"].get("llm_calls", 0) for row in baseline) / len(baseline)
    result["candidate_incremental_calls_per_task"] = sum(
        row["budget"].get("llm_calls", 0) for row in candidate) / len(candidate)
    result["total_calls_per_task"] = (
        result["baseline_calls_per_task"] +
        result["candidate_incremental_calls_per_task"])
    result["total_usd"] = sum(
        row["budget"].get("usd", 0) for row in baseline + candidate)
    result["settlement_overruns"] = sum(
        str(row["status"]).startswith("budget_exceeded")
        for row in baseline + candidate)
    if FREEZE.exists():
        result["frozen_selection"] = json.loads(FREEZE.read_text())
    manifest_path = EXP / "glm4flash_dcert_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        result["certificate_sha256"] = manifest["certificate_sha256"]
        result["n_prior_tasks"] = manifest["n_prior_tasks"]
        result["n_certificate_tasks"] = manifest["n_certificate_tasks"]
    attempt_path = EXP / "glm4flash_dcert_attempts.jsonl"
    provider_models = {
        row.get("provider_model")
        for row in load_jsonl(attempt_path)
        if row.get("status") == "completed"
    }
    provider_models.discard(None)
    if provider_models != {"glm-4-flash-250414"}:
        raise RuntimeError(f"unexpected provider model identities: {provider_models}")
    result["provider_models"] = sorted(provider_models)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


def select_and_freeze(
    baseline: list[dict],
    candidates: dict[str, list[dict]],
    workflows: dict[str, dict],
) -> dict:
    reports = {name: paired_metrics(baseline, rows)
               for name, rows in candidates.items()}
    reports["no_op_reference"] = {
        "n_tasks": len(baseline), "baseline_accuracy":
        sum(row["success"] for row in baseline) / len(baseline),
        "candidate_accuracy": sum(row["success"] for row in baseline) / len(baseline),
        "delta": 0.0, "delta_ci95": [0.0, 0.0], "repair": 0.0,
        "breakage": 0.0, "false_rejection_rate": 0.0,
        "acceptance_path_breakages": 0,
    }
    eligible = {
        name: report for name, report in reports.items()
        if report["breakage"] <= 0.02 and
        report["acceptance_path_breakages"] == 0
    }
    incremental_calls = {
        "no_op_reference": 0,
        "anchored_one_check": 1,
        "strict_one_check": 1,
    }
    selected = max(
        eligible,
        key=lambda name: (
            eligible[name]["delta"],
            -incremental_calls[name],
            name,
        ),
    )
    selected_workflow = workflows[selected]
    workflow_payload = json.dumps(
        selected_workflow, sort_keys=True, separators=(",", ":")
    ).encode()
    frozen = {
        "status": "frozen_before_dcert_calls",
        "selection_population": "150 previously archived MATH tasks",
        "selection_rule": "among candidates with empirical breakage <= 0.02 and zero acceptance-path breakage, maximize paired point delta; ties prefer fewer verifier calls",
        "selected_candidate": selected,
        "incremental_calls_per_task": incremental_calls[selected],
        "selected_workflow": selected_workflow,
        "selected_workflow_sha256": hashlib.sha256(workflow_payload).hexdigest(),
        "candidate_reports": reports,
        "primary_epsilon": EPSILON,
    }
    encoded = json.dumps(frozen, indent=2, sort_keys=True) + "\n"
    if FREEZE.exists() and FREEZE.read_text() != encoded:
        raise RuntimeError(f"refusing to change frozen selection: {FREEZE}")
    if not FREEZE.exists():
        FREEZE.write_text(encoded)
    print(json.dumps(frozen, indent=2), flush=True)
    return frozen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("dselect", "dcert_baseline",
                                            "dcert_candidate", "analyze",
                                            "all"), required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    env = provider_env(disable_seed=True)
    env["LLM_ATTEMPT_LOG"] = str(EXP / "glm4flash_dcert_attempts.jsonl")
    os.environ.update(env)

    from hbws.dsl import (wf_assign_anchored_refine, wf_assign_strict_refine,
                          wf_cot)

    dcert_tasks = load_jsonl(DCERT)
    dselect_tasks = load_jsonl(ROOT / "data" / "math_test.jsonl")
    stages = (("dselect", "dcert_baseline", "dcert_candidate", "analyze")
              if args.stage == "all" else (args.stage,))
    for stage in stages:
        if stage == "dselect":
            baseline = load_jsonl(
                EXP / "glm4flash_envelope_test" / "cot_math_loose" /
                "results_seed0.jsonl")
            anchored_workflow = wf_assign_anchored_refine()
            strict_workflow = wf_assign_strict_refine()
            anchored_rows = run_resumable(
                anchored_workflow,
                assigned_tasks(dselect_tasks, baseline),
                run_name="glm4flash_dselect_anchored/candidate",
                workers=args.workers)
            strict_rows = run_resumable(
                strict_workflow,
                assigned_tasks(dselect_tasks, baseline),
                run_name="glm4flash_dselect_strict/candidate",
                workers=args.workers)
            metrics = select_and_freeze(
                baseline, {"anchored_one_check": anchored_rows,
                           "strict_one_check": strict_rows},
                {
                    "anchored_one_check": anchored_workflow,
                    "strict_one_check": strict_workflow,
                    "no_op_reference": {
                        "nodes": [{"id": "a", "type": "assign"}],
                        "edges": [{"from": "a", "to": "END"}],
                    },
                })
            (EXP / "glm4flash_dselect_conservative_metrics.json").write_text(
                json.dumps(metrics, indent=2) + "\n")
        elif stage == "dcert_baseline":
            run_resumable(wf_cot(), baseline_tasks(dcert_tasks),
                          run_name="glm4flash_dcert/baseline",
                          workers=args.workers)
        elif stage == "dcert_candidate":
            baseline = load_jsonl(
                EXP / "glm4flash_dcert" / "baseline" / "results_seed0.jsonl")
            if len(baseline) != len(dcert_tasks):
                raise RuntimeError("complete Dcert baseline before candidate")
            if not FREEZE.exists():
                raise RuntimeError("freeze a candidate on Dselect before Dcert")
            frozen = json.loads(FREEZE.read_text())
            selected = frozen["selected_candidate"]
            if selected == "no_op_reference":
                out_dir = EXP / "glm4flash_dcert" / "candidate"
                out_dir.mkdir(parents=True, exist_ok=True)
                rows = []
                for row in baseline:
                    copied = dict(row)
                    copied["budget"] = {
                        "llm_calls": 0,
                        "in_tokens": 0,
                        "out_tokens": 0,
                        "tool_calls": 0,
                        "usd": 0.0,
                        "wall_sec": 0.0,
                    }
                    copied["trace"] = [{
                        "node": "a",
                        "type": "assign",
                        "source": "baseline",
                    }]
                    rows.append(copied)
                (out_dir / "results_seed0.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows))
            else:
                workflow = frozen["selected_workflow"]
                workflow_payload = json.dumps(
                    workflow, sort_keys=True, separators=(",", ":")
                ).encode()
                if hashlib.sha256(workflow_payload).hexdigest() != frozen[
                        "selected_workflow_sha256"]:
                    raise RuntimeError("frozen workflow hash mismatch")
                run_resumable(workflow,
                              assigned_tasks(dcert_tasks, baseline),
                              run_name="glm4flash_dcert/candidate",
                              workers=args.workers)
        elif stage == "analyze":
            analyze("glm4flash_dcert", EXP / "glm4flash_dcert_results.json")


if __name__ == "__main__":
    main()
