#!/usr/bin/env python3
"""Frozen five-refiner temporal-option study on the inspected GLM prefix.

This study does not claim a new independent deployment certificate.  It asks
what a common FRR screen can know before suffix generation, what direct
candidate-specific bounds know after generation, and what each timing choice
costs.  All five contracts are frozen together before their fresh calls.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "experiments/shared_refiner5_temporal_20260909"
DATA = ROOT / "data/math_dcert_glm4flash.jsonl"
BASE = ROOT / "experiments/glm4flash_dcert/baseline/results_seed0.jsonl"
GATE = ROOT / "experiments/glm4flash_dcert/candidate/results_seed0.jsonl"
MODEL = "glm-4-flash-250414"
FORMAT = "End with the final answer in \\boxed{...}."
ARMS = {
    "standard": [dict(max_tokens=1536, temperature=.7, prompt=
        "Problem:\n{task}\n\nStored solution:\n{reference}\n\nThe common verifier rejected this solution. "
        "Rework it carefully from the first questionable step. " + FORMAT)],
    "targeted": [dict(max_tokens=512, temperature=.3, prompt=
        "Problem:\n{task}\n\nStored solution:\n{reference}\n\nThe common verifier rejected this solution. "
        "Find the earliest concrete error and repair only dependent steps; preserve valid reasoning. "
        "If no error is substantiated, retain the stored answer. Be concise. " + FORMAT)],
    "conservative": [dict(max_tokens=768, temperature=0, prompt=
        "Problem:\n{task}\n\nStored solution:\n{reference}\n\nAudit the rejection. Change the answer only after "
        "identifying and checking a specific mathematical error; otherwise return the stored solution. " + FORMAT)],
    "long_form": [dict(max_tokens=2048, temperature=.3, prompt=
        "Problem:\n{task}\n\nStored solution:\n{reference}\n\nThe common verifier rejected this solution. "
        "Independently verify each essential step, repair any demonstrated error, and give a complete checked solution. "
        "If the stored answer survives the checks, retain it. " + FORMAT)],
    "independent_reconcile": [
        dict(max_tokens=1024, temperature=.7, prompt=
            "Problem:\n{task}\n\nSolve independently and check the main calculations. Be concise. " + FORMAT),
        dict(max_tokens=1024, temperature=.3, prompt=
            "Problem:\n{task}\n\nStored solution:\n{reference}\n\nIndependent solution:\n{draft}\n\n"
            "Resolve disagreements by checking the problem and calculations. Return the better supported solution. " + FORMAT),
    ],
}
LOCK = threading.Lock()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x]


def append(path: Path, row: dict) -> None:
    with LOCK, path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()


def inputs():
    tasks = {r["id"]: r for r in load(DATA)}
    base = {r["task_id"]: r for r in load(BASE)}
    gate = {r["task_id"]: r for r in load(GATE)}
    assert tasks.keys() == base.keys() == gate.keys()
    rejected = []
    for tid, row in gate.items():
        assert row["status"] == base[tid]["status"] == "completed"
        if any(n["type"] == "refine" for n in row["trace"]):
            rejected.append(tid)
        else:
            assert row["solution"] == base[tid]["solution"]
    assert len(tasks) == 1194 and len(rejected) == 100
    return tasks, base, gate, sorted(rejected)


def freeze() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tasks, base, gate, rejected = inputs()
    sources = [DATA, BASE, GATE, Path(__file__), ROOT / "hbws/verify.py"]
    payload = {
        "status": "frozen before all five fresh suffixes",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "post-hoc fixed inspected prefix; temporal/cost study, not independent deployment validation",
        "model": MODEL, "endpoint": "https://open.bigmodel.cn/api/paas/v4",
        "arms": ARMS, "rejected_task_ids": rejected,
        "task_count": len(tasks), "reference_correct": sum(bool(x["success"]) for x in base.values()),
        "alpha": .05, "m": len(ARMS), "epsilon_grid": [.01, .02, .03, .05, .08, .10],
        "analysis": "shared FRR before suffix; candidate-specific stored-prefix bound after suffix; task bootstrap; no exclusions",
        "screening": "common gate is class-level and therefore all-or-none at each epsilon",
        "planned_trajectories": len(rejected) * len(ARMS),
        "planned_calls": len(rejected) * sum(len(v) for v in ARMS.values()),
        "source_hashes": {str(p.relative_to(ROOT)): digest(p) for p in sources},
        "runtime": {p: importlib.metadata.version(p) for p in
                    ["openai", "numpy", "sympy", "antlr4-python3-runtime"]},
    }
    path = OUT / "manifest.json"
    if path.exists():
        old = json.loads(path.read_text())
        payload["frozen_at_utc"] = old["frozen_at_utc"]
        if payload != old:
            raise RuntimeError("Refusing to alter frozen five-refiner study")
    else:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("frozen", len(ARMS), "contracts and", len(rejected), "rejected tasks")


def manifest() -> dict:
    cfg = json.loads((OUT / "manifest.json").read_text())
    assert cfg["arms"] == ARMS
    for name, sha in cfg["source_hashes"].items():
        assert digest(ROOT / name) == sha, f"frozen source changed: {name}"
    return cfg


def render(template: str, *, task: str, reference: str, draft: str) -> str:
    return template.replace("{task}", task).replace("{reference}", reference).replace("{draft}", draft)


def run(limit: int | None, workers: int) -> None:
    cfg = manifest()
    from run_glm4flash_replication import load_key
    from openai import OpenAI
    os.environ.update(GLM_API_KEY=load_key())
    tasks, base, gate, rejected = inputs()
    done = {(r["task_id"], r["arm"]): r for r in load(OUT / "trajectories.jsonl")}
    stages = {(r["task_id"], r["arm"], r["stage"]): r for r in load(OUT / "stages.jsonl")}
    jobs = [(tid, arm) for tid in rejected for arm in ARMS if (tid, arm) not in done]
    random.Random(20260909).shuffle(jobs)
    if limit:
        jobs = jobs[:limit]

    def one(tid: str, arm: str) -> dict:
        client = OpenAI(api_key=os.environ["GLM_API_KEY"], base_url=cfg["endpoint"], timeout=90, max_retries=4)
        draft = ""
        records = []
        for index, stage in enumerate(ARMS[arm]):
            old = stages.get((tid, arm, index))
            if old:
                draft = old["content"]
                records.append(old)
                continue
            prompt = render(stage["prompt"], task=tasks[tid]["prompt"],
                            reference=base[tid]["solution"], draft=draft)
            started = time.monotonic()
            response = client.chat.completions.create(
                model=MODEL, messages=[{"role": "user", "content": prompt}],
                temperature=stage["temperature"], max_tokens=stage["max_tokens"])
            usage = response.usage
            if usage is None or response.model != MODEL:
                raise RuntimeError("missing usage or unexpected model identity")
            row = {"task_id": tid, "arm": arm, "stage": index,
                   "prompt": prompt, "content": response.choices[0].message.content or "",
                   "finish_reason": response.choices[0].finish_reason,
                   "in_tokens": usage.prompt_tokens, "out_tokens": usage.completion_tokens,
                   "seconds": time.monotonic() - started, "model": response.model}
            append(OUT / "stages.jsonl", row)
            draft = row["content"]
            records.append(row)
        row = {"task_id": tid, "arm": arm, "solution": draft, "status": "completed",
               "calls": len(records), "tokens": sum(x["in_tokens"] + x["out_tokens"] for x in records),
               "seconds": sum(x["seconds"] for x in records),
               "finish_reasons": [x["finish_reason"] for x in records]}
        append(OUT / "trajectories.jsonl", row)
        return row

    errors = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, *job): job for job in jobs}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except Exception as exc:
                errors.append({"job": futures[future], "type": type(exc).__name__, "message": str(exc)})
            if i % 25 == 0 or i == len(jobs):
                print(f"progress {i}/{len(jobs)} errors={len(errors)}", flush=True)
    if errors:
        (OUT / "errors.json").write_text(json.dumps(errors, indent=2) + "\n")
        raise SystemExit(2)


def analyze() -> None:
    cfg = manifest()
    import numpy as np
    from hbws.verify import grade_math
    tasks, base, gate, rejected = inputs()
    trajectories = {(r["task_id"], r["arm"]): r for r in load(OUT / "trajectories.jsonl")}
    assert len(trajectories) == len(rejected) * len(ARMS)
    n = len(tasks)
    correct = sum(bool(x["success"]) for x in base.values())
    false_reject = sum(bool(base[t]["success"]) for t in rejected)
    common_exact = false_reject / correct
    shared_hoeffding = min(1.0, (false_reject + math.sqrt(n * math.log(1 / cfg["alpha"]) / 2)) / correct)
    prefix = Counter(calls=0, tokens=0, seconds=0)
    for tid in tasks:
        b = base[tid]
        v = next(x for x in gate[tid]["trace"] if x["type"] == "verify")
        prefix.update(calls=b["budget"]["llm_calls"] + v["budget"]["llm_calls"],
                      tokens=b["budget"]["in_tokens"] + b["budget"]["out_tokens"] +
                             v["budget"]["in_tokens"] + v["budget"]["out_tokens"],
                      seconds=b["budget"]["wall_sec"] + v["sec"])
    reports = {}
    for arm in ARMS:
        all_rows = []
        for tid in tasks:
            if tid in rejected:
                row = dict(trajectories[tid, arm])
                row["success"] = bool(grade_math(row["solution"], tasks[tid]["gold_answer"]))
            else:
                row = {"task_id": tid, "success": bool(base[tid]["success"]),
                       "calls": 0, "tokens": 0, "seconds": 0}
            row["baseline_success"] = bool(base[tid]["success"])
            all_rows.append(row)
        k = sum(x["baseline_success"] and not x["success"] for x in all_rows)
        repair = sum(not x["baseline_success"] and x["success"] for x in all_rows)
        q = false_reject
        direct = min(common_exact,
                     (k + math.sqrt(q * math.log(len(ARMS) / cfg["alpha"]) / 2)) / correct)
        cost = {name: sum(x[name] for x in all_rows) for name in ("calls", "tokens", "seconds")}
        rng = np.random.default_rng(20260909)
        delta_values = np.array([int(x["success"]) - int(x["baseline_success"]) for x in all_rows])
        boot = rng.choice(delta_values, size=(10000, n), replace=True).mean(axis=1)
        threshold_m = math.exp(2 * (q - k) ** 2 / q) / cfg["alpha"] if q else 0
        reports[arm] = {"repair_events": repair, "breakage_events": k,
                        "delta": float(delta_values.mean()),
                        "delta_ci95": np.quantile(boot, [.025, .975]).tolist(),
                        "direct_stored_simultaneous_ucb": direct,
                        "common_exact_ceiling": common_exact,
                        "m_for_direct_to_tie_common_ceiling": threshold_m,
                        "cost": cost,
                        "length_terminated": sum("length" in x.get("finish_reasons", []) for x in all_rows)}
    decisions = []
    for epsilon in cfg["epsilon_grid"]:
        direct_pass = {arm: reports[arm]["direct_stored_simultaneous_ucb"] <= epsilon for arm in ARMS}
        for screen, value in (("shared_frr_hoeffding", shared_hoeffding),
                              ("shared_exact_fixed_prefix", common_exact)):
            shared_pass = value <= epsilon
            decisions.append({"epsilon": epsilon, "screen": screen,
                              "shared_class_pass": shared_pass,
                              "candidates_screened_in_before_suffix": len(ARMS) if shared_pass else 0,
                              "direct_candidates_passing_after_suffix": sum(direct_pass.values()),
                              "false_positive_count": sum(shared_pass and not x for x in direct_pass.values()),
                              "false_negative_count": sum((not shared_pass) and x for x in direct_pass.values())})
    suffix_total = {k: sum(v["cost"][k] for v in reports.values()) for k in ("calls", "tokens", "seconds")}
    independent = {k: len(ARMS) * prefix[k] + suffix_total[k] for k in suffix_total}
    shared_expost = {k: prefix[k] + suffix_total[k] for k in suffix_total}
    report = {"scope": cfg["scope"], "n": n, "reference_correct": correct,
              "false_rejections": false_reject, "shared_exact_ceiling": common_exact,
              "shared_frr_hoeffding_ucb": shared_hoeffding, "refiners": reports,
              "screen_decisions": decisions, "cost": {"one_common_prefix": dict(prefix),
              "suffix_total": suffix_total, "independent_prefix_plus_suffix": independent,
              "shared_prefix_plus_suffix": shared_expost,
              "shared_exante_screen_only": dict(prefix),
              "shared_expost_savings_fraction": {k: 1 - shared_expost[k] / independent[k] for k in independent}},
              "theorem_check": "direct stored-prefix bound is min(common ceiling, candidate term), hence never looser"}
    (OUT / "analysis.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["freeze", "run", "analyze"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.stage == "freeze":
        freeze()
    elif args.stage == "run":
        run(args.limit, args.workers)
    else:
        analyze()
