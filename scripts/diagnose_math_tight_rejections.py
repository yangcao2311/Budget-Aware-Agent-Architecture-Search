#!/usr/bin/env python3
"""Frozen replay diagnostic for original math/tight assignment rejections.

The original workflow trace records the verifier decision but not the
independent checker's text.  This script therefore cannot reconstruct the
historical rejection reason.  It freezes the 120 originally rejected
task--seed pairs, checks the stored incumbent parser offline, and replays the
same gold-free checker once with full response/finish-reason logging.  Replay
results are diagnostic only and never replace the original outcomes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hbws.data import load_split
from hbws.prompts import render
from hbws.verify import _norm, extract_boxed, math_equal

OUT = ROOT / "experiments/math_tight_rejection_diagnostic_20260911"
MANIFEST = OUT / "manifest.json"
RESULTS = OUT / "checker_replays.jsonl"
BASE = (ROOT / "experiments/glm4flash_repaired_matrix_clean_20260910_envelope_test"
        / "cot_math_tight")
ASSIGN = (ROOT / "experiments/glm4flash_repaired_matrix_final_20260910_causal"
          / "arm1_assign_math_tight")
MODEL = "glm-4-flash-250414"
SEEDS = (0, 1, 2)
LOCK = threading.Lock()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x]


def keyed(directory: Path) -> dict[tuple[str, int], dict]:
    out = {}
    for seed in SEEDS:
        for row in read_jsonl(directory / f"results_seed{seed}.jsonl"):
            out[(row["task_id"], seed)] = row
    return out


def initial_rejection(row: dict) -> bool:
    trace = row.get("trace") or []
    for index, item in enumerate(trace):
        if item.get("type") != "verify":
            continue
        if "reserve_rejected" in item:
            return False
        return index + 1 < len(trace) and trace[index + 1].get("type") == "refine"
    return False


def freeze() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base, assign = keyed(BASE), keyed(ASSIGN)
    targets = []
    for key in sorted(assign):
        if initial_rejection(assign[key]):
            b = base[key]
            targets.append({
                "task_id": key[0], "seed": key[1],
                "reference_correct": bool(b.get("success")),
                "candidate_boxed": extract_boxed(b.get("solution") or ""),
                "candidate_chars": len(b.get("solution") or ""),
                "refinement_executed": any(
                    x.get("type") == "refine" and "reserve_rejected" not in x
                    for x in assign[key].get("trace") or []),
            })
    assert len(targets) == 120
    assert sum(x["reference_correct"] for x in targets) == 47
    sources = [Path(__file__), ROOT / "hbws/prompts.py", ROOT / "hbws/verify.py"]
    inputs = [BASE / f"results_seed{s}.jsonl" for s in SEEDS]
    inputs += [ASSIGN / f"results_seed{s}.jsonl" for s in SEEDS]
    payload = {
        "status": "frozen before diagnostic checker replays",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "diagnostic replay; cannot reconstruct unlogged historical checker outputs",
        "model": MODEL,
        "endpoint": "https://open.bigmodel.cn/api/paas/v4",
        "temperature": 0.3,
        "max_tokens": 1536,
        "targets": targets,
        "input_hashes": {str(p.relative_to(ROOT)): digest(p) for p in inputs},
        "source_hashes": {str(p.relative_to(ROOT)): digest(p) for p in sources},
    }
    if MANIFEST.exists():
        old = json.loads(MANIFEST.read_text())
        payload["frozen_at_utc"] = old["frozen_at_utc"]
        if payload != old:
            raise RuntimeError("Refusing to alter frozen rejection diagnostic")
    else:
        MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("frozen", len(targets), "original rejections; reference-correct", sum(
        x["reference_correct"] for x in targets))


def config() -> dict:
    cfg = json.loads(MANIFEST.read_text())
    for group in ("input_hashes", "source_hashes"):
        for name, sha in cfg[group].items():
            assert digest(ROOT / name) == sha, f"frozen input changed: {name}"
    return cfg


def append(row: dict) -> None:
    with LOCK, RESULTS.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def run(limit: int | None = None) -> None:
    cfg = config()
    from openai import OpenAI
    from run_glm4flash_replication import load_key

    tasks = {x["id"]: x for x in load_split("math", "test")[:150]}
    done = {(x["task_id"], x["seed"]) for x in read_jsonl(RESULTS)
            if x.get("status") == "completed"}
    jobs = [x for x in cfg["targets"] if (x["task_id"], x["seed"]) not in done]
    random.Random(20260911).shuffle(jobs)
    if limit is not None:
        jobs = jobs[:limit]
    client = OpenAI(api_key=load_key(), base_url=cfg["endpoint"], timeout=90,
                    max_retries=5)
    for index, item in enumerate(jobs, 1):
        prompt = render("check_math", "math", tasks[item["task_id"]]["prompt"])
        started = time.monotonic()
        try:
            response = client.chat.completions.create(
                model=cfg["model"], messages=[{"role": "user", "content": prompt}],
                temperature=cfg["temperature"], max_tokens=cfg["max_tokens"],
                seed=item["seed"] * 100 + 50)
            usage = response.usage
            content = response.choices[0].message.content or ""
            append({**item, "status": "completed", "prompt": prompt,
                    "checker_text": content,
                    "checker_boxed": extract_boxed(content),
                    "finish_reason": response.choices[0].finish_reason,
                    "provider_model": response.model,
                    "in_tokens": usage.prompt_tokens if usage else None,
                    "out_tokens": usage.completion_tokens if usage else None,
                    "seconds": time.monotonic() - started})
        except Exception as exc:
            append({**item, "status": f"error:{type(exc).__name__}",
                    "error": str(exc), "seconds": time.monotonic() - started})
        print(index, "/", len(jobs), item["task_id"], item["seed"], flush=True)


def numeric_like(value: str | None) -> bool:
    if value is None:
        return False
    value = _norm(value)
    if re.search(r"[a-zA-Z]", value.replace("frac", "").replace("sqrt", "")
                 .replace("pi", "")):
        return False
    return bool(re.search(r"\d", value))


def reason(row: dict) -> str:
    candidate, checker = row.get("candidate_boxed"), row.get("checker_boxed")
    if candidate is None:
        return "candidate_parse_failure"
    if checker is None:
        return "checker_parse_failure"
    if math_equal(candidate, checker):
        return "replay_agreement"
    if numeric_like(candidate) and numeric_like(checker):
        return "numeric_disagreement"
    return "symbolic_disagreement"


def analyze() -> None:
    from collections import Counter

    cfg = config()
    rows = read_jsonl(RESULTS)
    good = [x for x in rows if x.get("status") == "completed"]
    counts = Counter(reason(x) for x in good)
    correct = Counter(reason(x) for x in good if x["reference_correct"])
    report = {
        "scope": cfg["scope"],
        "planned": len(cfg["targets"]), "completed": len(good),
        "errors": len(rows) - len(good),
        "all_original_rejections_replay_reasons": dict(counts),
        "reference_correct_replay_reasons": dict(correct),
        "candidate_parse_failures_offline": sum(
            x["candidate_boxed"] is None for x in cfg["targets"]),
        "reference_correct_candidate_parse_failures_offline": sum(
            x["reference_correct"] and x["candidate_boxed"] is None
            for x in cfg["targets"]),
        "finish_reasons": dict(Counter(x.get("finish_reason") for x in good)),
    }
    (OUT / "analysis.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "run", "analyze"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    {"freeze": freeze, "run": lambda: run(args.limit), "analyze": analyze}[args.stage]()


if __name__ == "__main__":
    main()
