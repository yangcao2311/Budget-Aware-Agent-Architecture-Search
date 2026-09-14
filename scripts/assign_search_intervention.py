#!/usr/bin/env python3
"""Freeze and run the actual HBWS search with stored assignment exposed.

The intervention keeps the search loop and objective unchanged.  It adds one
seed and one mutation that replace the entry generator by the zero-cost assign
node.  Every candidate receives the same previously materialized reference
output per task.  Logical GPT-4o prices preserve the original cost objective;
the external GLM provider remains free.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "experiments/assign_search_intervention_20260909"
MANIFEST = OUT / "manifest.json"
REFERENCE_PREFIX = "glm4flash_assign_search_20260909_"
RUN_NAME = "glm4flash_assign_exposed_code_s0"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), ROOT / "hbws/hbws.py", ROOT / "hbws/search.py",
               ROOT / "hbws/dsl.py", ROOT / "hbws/runner.py",
               ROOT / "hbws/protocol.py", ROOT / "scripts/run_envelope.py"]
    payload = {
        "status": "frozen before reference or search calls",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "glm-4-flash-250414",
        "family": "code", "split": "dev", "n": 120, "seed": 0,
        "reference": "one fresh direct/loose output per task, cache disabled",
        "intervention": "add assign_refine seed and generator-to-assign mutation; search loop/objective unchanged",
        "search_tiers": ["tight", "loose"], "logical_price_in_per_m": 2.5,
        "logical_price_out_per_m": 10.0, "search_cap_usd": 1.5,
        "source_hashes": {str(p.relative_to(ROOT)): digest(p) for p in sources},
    }
    if MANIFEST.exists():
        old = json.loads(MANIFEST.read_text())
        payload["frozen_at_utc"] = old["frozen_at_utc"]
        if payload != old:
            raise RuntimeError("Refusing to alter frozen assign-search intervention")
    else:
        MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("frozen", MANIFEST)


def cfg() -> dict:
    value = json.loads(MANIFEST.read_text())
    for name, sha in value["source_hashes"].items():
        assert digest(ROOT / name) == sha, f"frozen source changed: {name}"
    return value


def provider_env() -> dict[str, str]:
    from run_glm4flash_replication import load_key
    env = os.environ.copy()
    env.update(LLM_PROVIDER="glm", GLM_API_KEY=load_key(),
               GLM_BASE_URL="https://open.bigmodel.cn/api/paas/v4",
               GLM_MODEL="glm-4-flash-250414", LLM_DISABLE_SEED="0",
               LLM_BACKOFF_SCHEDULE="3,10,30", LLM_MAX_RETRIES="4")
    return env


def reference(workers: int) -> None:
    value = cfg()
    env = provider_env()
    env.update(LLM_PRICE_IN_PER_M="0", LLM_PRICE_OUT_PER_M="0")
    subprocess.run([sys.executable, "scripts/run_envelope.py", "--split", "dev",
                    "--families", "code", "--tiers", "loose", "--structures",
                    "direct", "--n", str(value["n"]), "--seed", "0", "--workers",
                    str(workers), "--tag-prefix", REFERENCE_PREFIX, "--no-cache"],
                   cwd=ROOT, env=env, check=True)


def has_assign(wf: dict) -> bool:
    return any(n.get("type") == "assign" for n in wf.get("nodes", []))


def search(workers: int) -> None:
    value = cfg()
    import random
    from hbws.data import load_split
    import hbws.dsl as dsl
    import hbws.search as search_mod
    import hbws.hbws as hbws_mod

    def mut_introduce_assign(wf: dict, rng: random.Random) -> dict:
        child = copy.deepcopy(wf)
        candidates = [n for n in child["nodes"] if n["type"] in {"generate", "vote"}]
        if not candidates:
            raise dsl.InvalidWorkflow("no generator to replace")
        node = candidates[0]
        keep = {"id": node["id"], "type": "assign"}
        node.clear(); node.update(keep)
        return child

    dsl.TEMPLATES["assign_refine"] = dsl.wf_assign_refine
    hbws_mod.TEMPLATES["assign_refine"] = dsl.wf_assign_refine
    hbws_mod.HUMAN_SEEDS[:] = ["assign_refine", "direct", "cot"]
    search_mod.OPERATORS.insert(0, mut_introduce_assign)
    tasks = load_split("code", "dev")[:value["n"]]
    ref_path = ROOT / "experiments" / f"{REFERENCE_PREFIX}envelope_dev/direct_code_loose/results_seed0.jsonl"
    if not ref_path.exists():
        # run_envelope historically names all non-test splits explicitly.
        alternatives = list((ROOT / "experiments").glob(
            f"{REFERENCE_PREFIX}*/direct_code_loose/results_seed0.jsonl"))
        if len(alternatives) != 1:
            raise RuntimeError("cannot resolve frozen reference results")
        ref_path = alternatives[0]
    refs = {r["task_id"]: r["solution"] for r in map(json.loads, ref_path.open())}
    assert len(refs) == len(tasks)
    tasks = [{**t, "_assign_solution": refs[t["id"]]} for t in tasks]
    env = provider_env()
    env.update(LLM_PRICE_IN_PER_M=str(value["logical_price_in_per_m"]),
               LLM_PRICE_OUT_PER_M=str(value["logical_price_out_per_m"]))
    os.environ.update(env)
    result = hbws_mod.hbws_search("code", tasks, cap_usd=value["search_cap_usd"],
                                  budget_contingent=True, seed=value["seed"],
                                  workers=workers, tiers=tuple(value["search_tiers"]),
                                  run_name=RUN_NAME)
    print("assign in archive:", any(has_assign(c["wf"]) for c in result["archive"]))


def analyze() -> None:
    value = cfg()
    path = ROOT / "experiments/search" / RUN_NAME / "search_result.json"
    result = json.loads(path.read_text())
    registry = result["archive"]
    rows = []
    for rank, cand in enumerate(registry, 1):
        rows.append({"rank_by_j_mean": rank, "cid": cand["cid"],
                     "origin": cand["origin"], "contains_assign": has_assign(cand["wf"]),
                     "j_mean": cand["j_mean"], "j_cb": cand["j_cb"],
                     "fidelity": cand["fidelity"], "per_tier":
                     cand["stats"][str(cand["fidelity"])]["per_tier"]})
    report = {"intervention": value["intervention"], "run": result["run"],
              "search_usd_logical": result["search_usd"],
              "n_candidates": result["n_candidates"], "archive": rows,
              "assign_in_archive": any(x["contains_assign"] for x in rows),
              "best_contains_assign": bool(rows and rows[0]["contains_assign"]),
              "interpretation": "unchanged objective sees accuracy and logical cost, not breakage as a separate term"}
    (OUT / "analysis.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["freeze", "reference", "search", "analyze"])
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    {"freeze": freeze, "reference": lambda: reference(args.workers),
     "search": lambda: search(args.workers), "analyze": analyze}[args.stage]()
