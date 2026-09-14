#!/usr/bin/env python3
"""Run exactly one cell of the frozen GLM-4.7 causal replication.

This operational driver prevents a rate-limited cell from cascading into the
next one before execution-status failures have been recovered.  It changes no
scientific design field.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hbws.data import load_split
from hbws.dsl import _wf_verify_refine, wf_assign_refine, wf_incumbent_refine_cot
from hbws.protocol import evaluate
import scripts.run_glm47_within_family as frozen
from scripts.run_provenance_causal import load_baseline_by_seed

EXP = ROOT / "experiments"
MANIFEST = EXP / "glm47flash_20260914_causal_cell_driver_manifest.json"
AMENDMENT = EXP / "glm47flash_20260913_concurrency_amendment.json"
ARMS = {
    "arm1_assign": wf_assign_refine,
    "arm2_samepolicy": wf_incumbent_refine_cot,
    "arm3_diffpolicy": lambda: _wf_verify_refine(3),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> None:
    sources = [Path(__file__), ROOT / "scripts/run_glm47_within_family.py",
               ROOT / "scripts/run_provenance_causal.py", ROOT / "hbws/runner.py",
               ROOT / "hbws/llm.py", ROOT / "hbws/dsl.py", ROOT / "hbws/prompts.py",
               ROOT / "hbws/verify.py", ROOT / "hbws/ledger.py", ROOT / "hbws/protocol.py"]
    payload = {
        "status": "frozen before any additional completed causal cell",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "operational scheduling only; run one arm/seed cell then stop for status recovery",
        "unchanged": ["model", "endpoint", "tasks", "arms", "seeds", "references",
                      "workflows", "caps", "cache", "workers"],
        "workers": 2,
        "note": "Abandoned arm2/seed0 calls wrote no result file and are not used.",
        "original_manifest_sha256": sha(frozen.MANIFEST),
        "concurrency_amendment_sha256": sha(AMENDMENT),
        "source_hashes": {str(path.relative_to(ROOT)): sha(path) for path in sources},
    }
    if MANIFEST.exists():
        old = json.loads(MANIFEST.read_text())
        payload["frozen_at_utc"] = old["frozen_at_utc"]
        if payload != old:
            raise RuntimeError("Refusing to alter frozen one-cell driver")
    else:
        MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("frozen:", MANIFEST)


def run(arm: str, seed: int) -> None:
    freeze()
    name = f"{frozen.TAG}/{arm}_math_tight_nonbinding"
    result = EXP / name / f"results_seed{seed}.jsonl"
    if result.exists() and sum(1 for line in result.open() if line.strip()) == 150:
        print("complete; skip", arm, "seed", seed)
        return
    tasks = load_split("math", "test")[:150]
    if arm == "arm1_assign":
        refs = load_baseline_by_seed("math", "tight", frozen.PREFIX, (seed,))[seed]
        tasks = [{**task, "_assign_solution": refs[task["id"]][0]} for task in tasks]
    os.environ.update(frozen.provider_env())
    evaluate(ARMS[arm](), tasks, frozen.CAPS, run_name=name, seed=seed,
             use_cache=False, workers=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "run"))
    parser.add_argument("--arm", choices=tuple(ARMS))
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze()
    else:
        if args.arm is None or args.seed is None:
            parser.error("run requires --arm and --seed")
        run(args.arm, args.seed)


if __name__ == "__main__":
    main()
