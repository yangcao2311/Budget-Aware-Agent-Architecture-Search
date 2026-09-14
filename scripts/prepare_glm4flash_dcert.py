#!/usr/bin/env python3
"""Create an untouched MATH certificate split and freeze its manifest.

The script reconstructs the eligible in-domain pool, excludes every task ID in
the frozen dev/val/test files, and orders the remainder by a seeded hash. It
must run before any D_cert model call.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from datasets import load_dataset

from hbws.data import (MATH_IN_SUBJECTS, MATH_LEVELS, SPLIT_SEED,
                       _extract_gold, load_split)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "math_dcert_glm4flash.jsonl"
MANIFEST = ROOT / "experiments" / "glm4flash_dcert_manifest.json"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    tasks = []
    for subject in MATH_IN_SUBJECTS:
        dataset = load_dataset("EleutherAI/hendrycks_math", subject,
                               split="test")
        for index, example in enumerate(dataset):
            if example["level"] not in MATH_LEVELS:
                continue
            gold = _extract_gold(example["solution"])
            if not gold:
                continue
            tasks.append({
                "id": f"math_{subject}_{index}",
                "family": "math",
                "prompt": example["problem"],
                "gold_answer": gold,
                "subject": subject,
                "level": example["level"],
            })
    used = []
    for split in ("dev", "val", "test"):
        used.extend(load_split("math", split))
    used_ids = {task["id"] for task in used}
    pool_ids = {task["id"] for task in tasks}
    if not used_ids <= pool_ids:
        raise RuntimeError("frozen split contains IDs absent from the source pool")
    certificate = [task for task in tasks if task["id"] not in used_ids]
    certificate.sort(key=lambda task: sha256_bytes(
        f"{SPLIT_SEED}:{task['id']}".encode()))
    if not certificate:
        raise RuntimeError("no unused certificate tasks remain")
    if used_ids & {task["id"] for task in certificate}:
        raise RuntimeError("certificate split overlaps prior in-domain splits")

    payload = "".join(json.dumps(task, sort_keys=True) + "\n"
                      for task in certificate).encode()
    if OUT.exists() and OUT.read_bytes() != payload:
        raise RuntimeError(f"refusing to overwrite frozen certificate file: {OUT}")
    if not OUT.exists():
        OUT.write_bytes(payload)

    manifest = {
        "status": "frozen_before_model_calls",
        "source": "EleutherAI/hendrycks_math test split",
        "subjects": MATH_IN_SUBJECTS,
        "levels": sorted(MATH_LEVELS),
        "split_seed": SPLIT_SEED,
        "exclusion_rule": "exclude every task ID in frozen dev/val/test; order remaining IDs by SHA256(split_seed:task_id)",
        "n_prior_tasks": len(used),
        "n_certificate_tasks": len(certificate),
        "prior_ids_sha256": sha256_bytes("\n".join(sorted(used_ids)).encode()),
        "certificate_file": str(OUT.relative_to(ROOT)),
        "certificate_sha256": sha256_bytes(payload),
        "model": "glm-4-flash-250414",
        "candidate_set": [
            "exact reuse; 1 anchored conservative check capped at 384 tokens; one refinement on a concrete disagreement",
            "exact reuse; 1 strict checked-error verifier capped at 384 tokens; one refinement on verified disagreement",
            "no-op exact reference reuse",
        ],
        "operational_withdrawal": "a 3-check candidate was stopped after 19/150 rows before aggregate correctness inspection because endpoint token-rate limits made full certification infeasible",
        "selection_rule": "on the prior 150-task archive, retain candidates with empirical breakage <= 0.02 and zero acceptance-path breakage; maximize paired point delta; ties prefer fewer verifier calls",
        "selection_data": "existing 150-task GLM math test archive, used only before Dcert",
        "primary_risk_tolerance": 0.05,
        "secondary_risk_tolerance": 0.10,
        "alpha": 0.05,
        "certificate_bound": "task-level Hoeffding ratio UCB from Proposition 1",
        "adoption_rule": "freeze the selected candidate before Dcert; adopt only if primary UCB <= 0.05; otherwise abstain",
    }
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if MANIFEST.exists() and MANIFEST.read_text() != encoded:
        raise RuntimeError(f"refusing to overwrite frozen manifest: {MANIFEST}")
    if not MANIFEST.exists():
        MANIFEST.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
