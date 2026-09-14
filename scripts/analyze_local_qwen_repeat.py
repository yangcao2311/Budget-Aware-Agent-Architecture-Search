#!/usr/bin/env python3
"""Exact same-request repeat audit for the local Qwen seed-0 baseline."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hbws.verify import extract_code


BASE = (ROOT / "experiments/qwen25coder7b_local_20260911_envelope_test"
        / "direct_code_loose/results_seed0.jsonl")
REPEAT = (ROOT / "experiments/qwen25coder7b_local_repeat_20260911_envelope_test"
          / "direct_code_loose/results_seed0.jsonl")
OUT = ROOT / "experiments/qwen25coder7b_local_repeat_20260911_analysis.json"


def load(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(rows) != 150:
        raise AssertionError(f"expected 150 rows in {path}, found {len(rows)}")
    result = {row["task_id"]: row for row in rows}
    if len(result) != 150:
        raise AssertionError(f"duplicate task ids in {path}")
    return result


def response_meta(row: dict) -> dict:
    for item in row.get("trace") or []:
        if item.get("type") == "generate":
            return item.get("llm_response") or {}
    return {}


def main() -> None:
    base, repeat = load(BASE), load(REPEAT)
    if set(base) != set(repeat):
        raise AssertionError("task ids differ")
    ids = sorted(base)
    exact = [task for task in ids
             if (base[task].get("solution") or "") ==
             (repeat[task].get("solution") or "")]
    code_exact = [task for task in ids
                  if extract_code(base[task].get("solution") or "") ==
                  extract_code(repeat[task].get("solution") or "")]
    correctness_same = [task for task in ids
                        if bool(base[task].get("success")) ==
                        bool(repeat[task].get("success"))]
    report = {
        "scope": "local seed-0 direct code/loose same-request repeat",
        "n": 150,
        "byte_identical": len(exact),
        "byte_identity_rate": len(exact) / 150,
        "extracted_code_identical": len(code_exact),
        "extracted_code_identity_rate": len(code_exact) / 150,
        "same_correctness": len(correctness_same),
        "base_accuracy": sum(bool(base[t].get("success")) for t in ids) / 150,
        "repeat_accuracy": sum(bool(repeat[t].get("success")) for t in ids) / 150,
        "base_finish_reasons": dict(Counter(
            response_meta(base[t]).get("finish_reason") for t in ids)),
        "repeat_finish_reasons": dict(Counter(
            response_meta(repeat[t]).get("finish_reason") for t in ids)),
        "base_fingerprints": dict(Counter(
            response_meta(base[t]).get("system_fingerprint") for t in ids)),
        "repeat_fingerprints": dict(Counter(
            response_meta(repeat[t]).get("system_fingerprint") for t in ids)),
        "nonidentical_task_ids": sorted(set(ids) - set(exact)),
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
