#!/usr/bin/env python3
"""Fail closed on silent provider/runtime failures in the assign-search run."""
from __future__ import annotations

from collections import Counter
import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
OUT = EXP / "assign_search_intervention_20260909"
RUN = EXP / "search/glm4flash_assign_exposed_code_s0"
REFERENCE_GLOB = "glm4flash_assign_search_20260909_*/direct_code_loose/results_seed0.jsonl"
FIDELITY_N = {0: 24, 1: 64, 2: 120}


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-only", action="store_true")
    args = parser.parse_args()
    references = list(EXP.glob(REFERENCE_GLOB))
    assert len(references) == 1, f"expected one reference file, found {references}"
    reference_rows = load(references[0])
    assert len(reference_rows) == 120
    assert len({row["task_id"] for row in reference_rows}) == 120
    reference_status = Counter(row.get("status") for row in reference_rows)
    assert reference_status == {"completed": 120}, reference_status
    assert all(row.get("solution") for row in reference_rows), "empty stored reference"
    if args.reference_only:
        print("assign-search reference audit passed: 120/120 completed and nonempty")
        return

    result_path = RUN / "search_result.json"
    analysis_path = OUT / "analysis.json"
    assert result_path.exists() and analysis_path.exists(), "search analysis incomplete"
    result = json.loads(result_path.read_text())
    analysis = json.loads(analysis_path.read_text())
    assert result["n_candidates"] == analysis["n_candidates"] > 0

    files = sorted(RUN.glob("c*_f*_*_es*/results_seed*.jsonl"))
    assert files, "no candidate evaluation files"
    status = Counter()
    rows_seen = 0
    for path in files:
        match = re.search(r"_f([0-2])_", path.parent.name)
        assert match, path
        expected = FIDELITY_N[int(match.group(1))]
        rows = load(path)
        assert len(rows) == expected, (path, len(rows), expected)
        assert len({row["task_id"] for row in rows}) == expected, path
        status.update(row.get("status") for row in rows)
        rows_seen += len(rows)
    defects = sum(count for name, count in status.items()
                  if name not in {"completed", "reserve_rejected"})
    assert defects == 0, f"search provider/runtime defects: {status}"

    report = {
        "status": "passed",
        "reference_rows": len(reference_rows),
        "reference_status_counts": dict(reference_status),
        "candidate_result_files": len(files),
        "candidate_rows": rows_seen,
        "candidate_status_counts": dict(status),
        "provider_runtime_defects": defects,
        "n_candidates": result["n_candidates"],
    }
    path = OUT / "integrity_audit.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
