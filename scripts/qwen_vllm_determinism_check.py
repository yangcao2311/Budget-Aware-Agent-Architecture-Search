#!/usr/bin/env python3
"""Determinism / byte-identical-rate check for the local Qwen/vLLM math/tight
nonbinding control.

For every task whose arm2 (same-policy regeneration, cache off, same seed)
trajectory was accepted at the first verify (no refinement fired), the arm2
first-draft solution is compared byte-for-byte against the stored cot
reference's solution for the same task/seed. Both come from an independent
`generate` call at temperature=0 with the same prompt and forwarded seed
against the local vLLM server (--max-num-seqs=1, prefix caching disabled),
so a byte-identical rate of 1.0 is the expected signature of deterministic
serving; anything less would indicate nondeterminism in this vLLM
configuration on this hardware.

Linux/vLLM-specific analysis script; reads existing JSONL results only, no
LLM calls.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEEDS = (0, 1, 2)


def load(path: Path) -> dict[str, dict]:
    return {json.loads(line)["task_id"]: json.loads(line)
            for line in path.read_text().splitlines() if line}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-dir", type=Path, required=True)
    ap.add_argument("--arm2-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    base = {s: load(args.baseline_dir / f"results_seed{s}.jsonl") for s in SEEDS}
    arm2 = {s: load(args.arm2_dir / f"results_seed{s}.jsonl") for s in SEEDS}

    identical, compared, mismatches = 0, 0, []
    for s in SEEDS:
        for tid, row in arm2[s].items():
            trace = row.get("trace") or []
            first_verify = next((t for t in trace if t.get("type") == "verify"), None)
            if first_verify is None:
                continue
            idx = trace.index(first_verify)
            following = trace[idx + 1] if idx + 1 < len(trace) else {}
            if following.get("type") == "refine":
                continue  # rejected at first verify: not a pure first-draft comparison
            ref_sol = base[s][tid].get("solution", "")
            arm_sol = row.get("solution", "")
            compared += 1
            if ref_sol == arm_sol:
                identical += 1
            else:
                mismatches.append({"seed": s, "task_id": tid})

    report = {
        "purpose": "byte-identical determinism check: same prompt, "
                   "temperature=0, same forwarded seed, independent local "
                   "vLLM calls (reference cot generation vs. arm2 "
                   "same-policy regeneration, cache disabled both times)",
        "compared": compared,
        "byte_identical": identical,
        "byte_identical_rate": identical / compared if compared else None,
        "mismatches": mismatches,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
