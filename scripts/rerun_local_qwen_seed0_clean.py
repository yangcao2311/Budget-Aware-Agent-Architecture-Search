#!/usr/bin/env python3
"""Frozen clean replacement for the locally contaminated seed-0 baseline."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from run_local_qwen_control import PREFIX, ROOT, env


PY = sys.executable
ARCHIVE = ROOT / "experiments/qwen25coder7b_local_20260911_concurrency_archive"
MANIFEST = ROOT / "experiments/qwen25coder7b_local_20260911_seed0_clean_recovery_manifest.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> None:
    sources = [
        Path(__file__),
        ROOT / "scripts/run_local_qwen_control.py",
        ROOT / "scripts/openai_stop_proxy.py",
        ROOT / "scripts/run_envelope.py",
        ROOT / "hbws/runner.py",
        ROOT / "hbws/llm.py",
        ROOT / "hbws/dsl.py",
        ROOT / "hbws/prompts.py",
        ROOT / "hbws/verify.py",
        ROOT / "hbws/ledger.py",
        ROOT / "hbws/protocol.py",
    ]
    payload = {
        "status": "frozen before clean seed-0 replacement calls",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "trigger": "19 unintended local repeat requests overlapped the original seed-0 baseline",
        "rule": "rerun all 150 code/loose direct seed-0 tasks, workers=1, cache disabled, with no other local inference client; preserve the contaminated archive",
        "archive_hashes": {
            path.name: sha(path) for path in (
                ARCHIVE / "results_seed0_concurrent.jsonl",
                ARCHIVE / "summary_seed0_concurrent.json",
                ARCHIVE / "repeat_overlap_attempts.jsonl",
                ARCHIVE / "manifest.json",
            )
        },
        "source_hashes": {
            str(path.relative_to(ROOT)): sha(path) for path in sources
        },
    }
    if MANIFEST.exists():
        old = json.loads(MANIFEST.read_text())
        payload["frozen_at_utc"] = old["frozen_at_utc"]
        if payload != old:
            raise RuntimeError("Refusing to alter frozen seed-0 recovery")
    else:
        MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("frozen:", MANIFEST)


def run() -> None:
    command = [
        PY, "scripts/run_envelope.py", "--split", "test", "--families", "code",
        "--tiers", "loose", "--structures", "direct", "--n", "150", "--seed",
        "0", "--workers", "1", "--tag-prefix", PREFIX, "--no-cache",
    ]
    subprocess.run(command, cwd=ROOT, env=env(), check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "run"))
    args = parser.parse_args()
    freeze()
    if args.stage == "run":
        run()


if __name__ == "__main__":
    main()
