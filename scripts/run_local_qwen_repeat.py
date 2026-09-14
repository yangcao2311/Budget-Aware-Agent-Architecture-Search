#!/usr/bin/env python3
"""Frozen same-request repeat for the local deterministic-serving control."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from run_local_qwen_control import MODEL_DIR, MODEL_ID, ROOT, env


PY = sys.executable
PREFIX = "qwen25coder7b_local_repeat_20260911_"
MANIFEST = ROOT / "experiments/qwen25coder7b_local_repeat_20260911_manifest.json"


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
        "status": "frozen before repeat calls",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "byte-identity audit under local single-request serving",
        "checkpoint": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
        "model_path": MODEL_ID,
        "model_sha256": sha(MODEL_DIR / "model.safetensors"),
        "design": {
            "family": "code",
            "profile": "loose",
            "structure": "direct",
            "tasks": 150,
            "seed": 0,
            "temperature": 0,
            "workers": 1,
            "cache": False,
            "comparison": "same seed-0 baseline from the frozen local control",
        },
        "source_hashes": {
            str(path.relative_to(ROOT)): sha(path) for path in sources
        },
    }
    if MANIFEST.exists():
        old = json.loads(MANIFEST.read_text())
        payload["frozen_at_utc"] = old["frozen_at_utc"]
        if payload != old:
            raise RuntimeError("Refusing to alter frozen local repeat")
    else:
        MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("frozen:", MANIFEST)


def main() -> None:
    freeze()
    command = [
        PY, "scripts/run_envelope.py", "--split", "test", "--families", "code",
        "--tiers", "loose", "--structures", "direct", "--n", "150", "--seed",
        "0", "--workers", "1", "--tag-prefix", PREFIX, "--no-cache",
    ]
    subprocess.run(command, cwd=ROOT, env=env(), check=True)


if __name__ == "__main__":
    main()
