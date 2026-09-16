#!/usr/bin/env python3
"""Load a second GLM API key for rate-limit fallback (operator-provided:
`glm2:` entry in api_key.txt, alongside the existing `glm:` entry that
scripts/run_glm4flash_replication.py's load_key() already reads).

Does not modify run_glm4flash_replication.py (existing, unmodified
source): this is purely additive. Never logs or prints either key.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_glm_keys() -> list[str]:
    """Returns [key1, key2] if both are present in api_key.txt (`glm:` and
    `glm2:` lines), else just [key1] if only the primary exists."""
    path = ROOT / "api_key.txt"
    if not path.exists():
        raise SystemExit("No api_key.txt found")
    keys: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        for prefix in ("glm2", "glm"):
            for sep in (":", "："):
                tag = f"{prefix}{sep}"
                if line.lower().startswith(tag) and prefix not in keys:
                    value = line.split(sep, 1)[1].strip()
                    if value:
                        keys[prefix] = value
    if "glm" not in keys:
        raise SystemExit("No primary `glm:` key found in api_key.txt")
    ordered = [keys["glm"]]
    if "glm2" in keys:
        ordered.append(keys["glm2"])
    return ordered


def is_rate_limit_exhausted(status: str) -> bool:
    s = status.lower()
    return ("ratelimiterror" in s or "code: 429" in s
            or "'code': '1302'" in s or "速率限制" in status)


def switch_glm_key(key: str) -> None:
    """Point every future hbws.llm call at `key`, forcing a fresh client
    (hbws.llm._client() caches one client per thread; clearing it here is
    the same, already-existing per-thread reset the module itself performs
    whenever the env changes are picked up on next access)."""
    import os
    import hbws.llm as llm_mod
    os.environ["GLM_API_KEY"] = key
    llm_mod._local.client = None
