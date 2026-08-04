"""Paired bootstrap CIs for method comparisons (pre-registered protocol §6).

Both result lists must be aligned per task (same task order, same split).
"""
from __future__ import annotations

import random


def paired_bootstrap_diff(a: list[bool], b: list[bool], *, n_boot: int = 10000,
                          seed: int = 0) -> dict:
    """95% CI on success-rate difference (a - b), paired by task."""
    assert len(a) == len(b) and a
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(a[i] for i in idx) / n - sum(b[i] for i in idx) / n)
    diffs.sort()
    point = sum(a) / n - sum(b) / n
    lo, hi = diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot)]
    return {"diff": round(point, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "significant": lo > 0 or hi < 0}


def holm_correction(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm-Bonferroni: name -> reject_null."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    reject, still = {}, True
    for i, (name, p) in enumerate(items):
        still = still and p < alpha / (m - i)
        reject[name] = still
    return reject
