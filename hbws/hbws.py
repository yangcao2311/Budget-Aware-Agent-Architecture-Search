"""HBWS: hierarchical budget-aware workflow search (方案 v4.0 Part II).

Search loop = three stages (topology -> configuration -> prompt), each using
multi-fidelity racing over nested task subsets, cross-budget joint evaluation
under J_CB, paired parent/child credit, and an empirical cross-budget archive.

Two modes, identical in every respect except one:
  budget_contingent=True   -> HBWS (M8): budget predicates available
  budget_contingent=False  -> Static Evolution Search (M7): they are not
This single flag is the causal contrast for H1b/H2.

Search dollars are metered against a cap; the loop stops when the cap is hit,
so anytime curves are directly comparable across methods (Protocol B).
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import ledger as ledger_mod
from .dsl import TEMPLATES, validate
from .ledger import BUDGET_TIERS, SEARCH_TIERS
from .search import cross_budget_evaluate, mutate, random_workflow

EXP_DIR = Path(__file__).resolve().parent.parent / "experiments"

# Human seeds available to search (PREREGISTRATION §6.5: max 3 per family).
HUMAN_SEEDS = ["direct", "cot", "strong_manual"]

FIDELITIES = [24, 64, 120]  # F1 / F2 / F3, nested prefixes of dev
# Search-phase estimates use ONE execution seed at the screening rungs and
# two at the top rung; three-seed averaging is reserved for the frozen
# confirmation runs. At n=24 the Hoeffding half-width is 0.277, so F0 can
# only screen out disasters -- paying for extra seeds there buys nothing.
EXEC_SEEDS = [(0,), (0,), (0, 1)]


@dataclass
class Candidate:
    wf: dict
    origin: str                 # "seed:<name>" or "mut:<op>"
    parent: int | None = None
    cid: int = -1
    stats: dict = field(default_factory=dict)   # fidelity level (int) -> eval dict
    j_cb: float = -1.0      # conservative (LCB): racing stop rule only
    j_mean: float = -1.0    # fidelity-comparable: selection and ranking
    j_ucb: float = 2.0      # optimistic: racing elimination
    feasible: bool = True
    fidelity: int = -1          # highest fidelity level reached
    credit: str = ""            # kept out of `stats`: that dict is int-keyed

    def record(self) -> dict:
        return {"cid": self.cid, "origin": self.origin, "parent": self.parent,
                "j_cb": self.j_cb, "j_mean": self.j_mean, "j_ucb": self.j_ucb,
                "feasible": self.feasible,
                "fidelity": self.fidelity, "credit": self.credit,
                "stats": self.stats, "wf": self.wf}


class SearchLedger:
    """Meters dollars spent by the search itself (design cost, distinct from
    deployment cost — the paper reports both plus the amortization point)."""

    def __init__(self, cap_usd: float):
        self.cap = cap_usd
        self.start = ledger_mod.GLOBAL.usd
        self.t0 = time.monotonic()

    @property
    def spent(self) -> float:
        return ledger_mod.GLOBAL.usd - self.start

    @property
    def exhausted(self) -> bool:
        return self.spent >= self.cap

    def snapshot(self) -> dict:
        return {"search_usd": round(self.spent, 4),
                "search_min": round((time.monotonic() - self.t0) / 60, 1)}


def _archive_update(archive: list[Candidate], cand: Candidate) -> bool:
    """Empirical cross-budget archive: non-dominated over (per-tier success
    up, per-tier $/task down). Returns True if the candidate entered it."""
    if not cand.feasible or not cand.stats:
        return False
    top = cand.stats[max(cand.stats)]

    def vec(c):
        t = c.stats[max(c.stats)]["per_tier"]
        out = []
        for tier in sorted(t):
            out += [t[tier]["success_rate"], -t[tier]["usd_per_task"]]
        return out

    v = vec(cand)
    for other in archive:
        o = vec(other)
        if len(o) == len(v) and all(a >= b for a, b in zip(o, v)) and any(a > b for a, b in zip(o, v)):
            return False
    archive[:] = [o for o in archive
                  if not (all(a >= b for a, b in zip(v, vec(o)))
                          and any(a > b for a, b in zip(v, vec(o))))]
    archive.append(cand)
    return True


def _credit(child: Candidate, parent: Candidate, eps: float = 0.01) -> str:
    """Cross-budget parent/child credit (方案 §3): positive requires no tier
    regressing beyond eps and at least one tier improving beyond eps."""
    if not child.feasible:
        return "negative"
    lvl = max(set(child.stats) & set(parent.stats), default=None)
    if lvl is None:
        return "neutral"
    ct, pt = child.stats[lvl]["per_tier"], parent.stats[lvl]["per_tier"]
    deltas = [ct[t]["success_rate"] - pt[t]["success_rate"] for t in ct if t in pt]
    if not deltas:
        return "neutral"
    if any(d < -eps for d in deltas):
        return "negative"
    if any(d > eps for d in deltas):
        return "positive"
    return "neutral"


def hbws_search(family: str, tasks: list[dict], *, cap_usd: float,
                budget_contingent: bool = True, seed: int = 0,
                run_name: str | None = None, workers: int = 8,
                tiers: tuple = SEARCH_TIERS, log_every: int = 1) -> dict:
    """Run one search. Returns archive + full registry + anytime curve."""
    rng = random.Random(20260805 + seed)
    mode = "hbws" if budget_contingent else "static"
    run_name = run_name or f"{mode}_{family}_s{seed}"
    out_dir = EXP_DIR / "search" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    sled = SearchLedger(cap_usd)
    tier_caps = {t: BUDGET_TIERS[t] for t in tiers}
    registry: list[Candidate] = []
    archive: list[Candidate] = []
    anytime: list[dict] = []
    next_id = 0

    def evaluate_candidate(c: Candidate, upto_level: int) -> None:
        """Evaluate through fidelity levels with racing; fills c.stats."""
        for lvl in range(upto_level + 1):
            if sled.exhausted:
                return
            n = FIDELITIES[lvl]
            res = cross_budget_evaluate(
                c.wf, tasks[:n], tier_caps,
                run_prefix=f"search/{run_name}/c{c.cid}_f{lvl}",
                seed=seed, exec_seeds=EXEC_SEEDS[lvl], workers=workers)
            c.stats[lvl] = res
            c.j_cb = res["j_cb"]
            c.j_mean = res["j_mean"]
            c.j_ucb = res["j_ucb"]
            c.feasible = c.feasible and res["feasible"]
            c.fidelity = lvl
            if not c.feasible:
                return
            # Racing: stop promoting if the optimistic bound cannot reach the
            # archive's incumbent J_CB (conservative slack of one CI width).
            if archive and lvl < upto_level:
                # Standard racing: eliminate only when even the OPTIMISTIC
                # view of this candidate cannot reach the incumbent's
                # conservative bound. Comparing LCB-to-LCB with a fixed slack
                # made promotion from F0 mathematically impossible.
                peers = [a.j_cb for a in archive if a.fidelity >= lvl]
                if peers and c.j_ucb < max(peers):
                    return

    # -- seed population ----------------------------------------------------
    pool = [n for n in HUMAN_SEEDS]
    for name in pool:
        c = Candidate(wf=TEMPLATES[name](), origin=f"seed:{name}", cid=next_id)
        next_id += 1
        registry.append(c)
    for _ in range(3):
        c = Candidate(wf=random_workflow(rng, 2, static=not budget_contingent),
                      origin="seed:random", cid=next_id)
        next_id += 1
        registry.append(c)

    for c in registry:
        if sled.exhausted:
            break
        evaluate_candidate(c, upto_level=1)
        _archive_update(archive, c)
        anytime.append({**sled.snapshot(), "cid": c.cid,
                        "best_j_cb": max((a.j_cb for a in archive), default=-1),
                        "best_j_mean": max((a.j_mean for a in archive), default=-1)})

    # -- evolutionary loop --------------------------------------------------
    generation = 0
    while not sled.exhausted:
        generation += 1
        parents = sorted([c for c in registry if c.feasible and c.stats],
                         key=lambda c: -c.j_mean)[:4]
        if not parents:
            break
        progressed = False
        for parent in parents:
            if sled.exhausted:
                break
            child_wf = mutate(parent.wf, rng, static=not budget_contingent)
            try:
                validate(child_wf)
            except Exception:
                continue
            if json.dumps(child_wf, sort_keys=True) == json.dumps(parent.wf, sort_keys=True):
                continue
            child = Candidate(wf=child_wf, origin=f"mut:gen{generation}",
                              parent=parent.cid, cid=next_id)
            next_id += 1
            registry.append(child)
            progressed = True
            # Children start at F1 and race upward to the parent's fidelity.
            evaluate_candidate(child, upto_level=min(2, parent.fidelity + 1))
            child.credit = _credit(child, parent)
            entered = _archive_update(archive, child)
            anytime.append({**sled.snapshot(), "cid": child.cid,
                            "credit": child.credit, "archive": entered,
                            "best_j_cb": max((a.j_cb for a in archive), default=-1),
                            "best_j_mean": max((a.j_mean for a in archive), default=-1)})
            if generation % log_every == 0:
                print(f"[{run_name}] gen{generation} c{child.cid} "
                      f"j_mean={child.j_mean:.3f} credit={child.credit} "
                      f"spent=${sled.spent:.2f}/{cap_usd}")
        if not progressed:
            break

    # -- promote archive members to full fidelity ---------------------------
    for c in sorted(archive, key=lambda c: -c.j_mean)[:3]:
        if not sled.exhausted and c.fidelity < 2:
            evaluate_candidate(c, upto_level=2)

    result = {
        "run": run_name, "family": family, "mode": mode, "seed": seed,
        "cap_usd": cap_usd, **sled.snapshot(),
        "n_candidates": len(registry),
        "archive": [c.record() for c in sorted(archive, key=lambda c: -c.j_mean)],
        "anytime": anytime,
    }
    with open(out_dir / "search_result.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(out_dir / "registry.jsonl", "w") as f:
        for c in registry:
            f.write(json.dumps(c.record()) + "\n")
    print(f"[{run_name}] done: {len(registry)} candidates, "
          f"{len(archive)} in archive, ${sled.spent:.2f} spent")
    return result
