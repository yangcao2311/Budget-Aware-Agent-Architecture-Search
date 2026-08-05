"""Stage-1 skeleton search: template-based mutation operators + successive
halving with cost-aware selection (方案 §3.2, Algorithm 1).

Week-2 skeleton: operators and halving are functional; the full 3-stage loop
(module config + prompt mutation) lands after gate G2.
"""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path

from .dsl import (MAX_LOOP_ITER, MAX_NODES, MAX_VOTE_K, TEMPLATES,
                  InvalidWorkflow, validate)
from .ledger import BudgetCaps
from .prompts import PROMPTS
from .protocol import evaluate

GENERATIVE_PROMPTS = [p for p in PROMPTS if p != "check_math"]
# Thresholds must cover the region where the remaining-budget signal actually
# separates the deployment tiers, otherwise the policy class cannot express
# tier-dependent behaviour at all. Measured 2026-08-05 (probe_budget_signal.py):
# after generate(+verify) the min remaining fraction is ~0.62 at tight, ~0.74 at
# unseen, ~0.81 at loose, so the discriminating band is 0.62-0.81; the original
# grid (0.3-0.6) fired identically in every tier.
BRANCH_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]


# -- mutation operators ------------------------------------------------------

def _fresh_id(wf):
    used = {n["id"] for n in wf["nodes"]}
    i = 0
    while f"n{i}" in used:
        i += 1
    return f"n{i}"


def mut_add_refine_loop(wf, rng):
    """Insert verify -> (budget-gated) refine loop before END."""
    wf = copy.deepcopy(wf)
    if len(wf["nodes"]) + 3 > MAX_NODES:
        raise InvalidWorkflow("no room")
    end_edges = [e for e in wf["edges"] if e["to"] == "END" and not e.get("loop")]
    if not end_edges:
        raise InvalidWorkflow("no END edge")
    e = rng.choice(end_edges)
    v, b, r = _fresh_id(wf), None, None
    wf["nodes"].append({"id": v, "type": "verify"})
    b = _fresh_id(wf)
    wf["nodes"].append({"id": b, "type": "branch"})
    r = _fresh_id(wf)
    wf["nodes"].append({"id": r, "type": "refine", "prompt_id": "refine_from_feedback",
                        "params": {"temperature": 0.7, "max_output_tokens": 1536}})
    thr = rng.choice(BRANCH_THRESHOLDS)
    e["to"] = v
    wf["edges"] += [
        {"from": v, "to": "END", "cond": "verify_passed"},
        {"from": v, "to": b, "cond": "verify_failed"},
        {"from": b, "to": r, "cond": f"budget_above:{thr}"},
        {"from": b, "to": "END", "cond": "always"},
        {"from": r, "to": v, "loop": True, "max_iter": rng.randint(1, MAX_LOOP_ITER)},
    ]
    return wf


def mut_swap_prompt(wf, rng):
    wf = copy.deepcopy(wf)
    cands = [n for n in wf["nodes"] if n.get("prompt_id") in GENERATIVE_PROMPTS
             and n["type"] in ("generate", "vote")]
    if not cands:
        raise InvalidWorkflow("no promptable node")
    node = rng.choice(cands)
    node["prompt_id"] = rng.choice([p for p in ("solve_direct", "solve_cot")
                                    if p != node["prompt_id"]] or ["solve_cot"])
    return wf


def mut_toggle_vote(wf, rng):
    """generate <-> vote-k conversion."""
    wf = copy.deepcopy(wf)
    cands = [n for n in wf["nodes"] if n["type"] in ("generate", "vote")]
    if not cands:
        raise InvalidWorkflow("no generate/vote node to toggle")
    node = rng.choice(cands)
    if node["type"] == "generate":
        node["type"], node["k"] = "vote", rng.randint(2, MAX_VOTE_K)
        node["aggregator"] = "majority"
    else:
        node["type"] = "generate"
        node.pop("k", None), node.pop("aggregator", None)
    return wf


def mut_tweak_branch_threshold(wf, rng):
    wf = copy.deepcopy(wf)
    edges = [e for e in wf["edges"] if str(e.get("cond", "")).startswith("budget_")]
    if not edges:
        raise InvalidWorkflow("no budget branch")
    e = rng.choice(edges)
    kind = e["cond"].split(":")[0]
    e["cond"] = f"{kind}:{rng.choice(BRANCH_THRESHOLDS)}"
    return wf


def mut_add_refine_loop_static(wf, rng):
    """Static variant of mut_add_refine_loop: verify->refine loop with NO
    budget-conditioned branch (M7 Static Evolution Search operator)."""
    wf = copy.deepcopy(wf)
    if len(wf["nodes"]) + 2 > MAX_NODES:
        raise InvalidWorkflow("no room")
    end_edges = [e for e in wf["edges"] if e["to"] == "END" and not e.get("loop")]
    if not end_edges:
        raise InvalidWorkflow("no END edge")
    e = rng.choice(end_edges)
    v = _fresh_id(wf)
    wf["nodes"].append({"id": v, "type": "verify"})
    r = _fresh_id(wf)
    wf["nodes"].append({"id": r, "type": "refine", "prompt_id": "refine_from_feedback",
                        "params": {"temperature": 0.7, "max_output_tokens": 1536}})
    e["to"] = v
    wf["edges"] += [
        {"from": v, "to": "END", "cond": "verify_passed"},
        {"from": v, "to": r, "cond": "verify_failed"},
        {"from": r, "to": v, "loop": True, "max_iter": rng.randint(1, MAX_LOOP_ITER)},
    ]
    return wf


OPERATORS = [mut_add_refine_loop, mut_swap_prompt, mut_toggle_vote,
             mut_tweak_branch_threshold]
# M7 Static Evolution Search: identical operator set except every budget-
# conditioned primitive is removed (the ONLY difference from HBWS).
OPERATORS_STATIC = [mut_add_refine_loop_static, mut_swap_prompt, mut_toggle_vote]


def _has_budget_cond(wf: dict) -> bool:
    return any(str(e.get("cond", "")).startswith("budget_") for e in wf["edges"])


def mutate(wf: dict, rng: random.Random, max_tries: int = 10, *,
           static: bool = False) -> dict:
    ops = OPERATORS_STATIC if static else OPERATORS
    for _ in range(max_tries):
        try:
            child = rng.choice(ops)(wf, rng)
            validate(child)
            if static and _has_budget_cond(child):
                continue
            return child
        except InvalidWorkflow:
            continue
    return copy.deepcopy(wf)


def random_workflow(rng: random.Random, n_mutations: int = 3, *,
                    static: bool = False) -> dict:
    pool = [n for n in TEMPLATES if not (static and n == "budget_adaptive")]
    wf = TEMPLATES[rng.choice(pool)]()
    for _ in range(n_mutations):
        wf = mutate(wf, rng, static=static)
    return wf


# -- cross-budget evaluation (方案 v4.0 §3) ---------------------------------

def hoeffding_lcb(successes: int, n: int, alpha: float = 0.05) -> float:
    """Conservative anytime-usable lower bound on success rate.
    Preregistered fallback bound (PREREGISTRATION §4); may be upgraded to
    empirical-Bernstein before prereg-freeze, never after."""
    import math as _m
    if n == 0:
        return 0.0
    return max(0.0, successes / n - _m.sqrt(_m.log(2 / alpha) / (2 * n)))


def cross_budget_evaluate(wf: dict, tasks: list[dict], tiers: dict, *,
                          run_prefix: str, seed: int = 0, exec_seeds=(0,),
                          workers: int = 6) -> dict:
    """Evaluate one candidate on identical tasks under every search tier.

    Returns per-tier stats plus J_CB. Any policy-attributable violation
    (over_budget > 0, i.e. settle overrun) marks the candidate infeasible.
    Tier keys must come from SEARCH_TIERS — never 'unseen'.
    """
    from .ledger import SEARCH_TIERS
    assert all(t in SEARCH_TIERS for t in tiers), "non-search tier in search path"
    per_tier, feasible = {}, True
    for tname, caps in tiers.items():
        succ_by_task = {t["id"]: [] for t in tasks}
        usd_total = 0.0
        for es in exec_seeds:
            s = evaluate(wf, tasks, caps, run_name=f"{run_prefix}_{tname}_es{es}",
                         seed=es, use_cache=True, workers=workers)
            usd_total += s["total_usd"]
            import json as _json
            from .protocol import EXP_DIR
            rows = [_json.loads(l) for l in
                    open(EXP_DIR / f"{run_prefix}_{tname}_es{es}" / f"results_seed{es}.jsonl")]
            for r in rows:
                succ_by_task[r["task_id"]].append(bool(r["success"]))
            if s["over_budget"] > 0:
                feasible = False
        per_task = [sum(v) / len(v) for v in succ_by_task.values() if v]
        n = len(per_task)
        mean = sum(per_task) / n if n else 0.0
        per_tier[tname] = {
            "success_rate": round(mean, 4),
            "lcb": round(hoeffding_lcb(int(round(mean * n)), n), 4),
            "usd_per_task": round(usd_total / max(1, n * len(exec_seeds)), 5),
            "n": n,
        }
    lcbs = [v["lcb"] for v in per_tier.values()]
    means = [v["success_rate"] for v in per_tier.values()]
    aubpc_lcb = sum(lcbs) / len(lcbs)
    # Two scores with different jobs. j_cb (LCB-based) is CONSERVATIVE and
    # drives the racing stop rule. j_mean is fidelity-comparable and drives
    # parent selection and archive ranking: the Hoeffding penalty shrinks
    # with n, so ranking candidates by j_cb across different fidelities
    # rewards having been evaluated more, not being better.
    j_cb = 0.5 * aubpc_lcb + 0.5 * min(lcbs)
    j_mean = 0.5 * (sum(means) / len(means)) + 0.5 * min(means)
    return {"per_tier": per_tier, "aubpc_lcb": round(aubpc_lcb, 4),
            "j_cb": round(j_cb, 4), "j_mean": round(j_mean, 4),
            "feasible": feasible}


# -- successive halving ------------------------------------------------------

def cost_aware_score(summary: dict, caps: BudgetCaps) -> float:
    """s - λ·c with λ = 1/max_usd: cost normalized as fraction of the cap."""
    return summary["success_rate"] - summary["usd_per_task"] / caps.max_usd


def successive_halving(candidates: list[dict], tasks: list[dict], caps: BudgetCaps,
                       *, rungs=(10, 30, 80), keep_frac=1 / 3, run_prefix="sh",
                       seed=0) -> list[tuple[dict, dict]]:
    """Returns surviving (workflow, last_summary) pairs, best first.
    Rung task subsets are nested prefixes of the (frozen, stratified) dev list."""
    alive = [(wf, None) for wf in candidates]
    for r, n in enumerate(rungs):
        scored = []
        for i, (wf, _) in enumerate(alive):
            s = evaluate(wf, tasks[:n], caps, run_name=f"{run_prefix}_r{r}_c{i}", seed=seed)
            scored.append((cost_aware_score(s, caps), wf, s))
        scored.sort(key=lambda x: -x[0])
        keep = max(1, int(len(scored) * keep_frac)) if r < len(rungs) - 1 else len(scored)
        alive = [(wf, s) for _, wf, s in scored[:keep]]
    return alive


def pareto_front(points: list[tuple[dict, dict]]) -> list[tuple[dict, dict]]:
    """Non-dominated set over (success_rate up, usd_per_task down)."""
    front = []
    for wf, s in points:
        dominated = any(
            o["success_rate"] >= s["success_rate"] and o["usd_per_task"] <= s["usd_per_task"]
            and (o["success_rate"] > s["success_rate"] or o["usd_per_task"] < s["usd_per_task"])
            for _, o in points)
        if not dominated:
            front.append((wf, s))
    return sorted(front, key=lambda x: -x[1]["success_rate"])


def save_registry(entries: list[dict], path: str | Path):
    with open(path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
