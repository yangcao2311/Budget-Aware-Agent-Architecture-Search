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
BRANCH_THRESHOLDS = [0.3, 0.4, 0.5, 0.6]


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
    node = rng.choice([n for n in wf["nodes"] if n["type"] in ("generate", "vote")])
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


OPERATORS = [mut_add_refine_loop, mut_swap_prompt, mut_toggle_vote,
             mut_tweak_branch_threshold]


def mutate(wf: dict, rng: random.Random, max_tries: int = 10) -> dict:
    for _ in range(max_tries):
        try:
            child = rng.choice(OPERATORS)(wf, rng)
            validate(child)
            return child
        except InvalidWorkflow:
            continue
    return copy.deepcopy(wf)


def random_workflow(rng: random.Random, n_mutations: int = 3) -> dict:
    wf = TEMPLATES[rng.choice(list(TEMPLATES))]()
    for _ in range(n_mutations):
        wf = mutate(wf, rng)
    return wf


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
