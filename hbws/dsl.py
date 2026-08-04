"""Restricted workflow DSL: schema, static validator, and hand-designed templates.

A workflow is a dict {nodes, edges, budget}. Static validation rejects
anything outside the whitelist BEFORE any API spend.

Edge conditions (whitelist):
  always | verify_passed | verify_failed | budget_below:<f> | budget_above:<f>
The budget_* conditions are the budget-adaptive primitive unique to HBWS.
"""
from __future__ import annotations

from .prompts import PROMPTS

NODE_TYPES = {"generate", "refine", "vote", "verify", "decompose", "aggregate", "branch"}
MAX_NODES = 8
MAX_LOOPS = 2
MAX_LOOP_ITER = 3
MAX_VOTE_K = 5


class InvalidWorkflow(Exception):
    pass


def _cond_ok(cond: str) -> bool:
    if cond in ("always", "verify_passed", "verify_failed"):
        return True
    for prefix in ("budget_below:", "budget_above:"):
        if cond.startswith(prefix):
            try:
                f = float(cond[len(prefix):])
                return 0.0 < f < 1.0
            except ValueError:
                return False
    return False


def validate(wf: dict) -> None:
    """Raise InvalidWorkflow on any violation. Costs zero API budget."""
    nodes = wf.get("nodes", [])
    edges = wf.get("edges", [])
    if not nodes or len(nodes) > MAX_NODES:
        raise InvalidWorkflow(f"node count {len(nodes)} not in [1,{MAX_NODES}]")
    ids = [n["id"] for n in nodes]
    if len(set(ids)) != len(ids):
        raise InvalidWorkflow("duplicate node ids")
    idset = set(ids)
    for n in nodes:
        if n["type"] not in NODE_TYPES:
            raise InvalidWorkflow(f"node type {n['type']} not whitelisted")
        if n["type"] == "vote" and not (2 <= n.get("k", 0) <= MAX_VOTE_K):
            raise InvalidWorkflow(f"vote k={n.get('k')} not in [2,{MAX_VOTE_K}]")
        pid = n.get("prompt_id")
        if pid is not None and pid not in PROMPTS:
            raise InvalidWorkflow(f"unknown prompt_id {pid}")

    n_loops = 0
    for e in edges:
        if e["from"] not in idset:
            raise InvalidWorkflow(f"edge from unknown node {e['from']}")
        if e["to"] != "END" and e["to"] not in idset:
            raise InvalidWorkflow(f"edge to unknown node {e['to']}")
        if not _cond_ok(e.get("cond", "always")):
            raise InvalidWorkflow(f"illegal condition {e.get('cond')}")
        if e.get("loop"):
            n_loops += 1
            if not (1 <= e.get("max_iter", 0) <= MAX_LOOP_ITER):
                raise InvalidWorkflow(f"loop max_iter {e.get('max_iter')} not in [1,{MAX_LOOP_ITER}]")
    if n_loops > MAX_LOOPS:
        raise InvalidWorkflow(f"{n_loops} loops > {MAX_LOOPS}")

    # Reachability: entry node must reach END via non-loop edges.
    entry = ids[0]
    adj: dict[str, list[str]] = {}
    for e in edges:
        if not e.get("loop"):
            adj.setdefault(e["from"], []).append(e["to"])
    seen, stack = set(), [entry]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(adj.get(cur, []))
    if "END" not in seen:
        raise InvalidWorkflow("END unreachable from entry via non-loop edges")


# ---------------------------------------------------------------------------
# Hand-designed skeletons. These serve three roles: (1) baselines M1-M3,
# (2) Stage-1 initial population, (3) unit tests for the runner.
# ---------------------------------------------------------------------------

def wf_direct() -> dict:  # M1
    return {
        "nodes": [{"id": "g", "type": "generate", "prompt_id": "solve_direct",
                   "params": {"temperature": 0.0, "max_output_tokens": 1024}}],
        "edges": [{"from": "g", "to": "END"}],
    }


def wf_cot_check() -> dict:  # M2
    return {
        "nodes": [
            {"id": "g", "type": "generate", "prompt_id": "solve_cot",
             "params": {"temperature": 0.0, "max_output_tokens": 1536}},
            {"id": "r", "type": "refine", "prompt_id": "self_check",
             "params": {"temperature": 0.0, "max_output_tokens": 1536}},
        ],
        "edges": [{"from": "g", "to": "r"}, {"from": "r", "to": "END"}],
    }


def wf_strong_manual() -> dict:  # M3: generate -> verify -> (refine loop x2) -> END
    return {
        "nodes": [
            {"id": "g", "type": "generate", "prompt_id": "solve_cot",
             "params": {"temperature": 0.7, "max_output_tokens": 1536}},
            {"id": "v", "type": "verify"},
            {"id": "r", "type": "refine", "prompt_id": "refine_from_feedback",
             "params": {"temperature": 0.7, "max_output_tokens": 1536}},
        ],
        "edges": [
            {"from": "g", "to": "v"},
            {"from": "v", "to": "END", "cond": "verify_passed"},
            {"from": "v", "to": "r", "cond": "verify_failed"},
            {"from": "r", "to": "v", "loop": True, "max_iter": 2},
        ],
    }


def wf_vote3() -> dict:
    return {
        "nodes": [
            {"id": "g", "type": "vote", "k": 3, "prompt_id": "solve_cot",
             "aggregator": "majority", "params": {"temperature": 0.8, "max_output_tokens": 1536}},
        ],
        "edges": [{"from": "g", "to": "END"}],
    }


def wf_budget_adaptive() -> dict:
    """Illustrative budget-adaptive skeleton: refine while budget remains,
    bail out to END when the remaining fraction drops below 0.4."""
    return {
        "nodes": [
            {"id": "g", "type": "generate", "prompt_id": "solve_cot",
             "params": {"temperature": 0.7, "max_output_tokens": 1536}},
            {"id": "v", "type": "verify"},
            {"id": "b", "type": "branch"},
            {"id": "r", "type": "refine", "prompt_id": "refine_from_feedback",
             "params": {"temperature": 0.7, "max_output_tokens": 1536}},
        ],
        "edges": [
            {"from": "g", "to": "v"},
            {"from": "v", "to": "END", "cond": "verify_passed"},
            {"from": "v", "to": "b", "cond": "verify_failed"},
            {"from": "b", "to": "r", "cond": "budget_above:0.4"},
            {"from": "b", "to": "END", "cond": "always"},
            {"from": "r", "to": "v", "loop": True, "max_iter": 3},
        ],
    }


TEMPLATES = {
    "direct": wf_direct,
    "cot_check": wf_cot_check,
    "strong_manual": wf_strong_manual,
    "vote3": wf_vote3,
    "budget_adaptive": wf_budget_adaptive,
}
