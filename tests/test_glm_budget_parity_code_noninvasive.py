"""Mock proof for the GLM code-domain budget-parity follow-up: the
suffix-parity arithmetic holds with code's own draft cost (1024
out-tokens, not math's 1536), and no suffix gating occurs under
SUFFIX_CAPS for the prescribed verify+refine flow. Never contacts a real
model or network endpoint.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from hbws.ledger import TaskLedger, llm_call_vec, tool_call_vec
from hbws.runner import run_workflow
from hbws.dsl import wf_assign_refine, wf_incumbent_refine
from hbws import llm as llm_mod

import glm_call_logger
import glm_exact_reservation
import run_glm_budget_parity_code as gpc


CODE_TASK = {
    "id": "code_test_1", "family": "code",
    "prompt": "Write a function `add(a, b)` that returns a + b.",
    "feedback_tests": "assert add(2, 3) == 5",
    "grading_tests": "assert add(2, 3) == 5\nassert add(-1, 1) == 0",
}


def test_budget_parity_arithmetic_matches_code_draft_cost():
    """After same_policy settles exactly one draft call at CODE's own
    worst-case cost (1024 out-tokens, not math's 1536), its remaining
    suffix headroom equals assign's full headroom."""
    assign_ledger = TaskLedger(gpc.ASSIGN_CAPS)
    same_policy_ledger = TaskLedger(gpc.SAME_POLICY_CAPS)

    draft_vec = llm_call_vec(100, gpc.DRAFT_EXTRA_OUT_TOKENS)
    lease = same_policy_ledger.reserve(draft_vec)
    assert lease is not None
    same_policy_ledger.settle(lease, {"llm_calls": 1, "in_tokens": 100,
                                      "out_tokens": gpc.DRAFT_EXTRA_OUT_TOKENS})

    assert gpc.DRAFT_EXTRA_OUT_TOKENS == 1024, (
        "code's draft-extra allowance must match wf_incumbent_refine's "
        "own 'g' node cap, not math's 1536")
    for dim in ("llm_calls", "out_tokens"):
        remaining = (same_policy_ledger.caps.as_vec()[dim]
                    - same_policy_ledger.used[dim])
        assert remaining == gpc.SUFFIX_CAPS.as_vec()[dim]
        assert assign_ledger.caps.as_vec()[dim] == gpc.SUFFIX_CAPS.as_vec()[dim]


def _fake_chat_factory(calls: list, always_fail: bool):
    def fake_chat(messages, ledger, *, temperature=0.7, max_tokens=1024,
                  seed=None, use_cache=True, max_retries=5, wall_est=0.0,
                  backoff_schedule=None):
        calls.append(1)
        llm_mod.clear_last_response_meta()
        content = ("def add(a, b):\n    return a - b"  # wrong on purpose
                   if always_fail else "def add(a, b):\n    return a + b")
        vec = llm_call_vec(llm_mod.estimate_in_tokens(messages), max_tokens)
        lease = ledger.reserve(vec)
        assert lease is not None, "suffix must not be gated under SUFFIX_CAPS"
        ledger.settle(lease, {"llm_calls": 1, "in_tokens": 50, "out_tokens": 30})
        llm_mod._local.last_response_meta = {
            "cached": False, "model": "fake", "finish_reason": "stop",
            "provider_model": "fake", "system_fingerprint": None,
            "in_tokens": 50, "out_tokens": 30, "requested_max_tokens": max_tokens,
        }
        return content
    return fake_chat


def test_same_policy_worst_case_refine_loop_never_gated_under_suffix_caps():
    """A same_policy run whose draft AND every refine attempt fails
    verification (forcing all 3 allowed refine iterations) must never hit
    ReserveRejected under SAME_POLICY_CAPS -- the prescribed flow is never
    suffix-budget-gated, exactly like the math version's guarantee."""
    calls: list = []
    original_chat = llm_mod.chat
    llm_mod.chat = _fake_chat_factory(calls, always_fail=True)
    original_estimate = llm_mod.estimate_in_tokens
    glm_exact_reservation.install()
    try:
        llm_mod.set_call_context(run_name="test", task_id=CODE_TASK["id"], seed=0)
        result = run_workflow(wf_incumbent_refine(), CODE_TASK, gpc.SAME_POLICY_CAPS,
                              use_cache=False, seed=0)
    finally:
        llm_mod.chat = original_chat
        llm_mod.estimate_in_tokens = original_estimate
        llm_mod.clear_call_context()

    assert not str(result["status"]).startswith("budget_exceeded")
    assert result["status"] != "reserve_rejected"
    # v->r has no loop cap (only r->v does, at max_iter=3), so a verifier
    # that never passes drives g + 4 refines = 5 LLM calls before the 4th
    # refine's own r->v edge is finally blocked by the loop counter and
    # the run ends; verify itself is a tool call for code, never an LLM
    # call, so it never touches llm_calls.
    assert len(calls) == 5


def test_assign_worst_case_refine_loop_never_gated_under_suffix_caps():
    calls: list = []
    original_chat = llm_mod.chat
    llm_mod.chat = _fake_chat_factory(calls, always_fail=True)
    original_estimate = llm_mod.estimate_in_tokens
    glm_exact_reservation.install()
    try:
        task = {**CODE_TASK, "_assign_solution": "def add(a, b):\n    return 0"}
        llm_mod.set_call_context(run_name="test", task_id=CODE_TASK["id"], seed=0)
        result = run_workflow(wf_assign_refine(), task, gpc.ASSIGN_CAPS,
                              use_cache=False, seed=0)
    finally:
        llm_mod.chat = original_chat
        llm_mod.estimate_in_tokens = original_estimate
        llm_mod.clear_call_context()

    assert not str(result["status"]).startswith("budget_exceeded")
    assert result["status"] != "reserve_rejected"
    # assign is free; the same 4-refine worst case as same_policy (minus
    # the draft) calls the LLM 4 times.
    assert len(calls) == 4
