"""Mock proof for the GLM budget-parity follow-up: installing
glm_exact_reservation + glm_call_logger changes neither the request
messages/sampling parameters/call order nor the reserved/settled budget
dimensions that matter for suffix reachability, versus the plain hbws code
path. Never contacts a real model or network endpoint.

Also checks the core budget-parity arithmetic itself: after the
same_policy arm's one extra draft call settles against SAME_POLICY_CAPS,
its REMAINING headroom for the verify+refine suffix is byte-for-byte the
same vector as the assign arm's full ASSIGN_CAPS headroom (before assign's
own zero-cost `assign` node, which never touches the ledger).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from hbws.dsl import wf_cot
from hbws.ledger import BudgetCaps, TaskLedger, llm_call_vec
from hbws.runner import run_workflow
from hbws import llm as llm_mod

import glm_call_logger
import glm_exact_reservation
import run_glm_budget_parity as gp


TASK = {"id": "m_test_1", "family": "math", "prompt": "What is 2+2?",
        "gold_answer": "4"}


def _fake_chat_factory(calls: list):
    def fake_chat(messages, ledger, *, temperature=0.7, max_tokens=1024,
                  seed=None, use_cache=True, max_retries=5, wall_est=0.0,
                  backoff_schedule=None):
        calls.append({"messages": messages, "temperature": temperature,
                      "max_tokens": max_tokens, "seed": seed})
        llm_mod.clear_last_response_meta()
        content = r"The answer is \boxed{4}."
        vec = llm_call_vec(llm_mod.estimate_in_tokens(messages), max_tokens)
        lease = ledger.reserve(vec)
        assert lease is not None
        ledger.settle(lease, {"llm_calls": 1, "in_tokens": 10, "out_tokens": 8})
        llm_mod._local.last_response_meta = {
            "cached": False, "model": "fake", "finish_reason": "stop",
            "provider_model": "fake", "system_fingerprint": None,
            "in_tokens": 10, "out_tokens": 8, "requested_max_tokens": max_tokens,
        }
        return content
    return fake_chat


def test_logging_and_exact_reservation_do_not_change_behavior(tmp_path):
    caps = BudgetCaps(4, 8000, 2000, 4, 90, 0.10)

    calls_a: list = []
    original_chat = llm_mod.chat
    llm_mod.chat = _fake_chat_factory(calls_a)
    try:
        llm_mod.set_call_context(run_name="baseline", task_id=TASK["id"], seed=0)
        result_a = run_workflow(wf_cot(), TASK, caps, use_cache=False, seed=0)
    finally:
        llm_mod.chat = original_chat
        llm_mod.clear_call_context()

    calls_b: list = []
    log_path = tmp_path / "calls.jsonl"
    original_estimate = llm_mod.estimate_in_tokens
    glm_exact_reservation.install()
    llm_mod.chat = _fake_chat_factory(calls_b)
    original_chat2 = glm_call_logger.install(log_path, condition="TEST")
    try:
        llm_mod.set_call_context(run_name="instrumented", task_id=TASK["id"], seed=0)
        result_b = run_workflow(wf_cot(), TASK, caps, use_cache=False, seed=0)
    finally:
        glm_call_logger.uninstall(original_chat2)
        llm_mod.chat = original_chat
        glm_exact_reservation.uninstall(original_estimate)
        llm_mod.clear_call_context()

    assert len(calls_a) == len(calls_b) == 1
    assert calls_a[0]["messages"] == calls_b[0]["messages"]
    assert calls_a[0]["temperature"] == calls_b[0]["temperature"]
    assert calls_a[0]["max_tokens"] == calls_b[0]["max_tokens"]
    assert calls_a[0]["seed"] == calls_b[0]["seed"]

    budget_a = {k: v for k, v in result_a["budget"].items() if k != "wall_sec"}
    budget_b = {k: v for k, v in result_b["budget"].items() if k != "wall_sec"}
    assert budget_a == budget_b

    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["condition"] == "TEST"
    assert rows[0]["response_text"] == result_b["solution"]


def test_exact_reservation_extracts_real_token_ids_not_dict_key_count():
    """Regression test: apply_chat_template returns a BatchEncoding (a dict
    of input_ids/attention_mask/position_ids), not a bare list -- an
    earlier version of exact_in_tokens took len() of that dict directly and
    silently returned ~3 (the key count) instead of the real token count,
    which under-reserved every real preflight call and produced
    budget_exceeded:settle_overrun:in_tokens on 10/10 tasks. A long prompt
    must estimate proportionally more tokens than a short one."""
    short = glm_exact_reservation.exact_in_tokens(
        [{"role": "user", "content": "Hi."}])
    long = glm_exact_reservation.exact_in_tokens(
        [{"role": "user", "content": "Solve this: " + "x " * 200 + "= 0."}])
    assert short > glm_exact_reservation._PROXY_SAFETY_MARGIN + 3, (
        "estimate must reflect real token count, not dict-key count")
    assert long > short + 100


def test_budget_parity_arithmetic():
    """After the same_policy arm settles exactly one draft call at its own
    worst-case cost, its remaining ledger headroom for the suffix equals
    the assign arm's full (untouched) headroom -- the point of the whole
    separate-accounting design."""
    assign_ledger = TaskLedger(gp.ASSIGN_CAPS)
    # assign's own node is zero-cost: nothing reserved/settled before verify.

    same_policy_ledger = TaskLedger(gp.SAME_POLICY_CAPS)
    draft_vec = llm_call_vec(100, gp.DRAFT_EXTRA_OUT_TOKENS)
    lease = same_policy_ledger.reserve(draft_vec)
    assert lease is not None
    same_policy_ledger.settle(lease, {"llm_calls": 1, "in_tokens": 100,
                                      "out_tokens": gp.DRAFT_EXTRA_OUT_TOKENS})

    for dim in ("llm_calls", "out_tokens"):
        assert (assign_ledger.caps.as_vec()[dim]
                == gp.SUFFIX_CAPS.as_vec()[dim])
        remaining_same_policy = (same_policy_ledger.caps.as_vec()[dim]
                                - same_policy_ledger.used[dim])
        assert remaining_same_policy == gp.SUFFIX_CAPS.as_vec()[dim], (
            f"{dim}: same_policy's post-draft remaining budget must equal "
            "assign's full suffix budget")
