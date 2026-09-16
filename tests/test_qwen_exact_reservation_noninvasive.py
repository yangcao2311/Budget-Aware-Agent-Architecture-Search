"""Mock proof that the new-driver additions for the Qwen serving-regime
follow-up (reviewer B1) do not change hbws's prompts, sampling parameters,
call order, budget accounting, or final result.

Runs the same task through the same frozen workflow (hbws.dsl.wf_cot) twice
against a fake llm.chat, once with the plain hbws code path and once with
qwen_call_logger.install() + qwen_exact_reservation.install() active, and
asserts the two runs are identical in every observable way except that the
second run additionally produced a call-log file. It never contacts a real
model or network endpoint.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from hbws.dsl import wf_cot
from hbws.ledger import BUDGET_TIERS
from hbws.runner import run_workflow
from hbws import llm as llm_mod

import qwen_call_logger
import qwen_exact_reservation


TASK = {"id": "m_test_1", "family": "math", "prompt": "What is 2+2?",
        "gold_answer": "4"}


def _fake_chat_factory(calls: list):
    def fake_chat(messages, ledger, *, temperature=0.7, max_tokens=1024,
                  seed=None, use_cache=True, max_retries=5, wall_est=0.0,
                  backoff_schedule=None):
        calls.append({"messages": messages, "temperature": temperature,
                      "max_tokens": max_tokens, "seed": seed,
                      "use_cache": use_cache})
        llm_mod.clear_last_response_meta()
        content = r"The answer is \boxed{4}."
        from hbws.ledger import llm_call_vec
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
    caps = BUDGET_TIERS["tight"]

    # Baseline run: plain hbws, fake chat installed directly.
    calls_a: list = []
    original_chat = llm_mod.chat
    llm_mod.chat = _fake_chat_factory(calls_a)
    try:
        llm_mod.set_call_context(run_name="baseline", task_id=TASK["id"], seed=0)
        result_a = run_workflow(wf_cot(), TASK, caps, use_cache=False, seed=0)
    finally:
        llm_mod.chat = original_chat
        llm_mod.clear_call_context()

    # Instrumented run: fake chat installed first (as the real driver's
    # "underlying" transport would be the real vLLM call), then the logger
    # wraps whatever hbws.llm.chat currently is -- exactly the order the
    # real driver uses (install the logger once, after hbws.llm.chat is
    # already pointing at the real transport).
    calls_b: list = []
    log_path = tmp_path / "calls.jsonl"
    original_estimate = llm_mod.estimate_in_tokens
    qwen_exact_reservation.install()
    llm_mod.chat = _fake_chat_factory(calls_b)
    original_chat2 = qwen_call_logger.install(log_path, condition="TEST")
    try:
        llm_mod.set_call_context(run_name="instrumented", task_id=TASK["id"], seed=0)
        result_b = run_workflow(wf_cot(), TASK, caps, use_cache=False, seed=0)
    finally:
        qwen_call_logger.uninstall(original_chat2)
        llm_mod.chat = original_chat  # fully restore, belt-and-suspenders
        qwen_exact_reservation.uninstall(original_estimate)
        llm_mod.clear_call_context()

    # 1. Identical call sequence and arguments (prompts, params, order).
    assert len(calls_a) == len(calls_b) == 1
    assert calls_a[0]["messages"] == calls_b[0]["messages"]
    assert calls_a[0]["temperature"] == calls_b[0]["temperature"]
    assert calls_a[0]["max_tokens"] == calls_b[0]["max_tokens"]
    assert calls_a[0]["seed"] == calls_b[0]["seed"]

    # 2. Identical budget/result.
    assert result_a["status"] == result_b["status"] == "completed"
    assert result_a["solution"] == result_b["solution"]
    # wall_sec is a continuously-metered wall clock, not a reserved
    # dimension (hbws/ledger.py), so it legitimately differs run to run
    # (e.g. first-call tokenizer warmup); every reserved/settled dimension
    # must still match exactly.
    budget_a = {k: v for k, v in result_a["budget"].items() if k != "wall_sec"}
    budget_b = {k: v for k, v in result_b["budget"].items() if k != "wall_sec"}
    assert budget_a == budget_b
    assert [t["node"] for t in result_a["trace"]] == \
        [t["node"] for t in result_b["trace"]]

    # 3. The logger produced exactly one row, matching the real call.
    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["condition"] == "TEST"
    assert rows[0]["task_id"] == TASK["id"]
    assert rows[0]["node_id"] == "g"
    assert rows[0]["response_text"] == result_b["solution"]
    assert rows[0]["request"]["messages"] == calls_b[0]["messages"]


def test_exact_reservation_gives_plausible_token_count():
    messages = [{"role": "user", "content": "Compute 17*24."}]
    n = qwen_exact_reservation.exact_in_tokens(messages)
    # Qwen's chat template injects a default system message plus role
    # wrapper tokens; a short user prompt should land well under 100 tokens
    # and well above a trivial floor, without being pinned to an exact
    # brittle count (the template text itself is not this test's concern).
    assert 10 < n < 100
