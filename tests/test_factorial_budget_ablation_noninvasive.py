"""Mock proof for the factorial budget ablation: the per-iteration logger
(glm_iteration_logger.py) changes neither the request/sampling
parameters/call order nor the reservation/settlement path versus the
plain, unmodified hbws code path, and correctly records both admitted and
refused reservations. Never contacts a real model or network endpoint.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from hbws.dsl import wf_incumbent_refine
from hbws.ledger import BudgetCaps, ReserveRejected
from hbws.runner import run_workflow
from hbws import llm as llm_mod

import glm_exact_reservation
import glm_iteration_logger
import run_factorial_budget_ablation as fba


CODE_TASK = {
    "id": "fba_test_1", "family": "code",
    "prompt": "Write a function `add(a, b)` that returns a + b.",
    "feedback_tests": "",  # fully masked
    "grading_tests": "assert add(2, 3) == 5",
}


def test_corner_cells_equal_budget_tiers_exactly():
    assert fba.CELLS["A"] == fba.TIGHT
    assert fba.CELLS["D"] == fba.LOOSE
    # off-diagonal cells mix exactly one named axis from each tier
    assert fba.CELLS["B"].max_llm_calls == fba.TIGHT.max_llm_calls
    assert fba.CELLS["B"].max_out_tokens == fba.LOOSE.max_out_tokens
    assert fba.CELLS["C"].max_llm_calls == fba.LOOSE.max_llm_calls
    assert fba.CELLS["C"].max_out_tokens == fba.TIGHT.max_out_tokens


def _fake_chat_factory(calls: list, fail_after: int | None = None):
    def fake_chat(messages, ledger, *, temperature=0.7, max_tokens=1024,
                  seed=None, use_cache=True, max_retries=5, wall_est=0.0,
                  backoff_schedule=None):
        calls.append(1)
        llm_mod.clear_last_response_meta()
        content = "def add(a, b):\n    return a + b"
        from hbws.ledger import llm_call_vec
        vec = llm_call_vec(llm_mod.estimate_in_tokens(messages), max_tokens)
        lease = ledger.reserve(vec)
        if lease is None:
            raise ReserveRejected(f"cannot reserve {vec}")
        ledger.settle(lease, {"llm_calls": 1, "in_tokens": 50, "out_tokens": 30})
        llm_mod._local.last_response_meta = {
            "cached": False, "model": "fake", "finish_reason": "stop",
            "provider_model": "fake", "system_fingerprint": None,
            "in_tokens": 50, "out_tokens": 30, "requested_max_tokens": max_tokens,
        }
        return content
    return fake_chat


def test_logger_does_not_change_request_or_budget(tmp_path):
    # Under full masking, verify() never passes regardless of correctness
    # (no-signal semantics, hbws/verify.py), so even a correct first draft
    # is forced through the full refine loop (g + 4 refines = 5 LLM
    # calls -- see run_factorial_budget_ablation.py's module docstring).
    # Use the loose corner (max_llm_calls=8) so this comparison isolates
    # "does the logger change behavior", not "does this cap gate at all".
    caps = BudgetCaps(8, 16000, 4000, 6, 180, 0.25)

    calls_a: list = []
    original_chat = llm_mod.chat
    llm_mod.chat = _fake_chat_factory(calls_a)
    try:
        llm_mod.set_call_context(run_name="baseline", task_id=CODE_TASK["id"], seed=0)
        result_a = run_workflow(wf_incumbent_refine(), CODE_TASK, caps, use_cache=False, seed=0)
    finally:
        llm_mod.chat = original_chat
        llm_mod.clear_call_context()

    calls_b: list = []
    log_path = tmp_path / "calls.jsonl"
    original_estimate = llm_mod.estimate_in_tokens
    glm_exact_reservation.install()
    llm_mod.chat = _fake_chat_factory(calls_b)
    original_chat2 = glm_iteration_logger.install(log_path, cell="TEST")
    try:
        llm_mod.set_call_context(run_name="instrumented", task_id=CODE_TASK["id"], seed=0)
        result_b = run_workflow(wf_incumbent_refine(), CODE_TASK, caps, use_cache=False, seed=0)
    finally:
        glm_iteration_logger.uninstall(original_chat2)
        llm_mod.chat = original_chat
        glm_exact_reservation.uninstall(original_estimate)
        llm_mod.clear_call_context()

    assert len(calls_a) == len(calls_b)
    budget_a = {k: v for k, v in result_a["budget"].items() if k != "wall_sec"}
    budget_b = {k: v for k, v in result_b["budget"].items() if k != "wall_sec"}
    assert budget_a == budget_b
    assert result_a["status"] == result_b["status"]

    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(rows) == len(calls_b)
    assert all(r["cell"] == "TEST" for r in rows)
    assert all(r["reservation_refused"] is False for r in rows)
    assert all("reservation_request" in r and "llm_calls" in r["reservation_request"]
              for r in rows)


def test_logger_records_reservation_refusal():
    """A cap too small for even one call must show up as a logged,
    reservation_refused row -- not silently vanish."""
    tiny_caps = BudgetCaps(max_llm_calls=0, max_in_tokens=8000, max_out_tokens=2000,
                           max_tool_calls=4, max_wall_sec=90, max_usd=0.10)
    calls: list = []
    log_path_holder = []
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "calls.jsonl"
        original_estimate = llm_mod.estimate_in_tokens
        real_chat = llm_mod.chat
        glm_exact_reservation.install()
        llm_mod.chat = _fake_chat_factory(calls)
        original_chat = glm_iteration_logger.install(log_path, cell="TEST2")
        try:
            llm_mod.set_call_context(run_name="t", task_id=CODE_TASK["id"], seed=0)
            result = run_workflow(wf_incumbent_refine(), CODE_TASK, tiny_caps,
                                  use_cache=False, seed=0)
        finally:
            glm_iteration_logger.uninstall(original_chat)
            llm_mod.chat = real_chat
            glm_exact_reservation.uninstall(original_estimate)
            llm_mod.clear_call_context()

        assert result["status"] == "reserve_rejected"
        rows = [json.loads(line) for line in log_path.read_text().splitlines()]
        assert len(rows) == 1
        assert rows[0]["reservation_refused"] is True
        assert rows[0]["response_text"] is None
