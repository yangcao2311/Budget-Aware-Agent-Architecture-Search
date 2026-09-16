#!/usr/bin/env python3
"""Full-request/response call logger for the local Qwen serving-regime
control (reviewer B1 follow-up).

Wraps hbws.llm.chat by monkeypatching, without editing hbws/llm.py or
hbws/runner.py. The wrapper calls the ORIGINAL chat() with the exact same
positional/keyword arguments it was given -- it does not alter the request
messages, temperature, max_tokens, seed, use_cache, or call order, and it
does not touch the reservation/settlement path at all (that stays inside
the original chat()). It only records, per call:

  - the outgoing request (model, messages, temperature, max_tokens, seed;
    no API key -- the client never sees one worth logging, since VLLM_API_KEY
    is the fixed placeholder "local"),
  - the exact response text returned,
  - finish_reason / usage / provider_model from llm.last_response_meta(),
  - the node context already tracked by hbws.runner (node_id, node_type)
    and by the driver (run_name, task_id, seed) via llm's own
    thread-local call-context mechanism (set_call_context /
    update_call_context -- both existing, unmodified hbws.llm functions).

The "first draft" for a task+seed+condition is the row with node_id == "g"
(the workflow's first node in both wf_incumbent_refine_cot and the assign
workflow's downstream is not used here -- this control only runs the
same-policy suffix, wf_incumbent_refine_cot, whose first node is always
"g") -- identified directly from this per-call log, never by inspecting or
subtracting the final `solution` field.

See tests/test_qwen_exact_reservation_noninvasive.py for the mock proof
that installing this wrapper changes neither the sequence of calls, their
arguments, budget usage, nor the final result, versus the same task run
without it.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

_lock = threading.Lock()
_installed = False
_log_path: Path | None = None
_condition: str | None = None


def _write(row: dict) -> None:
    assert _log_path is not None
    with _lock:
        with _log_path.open("a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def install(log_path: Path, condition: str):
    """Monkeypatch hbws.llm.chat in place. Returns the original function so
    the caller can uninstall() later. Idempotent per-process is NOT assumed
    -- call once per process."""
    global _installed, _log_path, _condition
    import hbws.llm as llm_mod

    _log_path = log_path
    _condition = condition
    original_chat = llm_mod.chat

    def logging_chat(messages, ledger, *, temperature=0.7, max_tokens=1024,
                      seed=None, use_cache=True, max_retries=5, wall_est=0.0,
                      backoff_schedule=None):
        context = dict(getattr(llm_mod._local, "call_context", {}))
        t0 = time.time()
        try:
            content = original_chat(
                messages, ledger, temperature=temperature,
                max_tokens=max_tokens, seed=seed, use_cache=use_cache,
                max_retries=max_retries, wall_est=wall_est,
                backoff_schedule=backoff_schedule)
            t1 = time.time()
            meta = llm_mod.last_response_meta() or {}
            _write({
                "condition": _condition,
                "run_name": context.get("run_name"),
                "task_id": context.get("task_id"),
                "seed": context.get("seed"),
                "node_id": context.get("node_id"),
                "node_type": context.get("node_type"),
                "request": {
                    "model": llm_mod._model_name(),
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "seed": seed,
                },
                "response_text": content,
                "finish_reason": meta.get("finish_reason"),
                "provider_model": meta.get("provider_model"),
                "cached": meta.get("cached"),
                "in_tokens": meta.get("in_tokens"),
                "out_tokens": meta.get("out_tokens"),
                "status": "completed",
                "request_start_unix": t0,
                "request_end_unix": t1,
            })
            return content
        except Exception as e:
            t1 = time.time()
            _write({
                "condition": _condition,
                "run_name": context.get("run_name"),
                "task_id": context.get("task_id"),
                "seed": context.get("seed"),
                "node_id": context.get("node_id"),
                "node_type": context.get("node_type"),
                "request": {
                    "model": llm_mod._model_name(),
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "seed": seed,
                },
                "response_text": None,
                "status": f"error:{type(e).__name__}:{e}",
                "request_start_unix": t0,
                "request_end_unix": t1,
            })
            raise

    llm_mod.chat = logging_chat
    globals()["_installed"] = True
    return original_chat


def uninstall(original_chat) -> None:
    import hbws.llm as llm_mod
    llm_mod.chat = original_chat
    global _installed
    _installed = False
