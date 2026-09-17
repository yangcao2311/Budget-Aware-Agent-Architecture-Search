#!/usr/bin/env python3
"""Per-iteration call logger for the factorial budget ablation.

Current campaigns' logs (glm_call_logger.py) keep the request/response
per call but not an explicit reservation-request/refusal record -- a
refused call never reaches hbws.llm.chat at all (hbws.runner catches
ReserveRejected one level up), so it never appears in a plain call log.
This module fixes that: it recomputes the SAME reservation vector
hbws.llm.chat is about to compute internally (same installed estimator,
same messages/max_tokens), logs it as `reservation_request` BEFORE
calling the real chat(), and marks `reservation_refused: true` if
ReserveRejected is raised -- so every iteration's admission decision is
on record, not just the ones that were admitted.

Wraps hbws.llm.chat by monkeypatching, without editing hbws/llm.py or
hbws/runner.py. The wrapper still calls the ORIGINAL chat() with the
exact same arguments it was given; it does not alter the request,
sampling parameters, or call order -- see
tests/test_factorial_budget_ablation_noninvasive.py.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from hbws.ledger import ReserveRejected, llm_call_vec

_lock = threading.Lock()
_log_path: Path | None = None
_cell: str | None = None


def _write(row: dict) -> None:
    assert _log_path is not None
    with _lock:
        with _log_path.open("a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def install(log_path: Path, cell: str):
    global _log_path, _cell
    import hbws.llm as llm_mod

    _log_path = log_path
    _cell = cell
    original_chat = llm_mod.chat

    def logging_chat(messages, ledger, *, temperature=0.7, max_tokens=1024,
                      seed=None, use_cache=True, max_retries=5, wall_est=0.0,
                      backoff_schedule=None):
        context = dict(getattr(llm_mod._local, "call_context", {}))
        in_tok_est = llm_mod.estimate_in_tokens(messages)
        reservation_request = llm_call_vec(in_tok_est, max_tokens, wall_est)
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
                "cell": _cell,
                "run_name": context.get("run_name"),
                "task_id": context.get("task_id"),
                "seed": context.get("seed"),
                "node_id": context.get("node_id"),
                "node_type": context.get("node_type"),
                "reservation_request": reservation_request,
                "reservation_refused": False,
                "request": {"messages": messages, "temperature": temperature,
                           "max_tokens": max_tokens, "seed": seed},
                "response_text": content,
                "finish_reason": meta.get("finish_reason"),
                "provider_model": meta.get("provider_model"),
                "in_tokens": meta.get("in_tokens"),
                "out_tokens": meta.get("out_tokens"),
                "status": "completed",
                "request_start_unix": t0, "request_end_unix": t1,
            })
            return content
        except ReserveRejected as e:
            t1 = time.time()
            _write({
                "cell": _cell,
                "run_name": context.get("run_name"),
                "task_id": context.get("task_id"),
                "seed": context.get("seed"),
                "node_id": context.get("node_id"),
                "node_type": context.get("node_type"),
                "reservation_request": reservation_request,
                "reservation_refused": True,
                "request": {"messages": messages, "temperature": temperature,
                           "max_tokens": max_tokens, "seed": seed},
                "response_text": None,
                "status": f"reservation_refused:{e}",
                "request_start_unix": t0, "request_end_unix": t1,
            })
            raise
        except Exception as e:
            t1 = time.time()
            _write({
                "cell": _cell,
                "run_name": context.get("run_name"),
                "task_id": context.get("task_id"),
                "seed": context.get("seed"),
                "node_id": context.get("node_id"),
                "node_type": context.get("node_type"),
                "reservation_request": reservation_request,
                "reservation_refused": False,
                "request": {"messages": messages, "temperature": temperature,
                           "max_tokens": max_tokens, "seed": seed},
                "response_text": None,
                "status": f"error:{type(e).__name__}:{e}",
                "request_start_unix": t0, "request_end_unix": t1,
            })
            raise

    llm_mod.chat = logging_chat
    return original_chat


def uninstall(original_chat) -> None:
    import hbws.llm as llm_mod
    llm_mod.chat = original_chat
