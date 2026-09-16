#!/usr/bin/env python3
"""Exact-tokenizer input-token reservation for the local Qwen serving-regime
control (reviewer B1 follow-up).

hbws/llm.py's estimate_in_tokens() approximates prompt tokens with tiktoken
(an OpenAI tokenizer) plus a flat +150-token buffer for any non-default
LLM_PROVIDER. That is a cross-provider proxy, not an exact count, and it
under-estimated 14/1350 causal rows in the original math/tight nonbinding
campaign (experiments/qwen25coder7b_server_math_tight_20260914_causal),
producing deterministic `budget_exceeded:settle_overrun:in_tokens` rows on
long verify+refine prompts. Those 14 rows are left untouched in the
archived campaign -- this module exists ONLY for the new serving-regime
control and is installed by monkeypatching, never by editing hbws/llm.py,
so the original campaign's source hashes stay valid.

This computes the exact prompt token count Qwen2.5-Coder-7B-Instruct's own
tokenizer + chat template produce for a given messages list -- the same
computation vLLM's OpenAI-compatible server performs server-side for
/v1/chat/completions -- instead of approximating it. It changes ONLY the
ledger's admission-reservation estimate: it does not touch the request
messages, sampling parameters, max_tokens, seed, or call order, and it does
not change hbws.llm.chat's control flow at all (see
tests/test_qwen_exact_reservation_noninvasive.py for the mock proof).
"""
from __future__ import annotations

import threading

_local = threading.local()
_MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"


def _tokenizer():
    tok = getattr(_local, "tok", None)
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(_MODEL_ID)
        _local.tok = tok
    return tok


def exact_in_tokens(messages: list[dict]) -> int:
    """Exact prompt-token count for `messages` under Qwen's own chat
    template, matching what vLLM computes server-side. A 1-token margin
    covers any float/rounding-free tokenizer nondeterminism across
    processes (there should be none, but the reservation is worst-case by
    design, per hbws/ledger.py's own convention -- never a hard cap on the
    real request, which vLLM enforces separately via max_tokens)."""
    tok = _tokenizer()
    ids = tok.apply_chat_template(messages, tokenize=True,
                                   add_generation_prompt=True)["input_ids"]
    return len(ids) + 1


_installed = False


def install() -> None:
    """Monkeypatch hbws.llm.estimate_in_tokens in place. Idempotent.

    Pre-warms the tokenizer (and, transitively, `transformers`'s own lazy
    module machinery) synchronously in the calling thread before any
    ThreadPoolExecutor workers start. Without this, concurrent first-time
    `from transformers import AutoTokenizer` calls from multiple worker
    threads under condition B (workers=8) raced and raised
    `ImportError: cannot import name 'AutoTokenizer'` on 6/150 seed-0 tasks
    -- a one-time cold-start infrastructure defect in this new module, not
    a change to any request, sampling parameter, or call order. Pre-warming
    here (and each recovered row re-uses the already-imported, per-process
    cached tokenizer) fixes it structurally instead of retrying into the
    same race.
    """
    global _installed
    if _installed:
        return
    exact_in_tokens([{"role": "user", "content": "warmup"}])
    import hbws.llm as llm_mod
    llm_mod.estimate_in_tokens = exact_in_tokens
    _installed = True


def uninstall(original) -> None:
    import hbws.llm as llm_mod
    llm_mod.estimate_in_tokens = original
    global _installed
    _installed = False
