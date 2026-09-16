#!/usr/bin/env python3
"""Matching-model token reservation for the GLM budget-parity follow-up.

hbws/llm.py's estimate_in_tokens() approximates prompt tokens with tiktoken
(an OpenAI tokenizer) plus a flat +150-token buffer for any non-default
LLM_PROVIDER. Zhipu does not publish the exact tokenizer for
glm-4-flash-250414, so this uses the tokenizer for THUDM/glm-4-9b-chat (the
closest openly-available same-family GLM-4 tokenizer) as a matching-model
proxy, with an explicit extra safety margin on top precisely because it is
a proxy, not a byte-exact match -- documented here rather than silently
assumed exact. This is still materially closer to GLM's real tokenization
than an OpenAI tokenizer.

Installed by monkeypatching hbws.llm.estimate_in_tokens, exactly like
scripts/qwen_exact_reservation.py for the Qwen serving-regime follow-up.
Changes ONLY the ledger's admission-reservation estimate: never the
request messages, sampling parameters, per-call output cap, or call order.
"""
from __future__ import annotations

import threading

_local = threading.local()
_MODEL_ID = "THUDM/glm-4-9b-chat"
_PROXY_SAFETY_MARGIN = 64  # extra tokens, because this is a same-family
                           # proxy tokenizer, not GLM-4-Flash's own


def _tokenizer():
    tok = getattr(_local, "tok", None)
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(_MODEL_ID, trust_remote_code=True)
        _local.tok = tok
    return tok


def exact_in_tokens(messages: list[dict]) -> int:
    """Matching-family-tokenizer prompt-token estimate for `messages`,
    plus a fixed proxy-tokenizer safety margin (see module docstring)."""
    tok = _tokenizer()
    encoding = tok.apply_chat_template(messages, tokenize=True,
                                       add_generation_prompt=True)
    ids = encoding["input_ids"] if hasattr(encoding, "keys") else encoding
    return len(ids) + _PROXY_SAFETY_MARGIN


_installed = False


def install() -> None:
    """Monkeypatch hbws.llm.estimate_in_tokens in place. Idempotent.

    Pre-warms the tokenizer synchronously in the calling thread before any
    ThreadPoolExecutor workers start -- see qwen_exact_reservation.install's
    docstring for why: a concurrent first-time `transformers` import from
    multiple worker threads can race.
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
