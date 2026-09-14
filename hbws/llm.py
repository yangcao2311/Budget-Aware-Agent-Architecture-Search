"""LLM client with retries, sqlite response cache, and ledger hooks.

Cache is keyed on (model, messages, params). It is ON during search and MUST
be OFF for final test runs (protocol §5.3): pass use_cache=False.

Provider selection:
  - LLM_PROVIDER=<NAME> selects <NAME>_API_KEY / <NAME>_BASE_URL / <NAME>_MODEL
    (e.g. LLM_PROVIDER=kimi -> KIMI_API_KEY, KIMI_BASE_URL, KIMI_MODEL). Lets
    separate model campaigns (e.g. a GPT-4o run and a Kimi run) live in the
    same .env without hand-editing keys between runs.
  - Unset LLM_PROVIDER falls back to the original two-provider behavior:
    CHATANYWHERE_API_KEY (OpenAI-compatible proxy) if present, else
    AZURE_OPENAI_KEY (the original Azure deployment).
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from pathlib import Path

from openai import (AzureOpenAI, OpenAI, APIConnectionError, APITimeoutError,
                    InternalServerError, RateLimitError)

from .ledger import ReserveRejected, TaskLedger, llm_call_vec

ROOT = Path(__file__).resolve().parent.parent
_local = threading.local()


def _load_env():
    envfile = ROOT / ".env"
    if envfile.exists():
        for line in envfile.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def _provider_prefix() -> str | None:
    name = os.environ.get("LLM_PROVIDER")
    return name.upper() if name else None


def _model_name() -> str:
    _load_env()
    prefix = _provider_prefix()
    if prefix and os.environ.get(f"{prefix}_MODEL"):
        return os.environ[f"{prefix}_MODEL"]
    return (os.environ.get("CHATANYWHERE_MODEL")
            or os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"))


def _client():
    if getattr(_local, "client", None) is None:
        _load_env()
        prefix = _provider_prefix()
        if prefix and os.environ.get(f"{prefix}_API_KEY"):
            _local.client = OpenAI(
                api_key=os.environ[f"{prefix}_API_KEY"],
                base_url=os.environ[f"{prefix}_BASE_URL"],
                timeout=90.0,
            )
        elif os.environ.get("CHATANYWHERE_API_KEY"):
            _local.client = OpenAI(
                api_key=os.environ["CHATANYWHERE_API_KEY"],
                base_url=os.environ.get(
                    "CHATANYWHERE_BASE_URL", "https://api.chatanywhere.tech/v1"),
                timeout=90.0,
            )
        else:
            _local.client = AzureOpenAI(
                api_version=os.environ["AZURE_OPENAI_API_VERSION"],
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_key=os.environ["AZURE_OPENAI_KEY"],
                timeout=90.0,
            )
    return _local.client


class Cache:
    def __init__(self, path: Path = ROOT / "cache.sqlite"):
        self.path = str(path)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT)")

    def _conn(self):
        return sqlite3.connect(self.path, timeout=30)

    def get(self, k: str):
        with self._lock, self._conn() as c:
            row = c.execute("SELECT v FROM cache WHERE k=?", (k,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, k: str, v: dict):
        with self._lock, self._conn() as c:
            c.execute("INSERT OR REPLACE INTO cache VALUES (?,?)", (k, json.dumps(v)))


CACHE = Cache()
RETRYABLE = (RateLimitError, APITimeoutError, APIConnectionError,
             InternalServerError, ConnectionError)
_attempt_log_lock = threading.Lock()


def clear_last_response_meta() -> None:
    """Clear thread-local metadata from the preceding provider call."""
    _local.last_response_meta = None


def last_response_meta() -> dict | None:
    """Return metadata for the most recent call in this worker thread.

    The completion text remains the public return value of :func:`chat`; this
    side channel lets experiment traces retain provider termination metadata
    without changing every existing caller.
    """
    value = getattr(_local, "last_response_meta", None)
    return dict(value) if value is not None else None


def set_call_context(**fields) -> None:
    """Attach non-sensitive experiment identifiers to attempt-log rows."""
    _local.call_context = {k: v for k, v in fields.items() if v is not None}


def update_call_context(**fields) -> None:
    context = dict(getattr(_local, "call_context", {}))
    context.update({k: v for k, v in fields.items() if v is not None})
    _local.call_context = context


def clear_call_context() -> None:
    _local.call_context = {}


def _encoder():
    if getattr(_local, "enc", None) is None:
        import tiktoken
        try:
            _local.enc = tiktoken.encoding_for_model(_model_name())
        except KeyError:
            _local.enc = tiktoken.get_encoding("o200k_base")
    return _local.enc


def estimate_in_tokens(messages: list[dict]) -> int:
    """Upper bound on prompt tokens, for reservation only (never billing).

    Exact tiktoken count plus per-message chat overhead and a 5% margin.
    A char/3 heuristic was used first and UNDER-estimated punctuation-heavy
    code by ~55% (71 chars -> 36 real tokens vs 23 estimated), producing 124
    settle-overruns that were wrongly scored as infeasible candidates.

    Non-default providers add a fixed buffer on top: measured directly against
    tokenrouter.com's Kimi K3 endpoint, a one-token message ("pong") billed
    prompt_tokens=90 -- a near-constant server-side overhead (likely an
    injected system prompt we cannot see or account for by encoding our own
    message text), not proportional to content length, so scaling the
    tiktoken estimate cannot fix it. o200k_base is also not Kimi's real
    tokenizer, so the content-based estimate is separately approximate.
    """
    enc = _encoder()
    n = sum(len(enc.encode(m.get("content", ""))) + 4 for m in messages) + 3
    est = int(n * 1.05) + 16
    if _provider_prefix():
        est += 150
    return est


def _env_backoff_schedule() -> list[float] | None:
    """LLM_BACKOFF_SCHEDULE='10,30,90' overrides retry sleep times for every
    call in the process, without touching call sites in runner.py/protocol.py.
    Unset by default; the Kimi backfill sets it, GPT-4o runs never do."""
    raw = os.environ.get("LLM_BACKOFF_SCHEDULE")
    if not raw:
        return None
    return [float(x) for x in raw.split(",")]


def _append_attempt_log(row: dict) -> None:
    """Append provider-attempt metadata without prompts, responses, or keys."""
    path = os.environ.get("LLM_ATTEMPT_LOG")
    if not path:
        return
    safe = {"time": time.time(), **getattr(_local, "call_context", {}), **row}
    with _attempt_log_lock:
        with open(path, "a") as f:
            f.write(json.dumps(safe, sort_keys=True) + "\n")


def chat(messages: list[dict], ledger: TaskLedger, *,
         temperature: float = 0.7, max_tokens: int = 1024, seed: int | None = None,
         use_cache: bool = True, max_retries: int = 5, wall_est: float = 0.0,
         backoff_schedule: list[float] | None = None) -> str:
    """One chat completion under reserve->call->settle discipline.

    Raises ReserveRejected if the worst-case cost cannot be reserved — the
    caller must treat that as a control-flow signal (fallback/END), never as
    a task error. Cached hits still reserve and settle: they count against
    the task budget; caching only saves real dollars during search.

    backoff_schedule overrides the default exponential backoff with a fixed
    sequence of sleep times (seconds), one per retry, the last value repeated
    if there are more retries than entries. Used for provider backfill passes
    (e.g. Kimi K3: 10, 30, 90s, max 3 attempts) where a third-party proxy's
    rate-limit window is on a different clock than the default schedule was
    tuned for; the default (uncalled) path is unaffected.
    """
    clear_last_response_meta()
    env_retries = os.environ.get("LLM_MAX_RETRIES")
    if env_retries:
        max_retries = int(env_retries)
    deployment = _model_name()
    disable_seed = os.environ.get("LLM_DISABLE_SEED", "0") == "1"
    reasoning_effort = os.environ.get("LLM_REASONING_EFFORT")
    params = {"temperature": temperature, "max_tokens": max_tokens,
              "seed": None if disable_seed else seed,
              "reasoning_effort": reasoning_effort}
    key = hashlib.sha256(json.dumps(
        [deployment, messages, params], sort_keys=True).encode()).hexdigest()

    vec = llm_call_vec(estimate_in_tokens(messages), max_tokens, wall_est)
    lease = ledger.reserve(vec)
    if lease is None:
        raise ReserveRejected(f"cannot reserve {vec}")

    try:
        if use_cache:
            hit = CACHE.get(key)
            if hit is not None:
                ledger.settle(lease, {"llm_calls": 1,
                                      "in_tokens": hit["in_tokens"],
                                      "out_tokens": hit["out_tokens"]})
                lease = None
                _local.last_response_meta = {
                    "cached": True,
                    "model": deployment,
                    "finish_reason": hit.get("finish_reason"),
                    "provider_model": hit.get("provider_model"),
                    "system_fingerprint": hit.get("system_fingerprint"),
                    "in_tokens": hit["in_tokens"],
                    "out_tokens": hit["out_tokens"],
                    "requested_max_tokens": max_tokens,
                }
                return hit["content"]

        last_err = None
        for attempt in range(max_retries):
            try:
                request = {"model": deployment, "messages": messages,
                           "temperature": temperature, "max_tokens": max_tokens}
                if seed is not None and not disable_seed:
                    request["seed"] = seed
                if reasoning_effort:
                    # Z.AI exposes this as a top-level extension. extra_body
                    # keeps the generic OpenAI client compatible with it.
                    request["extra_body"] = {"reasoning_effort": reasoning_effort}
                r = _client().chat.completions.create(**request)
                content = r.choices[0].message.content or ""
                if r.usage is None:
                    raise RuntimeError("provider returned no usage object")
                usage = {"in_tokens": r.usage.prompt_tokens,
                         "out_tokens": r.usage.completion_tokens,
                         "content": content,
                         "finish_reason": getattr(r.choices[0], "finish_reason", None),
                         "provider_model": getattr(r, "model", None),
                         "system_fingerprint": getattr(r, "system_fingerprint", None)}
                # No clamping: if actual exceeds the reservation, settle
                # raises settle_overrun — a defect in the estimator we must
                # see and fix, not silently absorb.
                ledger.settle(lease, {"llm_calls": 1,
                                      "in_tokens": usage["in_tokens"],
                                      "out_tokens": usage["out_tokens"]})
                lease = None
                if use_cache:
                    CACHE.put(key, usage)
                _local.last_response_meta = {
                    "cached": False,
                    "model": deployment,
                    "finish_reason": usage["finish_reason"],
                    "provider_model": usage["provider_model"],
                    "system_fingerprint": usage["system_fingerprint"],
                    "in_tokens": usage["in_tokens"],
                    "out_tokens": usage["out_tokens"],
                    "requested_max_tokens": max_tokens,
                }
                _append_attempt_log({"model": deployment, "attempt": attempt + 1,
                                     "status": "completed",
                                     "provider_model": usage["provider_model"],
                                     "system_fingerprint": usage["system_fingerprint"],
                                     "finish_reason": usage["finish_reason"],
                                     "requested_max_tokens": max_tokens,
                                     "in_tokens": usage["in_tokens"],
                                     "out_tokens": usage["out_tokens"]})
                return content
            except RETRYABLE as e:
                last_err = e
                _append_attempt_log({"model": deployment, "attempt": attempt + 1,
                                     "status": "retryable_error",
                                     "error_type": type(e).__name__})
                sched = backoff_schedule or _env_backoff_schedule()
                if sched:
                    delay = sched[min(attempt, len(sched) - 1)]
                else:
                    delay = min(2 ** attempt * 1.5, 30)
                time.sleep(delay)
            except Exception as e:
                _append_attempt_log({"model": deployment, "attempt": attempt + 1,
                                     "status": "nonretryable_error",
                                     "error_type": type(e).__name__})
                raise
        raise RuntimeError(f"LLM call failed after {max_retries} retries: {last_err}")
    finally:
        if lease is not None:
            ledger.cancel(lease)
