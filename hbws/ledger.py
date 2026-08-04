"""Five-dimensional budget ledger: llm_calls, tool_calls, tokens, wall time, USD.

Per-task caps are hard constraints enforced by the runner (BudgetExceeded ->
task counted as failure, cost still counted). A global ledger aggregates all
spend for the circuit breaker.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

# Azure GPT-4o global deployment pricing, USD per 1M tokens.
# TODO: verify against the Azure portal for this subscription (方案 §7.1).
PRICE_IN_PER_M = 2.50
PRICE_OUT_PER_M = 10.00

GLOBAL_HARD_CAP_USD = 2000.0
GLOBAL_WARN_FRACTION = 0.8


class BudgetExceeded(Exception):
    def __init__(self, dimension: str, spent, cap):
        self.dimension = dimension
        super().__init__(f"budget exceeded on {dimension}: {spent} > {cap}")


@dataclass
class BudgetCaps:
    max_llm_calls: int = 8
    max_in_tokens: int = 16000
    max_out_tokens: int = 4000
    max_tool_calls: int = 6
    max_wall_sec: float = 120.0
    max_usd: float = 0.10

    @classmethod
    def from_dict(cls, d: dict) -> "BudgetCaps":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TaskLedger:
    caps: BudgetCaps
    llm_calls: int = 0
    in_tokens: int = 0
    out_tokens: int = 0
    tool_calls: int = 0
    usd: float = 0.0
    started_at: float = field(default_factory=time.monotonic)

    @property
    def wall_sec(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def remaining_frac(self) -> float:
        """Most-binding remaining budget fraction across dimensions.

        This is the runtime signal available to budget-conditioned branches.
        """
        fracs = [
            1 - self.llm_calls / self.caps.max_llm_calls,
            1 - self.in_tokens / self.caps.max_in_tokens,
            1 - self.out_tokens / self.caps.max_out_tokens,
            1 - self.usd / self.caps.max_usd,
            1 - self.wall_sec / self.caps.max_wall_sec,
        ]
        return max(0.0, min(fracs))

    def check(self):
        c = self.caps
        for dim, spent, cap in [
            ("llm_calls", self.llm_calls, c.max_llm_calls),
            ("in_tokens", self.in_tokens, c.max_in_tokens),
            ("out_tokens", self.out_tokens, c.max_out_tokens),
            ("tool_calls", self.tool_calls, c.max_tool_calls),
            ("usd", self.usd, c.max_usd),
            ("wall_sec", self.wall_sec, c.max_wall_sec),
        ]:
            if spent > cap:
                raise BudgetExceeded(dim, spent, cap)

    def add_llm(self, in_tok: int, out_tok: int):
        self.llm_calls += 1
        self.in_tokens += in_tok
        self.out_tokens += out_tok
        self.usd += in_tok * PRICE_IN_PER_M / 1e6 + out_tok * PRICE_OUT_PER_M / 1e6
        GLOBAL.add(in_tok, out_tok)
        self.check()

    def add_tool(self):
        self.tool_calls += 1
        self.check()

    def snapshot(self) -> dict:
        return {
            "llm_calls": self.llm_calls,
            "in_tokens": self.in_tokens,
            "out_tokens": self.out_tokens,
            "tool_calls": self.tool_calls,
            "usd": round(self.usd, 6),
            "wall_sec": round(self.wall_sec, 2),
        }


class GlobalLedger:
    """Process-wide spend accumulator with the $2k circuit breaker."""

    def __init__(self):
        self._lock = threading.Lock()
        self.in_tokens = 0
        self.out_tokens = 0
        self.usd = 0.0
        self.warned = False

    def add(self, in_tok: int, out_tok: int):
        with self._lock:
            self.in_tokens += in_tok
            self.out_tokens += out_tok
            self.usd += in_tok * PRICE_IN_PER_M / 1e6 + out_tok * PRICE_OUT_PER_M / 1e6
            if self.usd > GLOBAL_HARD_CAP_USD:
                raise BudgetExceeded("GLOBAL_usd", self.usd, GLOBAL_HARD_CAP_USD)
            if not self.warned and self.usd > GLOBAL_WARN_FRACTION * GLOBAL_HARD_CAP_USD:
                self.warned = True
                print(f"[ledger] WARNING: global spend ${self.usd:.2f} passed "
                      f"{GLOBAL_WARN_FRACTION:.0%} of ${GLOBAL_HARD_CAP_USD}")


GLOBAL = GlobalLedger()
