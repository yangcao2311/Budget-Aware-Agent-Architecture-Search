"""Reservation-based budget ledger (方案 v4.0 §4.2).

Discipline: reserve -> call -> settle. A node may only execute if its
worst-case billable vector can be reserved against the remaining task caps;
reservation failure never raises — it returns None, and the runner routes to
fallback/END. Settlement records actual usage and releases the surplus.

Safety invariant (proved by induction over settles, paper appendix): if every
node's actual metered usage <= its reservation and all calls go through
reserve(), cumulative usage never exceeds any per-task cap.

Budget signals exposed to policies:
  - remaining_frac: per-dimension (or most-binding) fraction of *unreserved,
    unused* budget — the rho_t vector for budget predicates;
  - can_reserve(vec): the reserve-feasibility mask m_t.
"""
from __future__ import annotations

import itertools
import threading
import time
from dataclasses import dataclass, field

# Azure GPT-4o global deployment pricing, USD per 1M tokens.
# Frozen price table: verify against the Azure portal in W1 and archive.
PRICE_IN_PER_M = 2.50
PRICE_OUT_PER_M = 10.00

GLOBAL_HARD_CAP_USD = 2000.0
GLOBAL_WARN_FRACTION = 0.8

DIMS = ("llm_calls", "in_tokens", "out_tokens", "tool_calls", "wall_sec", "usd")


class ReserveRejected(Exception):
    """Control-flow signal: a node could not reserve its worst-case cost.
    The runner catches this and routes to fallback/END — it is NOT an error
    and NOT a budget violation (the violation was prevented)."""


class BudgetExceeded(Exception):
    """Raised only on global circuit-break or a settle overrunning its
    reservation (the latter is a runtime defect, not a policy failure)."""

    def __init__(self, dimension: str, spent, cap):
        self.dimension = dimension
        super().__init__(f"budget exceeded on {dimension}: {spent} > {cap}")


@dataclass
class BudgetCaps:
    max_llm_calls: int = 4
    max_in_tokens: int = 8000
    max_out_tokens: int = 2000
    max_tool_calls: int = 4
    max_wall_sec: float = 90.0
    max_usd: float = 0.10

    def as_vec(self) -> dict:
        return {"llm_calls": self.max_llm_calls, "in_tokens": self.max_in_tokens,
                "out_tokens": self.max_out_tokens, "tool_calls": self.max_tool_calls,
                "wall_sec": self.max_wall_sec, "usd": self.max_usd}

    @classmethod
    def from_dict(cls, d: dict) -> "BudgetCaps":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# Budget profiles (方案 v4.0 §5.2). E1/E2 envelope-only; tight/loose are the
# search budgets; "unseen" ($0.15) is final-test only and must never be
# touched by any search or selection code path.
BUDGET_TIERS = {
    "E1":     BudgetCaps(1, 3000, 1000, 2, 45, 0.02),
    "E2":     BudgetCaps(2, 5000, 1500, 3, 60, 0.05),
    "tight":  BudgetCaps(4, 8000, 2000, 4, 90, 0.10),
    "unseen": BudgetCaps(6, 12000, 3000, 5, 135, 0.15),
    "loose":  BudgetCaps(8, 16000, 4000, 6, 180, 0.25),
}
SEARCH_TIERS = ("tight", "loose")  # the only tiers any search code may see


def usd_of(in_tok: float, out_tok: float) -> float:
    return in_tok * PRICE_IN_PER_M / 1e6 + out_tok * PRICE_OUT_PER_M / 1e6


# Wall clock is NOT a reserved dimension (方案 §4.2 / external v3 §5.3):
# countable resources are reserved; wall time is enforced by continuous
# metering (can_reserve rejects once elapsed exceeds the cap) plus per-call
# deadlines. Reservation vectors therefore carry wall_sec = 0.

def llm_call_vec(in_tokens_est: float, max_out_tokens: int,
                 wall_est: float = 0.0) -> dict:
    """Worst-case billable vector for one LLM call."""
    return {"llm_calls": 1, "in_tokens": in_tokens_est,
            "out_tokens": max_out_tokens, "tool_calls": 0,
            "wall_sec": wall_est, "usd": usd_of(in_tokens_est, max_out_tokens)}


def tool_call_vec(wall_est: float = 0.0) -> dict:
    return {"llm_calls": 0, "in_tokens": 0, "out_tokens": 0,
            "tool_calls": 1, "wall_sec": wall_est, "usd": 0.0}


@dataclass
class TaskLedger:
    caps: BudgetCaps
    used: dict = field(default_factory=lambda: {d: 0.0 for d in DIMS})
    reserved: dict = field(default_factory=dict)  # lease_id -> vec
    started_at: float = field(default_factory=time.monotonic)
    _ids: itertools.count = field(default_factory=itertools.count)

    # -- wall clock is metered continuously, not per-lease ------------------
    @property
    def wall_sec(self) -> float:
        return time.monotonic() - self.started_at

    def _reserved_total(self, dim: str) -> float:
        return sum(v[dim] for v in self.reserved.values())

    # -- reservation protocol ----------------------------------------------
    def can_reserve(self, vec: dict) -> bool:
        caps = self.caps.as_vec()
        for d in DIMS:
            used = self.used[d] if d != "wall_sec" else self.wall_sec
            if used + self._reserved_total(d) + vec.get(d, 0) > caps[d]:
                return False
        return True

    def reserve(self, vec: dict) -> int | None:
        """Returns a lease id, or None (REJECT) — never raises."""
        if not self.can_reserve(vec):
            return None
        lease = next(self._ids)
        self.reserved[lease] = dict(vec)
        return lease

    def settle(self, lease: int, actual: dict):
        res = self.reserved.pop(lease)
        for d in ("llm_calls", "in_tokens", "out_tokens", "tool_calls"):
            a = actual.get(d, 0)
            if a > res[d] + 1e-9:
                # Overrunning a reservation is a runtime defect (方案 §4.2).
                raise BudgetExceeded(f"settle_overrun:{d}", a, res[d])
            self.used[d] += a
        actual_usd = usd_of(actual.get("in_tokens", 0), actual.get("out_tokens", 0))
        self.used["usd"] += actual_usd
        GLOBAL.add(actual.get("in_tokens", 0), actual.get("out_tokens", 0))

    def cancel(self, lease: int):
        self.reserved.pop(lease, None)

    # -- policy-visible signals --------------------------------------------
    def remaining_frac_dim(self, dim: str) -> float:
        caps = self.caps.as_vec()
        used = self.used[dim] if dim != "wall_sec" else self.wall_sec
        return max(0.0, 1 - (used + self._reserved_total(dim)) / caps[dim])

    @property
    def remaining_frac(self) -> float:
        """Most-binding remaining fraction — default rho signal."""
        return min(self.remaining_frac_dim(d) for d in DIMS)

    def snapshot(self) -> dict:
        return {"llm_calls": int(self.used["llm_calls"]),
                "in_tokens": int(self.used["in_tokens"]),
                "out_tokens": int(self.used["out_tokens"]),
                "tool_calls": int(self.used["tool_calls"]),
                "usd": round(self.used["usd"], 6),
                "wall_sec": round(self.wall_sec, 2)}


class GlobalLedger:
    """Process-wide spend accumulator with the $2k circuit breaker."""

    def __init__(self):
        self._lock = threading.Lock()
        self.in_tokens = 0
        self.out_tokens = 0
        self.usd = 0.0
        self.warned = False

    def add(self, in_tok: float, out_tok: float):
        with self._lock:
            self.in_tokens += in_tok
            self.out_tokens += out_tok
            self.usd += usd_of(in_tok, out_tok)
            if self.usd > GLOBAL_HARD_CAP_USD:
                raise BudgetExceeded("GLOBAL_usd", self.usd, GLOBAL_HARD_CAP_USD)
            if not self.warned and self.usd > GLOBAL_WARN_FRACTION * GLOBAL_HARD_CAP_USD:
                self.warned = True
                print(f"[ledger] WARNING: global spend ${self.usd:.2f} passed "
                      f"{GLOBAL_WARN_FRACTION:.0%} of ${GLOBAL_HARD_CAP_USD}")


GLOBAL = GlobalLedger()
