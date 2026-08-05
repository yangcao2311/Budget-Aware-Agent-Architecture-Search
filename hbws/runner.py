"""Workflow graph executor with hard budget enforcement.

Task dict contract:
  {"id", "family": "code"|"math", "prompt", "feedback_tests" (code: visible
   asserts), "grading_tests" (code: held-out tests), "gold_answer" (math)}

The runner never sees grading_tests / gold_answer — those live only in
eval/protocol grading. Internal `verify`:
  code: run feedback_tests in the sandbox (tool call, charged)
  math: independent re-derivation by the LLM (charged), answers compared.

Returns a trace: node visits, budget snapshot, final solution.
"""
from __future__ import annotations

import time

from . import llm, verify
from .dsl import validate
from .ledger import (BudgetCaps, BudgetExceeded, ReserveRejected, TaskLedger,
                     llm_call_vec, tool_call_vec)
from .prompts import render


class WorkflowRun:
    def __init__(self, wf: dict, task: dict, caps: BudgetCaps, *,
                 use_cache: bool = True, seed: int | None = None):
        validate(wf)
        self.wf, self.task, self.caps = wf, task, caps
        self.use_cache, self.seed = use_cache, seed
        self.nodes = {n["id"]: n for n in wf["nodes"]}
        self.ledger = TaskLedger(caps)
        self.state = {"solution": "", "verify_passed": None, "feedback": ""}
        self.trace: list[dict] = []
        self.loop_counts: dict[int, int] = {}

    # -- node semantics ------------------------------------------------------

    def _chat(self, prompt: str, params: dict) -> str:
        return llm.chat(
            [{"role": "user", "content": prompt}], self.ledger,
            temperature=params.get("temperature", 0.7),
            max_tokens=params.get("max_output_tokens", 1024),
            seed=self.seed, use_cache=self.use_cache)

    def _exec_node(self, node: dict):
        t, fam, task_text = node["type"], self.task["family"], self.task["prompt"]
        params = node.get("params", {})
        if t in ("generate", "decompose"):
            self.state["solution"] = self._chat(
                render(node["prompt_id"], fam, task_text), params)
            self.state["verify_passed"] = None
        elif t in ("refine", "aggregate"):
            self.state["solution"] = self._chat(
                render(node["prompt_id"], fam, task_text,
                       self.state["solution"], self.state["feedback"]), params)
            self.state["verify_passed"] = None
        elif t == "vote":
            # All-k upfront feasibility (方案 §4.2): a vote either fully fits
            # in the remaining budget or is rejected before any call starts.
            prompt = render(node["prompt_id"], fam, task_text)
            one = llm_call_vec(llm.estimate_in_tokens(
                [{"role": "user", "content": prompt}]),
                params.get("max_output_tokens", 1024))
            k = node["k"]
            all_k = {d: v * k for d, v in one.items()}
            if not self.ledger.can_reserve(all_k):
                raise ReserveRejected(f"vote-{k} cannot fully reserve")
            answers = []
            for i in range(k):
                answers.append(llm.chat(
                    [{"role": "user", "content": prompt}],
                    self.ledger, temperature=params.get("temperature", 0.8),
                    max_tokens=params.get("max_output_tokens", 1024),
                    seed=(self.seed or 0) * 100 + i, use_cache=self.use_cache))
            self.state["solution"] = self._majority(answers)
            self.state["verify_passed"] = None
        elif t == "verify":
            if fam == "code":
                lease = self.ledger.reserve(tool_call_vec())
                if lease is None:
                    raise ReserveRejected("verify tool call")
                ok, fb = verify.run_code_tests(
                    self.state["solution"], self.task["feedback_tests"])
                self.ledger.settle(lease, {"tool_calls": 1})
            else:
                # heuristic_critic: k independent gold-free re-derivations;
                # PASS iff a majority agrees with the candidate's answer.
                k = node.get("k", 1)
                a1 = verify.extract_boxed(self.state["solution"])
                agree = 0
                for i in range(k):
                    check = llm.chat(
                        [{"role": "user", "content": render("check_math", fam, task_text)}],
                        self.ledger, temperature=0.3 + 0.2 * i,
                        max_tokens=1536, seed=(self.seed or 0) * 100 + 50 + i,
                        use_cache=self.use_cache)
                    a2 = verify.extract_boxed(check)
                    if a1 is None or a2 is None:
                        continue
                    same = (verify.same_choice(a1, a2) if fam == "logic"
                            else verify.math_equal(a1, a2))
                    if same:
                        agree += 1
                ok = agree * 2 > k
                fb = ("independent check agrees" if ok else
                      "an independent solution reached a different answer; "
                      "re-examine your reasoning step by step")
            self.state["verify_passed"], self.state["feedback"] = ok, fb
        elif t == "branch":
            pass  # routing happens on edges
        else:
            raise AssertionError(t)

    def _majority(self, answers: list[str]) -> str:
        fam = self.task["family"]
        if fam in ("math", "logic"):
            keyf = lambda a: verify._norm(verify.extract_boxed(a) or "")
        else:
            keyf = lambda a: verify.extract_code(a)
        buckets: dict[str, list[str]] = {}
        for a in answers:
            buckets.setdefault(keyf(a), []).append(a)
        best = max(buckets.values(), key=len)
        return best[0]

    # -- routing -------------------------------------------------------------

    def _cond_met(self, cond: str) -> bool:
        if cond == "always":
            return True
        if cond == "verify_passed":
            return self.state["verify_passed"] is True
        if cond == "verify_failed":
            return self.state["verify_passed"] is False
        if cond.startswith("budget_below:"):
            return self.ledger.remaining_frac < float(cond.split(":")[1])
        if cond.startswith("budget_above:"):
            return self.ledger.remaining_frac >= float(cond.split(":")[1])
        raise AssertionError(cond)

    def _next(self, node_id: str) -> str | None:
        for i, e in enumerate(self.wf["edges"]):
            if e["from"] != node_id or not self._cond_met(e.get("cond", "always")):
                continue
            if e.get("loop"):
                if self.loop_counts.get(i, 0) >= e["max_iter"]:
                    continue
                self.loop_counts[i] = self.loop_counts.get(i, 0) + 1
            return e["to"]
        return None  # no matching edge -> terminate with current solution

    # -- main loop -----------------------------------------------------------

    def run(self) -> dict:
        cur = self.wf["nodes"][0]["id"]
        status = "completed"
        try:
            for _ in range(64):  # absolute step bound
                node = self.nodes[cur]
                t0 = time.monotonic()
                try:
                    self._exec_node(node)
                except ReserveRejected as e:
                    # Not a violation: the ledger refused the call, we end
                    # gracefully with the incumbent solution (safe fallback).
                    status = "reserve_rejected"
                    self.trace.append({"node": cur, "type": node["type"],
                                       "reserve_rejected": str(e),
                                       "budget": self.ledger.snapshot()})
                    break
                self.trace.append({"node": cur, "type": node["type"],
                                   "sec": round(time.monotonic() - t0, 2),
                                   "budget": self.ledger.snapshot(),
                                   "remaining_frac": round(self.ledger.remaining_frac, 3)})
                nxt = self._next(cur)
                if nxt in (None, "END"):
                    break
                cur = nxt
            else:
                status = "step_bound_hit"
        except BudgetExceeded as e:
            # With the reservation ledger this indicates a runtime defect
            # (settle overrun) or the global circuit breaker — surface loudly.
            status = f"budget_exceeded:{e.dimension}"
        return {"task_id": self.task["id"], "status": status,
                "solution": self.state["solution"],
                "budget": self.ledger.snapshot(), "trace": self.trace}


def run_workflow(wf: dict, task: dict, caps: BudgetCaps, **kw) -> dict:
    return WorkflowRun(wf, task, caps, **kw).run()
