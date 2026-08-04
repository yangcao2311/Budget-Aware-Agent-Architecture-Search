#!/usr/bin/env python
"""G1 gate: fuzz the reservation ledger's safety invariant (方案 v4.0 §9).

Runs thousands of random workflows against a mocked LLM/verifier and asserts
that no execution ever exceeds any per-task cap in any dimension. No API cost.
"""
import random
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hbws import llm, verify  # noqa: E402
from hbws.dsl import TEMPLATES  # noqa: E402
from hbws.ledger import BUDGET_TIERS, GLOBAL  # noqa: E402
from hbws.runner import run_workflow  # noqa: E402
from hbws.search import random_workflow  # noqa: E402

rng = random.Random(20260804)

SNIPPETS = [
    "Reasoning...\n```python\ndef f(x):\n    return x\n```",
    "Step by step, the answer is \\boxed{42}.",
    "I think \\boxed{\\frac{1}{2}} is right.",
    "```python\ndef g(a, b):\n    return a + b\n```",
    "no structured answer here " * 20,
]


def install_mocks():
    class FakeUsage:
        def __init__(self, i, o):
            self.prompt_tokens, self.completion_tokens = i, o

    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(model, messages, temperature, max_tokens, seed):
                    est = llm.estimate_in_tokens(messages)
                    usage = FakeUsage(rng.randint(est // 2, est), rng.randint(1, max_tokens))
                    msg = types.SimpleNamespace(content=rng.choice(SNIPPETS))
                    return types.SimpleNamespace(
                        choices=[types.SimpleNamespace(message=msg)], usage=usage)

    llm._client = lambda: FakeClient()
    verify.run_code_tests = lambda sol, tests: (rng.random() < 0.5, "mock feedback")


def main(n=3000):
    install_mocks()
    GLOBAL.usd = -1e9  # disable global breaker for fuzzing
    violations, statuses = [], {}
    tiers = list(BUDGET_TIERS.items())
    for i in range(n):
        wf = random_workflow(rng, n_mutations=rng.randint(0, 6)) \
            if rng.random() < 0.7 else TEMPLATES[rng.choice(list(TEMPLATES))]()
        name, caps = rng.choice(tiers)
        task = {"id": f"fuzz_{i}", "family": rng.choice(["code", "math"]),
                "prompt": "fuzz task " * rng.randint(1, 60),
                "feedback_tests": "assert True"}
        r = run_workflow(wf, task, caps, use_cache=False, seed=i)
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
        b, cv = r["budget"], caps.as_vec()
        for dim, cap in [("llm_calls", cv["llm_calls"]), ("in_tokens", cv["in_tokens"]),
                         ("out_tokens", cv["out_tokens"]), ("tool_calls", cv["tool_calls"]),
                         ("usd", cv["usd"])]:
            if b[dim] > cap + 1e-9:
                violations.append((i, name, dim, b[dim], cap))
        if r["status"].startswith("budget_exceeded"):
            violations.append((i, name, "settle_overrun_or_breaker", r["status"], ""))
    print(f"runs={n} statuses={statuses}")
    if violations:
        print(f"FAIL: {len(violations)} violations, first 5: {violations[:5]}")
        sys.exit(1)
    print("PASS: zero cap violations in any dimension — G1 gate satisfied")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3000)
