from hbws.ledger import BUDGET_TIERS
from hbws.runner import WorkflowRun


def _runner(monkeypatch, answers):
    remaining = iter(answers)

    def fake_chat(*args, **kwargs):
        return next(remaining)

    monkeypatch.setattr("hbws.runner.llm.chat", fake_chat)
    workflow = {
        "nodes": [{"id": "v", "type": "verify", "k": 3,
                   "decision_rule": "any_agree"}],
        "edges": [{"from": "v", "to": "END"}],
    }
    task = {"id": "m1", "family": "math", "prompt": "1+1?",
            "gold_answer": "2"}
    runner = WorkflowRun(workflow, task, BUDGET_TIERS["loose"])
    runner.state["solution"] = r"\boxed{2}"
    return runner


def test_any_agree_verifier_accepts_one_matching_check(monkeypatch):
    runner = _runner(monkeypatch, [r"\boxed{3}", r"\boxed{4}", r"\boxed{2}"])
    runner._exec_node(runner.nodes["v"])
    assert runner.state["verify_passed"] is True


def test_majority_remains_default(monkeypatch):
    runner = _runner(monkeypatch, [r"\boxed{3}", r"\boxed{4}", r"\boxed{2}"])
    runner.nodes["v"].pop("decision_rule")
    runner._exec_node(runner.nodes["v"])
    assert runner.state["verify_passed"] is False


def test_verifier_prompt_can_receive_incumbent(monkeypatch):
    seen = []

    def fake_chat(messages, *args, **kwargs):
        seen.append(messages[0]["content"])
        return r"\boxed{2}"

    monkeypatch.setattr("hbws.runner.llm.chat", fake_chat)
    workflow = {
        "nodes": [{"id": "v", "type": "verify", "k": 1,
                   "prompt_id": "conservative_check_math"}],
        "edges": [{"from": "v", "to": "END"}],
    }
    task = {"id": "m1", "family": "math", "prompt": "1+1?",
            "gold_answer": "2"}
    runner = WorkflowRun(workflow, task, BUDGET_TIERS["loose"])
    runner.state["solution"] = r"work... \boxed{2}"
    runner._exec_node(runner.nodes["v"])
    assert r"work... \boxed{2}" in seen[0]
    assert runner.state["verify_passed"] is True


def test_verifier_trace_retains_parse_and_provider_metadata(monkeypatch):
    monkeypatch.setattr("hbws.runner.llm.chat", lambda *a, **k: r"\boxed{3}")
    monkeypatch.setattr(
        "hbws.runner.llm.last_response_meta",
        lambda: {"finish_reason": "stop", "out_tokens": 7})
    workflow = {
        "nodes": [{"id": "v", "type": "verify", "k": 1}],
        "edges": [{"from": "v", "to": "END"}],
    }
    task = {"id": "m1", "family": "math", "prompt": "1+1?",
            "gold_answer": "2"}
    runner = WorkflowRun(workflow, task, BUDGET_TIERS["loose"])
    runner.state["solution"] = r"\boxed{2}"
    result = runner.run()
    diag = result["trace"][0]["verifier"]
    assert diag["candidate_answer"] == "2"
    assert diag["checks"][0]["checker_answer"] == "3"
    assert diag["checks"][0]["parse_status"] == "parsed"
    assert diag["checks"][0]["response"]["finish_reason"] == "stop"
