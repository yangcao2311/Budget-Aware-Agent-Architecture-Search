"""Deterministic verifiers.

- Code family: run the extracted program against unit tests in a resource-
  limited subprocess (5s CPU, 1GB address space). Model output is executed
  ONLY here, inside the sandbox, never in the search/runner process.
- Math family: extract \\boxed answer and compare after normalization
  (exact string, numeric, or sympy equivalence when available).

Two grading modes per protocol:
- dev feedback tests (visible to the workflow's `verify` node)
- final grading tests (held-out MBPP+ extended tests; only in eval, never
  fed back to the model)
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

SANDBOX_TIMEOUT = 8  # wall seconds per test run
_PY_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    blocks = _PY_FENCE.findall(text)
    return blocks[-1].strip() if blocks else text.strip()


_SANDBOX_PRELUDE = """\
import os, resource, sys
os.environ["OPENBLAS_NUM_THREADS"] = "1"
resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
# 4GB: the evalplus harness imports numpy, whose BLAS needs address space.
resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
sys.setrecursionlimit(10000)
"""


def run_code_tests(solution_text: str, test_code: str) -> tuple[bool, str]:
    """Returns (passed, feedback). feedback is truncated stderr/assertion info.

    Empty test_code means NO verification signal — that must read as
    not-passed with a no-signal message (blind refinement), never as a
    vacuous pass (Fig.3 mask=0.0 semantics)."""
    if not test_code.strip():
        return False, ("no tests are available; carefully review your solution "
                       "for correctness and edge cases")
    code = extract_code(solution_text)
    program = _SANDBOX_PRELUDE + code + "\n\n" + test_code + "\nprint('ALL_TESTS_PASSED')\n"
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "prog.py"
        path.write_text(program)
        try:
            r = subprocess.run([sys.executable, "-I", str(path)], cwd=td,
                               capture_output=True, text=True, timeout=SANDBOX_TIMEOUT)
        except subprocess.TimeoutExpired:
            return False, "TIMEOUT: execution exceeded time limit"
    if r.returncode == 0 and "ALL_TESTS_PASSED" in r.stdout:
        return True, "all tests passed"
    err = (r.stderr or r.stdout).strip()
    return False, err[-1500:] if err else "tests failed with no output"


# ---------------------------------------------------------------------------
# Math grading (normalization adapted from the Hendrycks MATH conventions)
# ---------------------------------------------------------------------------

def extract_boxed(text: str) -> str | None:
    idx = text.rfind("\\boxed")
    if idx == -1:
        m = re.search(r"[Aa]nswer[:\s]+([^\n]+)", text)
        return m.group(1).strip() if m else None
    i = text.find("{", idx)
    if i == -1:
        return None
    depth, j = 0, i
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    return None


def _norm(a: str) -> str:
    a = a.strip().strip("$").strip()
    a = a.replace("\\left", "").replace("\\right", "")
    a = a.replace("\\!", "").replace("\\,", "").replace(" ", "")
    a = a.replace("dfrac", "frac").replace("tfrac", "frac")
    a = re.sub(r"\\text\{[^}]*\}", "", a)
    a = a.replace("^{\\circ}", "").replace("^\\circ", "")
    a = a.rstrip(".")
    if a.startswith("\\frac") is False:
        a = a.lstrip("0") or "0" if re.fullmatch(r"0+\d+", a) else a
    return a


def math_equal(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    p, g = _norm(pred), _norm(gold)
    if p == g:
        return True
    try:
        if abs(float(p) - float(g)) < 1e-6:
            return True
    except ValueError:
        pass
    try:
        from sympy.parsing.latex import parse_latex
        import sympy
        return sympy.simplify(parse_latex(p) - parse_latex(g)) == 0
    except Exception:
        return False


_CHOICE = re.compile(r"\(([A-Z])\)")


def same_choice(a: str, b: str) -> bool:
    """Do two answer strings name the same multiple-choice option?
    Tolerates "(A)" vs "A" vs "Option A"."""
    ca = _CHOICE.findall(a) or re.findall(r"\b([A-Z])\b", a)
    cb = _CHOICE.findall(b) or re.findall(r"\b([A-Z])\b", b)
    return bool(ca) and bool(cb) and ca[-1] == cb[-1]


def grade_choice(solution_text: str, gold_answer: str) -> bool:
    """Multiple-choice grading for the logic family: compare the boxed
    choice label, falling back to the last (X) mentioned."""
    pred = extract_boxed(solution_text)
    cand = _CHOICE.findall(pred or "") or _CHOICE.findall(solution_text or "")
    gold = _CHOICE.findall(gold_answer or "")
    if not cand or not gold:
        return False
    return cand[-1] == gold[-1]


def grade_math(solution_text: str, gold_answer: str) -> bool:
    """FINAL grading only. Gold answers must never reach a workflow's verify
    node (label leakage): the runner's internal math verifier is an LLM
    checker charged to the task budget (see runner.py / prompts check_math)."""
    return math_equal(extract_boxed(solution_text), gold_answer)
