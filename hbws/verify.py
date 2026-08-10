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
    return _symbolic_equal(p, g)


def _symbolic_compute(p: str, g: str, conn) -> None:
    try:
        from sympy.parsing.latex import parse_latex
        import sympy
        conn.send(bool(sympy.simplify(parse_latex(p) - parse_latex(g)) == 0))
    except Exception:
        conn.send(False)
    finally:
        conn.close()


def _symbolic_equal(p: str, g: str, timeout_s: int = 5) -> bool:
    """Symbolic fallback with a hard timeout.

    sympy.simplify can hang indefinitely on adversarial expressions, so the
    call runs under a timeout and a timeout counts as "not equal" rather than
    blocking the grader. (Found the hard way: an unguarded version stalled a
    full re-grading pass.)

    History of this function's timeout mechanism, kept because each version
    fixed a real incident:
      1. SIGALRM-based. Only works in the main thread of the main
         interpreter. Every historical call site was single-threaded
         (apply_regrade.py, or generation runs whose answers never reached
         this fallback), so it never surfaced -- until protocol.evaluate()'s
         ThreadPoolExecutor first needed it from a worker thread:
         signal.signal() raised immediately, discarding the just-generated,
         just-paid-for solution via the caller's blanket except-and-replace.
      2. Thread-based (concurrent.futures), one pool per call. Correct, but
         a fresh ThreadPoolExecutor per call was expensive at regrade-pass
         volume (thousands of spin-up/tear-downs).
      3. Thread-based, one shared pool. Fixed the overhead, but exposed the
         real problem: Python cannot force-kill a thread, so a genuinely
         hung sympy call leaves its worker permanently occupied. Enough
         adversarial inputs across a 24k-call regrade pass exhausted the
         pool and the process never exited (hung at 100% CPU, memory
         climbing) even though every result had already been computed and
         written to disk.
    A subprocess CAN be force-killed, so this now runs the computation in a
    fresh process and terminates it on timeout -- the resources are always
    reclaimed, at the cost of one process spawn per fallback call (this path
    is the minority case; exact and float matches short-circuit before it).
    """
    import multiprocessing

    global _SYMPY_WARMED
    if not _SYMPY_WARMED:
        # fork-mode children inherit the parent's already-imported modules via
        # copy-on-write, so warming sympy here once makes every subsequent
        # fork's copy free instead of a few hundred ms of cold import each.
        try:
            import sympy  # noqa: F401
            from sympy.parsing.latex import parse_latex  # noqa: F401
        except Exception:
            pass
        _SYMPY_WARMED = True

    ctx = multiprocessing.get_context("fork")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_symbolic_compute, args=(p, g, child_conn))
    proc.start()
    child_conn.close()
    try:
        if parent_conn.poll(timeout_s):
            return parent_conn.recv()
        return False
    except Exception:
        return False
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(1.0)
            if proc.is_alive():
                proc.kill()
                proc.join()
        else:
            proc.join()
        parent_conn.close()


_SYMPY_WARMED = False
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
