#!/usr/bin/env python
"""Decouple two things the first OOD run confounded.

The original OOD code split (HumanEval+) shipped with EMPTY visible tests, so
its verify node ran in the no-signal regime while the in-domain parameters
came from the oracle regime. The locked transfer prediction therefore tested
"new domain AND no verifier" at once, and failed. This script builds visible
tests for the same OOD tasks by parsing the docstring examples, producing a
second OOD condition that differs from in-domain ONLY in the domain.

Docstring example forms handled:
    f(args) ➞ value        (HumanEval "Examples" style)
    f(args) == value
    >>> f(args)  /  value  (doctest style)
Anything not parsed is skipped; tasks with zero parsed examples are recorded
so the coverage is reported rather than hidden.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "code_ood.jsonl"
DST = ROOT / "data" / "code_ood_visible.jsonl"

ARROW = re.compile(r"^\s*([A-Za-z_]\w*\s*\(.*\))\s*(?:➞|==|=>|->)\s*(.+?)\s*$")
DOCTEST = re.compile(r"^\s*>>>\s*([A-Za-z_]\w*\s*\(.*\))\s*$")


def entry_point(prompt: str) -> str | None:
    m = re.search(r"^\s*def\s+([A-Za-z_]\w*)\s*\(", prompt, re.M)
    return m.group(1) if m else None


def extract(prompt: str, fn: str) -> list[str]:
    lines = prompt.splitlines()
    out = []
    for i, line in enumerate(lines):
        m = ARROW.match(line)
        if m and m.group(1).lstrip().startswith(fn):
            call, want = m.group(1).strip(), m.group(2).strip().rstrip(",.")
            if want and not want.startswith("#"):
                out.append(f"assert {call} == {want}")
            continue
        m = DOCTEST.match(line)
        if m and m.group(1).lstrip().startswith(fn) and i + 1 < len(lines):
            want = lines[i + 1].strip()
            if want and not want.startswith(">>>"):
                out.append(f"assert {m.group(1).strip()} == {want}")
    # keep only syntactically valid assertions
    ok = []
    for a in out:
        try:
            compile(a, "<t>", "exec")
            ok.append(a)
        except SyntaxError:
            pass
    return ok


def main():
    rows = [json.loads(l) for l in open(SRC)]
    kept, counts = [], []
    for r in rows:
        fn = entry_point(r["prompt"])
        tests = extract(r["prompt"], fn) if fn else []
        counts.append(len(tests))
        r = dict(r)
        r["feedback_tests"] = "\n".join(tests)
        kept.append(r)
    with_tests = sum(1 for c in counts if c)
    print(f"parsed visible tests for {with_tests}/{len(rows)} OOD tasks "
          f"(mean {sum(counts)/len(counts):.1f} asserts, max {max(counts)})")
    if DST.exists():
        print(f"{DST} already exists; refusing to overwrite")
        return
    with open(DST, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    print(f"written {DST}")


if __name__ == "__main__":
    main()
