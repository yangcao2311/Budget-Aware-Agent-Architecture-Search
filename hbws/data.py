"""Dataset preparation and frozen splits.

Splits are generated ONCE with SPLIT_SEED, written to data/*.jsonl, and never
regenerated (方案 §4). Grouped splitting: MBPP+ split by task_id order blocks;
MATH split by subject (in-domain subjects for dev/val/test, held-out subjects
form the OOD frozen set).
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SPLIT_SEED = 20260804

MATH_IN_SUBJECTS = ["algebra", "counting_and_probability", "number_theory", "prealgebra"]
MATH_OOD_SUBJECTS = ["intermediate_algebra", "precalculus"]
MATH_LEVELS = {"Level 4", "Level 5"}  # v4.0 §5.1: L3 dropped for headroom

# v4.0 one-time authorized resplit (2026-08-04, before any real search run);
# frozen permanently afterwards. F1/F2 fidelity subsets are nested prefixes
# of dev (which is stratified-shuffled at generation time).
SPLIT_SIZES = {"dev": 120, "val": 40, "test": 150}


def _extract_gold(solution: str) -> str | None:
    from .verify import extract_boxed
    return extract_boxed(solution)


def prepare_all():
    from datasets import load_dataset
    DATA_DIR.mkdir(exist_ok=True)
    rng = random.Random(SPLIT_SEED)

    # ---- code family: MBPP+ ----
    d = load_dataset("evalplus/mbppplus", split="test")
    tasks = []
    for ex in d:
        tasks.append({
            "id": f"mbpp_{ex['task_id']}",
            "family": "code",
            "prompt": (ex["prompt"] + "\n\nYour function must satisfy tests like:\n"
                       + "\n".join(ex["test_list"][:1])),
            "feedback_tests": "\n".join(ex["test_imports"]) + "\n" + "\n".join(ex["test_list"]),
            "grading_tests": ex["test"],
        })
    rng.shuffle(tasks)
    _write_splits("code", tasks)

    # ---- math family: MATH L3-5, grouped by subject ----
    def load_subjects(subjects):
        out = []
        for s in subjects:
            ds = load_dataset("EleutherAI/hendrycks_math", s, split="test")
            for i, ex in enumerate(ds):
                if ex["level"] not in MATH_LEVELS:
                    continue
                gold = _extract_gold(ex["solution"])
                if not gold:
                    continue
                out.append({"id": f"math_{s}_{i}", "family": "math",
                            "prompt": ex["problem"], "gold_answer": gold,
                            "subject": s, "level": ex["level"]})
        return out

    in_tasks = load_subjects(MATH_IN_SUBJECTS)
    rng.shuffle(in_tasks)
    _write_splits("math", in_tasks)

    ood = load_subjects(MATH_OOD_SUBJECTS)
    rng.shuffle(ood)
    _dump("math_ood.jsonl", ood[:100])

    # ---- code OOD: HumanEval+ ----
    he = load_dataset("evalplus/humanevalplus", split="test")
    ood_code = [{
        "id": ex["task_id"].replace("/", "_"),
        "family": "code",
        "prompt": ("Complete the following Python function. Return the full "
                   "function in one ```python block.\n\n" + ex["prompt"]),
        "feedback_tests": "",  # HumanEval+ has no visible asserts; verify node degrades
        "grading_tests": ex["test"] + f"\ncheck({ex['entry_point']})",
    } for ex in he]
    rng.shuffle(ood_code)
    _dump("code_ood.jsonl", ood_code[:100])
    print("all splits written to", DATA_DIR)


def _write_splits(family: str, tasks: list[dict]):
    need = sum(SPLIT_SIZES.values())
    assert len(tasks) >= need, f"{family}: {len(tasks)} < {need}"
    i = 0
    for split, n in SPLIT_SIZES.items():
        _dump(f"{family}_{split}.jsonl", tasks[i:i + n])
        i += n


def _dump(name: str, rows: list[dict]):
    path = DATA_DIR / name
    if path.exists():
        raise RuntimeError(f"{path} exists — frozen splits are never regenerated")
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"  {name}: {len(rows)} tasks")


def load_split(family: str, split: str) -> list[dict]:
    path = DATA_DIR / f"{family}_{split}.jsonl"
    return [json.loads(l) for l in open(path)]


def load_ood_visible() -> list[dict]:
    """OOD code tasks with visible tests recovered from docstring examples
    (scripts/build_ood_visible_tests.py). Same 100 tasks as code_ood, but
    the verifier has signal, so domain transfer can be measured without
    being confounded with verifier availability. Only the 68 tasks that
    yielded parsable examples are returned."""
    rows = [json.loads(l) for l in open(DATA_DIR / "code_ood_visible.jsonl")]
    return [r for r in rows if r["feedback_tests"].strip()]


# ---------------------------------------------------------------------------
# Third domain, added 2026-08-05 for PROSPECTIVE validation only.
# BIG-Bench Hard logical-deduction and related deterministic reasoning tasks:
# neither code nor math, multiple-choice, exact-match grading. Never used in
# any search, tuning, or selection.
# ---------------------------------------------------------------------------

BBH_SUBTASKS = ["logical_deduction_five_objects",
                "logical_deduction_seven_objects",
                "tracking_shuffled_objects_five_objects",
                "date_understanding"]


def prepare_logic(n: int = 120):
    from datasets import load_dataset
    rng = random.Random(SPLIT_SEED)
    rows = []
    for sub in BBH_SUBTASKS:
        ds = load_dataset("lukaemon/bbh", sub, split="test")
        for i, ex in enumerate(ds):
            if not re.match(r"^\([A-Z]\)$", ex["target"].strip()):
                continue
            rows.append({"id": f"bbh_{sub}_{i}", "family": "logic",
                         "prompt": ex["input"], "gold_answer": ex["target"].strip(),
                         "subject": sub})
    rng.shuffle(rows)
    _dump("logic_prospective.jsonl", rows[:n])


def load_logic() -> list[dict]:
    return [json.loads(l) for l in open(DATA_DIR / "logic_prospective.jsonl")]
