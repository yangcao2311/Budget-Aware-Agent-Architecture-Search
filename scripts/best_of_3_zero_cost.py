#!/usr/bin/env python
"""A matched-realized-cost self-consistency / test-filtered-selection
baseline, built entirely from data already on disk -- no new inference.

Each task in the frozen test split was already drafted independently three
times (once per execution seed) to compute the seed-averaged baseline. Those
three drafts are exactly a k=3 sampling pool. This script selects one answer
per task from that pool -- the way a best-of-3 baseline would -- and grades
the selection, so its logical cost is exactly three generate calls per task,
identical to what the baseline already spent, and its accuracy is directly
comparable to Table 1's protected-arm figures at that same logical cost.

Selection rules, fixed in advance (not tuned on the outcome):
  math:  normalize each seed's boxed answer; majority vote; ties broken by
         lowest seed index. Grading reuses that seed's own precomputed
         success_symbolic (deterministic given the same solution text).
  code:  run each seed's solution against the VISIBLE feedback tests only
         (zero-cost, a local subprocess, not a re-grading against the held-out
         suite). If any pass, pick the lowest-seed-index passer; if none
         pass, fall back to seed 0. The HELD-OUT grading tests are used only
         to score the one selection made this way, never to choose among
         candidates -- matching the paper's own baseline/verifier protocol.

Limitation this script does not hide: this is one pool of three dependent
seeds per task, not three independently-drawn best-of-3 pools, so its
uncertainty likely understates a true repeated-sampling design. We report a
task-clustered bootstrap CI, which is the same caveat Appendix L already
states about seed non-independence within a task.
"""
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hbws import verify
from hbws.data import load_split

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
SEEDS = [0, 1, 2]
NB = 10000


def load_raw(rel):
    """task_id -> {seed: (solution_text, success, success_symbolic)}"""
    out = defaultdict(dict)
    for s in SEEDS:
        for r in map(json.loads, open(EXP / rel / f"results_seed{s}.jsonl")):
            out[r["task_id"]][s] = (
                r.get("solution") or "",
                bool(r["success"]),
                bool(r.get("success_symbolic", r["success"])),
            )
    return out


def best_of_3_math():
    tasks = {t["id"]: t for t in load_split("math", "test")[:150]}
    raw = load_raw("envelope_test/cot_math_loose")
    per_task_correct = {}
    for tid, by_seed in raw.items():
        if len(by_seed) != 3:
            continue
        norm_answers = {}
        for s, (sol, ok, ok_sym) in by_seed.items():
            boxed = verify.extract_boxed(sol)
            norm_answers[s] = verify._norm(boxed) if boxed else None
        counts = Counter(v for v in norm_answers.values() if v is not None)
        if not counts:
            chosen = 0
        else:
            best = max(counts.values())
            winners = sorted(s for s in SEEDS if norm_answers.get(s) in
                              {a for a, c in counts.items() if c == best})
            chosen = winners[0]
        per_task_correct[tid] = by_seed[chosen][2]
    return per_task_correct


def best_of_3_code():
    tasks = {t["id"]: t for t in load_split("code", "test")[:150]}
    raw = load_raw("envelope_test/direct_code_loose")
    per_task_correct = {}
    for tid, by_seed in raw.items():
        if len(by_seed) != 3 or tid not in tasks:
            continue
        task = tasks[tid]
        passers = []
        for s in SEEDS:
            sol = by_seed[s][0]
            ok, _ = verify.run_code_tests(sol, task["feedback_tests"])
            if ok:
                passers.append(s)
        chosen = passers[0] if passers else 0
        per_task_correct[tid] = by_seed[chosen][2]
    return per_task_correct


def cluster_ci(vals, seed=0):
    if not vals:
        return float("nan"), float("nan"), float("nan")
    n = len(vals)
    rng = random.Random(seed)
    point = sum(vals) / n
    boots = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(NB))
    return point, boots[int(0.025 * NB)], boots[int(0.975 * NB)]


def baseline_p(rel):
    raw = load_raw(rel)
    return sum(sum(v[2] for v in by.values()) / len(by) for by in raw.values() if by) / len(raw)


def ref_preserving_per_task(rel):
    """Per-task seed-averaged accuracy of the reference-preserving arm."""
    raw = load_raw(rel)
    return {tid: sum(v[2] for v in by.values()) / len(by)
            for tid, by in raw.items() if by}


def paired_diff_ci(ref_by_task, alt_by_task, seed=0):
    """Task-paired bootstrap CI on mean(ref) - mean(alt), resampling tasks."""
    common = sorted(set(ref_by_task) & set(alt_by_task))
    diffs = [ref_by_task[t] - alt_by_task[t] for t in common]
    n = len(diffs)
    rng = random.Random(seed)
    point = sum(diffs) / n
    boots = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(NB))
    return point, boots[int(0.025 * NB)], boots[int(0.975 * NB)], n


def main():
    configs = [
        ("math", best_of_3_math, "envelope_test/cot_math_loose",
         "envelope_test/incumbent_refine_cot_math_loose", "self-consistency majority vote"),
        ("code", best_of_3_code, "envelope_test/direct_code_loose",
         "envelope_test/incumbent_refine_code_loose", "visible-test-filtered selection"),
    ]
    for fam, fn, base_rel, ref_rel, method_name in configs:
        per_task = fn()
        alt_by_task = {t: (1.0 if v else 0.0) for t, v in per_task.items()}
        vals = list(alt_by_task.values())
        point, lo, hi = cluster_ci(vals)
        p = baseline_p(base_rel)
        ref_by_task = ref_preserving_per_task(ref_rel)
        ref_point, ref_lo, ref_hi = cluster_ci(list(ref_by_task.values()))
        d_point, d_lo, d_hi, n_common = paired_diff_ci(ref_by_task, alt_by_task)
        print(f"{fam} ({method_name}): n={len(vals)}  baseline p={p:.4f}")
        print(f"  three-sample baseline acc={point:.4f}  [95% CI {lo:.4f}, {hi:.4f}]")
        print(f"  reference-preserving acc={ref_point:.4f}  [95% CI {ref_lo:.4f}, {ref_hi:.4f}]")
        print(f"  paired diff (ref - three-sample) = {d_point:+.4f}  "
              f"[95% CI {d_lo:+.4f}, {d_hi:+.4f}]  (n={n_common} paired tasks)")


if __name__ == "__main__":
    main()
