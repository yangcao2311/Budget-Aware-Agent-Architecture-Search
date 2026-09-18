#!/usr/bin/env python3
"""Design-based precision and detectability analysis for the provenance matrix.

Two questions the null results cannot answer on their own:

1.  *Precision.*  Given the observed task-clustered intervals, which
    assignment-versus-same-policy differences remain compatible with the data?
    We report the interval endpoints as the tolerable-difference statement
    rather than repeating that zero is included.
2.  *Detectability.*  For a prespecified effect, what does a paired
    task-clustered design of this shape actually detect?  We simulate the
    design under an explicit task-level random effect, so the unit of
    replication is the task and the three execution seeds are exchangeable
    within it.  There is no single minimum detectable effect: it depends
    jointly on task count, within-task correlation, and event frequency, and
    we report the surface rather than one number.

No observed-power calculation is performed: every reported detectability
number comes from a prespecified effect under a simulated design, never from
the realised estimate of an executed contrast.

No model calls.  Reads only frozen result rows.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
BASE = EXP / "glm4flash_repaired_matrix_clean_20260910_envelope_test"
CAUSAL = EXP / "glm4flash_repaired_matrix_final_20260910_causal"
OUT_JSON = EXP / "effect_precision_20260916.json"
PAPER = ROOT / "paper"

# Cells that are not censored by the reservation gate.
CELLS = {
    "code/tight": ("direct_code_tight", "arm1_assign_code_tight",
                   "arm2_samepolicy_code_tight"),
    "code/loose": ("direct_code_loose", "arm1_assign_code_loose",
                   "arm2_samepolicy_code_loose"),
    "math/loose": ("cot_math_loose", "arm1_assign_math_loose",
                   "arm2_samepolicy_math_loose"),
}
SEEDS = 3
BOOT_DRAWS = 10_000
BOOT_SEED = 20260916


def load_unique(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for line in path.open():
        row = json.loads(line)
        if row["task_id"] in rows:
            raise ValueError(f"duplicate task_id in {path}")
        rows[row["task_id"]] = row
    if len(rows) != 150:
        raise ValueError(f"expected 150 rows in {path}, found {len(rows)}")
    return rows


def ok(row: dict) -> int:
    return int(bool(row.get("success")))


def collect(cell: str) -> dict[str, list[tuple[int, int, int]]]:
    """task_id -> [(baseline, assignment, same-policy) correctness] per seed."""
    base_run, assign_run, same_run = CELLS[cell]
    per_task: dict[str, list[tuple[int, int, int]]] = {}
    for seed in range(SEEDS):
        base = load_unique(BASE / base_run / f"results_seed{seed}.jsonl")
        assign = load_unique(CAUSAL / assign_run / f"results_seed{seed}.jsonl")
        same = load_unique(CAUSAL / same_run / f"results_seed{seed}.jsonl")
        if base.keys() != assign.keys() or base.keys() != same.keys():
            raise ValueError(f"task mismatch in {cell} seed {seed}")
        for task_id in base:
            per_task.setdefault(task_id, []).append(
                (ok(base[task_id]), ok(assign[task_id]), ok(same[task_id])))
    return per_task


# --------------------------------------------------------------------------
# Part 1: observed precision, task-clustered
# --------------------------------------------------------------------------

def estimates(tasks: list[list[tuple[int, int, int]]],
              lambdas=(1.0, 3.0, 5.0)) -> dict[str, float]:
    rows = [r for task in tasks for r in task]
    wrong = [r for r in rows if r[0] == 0]
    right = [r for r in rows if r[0] == 1]
    repair_a = sum(a for _, a, _ in wrong)
    repair_s = sum(s for _, _, s in wrong)
    break_a = sum(1 - a for _, a, _ in right)
    break_s = sum(1 - s for _, _, s in right)
    out = {
        "repair_difference": (repair_a - repair_s) / max(1, len(wrong)),
        "breakage_difference": (break_a - break_s) / max(1, len(right)),
        "accuracy_difference": sum(a - s for _, a, s in rows) / len(rows),
    }
    for lam in lambdas:
        out[f"gross_value_lambda{lam:g}"] = (
            (repair_a - repair_s) - lam * (break_a - break_s)) / len(rows)
    return out


def clustered_intervals(per_task: dict[str, list[tuple[int, int, int]]]) -> dict:
    tasks = list(per_task.values())
    point = estimates(tasks)
    rng = random.Random(BOOT_SEED)
    draws: dict[str, list[float]] = {k: [] for k in point}
    n = len(tasks)
    for _ in range(BOOT_DRAWS):
        sample = [tasks[rng.randrange(n)] for _ in range(n)]
        for k, v in estimates(sample).items():
            draws[k].append(v)
    result = {}
    for key, value in point.items():
        ordered = sorted(draws[key])
        lo = ordered[int(0.025 * BOOT_DRAWS)]
        hi = ordered[min(BOOT_DRAWS - 1, int(0.975 * BOOT_DRAWS))]
        result[key] = {"point": value, "ci95": [lo, hi],
                       "half_width": (hi - lo) / 2}
    return result


def observed_rates(per_task: dict[str, list[tuple[int, int, int]]]) -> dict:
    rows = [r for task in per_task.values() for r in task]
    right = [r for r in rows if r[0] == 1]
    wrong = [r for r in rows if r[0] == 0]
    # Within-task correlation of the same-policy breakage indicator, estimated
    # by a one-way random-effects ANOVA intraclass correlation on the
    # reference-correct pairs of each task.
    groups = []
    for task in per_task.values():
        vals = [1 - s for b, _, s in task if b == 1]
        if len(vals) >= 2:
            groups.append(vals)
    icc = anova_icc(groups)
    return {
        "reference_correct_pairs": len(right),
        "reference_wrong_pairs": len(wrong),
        "same_policy_breakage_rate": sum(1 - s for _, _, s in right) / max(1, len(right)),
        "assignment_breakage_rate": sum(1 - a for _, a, _ in right) / max(1, len(right)),
        "same_policy_repair_rate": sum(s for _, _, s in wrong) / max(1, len(wrong)),
        "assignment_repair_rate": sum(a for _, a, _ in wrong) / max(1, len(wrong)),
        "breakage_within_task_icc": icc,
        "icc_groups": len(groups),
    }


def anova_icc(groups: list[list[int]]) -> float | None:
    """One-way random-effects ICC(1) for equal-ish group sizes."""
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return None
    k = len(groups)
    sizes = [len(g) for g in groups]
    total = sum(sizes)
    grand = sum(sum(g) for g in groups) / total
    ss_between = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups)
    ss_within = sum(sum((x - np.mean(g)) ** 2 for x in g) for g in groups)
    df_b, df_w = k - 1, total - k
    if df_w <= 0 or df_b <= 0:
        return None
    ms_b, ms_w = ss_between / df_b, ss_within / df_w
    n0 = (total - sum(s ** 2 for s in sizes) / total) / df_b
    denom = ms_b + (n0 - 1) * ms_w
    if denom <= 0:
        return 0.0
    return float(max(0.0, (ms_b - ms_w) / denom))


# --------------------------------------------------------------------------
# Part 2: prespecified-effect detectability under a clustered paired design
# --------------------------------------------------------------------------

def beta_params(pi: float, rho: float) -> tuple[float, float]:
    a = pi * (1 - rho) / rho
    b = (1 - pi) * (1 - rho) / rho
    return a, b


def student_crit(df: int, alpha: float) -> float:
    """Exact two-sided Student-t critical value for ``df`` task clusters."""
    from scipy import stats
    return float(stats.t.ppf(1 - alpha / 2, df))


def eligible_profile(per_task: dict[str, list[tuple[int, int, int]]]) -> np.ndarray:
    """Per-task count of reference-correct seeds, including tasks with none.

    Breakage is defined only where the stored reference is correct, so this,
    and not three seeds per task, is the eligible denominator.  A task with no
    reference-correct seed contributes no cluster to the breakage endpoint.
    """
    return np.array([sum(1 for b, _, _ in seeds if b == 1)
                     for seeds in per_task.values()], dtype=int)


MAX_CELLS = 24_000_000  # element budget per simulated array block


def _one_block(n_tasks: int, profile: np.ndarray, pi: float, rho_task: float,
               rel_reduction: float, n_sims: int, rng: np.random.Generator,
               rho_arm: float, alpha: float) -> tuple[np.ndarray, np.ndarray,
                                                      np.ndarray]:
    m = rng.choice(profile, size=(n_sims, n_tasks))
    max_m = max(int(profile.max()), 1)
    if rho_task <= 0:
        p = np.full((n_sims, n_tasks), float(pi))
    else:
        a, b = beta_params(pi, rho_task)
        p = rng.beta(a, b, size=(n_sims, n_tasks))

    mask = np.arange(max_m)[None, None, :] < m[:, :, None]
    control = (rng.random((n_sims, n_tasks, max_m)) < p[:, :, None]) & mask
    thinned = control & (rng.random((n_sims, n_tasks, max_m))
                         < (1.0 - rel_reduction))
    q = np.clip(p * (1.0 - rel_reduction), 0.0, 1.0)
    independent = (rng.random((n_sims, n_tasks, max_m)) < q[:, :, None]) & mask
    couple = rng.random((n_sims, n_tasks, max_m)) < rho_arm
    treat = np.where(couple, thinned, independent)

    counts = m.astype(np.float32)
    contributing = counts > 0
    safe = np.where(contributing, counts, 1.0)
    d = (treat.sum(axis=2) - control.sum(axis=2)) / safe
    d = np.where(contributing, d, np.nan)

    k = contributing.sum(axis=1).astype(float)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(d, axis=1)
        sd = np.nanstd(d, axis=1, ddof=1)
    se = np.where(k > 1, sd / np.sqrt(np.maximum(k, 1.0)), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where((se > 0) & np.isfinite(se) & np.isfinite(mean),
                     mean / se, 0.0)
    crit_map = {kk: student_crit(int(max(kk - 1, 1)), alpha)
                for kk in np.unique(k)}
    crit = np.array([crit_map[kk] for kk in k])
    return np.abs(t) > crit, k, counts.sum(axis=1)


def simulate_power(n_tasks: int, profile: np.ndarray, pi: float,
                   rho_task: float, rel_reduction: float, n_sims: int,
                   rng: np.random.Generator, rho_arm: float = 0.0,
                   alpha: float = 0.05) -> dict:
    """Power of a task-clustered paired test for a prespecified effect.

    ``profile`` is resampled so that each simulated task receives a number of
    reference-correct seeds drawn from the executed design, which fixes both
    the eligible denominator and the share of tasks contributing no eligible
    position at all.  Each task carries a latent breakage propensity with
    intraclass correlation ``rho_task``.  ``rho_arm`` is the share of the
    protective effect realised as a within-position thinning of the control
    arm's breakages rather than as an independent draw: it spans paired
    structures that the design permits and the data do not identify, so every
    reported figure is conditional on it.  The test is a paired t-test on
    task-level mean differences over contributing clusters.

    Simulations are evaluated in blocks so that peak memory stays bounded
    regardless of ``n_tasks``.
    """
    max_m = max(int(profile.max()), 1)
    per_sim = max(n_tasks * max_m, 1)
    block = max(1, min(n_sims, MAX_CELLS // per_sim))
    rejects, ks, positions, done = [], [], [], 0
    while done < n_sims:
        take = min(block, n_sims - done)
        r, k, pos = _one_block(n_tasks, profile, pi, rho_task, rel_reduction,
                              take, rng, rho_arm, alpha)
        rejects.append(r); ks.append(k); positions.append(pos)
        done += take
    return {
        "power": float(np.concatenate(rejects).mean()),
        "mean_contributing_clusters": float(np.concatenate(ks).mean()),
        "mean_eligible_positions": float(np.concatenate(positions).mean()),
    }


def power_only(*args, **kwargs) -> float:
    return simulate_power(*args, **kwargs)["power"]


def mde(n_tasks: int, profile: np.ndarray, pi: float, rho_task: float,
        n_sims: int, rng: np.random.Generator, rho_arm: float = 0.0,
        target: float = 0.80) -> float | None:
    """Smallest prespecified proportional reduction reaching ``target`` power."""
    search_sims = max(1000, n_sims // 4)
    for rel in [0.05 * k for k in range(1, 21)]:
        if power_only(n_tasks, profile, pi, rho_task, rel, search_sims, rng,
                      rho_arm=rho_arm) >= target:
            return round(rel, 2)
    return None


def tasks_for_power(profile: np.ndarray, pi: float, rho_task: float,
                    rel_reduction: float, n_sims: int,
                    rng: np.random.Generator, rho_arm: float = 0.0,
                    target: float = 0.80, cap: int = 16_000) -> int | None:
    """Screened-task count reaching ``target`` power, as a scenario.

    This is conditional on the event rate, the intraclass correlation and the
    paired structure ``rho_arm``; it is not a design requirement, and it moves
    substantially with ``rho_arm``, which our data do not identify.
    """
    search_sims = max(1000, n_sims // 4)
    lo, hi = 50, 200
    while hi <= cap:
        if power_only(hi, profile, pi, rho_task, rel_reduction, search_sims,
                      rng, rho_arm=rho_arm) >= target:
            break
        lo, hi = hi, hi * 2
    else:
        return None
    while lo + 50 < hi:
        mid = (lo + hi) // 2
        if power_only(mid, profile, pi, rho_task, rel_reduction, search_sims,
                      rng, rho_arm=rho_arm) >= target:
            hi = mid
        else:
            lo = mid
    return hi


# --------------------------------------------------------------------------
# Part 3: certification feasibility under the prespecified Hoeffding rule
# --------------------------------------------------------------------------

def hoeffding_frr_bound(false_rejections: int, correct: int, n: int,
                        alpha: float = 0.05) -> float:
    slack = math.sqrt(n * math.log(1 / alpha) / 2)
    return min(1.0, (false_rejections + slack) / correct)


def tasks_needed_for_zero_rejections(correct_fraction: float, epsilon: float,
                                     alpha: float = 0.05) -> int:
    """Smallest n with a certifiable bound when zero false rejections occur."""
    n = 1
    while True:
        correct = correct_fraction * n
        if hoeffding_frr_bound(0, correct, n, alpha) <= epsilon:
            return n
        n += 1


def main() -> None:
    report: dict = {
        "generator": "scripts/analyze_effect_precision.py",
        "bootstrap_draws": BOOT_DRAWS,
        "bootstrap_seed": BOOT_SEED,
        "cluster_unit": "task; breakage clusters are tasks with at least one "
                        "reference-correct seed",
        "note": ("Detectability figures come from prespecified effects under a "
                 "simulated clustered design; no observed-power calculation is "
                 "reported for any executed contrast.  Every figure is "
                 "conditional on an assumed paired structure (rho_arm), which "
                 "the design does not identify, so task counts are scenarios "
                 "rather than requirements."),
        "cells": {},
    }

    profiles: dict[str, np.ndarray] = {}
    for cell in CELLS:
        per_task = collect(cell)
        profiles[cell] = eligible_profile(per_task)
        prof = profiles[cell]
        report["cells"][cell] = {
            "rates": observed_rates(per_task),
            "assignment_minus_same_policy": clustered_intervals(per_task),
            "eligible_positions": int(prof.sum()),
            "tasks_screened": int(prof.size),
            "tasks_with_an_eligible_seed": int((prof > 0).sum()),
            "eligible_seeds_per_task": {str(k): int(v) for k, v in
                                        zip(*np.unique(prof, return_counts=True))},
        }

    rng = np.random.default_rng(20260916)
    n_sims = 4000
    # A reference eligibility profile for the generic surface: the pooled
    # per-task reference-correct seed counts of the uncensored cells.
    pooled = np.concatenate([profiles[c] for c in CELLS])
    report["surface_eligibility_profile"] = {
        "source": "pooled per-task reference-correct seed counts, three "
                  "uncensored cells",
        "mean_eligible_seeds_per_screened_task": float(pooled.mean()),
        "share_of_tasks_with_no_eligible_seed": float((pooled == 0).mean()),
    }

    RHO_ARM = (0.0, 0.5, 0.9)
    surface = []
    for pi in (0.005, 0.01, 0.02, 0.04, 0.08):
        for rho in (0.0, 0.2, 0.5):
            for n_tasks in (150, 300, 600, 1200, 2400):
                row = {
                    "breakage_rate": pi,
                    "within_task_icc": rho,
                    "tasks_screened": n_tasks,
                    "power_half_reduction": {},
                    "mde_relative_at_80pct": {},
                    "type_i_error_at_null": {},
                }
                for ra in RHO_ARM:
                    key = f"rho_arm={ra:g}"
                    res = simulate_power(n_tasks, pooled, pi, rho, 0.50,
                                         n_sims, rng, rho_arm=ra)
                    row["power_half_reduction"][key] = res["power"]
                    row["mde_relative_at_80pct"][key] = mde(
                        n_tasks, pooled, pi, rho, n_sims, rng, rho_arm=ra)
                    row["type_i_error_at_null"][key] = power_only(
                        n_tasks, pooled, pi, rho, 0.0, n_sims, rng, rho_arm=ra)
                    if ra == 0.0:
                        row["mean_contributing_clusters"] = res[
                            "mean_contributing_clusters"]
                        row["mean_eligible_positions"] = res[
                            "mean_eligible_positions"]
                surface.append(row)
    report["detectability_surface"] = surface

    # What each executed cell could have detected, at its own observed event
    # rate, intraclass correlation and eligibility profile.  Prespecified
    # effects only; never the realised estimate.
    executed = []
    for cell, payload in report["cells"].items():
        rates = payload["rates"]
        pi = max(rates["same_policy_breakage_rate"], 1e-4)
        rho = min(0.9, max(0.0, rates["breakage_within_task_icc"] or 0.0))
        prof = profiles[cell]
        entry = {
            "cell": cell,
            "observed_breakage_rate": rates["same_policy_breakage_rate"],
            "observed_within_task_icc": rates["breakage_within_task_icc"],
            "tasks_screened": int(prof.size),
            "tasks_with_an_eligible_seed": int((prof > 0).sum()),
            "eligible_positions": int(prof.sum()),
            "power_half_reduction_as_executed": {},
            "power_full_elimination_as_executed": {},
            "tasks_scenario_80pct_half_reduction": {},
            "tasks_scenario_80pct_full_elimination": {},
            "type_i_error_at_null_as_executed": {},
        }
        for ra in RHO_ARM:
            key = f"rho_arm={ra:g}"
            entry["power_half_reduction_as_executed"][key] = power_only(
                int(prof.size), prof, pi, rho, 0.50, n_sims, rng, rho_arm=ra)
            entry["power_full_elimination_as_executed"][key] = power_only(
                int(prof.size), prof, pi, rho, 1.00, n_sims, rng, rho_arm=ra)
            entry["tasks_scenario_80pct_half_reduction"][key] = tasks_for_power(
                prof, pi, rho, 0.50, n_sims, rng, rho_arm=ra)
            entry["tasks_scenario_80pct_full_elimination"][key] = tasks_for_power(
                prof, pi, rho, 1.00, n_sims, rng, rho_arm=ra)
            entry["type_i_error_at_null_as_executed"][key] = power_only(
                int(prof.size), prof, pi, rho, 0.0, n_sims, rng, rho_arm=ra)
        executed.append(entry)
    report["executed_cell_detectability"] = executed

    # Certification feasibility.
    correct, n_heldout, false_rej = 835, 1194, 22
    report["certification_feasibility"] = {
        "held_out_tasks": n_heldout,
        "reference_correct": correct,
        "reference_correct_fraction": correct / n_heldout,
        "observed_false_rejections": false_rej,
        "observed_bound": hoeffding_frr_bound(false_rej, correct, n_heldout),
        "zero_rejection_floor_at_this_n": hoeffding_frr_bound(0, correct, n_heldout),
        "epsilon": 0.05,
        "tasks_needed_if_zero_false_rejections": tasks_needed_for_zero_rejections(
            correct / n_heldout, 0.05),
    }

    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", OUT_JSON)
    for e in report["executed_cell_detectability"]:
        print(f"{e['cell']:12s} clusters={e['tasks_with_an_eligible_seed']:3d} "
              f"eligible={e['eligible_positions']:3d} "
              f"power(half) {e['power_half_reduction_as_executed']} "
              f"scenario_n {e['tasks_scenario_80pct_half_reduction']}")
    fe = report["certification_feasibility"]
    print(f"zero-rejection floor at n={n_heldout}: "
          f"{fe['zero_rejection_floor_at_this_n']:.5f}; tasks needed "
          f"{fe['tasks_needed_if_zero_false_rejections']}")


if __name__ == "__main__":
    main()
