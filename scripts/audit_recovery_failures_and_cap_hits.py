#!/usr/bin/env python3
"""Audit GLM matrix provider failures and an output-cap-hit proxy.

This script is deliberately read-only with respect to raw experiment logs.  It
recomputes (1) status distributions in the original repaired-executor baseline
and causal runs and (2) a completed-call output-cap-hit proxy in the final
materialized matrix, separating frozen recovery rows from untouched rows.

IMPORTANT: the generic runner did not persist the provider's ``finish_reason``.
Consequently, ``completion_tokens == requested_max_tokens`` is reported only as
a cap-hit proxy; it must not be described as an observed
``finish_reason=length`` event.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
PAPER = ROOT / "paper"

BASE_ORIGINAL = EXP / "glm4flash_repaired_code_20260909_envelope_test"
CAUSAL_ORIGINAL = EXP / "glm4flash_repaired_code_20260909_causal"
BASE_RECOVERY = EXP / "glm4flash_repaired_matrix_recovery_20260910"
CAUSAL_RECOVERY = EXP / "glm4flash_repaired_matrix_causal_recovery_20260910"
BASE_FINAL = EXP / "glm4flash_repaired_matrix_clean_20260910_envelope_test"
CAUSAL_FINAL = EXP / "glm4flash_repaired_matrix_final_20260910_causal"
REPORT_PATH = EXP / "glm4flash_failure_length_proxy_audit_20260911.json"
LATEX_PATH = PAPER / "generated_failure_length_proxy_rows.tex"
FAILURE_LATEX_PATH = PAPER / "generated_failure_status_rows.tex"

CELLS = (("code", "tight"), ("code", "loose"),
         ("math", "tight"), ("math", "loose"))
SEEDS = (0, 1, 2)
ARMS = ("arm1_assign", "arm2_samepolicy", "arm3_diffpolicy")
BASE_STRUCTURE = {"code": "direct", "math": "cot"}
DISPLAY_ARM = {
    "direct": "Direct",
    "cot": "CoT",
    "arm1_assign": "Assign",
    "arm2_samepolicy": "Same policy",
    "arm3_diffpolicy": "Different policy",
}


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def status_class(status: str) -> str:
    low = status.lower()
    if status in {"completed", "reserve_rejected"}:
        return status
    if "429" in low or "rate limit" in low or "'1302'" in low:
        return "provider_429"
    if "connection" in low:
        return "provider_connection"
    if "timeout" in low or "timed out" in low:
        return "provider_timeout"
    if status.startswith("budget_exceeded"):
        return "budget_exceeded"
    return "other_error"


def sorted_counts(counter: Counter) -> dict[str, int]:
    order = ("completed", "reserve_rejected", "provider_429",
             "provider_connection", "provider_timeout", "budget_exceeded",
             "other_error")
    return {key: counter[key] for key in order if counter[key]}


def baseline_original_statuses() -> tuple[list[dict], Counter]:
    groups, total = [], Counter()
    for family, tier in CELLS:
        arm = BASE_STRUCTURE[family]
        for seed in SEEDS:
            path = BASE_ORIGINAL / f"{arm}_{family}_{tier}" / f"results_seed{seed}.jsonl"
            rows = read_rows(path)
            counts = Counter(status_class(row.get("status", "")) for row in rows)
            total.update(counts)
            groups.append({
                "family": family,
                "tier": tier,
                "seed": seed,
                "arm": arm,
                "n_rows": len(rows),
                "status_counts": sorted_counts(counts),
                "source": str(path.relative_to(ROOT)),
            })
    return groups, total


def causal_original_statuses() -> tuple[list[dict], Counter, list[dict]]:
    groups, total, failures = [], Counter(), []
    for family, tier in CELLS:
        for arm in ARMS:
            for seed in SEEDS:
                path = CAUSAL_ORIGINAL / f"{arm}_{family}_{tier}" / f"results_seed{seed}.jsonl"
                rows = read_rows(path)
                counts = Counter(status_class(row.get("status", "")) for row in rows)
                total.update(counts)
                groups.append({
                    "family": family,
                    "tier": tier,
                    "seed": seed,
                    "arm": arm,
                    "n_rows": len(rows),
                    "status_counts": sorted_counts(counts),
                    "source": str(path.relative_to(ROOT)),
                })
                for row in rows:
                    cls = status_class(row.get("status", ""))
                    if cls.startswith("provider_") or cls == "other_error":
                        failures.append({
                            "family": family,
                            "tier": tier,
                            "seed": seed,
                            "arm": arm,
                            "task_id": row["task_id"],
                            "status_class": cls,
                            "status": row.get("status", ""),
                        })
    return groups, total, failures


def target_pairs(manifest: dict) -> set[tuple[str, str, int, str]]:
    pairs = set()
    for key, tasks in manifest["targets"].items():
        family, tier, seed_text = key.split("/")
        seed = int(seed_text.removeprefix("seed"))
        pairs.update((family, tier, seed, task_id) for task_id in tasks)
    return pairs


def recovery_statuses(root: Path) -> tuple[list[dict], Counter]:
    pattern = re.compile(
        r"^(baseline|arm1_assign|arm2_samepolicy|arm3_diffpolicy)_"
        r"(code|math)_(tight|loose)_seed([0-2])_pass(\d+)$"
    )
    groups, total = [], Counter()
    for path in sorted((root / "runs").glob("*/results_seed*.jsonl")):
        match = pattern.match(path.parent.name)
        if not match:
            raise RuntimeError(f"unrecognized recovery directory: {path.parent.name}")
        kind, family, tier, seed_text, pass_text = match.groups()
        rows = read_rows(path)
        counts = Counter(status_class(row.get("status", "")) for row in rows)
        total.update(counts)
        groups.append({
            "family": family,
            "tier": tier,
            "seed": int(seed_text),
            "kind": kind,
            "pass": int(pass_text),
            "n_rows": len(rows),
            "status_counts": sorted_counts(counts),
            "source": str(path.relative_to(ROOT)),
        })
    return groups, total


def contains_key(value, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(contains_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(contains_key(item, key) for item in value)
    return False


def completed_calls(row: dict, *, family: str, arm: str) -> list[dict]:
    """Recover completed calls from cumulative trace budgets.

    The four audited workflows have at most one provider call per trace entry.
    The requested cap is 1,024 only for a code direct/same-policy generation;
    every other provider call in these workflows requests 1,536 tokens.
    """
    calls = []
    previous_calls = 0
    previous_out = 0
    for index, trace in enumerate(row.get("trace", [])):
        budget = trace.get("budget", {})
        cumulative_calls = int(budget.get("llm_calls", previous_calls))
        cumulative_out = int(budget.get("out_tokens", previous_out))
        delta_calls = cumulative_calls - previous_calls
        delta_out = cumulative_out - previous_out
        if delta_calls not in (0, 1):
            raise AssertionError(
                f"expected <=1 call per trace entry, got {delta_calls}: "
                f"{family}/{arm}/{row.get('task_id')}/trace{index}"
            )
        if delta_calls == 1:
            trace_type = trace.get("type")
            max_tokens = (
                1024
                if family == "code" and arm in {"direct", "arm2_samepolicy"}
                and trace_type == "generate"
                else 1536
            )
            if delta_out > max_tokens:
                raise AssertionError(
                    f"observed output {delta_out} exceeds requested cap {max_tokens}: "
                    f"{family}/{arm}/{row.get('task_id')}/trace{index}"
                )
            calls.append({
                "trace_index": index,
                "node_type": trace_type,
                "completion_tokens": delta_out,
                "requested_max_tokens": max_tokens,
                "cap_hit": delta_out == max_tokens,
            })
        previous_calls = cumulative_calls
        previous_out = cumulative_out
    return calls


def empty_proxy_counter() -> dict[str, int]:
    return {"rows": 0, "provider_calls": 0, "cap_hit_calls": 0,
            "rows_with_any_cap_hit": 0}


def add_proxy(counter: dict[str, int], row: dict, *, family: str, arm: str) -> None:
    calls = completed_calls(row, family=family, arm=arm)
    counter["rows"] += 1
    counter["provider_calls"] += len(calls)
    counter["cap_hit_calls"] += sum(call["cap_hit"] for call in calls)
    counter["rows_with_any_cap_hit"] += int(any(call["cap_hit"] for call in calls))


def rate_record(key: tuple, counter: dict[str, int], key_names: tuple[str, ...]) -> dict:
    result = dict(zip(key_names, key))
    result.update(counter)
    result["cap_hit_call_rate"] = (
        counter["cap_hit_calls"] / counter["provider_calls"]
        if counter["provider_calls"] else None
    )
    result["row_any_cap_hit_rate"] = (
        counter["rows_with_any_cap_hit"] / counter["rows"]
        if counter["rows"] else None
    )
    return result


def proxy_audit(base_targets: set, causal_targets: set) -> dict:
    base_seed = defaultdict(empty_proxy_counter)
    base_aggregate = defaultdict(empty_proxy_counter)
    causal_seed = defaultdict(empty_proxy_counter)
    causal_aggregate_detailed = defaultdict(empty_proxy_counter)
    causal_aggregate_binary = defaultdict(empty_proxy_counter)
    checked_files: set[Path] = set()
    checked_rows = 0
    finish_reason_rows = 0

    for family, tier in CELLS:
        arm = BASE_STRUCTURE[family]
        for seed in SEEDS:
            path = BASE_FINAL / f"{arm}_{family}_{tier}" / f"results_seed{seed}.jsonl"
            checked_files.add(path)
            for row in read_rows(path):
                checked_rows += 1
                finish_reason_rows += int(contains_key(row, "finish_reason"))
                pair = (family, tier, seed, row["task_id"])
                source = "baseline_recovery" if pair in base_targets else "untouched"
                add_proxy(base_seed[(family, tier, arm, seed, source)], row,
                          family=family, arm=arm)
                binary = "recovered" if source != "untouched" else "untouched"
                add_proxy(base_aggregate[(family, tier, arm, binary)], row,
                          family=family, arm=arm)

    for family, tier in CELLS:
        for arm in ARMS:
            for seed in SEEDS:
                path = CAUSAL_FINAL / f"{arm}_{family}_{tier}" / f"results_seed{seed}.jsonl"
                checked_files.add(path)
                for row in read_rows(path):
                    checked_rows += 1
                    finish_reason_rows += int(contains_key(row, "finish_reason"))
                    pair = (family, tier, seed, row["task_id"])
                    if pair in base_targets:
                        source = "baseline_recovery"
                    elif pair in causal_targets:
                        source = "causal_recovery"
                    else:
                        source = "untouched"
                    add_proxy(causal_seed[(family, tier, arm, seed, source)], row,
                              family=family, arm=arm)
                    add_proxy(causal_aggregate_detailed[(family, tier, arm, source)], row,
                              family=family, arm=arm)
                    binary = "recovered" if source != "untouched" else "untouched"
                    add_proxy(causal_aggregate_binary[(family, tier, arm, binary)], row,
                              family=family, arm=arm)

    base_seed_rows = [
        rate_record(key, value, ("family", "tier", "arm", "seed", "source"))
        for key, value in sorted(base_seed.items())
    ]
    base_aggregate_rows = [
        rate_record(key, value, ("family", "tier", "arm", "source"))
        for key, value in sorted(base_aggregate.items())
    ]
    causal_seed_rows = [
        rate_record(key, value, ("family", "tier", "arm", "seed", "source"))
        for key, value in sorted(causal_seed.items())
    ]
    causal_detailed_rows = [
        rate_record(key, value, ("family", "tier", "arm", "source"))
        for key, value in sorted(causal_aggregate_detailed.items())
    ]
    causal_binary_rows = [
        rate_record(key, value, ("family", "tier", "arm", "source"))
        for key, value in sorted(causal_aggregate_binary.items())
    ]
    return {
        "finish_reason_field_check": {
            "checked_files": len(checked_files),
            "checked_rows": checked_rows,
            "rows_containing_finish_reason_at_any_depth": finish_reason_rows,
            "finish_reason_logged": finish_reason_rows > 0,
        },
        "baseline": {
            "by_seed_and_source": base_seed_rows,
            "aggregate_over_seeds_recovered_vs_untouched": base_aggregate_rows,
        },
        "causal": {
            "by_seed_and_source": causal_seed_rows,
            "aggregate_over_seeds_by_recovery_source": causal_detailed_rows,
            "aggregate_over_seeds_recovered_vs_untouched": causal_binary_rows,
        },
    }


def build_report() -> dict:
    baseline_groups, baseline_total = baseline_original_statuses()
    causal_groups, causal_total, causal_failures = causal_original_statuses()
    baseline_manifest_path = BASE_RECOVERY / "manifest.json"
    causal_manifest_path = CAUSAL_RECOVERY / "manifest.json"
    baseline_manifest = json.loads(baseline_manifest_path.read_text())
    causal_manifest = json.loads(causal_manifest_path.read_text())
    baseline_targets = target_pairs(baseline_manifest)
    causal_targets = target_pairs(causal_manifest)
    base_recovery_groups, base_recovery_total = recovery_statuses(BASE_RECOVERY)
    causal_recovery_groups, causal_recovery_total = recovery_statuses(CAUSAL_RECOVERY)
    proxy = proxy_audit(baseline_targets, causal_targets)

    baseline_failed = sum(
        count for key, count in baseline_total.items()
        if key not in {"completed", "reserve_rejected"}
    )
    causal_failed = sum(
        count for key, count in causal_total.items()
        if key not in {"completed", "reserve_rejected"}
    )
    if baseline_failed != 183:
        raise AssertionError(f"expected 183 baseline failures, found {baseline_failed}")
    if sum(baseline_total.values()) != 1800:
        raise AssertionError("expected 1,800 original baseline rows")
    if baseline_total["provider_429"] != 180:
        raise AssertionError("expected 180 baseline 429 failures")
    if causal_failed != 9:
        raise AssertionError(f"expected 9 original causal failures, found {causal_failed}")
    if sum(causal_total.values()) != 5400:
        raise AssertionError("expected 5,400 original causal rows")
    if len(baseline_targets) != 183 or len(causal_targets) != 8:
        raise AssertionError("frozen recovery target counts changed")
    if proxy["finish_reason_field_check"]["finish_reason_logged"]:
        raise AssertionError("finish_reason unexpectedly appeared; use it instead of this proxy")
    if proxy["finish_reason_field_check"]["checked_rows"] != 7200:
        raise AssertionError("expected 7,200 rows in the materialized matrix")
    if sum(base_recovery_total.values()) != 732:
        raise AssertionError("expected 732 baseline-recovery rows")
    if sum(causal_recovery_total.values()) != 24:
        raise AssertionError("expected 24 causal-only recovery rows")

    causal_failure_pairs = {
        (row["family"], row["tier"], row["seed"], row["task_id"])
        for row in causal_failures
    }
    overlapping_failure_rows = sum(
        (row["family"], row["tier"], row["seed"], row["task_id"])
        in baseline_targets
        for row in causal_failures
    )

    return {
        "artifact": "GLM repaired-matrix provider-failure and output-cap-hit audit",
        "definitions_and_caveats": {
            "finish_reason_observed": False,
            "cap_hit_proxy": "For each completed provider call reconstructed from consecutive cumulative trace budgets: completion_tokens == requested_max_tokens.",
            "non_equivalence_warning": "A cap hit is only a post-hoc proxy. The provider finish_reason was not logged, so these counts must not be presented as observed finish_reason=length events.",
            "call_denominator": "Completed provider calls, including calls preceding a later reserve_rejected control-flow stop; reserve rejections themselves are not calls.",
            "row_denominator": "Materialized task-seed workflow rows; rows_with_any_cap_hit counts a row once even if multiple calls hit their caps.",
            "recovered_definition": "A final row overlaid from either frozen baseline recovery or frozen causal-only recovery; untouched means copied from the original run.",
        },
        "source_manifests": {
            "baseline_recovery": {
                "path": str(baseline_manifest_path.relative_to(ROOT)),
                "sha256": sha256(baseline_manifest_path),
                "n_target_pairs": len(baseline_targets),
            },
            "causal_only_recovery": {
                "path": str(causal_manifest_path.relative_to(ROOT)),
                "sha256": sha256(causal_manifest_path),
                "n_target_pairs": len(causal_targets),
            },
        },
        "original_baseline_status_audit": {
            "groups": baseline_groups,
            "total": sorted_counts(baseline_total),
            "provider_failure_rows": baseline_failed,
        },
        "original_causal_status_audit": {
            "groups": causal_groups,
            "total": sorted_counts(causal_total),
            "provider_failure_rows": causal_failed,
            "unique_provider_failure_pairs": len(causal_failure_pairs),
            "failure_rows_also_in_baseline_recovery_targets": overlapping_failure_rows,
            "causal_only_unique_failed_pairs": len(causal_targets),
            "provider_failure_details": causal_failures,
        },
        "recovery_execution_status_audit": {
            "baseline_failure_recovery": {
                "groups": base_recovery_groups,
                "total": sorted_counts(base_recovery_total),
            },
            "causal_only_recovery": {
                "groups": causal_recovery_groups,
                "total": sorted_counts(causal_recovery_total),
            },
        },
        "completed_call_cap_hit_proxy": proxy,
        "interpretation": {
            "service_failure_clustering": "The original unusable rows are overwhelmingly 429 rate-limit failures and are concentrated by cell and seed, consistent with temporally clustered provider availability/throttling rather than outcome-level missingness.",
            "length_vs_recovery": "The cap-hit proxy has no consistent direction across cells or arms. It cannot establish that provider length termination was or was not associated with recovery timing or serving load.",
            "required_future_logging": [
                "provider finish_reason",
                "run/cell/task/seed/arm/node/call_index",
                "requested max_tokens",
                "provider timestamp and request identifier",
            ],
        },
    }


def percent(hit: int, total: int) -> str:
    return f"{100 * hit / total:.1f}\\%" if total else "--"


def fraction(record: dict | None) -> str:
    if record is None or not record["provider_calls"]:
        return "--"
    hit, total = record["cap_hit_calls"], record["provider_calls"]
    return f"{hit}/{total} ({percent(hit, total)})"


def render_latex(report: dict) -> str:
    base = report["completed_call_cap_hit_proxy"]["baseline"][
        "aggregate_over_seeds_recovered_vs_untouched"
    ]
    causal = report["completed_call_cap_hit_proxy"]["causal"][
        "aggregate_over_seeds_recovered_vs_untouched"
    ]
    lookup = {
        (row["family"], row["tier"], row["arm"], row["source"]): row
        for row in base + causal
    }
    lines = [
        "% AUTO-GENERATED by scripts/audit_recovery_failures_and_cap_hits.py.",
        "% IMPORTANT: provider finish_reason was not logged. Cap hit means only",
        "% completion_tokens == requested_max_tokens for a completed call; it is",
        "% not an observed finish_reason=length event.",
        ("\\multicolumn{5}{p{0.96\\linewidth}}{\\footnotesize\\emph{Caveat:} "
         "Provider finish reasons were not logged. ``Cap hit'' is only the "
         "post-hoc equality of completion tokens and the requested maximum; "
         "it is not an observed length termination.} \\\\"),
    ]
    for stage, family, tier, arm in [
        ("Baseline", "code", "tight", "direct"),
        ("Baseline", "code", "loose", "direct"),
        ("Baseline", "math", "tight", "cot"),
        ("Baseline", "math", "loose", "cot"),
        ("Causal", "code", "tight", "arm1_assign"),
        ("Causal", "code", "tight", "arm2_samepolicy"),
        ("Causal", "code", "tight", "arm3_diffpolicy"),
        ("Causal", "code", "loose", "arm1_assign"),
        ("Causal", "code", "loose", "arm2_samepolicy"),
        ("Causal", "code", "loose", "arm3_diffpolicy"),
        ("Causal", "math", "tight", "arm1_assign"),
        ("Causal", "math", "tight", "arm2_samepolicy"),
        ("Causal", "math", "tight", "arm3_diffpolicy"),
        ("Causal", "math", "loose", "arm1_assign"),
        ("Causal", "math", "loose", "arm2_samepolicy"),
        ("Causal", "math", "loose", "arm3_diffpolicy"),
    ]:
        recovered = lookup.get((family, tier, arm, "recovered"))
        untouched = lookup.get((family, tier, arm, "untouched"))
        lines.append(
            f"{stage} & {family}/{tier} & {DISPLAY_ARM[arm]} & "
            f"{fraction(recovered)} & {fraction(untouched)} \\\\"
        )
    lines = [line if line.startswith("%") else line.rstrip("\\") + r"\\"
             for line in lines if not line.startswith("\\multicolumn")]
    lines[-1] = lines[-1].rstrip("\\").rstrip()
    return "\n".join(lines) + "\n"


def failure_fields(counts: dict[str, int]) -> tuple[int, int, int, int]:
    rate_limit = counts.get("provider_429", 0)
    connection = counts.get("provider_connection", 0)
    timeout = counts.get("provider_timeout", 0)
    provider_failures = rate_limit + connection + timeout
    return provider_failures, rate_limit, connection, timeout


def render_failure_latex(report: dict) -> str:
    """Render rows for columns: stage, cell, seed/arm, n, fail, 429, conn, timeout."""
    lines = [
        "% AUTO-GENERATED by scripts/audit_recovery_failures_and_cap_hits.py.",
        ("\\multicolumn{8}{p{0.96\\linewidth}}{\\footnotesize "
         "Provider failures are unusable status-level rows. "
         "\\texttt{reserve\\_rejected} is intended budget control flow and "
         "is not counted as a provider failure.} \\\\"),
    ]
    baseline_groups = report["original_baseline_status_audit"]["groups"]
    for row in baseline_groups:
        failures, rate_limit, connection, timeout = failure_fields(
            row["status_counts"]
        )
        lines.append(
            f"Baseline & {row['family']}/{row['tier']} & seed {row['seed']} & "
            f"{row['n_rows']} & {failures} & {rate_limit} & {connection} & "
            f"{timeout} \\\\"
        )

    causal_aggregate: dict[tuple[str, str, str], dict] = {}
    for row in report["original_causal_status_audit"]["groups"]:
        key = (row["family"], row["tier"], row["arm"])
        item = causal_aggregate.setdefault(key, {"n_rows": 0, "counts": Counter()})
        item["n_rows"] += row["n_rows"]
        item["counts"].update(row["status_counts"])
    for family, tier in CELLS:
        for arm in ARMS:
            item = causal_aggregate[(family, tier, arm)]
            failures, rate_limit, connection, timeout = failure_fields(item["counts"])
            lines.append(
                f"Causal & {family}/{tier} & {DISPLAY_ARM[arm]} & "
                f"{item['n_rows']} & {failures} & {rate_limit} & {connection} & "
                f"{timeout} \\\\"
            )
    lines = [line if line.startswith("%") else line.rstrip("\\") + r"\\"
             for line in lines if not line.startswith("\\multicolumn")]
    lines[-1] = lines[-1].rstrip("\\").rstrip()
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true",
        help="Recompute and fail if checked-in JSON/LaTeX artifacts differ.",
    )
    args = parser.parse_args()
    report = build_report()
    report_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    latex_text = render_latex(report)
    failure_latex_text = render_failure_latex(report)
    if args.check:
        for path, expected in (
            (REPORT_PATH, report_text),
            (LATEX_PATH, latex_text),
            (FAILURE_LATEX_PATH, failure_latex_text),
        ):
            if not path.exists():
                raise SystemExit(f"missing generated artifact: {path}")
            if path.read_text() != expected:
                raise SystemExit(f"stale generated artifact: {path}")
        print("failure/cap-hit audit artifacts are current")
        return
    REPORT_PATH.write_text(report_text)
    LATEX_PATH.write_text(latex_text)
    FAILURE_LATEX_PATH.write_text(failure_latex_text)
    print(f"wrote {REPORT_PATH.relative_to(ROOT)}")
    print(f"wrote {LATEX_PATH.relative_to(ROOT)}")
    print(f"wrote {FAILURE_LATEX_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
