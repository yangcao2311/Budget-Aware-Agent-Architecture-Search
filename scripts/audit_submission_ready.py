#!/usr/bin/env python3
"""Fail closed if the revised submission is internally incomplete."""
from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
EXP = ROOT / "experiments"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    clean_path = EXP / "glm4flash_repaired_matrix_final_20260910_analysis.json"
    search_path = EXP / "assign_search_intervention_20260909/analysis.json"
    search_integrity_path = (EXP /
                             "assign_search_intervention_20260909/integrity_audit.json")
    refiner_path = EXP / "shared_refiner5_temporal_20260909/analysis_audited.json"
    for path in (clean_path, search_path, search_integrity_path, refiner_path):
        require(path.exists(), f"missing required analysis: {path}")

    clean = json.loads(clean_path.read_text())
    require(set(clean["cells"]) ==
            {"code/tight", "code/loose", "math/tight", "math/loose"},
            "clean matrix does not contain exactly four planned cells")
    for key, cell in clean["cells"].items():
        require(cell["n_tasks"] == 150 and cell["n_seeds"] == 3,
                f"wrong cluster dimensions in {key}")
        require(cell["reference"]["rows"] == 450,
                f"wrong reference row count in {key}")
        require(cell["executor_defects"]["reference"] == 0,
                f"reference defects remain in {key}")
        for arm, row in cell["arms"].items():
            require(row["effective_tasks"] == 150 and row["matched_pairs"] == 450,
                    f"incomplete matched population: {key}/{arm}")
            require(cell["executor_defects"][arm] == 0,
                    f"workflow defects remain: {key}/{arm}")
    require(sum(cell["arms"]["arm1_assign"]["acceptance_path_breakage_events"]
                for cell in clean["cells"].values()) == 0,
            "assignment acceptance-path contract violation")

    search = json.loads(search_path.read_text())
    require(search["n_candidates"] > 0, "assign-search archive is empty")
    search_integrity = json.loads(search_integrity_path.read_text())
    require(search_integrity["status"] == "passed" and
            search_integrity["provider_runtime_defects"] == 0,
            "assign-search integrity audit did not pass")
    refiners = json.loads(refiner_path.read_text())
    require(refiners["math_audit"]["status"] == "passed",
            "five-refiner mathematical audit did not pass")
    require(len(refiners["refiners"]) == 5, "five-refiner study incomplete")
    require(all(item["false_positive_count"] == 0
                for item in refiners["screen_decisions"]),
            "shared screen produced a direct-bound false positive")

    generated = (
        "generated_main_numbers.tex",
        "generated_clean_matrix_rows.tex",
        "generated_lambda_rows.tex",
        "generated_refiner5_rows.tex",
        "generated_clean_cost_rows.tex",
        "generated_clean_utility_rows.tex",
        "generated_precision_rows.tex",
        "generated_design_surface_rows.tex",
        "generated_precision_macros.tex",
        "generated_budget_parity_rows.tex",
        "generated_serving_regime_rows.tex",
        "generated_stratification_rows.tex",
        "generated_parity_cost_rows.tex",
        "generated_followup_macros.tex",
        "generated_clean_matrix_main_rows.tex",
        "generated_clean_matrix_censored_rows.tex",
        "generated_signal_path_rows.tex",
        "generated_signal_macros.tex",
        "generated_blocked_rows.tex",
        "generated_blocked_macros.tex",
    )
    for name in generated:
        path = PAPER / name
        require(path.exists() and path.stat().st_size > 0,
                f"missing generated LaTeX: {name}")

    tex = (PAPER / "main.tex").read_text().split(r"\end{document}", 1)[0]
    lean_appendix = (PAPER / "lean_appendix.tex").read_text()
    before_bib = tex.split(r"\bibliography", 1)[0]
    revised_body = (PAPER / "revised_results_body.tex").read_text()
    five_refiner_appendix = (PAPER / "five_refiner_appendix.tex").read_text()
    qwen_appendix = (PAPER / "qwen_server_appendix.tex").read_text()
    require(r"\input{revised_results_body.tex}" in before_bib,
            "revised results body is not in the main paper")
    restored_appendix = (PAPER / "restored_appendix.tex").read_text()
    evidence_map = (PAPER / "evidence_map.tex").read_text()
    followup_appendix = (PAPER / "followup_appendix.tex").read_text()
    submission_body = (before_bib + "\n" + revised_body + "\n" +
                       five_refiner_appendix + "\n" + qwen_appendix + "\n" +
                       lean_appendix + "\n" + restored_appendix + "\n" +
                       evidence_map + "\n" + followup_appendix)
    require(r"\input{generated_clean_matrix_main_rows.tex}" in revised_body,
            "clean matrix is not in the main paper")
    require(r"\input{generated_clean_matrix_censored_rows.tex}" in lean_appendix,
            "censored math/tight cell is not in the appendix")
    require(r"\input{generated_refiner5_rows.tex}" in submission_body,
            "five-refiner evidence is not in the compiled submission")
    require(r"\input{lean_appendix.tex}" in tex,
            "lean appendix is not included in the active submission")
    require(r"\input{five_refiner_appendix.tex}" in lean_appendix,
            "five-refiner appendix is not included")
    require(r"\input{qwen_server_appendix.tex}" in lean_appendix,
            "Qwen boundary-control appendix is not included")
    require(r"\input{evidence_map.tex}" in lean_appendix,
            "claim-evidence map is not in the compiled submission")
    require(r"\input{restored_appendix.tex}" in lean_appendix,
            "restored GPT-4o/Kimi appendix is not included")
    # The GPT-4o confirmatory table was reinstated in the main text on
    # 2026-09-16: it carries the paper's largest measured effects.
    require(r"\label{tab:parti}" in revised_body,
            "GPT-4o confirmatory table is missing from the main paper")
    require(r"\label{app:kimi}" in restored_appendix,
            "Kimi K3 replication is missing from the appendix")
    require(r"\label{tab:precision}" in restored_appendix,
            "effect-resolution table is missing from the appendix")
    # 2026-09-17 follow-ups: the budget-parity contrast is a main-text result,
    # the serving-regime intervention motivates the provenance arm in setup.
    require(r"\label{tab:parity}" in revised_body,
            "budget-parity table is missing from the main paper")
    require(r"\label{sec:parity}" in revised_body,
            "budget-parity section is missing from the main paper")
    require(r"\label{tab:blocked}" in followup_appendix,
            "blocked terminal-path table is missing from the appendix")
    require(r"\label{tab:signalpaths}" in followup_appendix,
            "verifier-signal path decomposition is missing from the appendix")
    require(r"\label{tab:revert}" in followup_appendix,
            "best-so-far retention control is missing from the appendix")
    require(r"\label{app:revert}" in followup_appendix,
            "revert-control appendix label is missing")
    # the limitations must report the control, not concede it as untested.
    require("Best-so-far retention adds nothing" in revised_body,
            "limitations still concede best-so-far retention as untested")
    require("do not compare assignment with best-so-far" not in revised_body,
            "retired best-so-far concession is still present")
    require(r"K_{\rm pre}" in before_bib,
            "the three-way breakage decomposition is missing from the theory")
    require(r"\label{tab:serving}" in followup_appendix,
            "serving-regime table is missing from the appendix")
    require(r"\label{tab:stratification}" in followup_appendix,
            "first-incumbent stratification is missing from the appendix")
    require(r"\input{followup_appendix.tex}" in lean_appendix,
            "follow-up appendix is not included")
    require("post-hoc" in revised_body and "post-hoc" in followup_appendix,
            "follow-up results must be labelled post-hoc")
    require("prior-\\texttt{tight}" in revised_body,
            "budget-parity cohorts must be named prior-tight/prior-loose")
    for stale in ("Every same-policy contrast above includes zero",
                  "A byte-stable serving regime removes the distinction",
                  "isolates no serving cause"):
        require(stale not in submission_body,
                f"superseded claim still present: {stale!r}")
    require(r"\label{tab:design-surface}" in restored_appendix,
            "detectability surface is missing from the appendix")
    require("observed-power" in restored_appendix,
            "detectability appendix must disclaim observed power")
    require(r"\label{tab:sharedmain}" not in submission_body,
            "historical three-refiner table remains in the main paper")

    log = (PAPER / "main.log").read_text(errors="replace")
    require("undefined references" not in log.lower(), "undefined references")
    require("undefined citations" not in log.lower(), "undefined citations")
    require("overfull \\hbox" not in log.lower(), "overfull hbox remains")
    require((PAPER / "main.pdf").exists(), "main.pdf is missing")
    aux = (PAPER / "main.aux").read_text()
    main_end = re.search(r"\\newlabel\{sec:main-end\}\{\{[^}]*\}\{(\d+)\}", aux)
    require(main_end is not None, "main-text page marker is missing")
    require(int(main_end.group(1)) <= 9, "main text exceeds nine pages")
    require((PAPER / "fig1_decomposition.pdf").exists(), "Figure 1 missing")
    for name in ("fig2_risk_adoption.pdf", "fig3_budget_floor.pdf",
                 "fig4_dose_response.pdf", "fig5_regimes.pdf"):
        require((PAPER / name).exists(), f"{name} missing from the active paper")

    deck = (PAPER / "ppt_projects/figure2_risk_adoption_simplified_20260911/"
            "exports/Figure2_Risk_Constrained_Adoption_ReviewerResolved_Editable.pptx")
    require(deck.exists(), "editable Figure 2 deck missing")
    with zipfile.ZipFile(deck) as archive:
        require(archive.testzip() is None, "editable Figure 2 deck is corrupt")

    numbers = re.findall(r"\\newcommand\{\\([A-Za-z]+)\}",
                         (PAPER / "generated_main_numbers.tex").read_text())
    require(len(numbers) == len(set(numbers)), "duplicate generated macros")
    print("submission readiness audit passed")


if __name__ == "__main__":
    main()
