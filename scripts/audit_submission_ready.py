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
    )
    for name in generated:
        path = PAPER / name
        require(path.exists() and path.stat().st_size > 0,
                f"missing generated LaTeX: {name}")

    tex = (PAPER / "main.tex").read_text()
    before_bib = tex.split(r"\bibliography", 1)[0]
    revised_body = (PAPER / "revised_results_body.tex").read_text()
    require(r"\input{revised_results_body.tex}" in before_bib,
            "revised results body is not in the main paper")
    submission_body = before_bib + "\n" + revised_body
    require(r"\input{generated_clean_matrix_rows.tex}" in submission_body,
            "clean matrix is not in the main paper")
    require(r"\input{generated_refiner5_rows.tex}" in submission_body,
            "five-refiner evidence is not in the main paper")
    require(r"\input{evidence_map.tex}" in submission_body,
            "claim-evidence map is not in the main paper")
    require(r"\label{tab:parti}" not in submission_body,
            "historical GPT table remains in the main paper")
    require(r"\label{tab:sharedmain}" not in submission_body,
            "historical three-refiner table remains in the main paper")

    log = (PAPER / "main.log").read_text(errors="replace")
    require("undefined references" not in log.lower(), "undefined references")
    require("undefined citations" not in log.lower(), "undefined citations")
    require("overfull \\hbox" not in log.lower(), "overfull hbox remains")
    require((PAPER / "main.pdf").exists(), "main.pdf is missing")
    require((PAPER / "fig1_decomposition.pdf").exists(), "Figure 1 missing")
    require((PAPER / "fig2_risk_adoption.pdf").exists(), "Figure 2 missing")

    deck = (PAPER / "ppt_projects/figure2_risk_adoption_temporal_revision_20260909/"
            "exports/Figure2_Risk_Constrained_Adoption_TwoStage_Editable.pptx")
    require(deck.exists(), "editable Figure 2 deck missing")
    with zipfile.ZipFile(deck) as archive:
        require(archive.testzip() is None, "editable Figure 2 deck is corrupt")

    numbers = re.findall(r"\\newcommand\{\\([A-Za-z]+)\}",
                         (PAPER / "generated_main_numbers.tex").read_text())
    require(len(numbers) == len(set(numbers)), "duplicate generated macros")
    print("submission readiness audit passed")


if __name__ == "__main__":
    main()
