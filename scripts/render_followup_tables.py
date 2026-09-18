#!/usr/bin/env python3
"""Render LaTeX fragments from the provenance follow-up recomputation."""
from __future__ import annotations
import json
import re as _re  # _tighten
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REP = json.loads((ROOT / "experiments/provenance_followup_20260917_analysis.json").read_text())
PAPER = ROOT / "paper"


def d3(v):
    s = f"{abs(v):.3f}"
    return ("+" if v >= 0 else "-") + (s[1:] if s.startswith("0") else s)


def ci(e):
    lo, hi = e["ci95"]
    return f"${d3(e['point'])}$ [{d3(lo)}, {d3(hi)}]"


a = REP["part_a_serving_regime"]
c = a["conditions"]
LABEL = {"S2_vs_S": r"repeat serial (S2)", "B_vs_S_all": r"batched, all rows",
         "B_vs_S_excl_recovered": r"batched, excl.\ recovered",
         "B_vs_S_excl_never_concurrent": r"batched, excl.\ non-overlapping",
         "P_vs_S": r"prefix cache"}
rows = []
for k in ("S2_vs_S", "B_vs_S_all", "B_vs_S_excl_recovered",
          "B_vs_S_excl_never_concurrent", "P_vs_S"):
    v = c[k]
    lo, hi = v["byte_identity_ci95"]
    rows.append(
        f"{LABEL[k]} & {v['n_pairs']} & {v['identical_request_bodies']} & "
        f"{100*v['byte_identity_rate']:.1f}\\% [{100*lo:.1f}, {100*hi:.1f}] & "
        f"{100*v['workflow_outcome_agreement']:.1f}\\% & "
        f"{v['flips_correct_to_wrong']}\\,/\\,{v['flips_wrong_to_correct']} \\\\")
(PAPER / "generated_serving_regime_rows.tex").write_text(
    "\n".join(rows + [r"\bottomrule"]) + "\n")

b = REP["part_b_budget_parity"]
rows = []
for tier in ("tight", "loose"):
    t = b[tier]
    p = t["paired_differences_assign_minus_same_policy"]
    rows.append(
        f"prior-\\texttt{{{tier}}} & {t['assign']['n']} & "
        f"{t['assign']['accepted_wrong'] + t['assign']['rejected_wrong']}/"
        f"{t['same_policy']['accepted_wrong'] + t['same_policy']['rejected_wrong']} & "
        f"{t['assign']['accepted_wrong']}/{t['same_policy']['accepted_wrong']} & "
        f"{ci(p['breakage_rate'])} & "
        f"{ci(p['accuracy'])} \\\\")
(PAPER / "generated_budget_parity_rows.tex").write_text(
    _re.sub(r"\[(\S+?), (\S+?)\]", r"[\1,\2]",
            "\n".join(rows + [r"\bottomrule"])) + "\n")

rows = []
for tier in ("tight", "loose"):
    t = b[tier]
    w = t["positions_with_wrong_regenerated_incumbent"]
    for key, lbl in (("same_policy_incumbent_correct", "incumbent correct"),
                     ("same_policy_incumbent_wrong", "incumbent wrong")):
        v = t[key]
        pr = v["post_rejection_failure_rate"]
        rows.append(
            f"prior-\\texttt{{{tier}}} & {lbl} & {v['n']} & {v['accepted']} & "
            f"{v['accepted_wrong']} & {v['rejected']} & {v['rejected_wrong']} & "
            + ("--" if pr is None else f"{100*pr:.1f}\\%") + r" \\")
(PAPER / "generated_stratification_rows.tex").write_text(
    "\n".join(rows + [r"\bottomrule"]) + "\n")

rows = []
for tier in ("tight", "loose"):
    for arm, lbl in (("assign", "assignment"), ("same_policy", "same-policy")):
        co = b[tier]["cost"][arm]
        rows.append(
            f"prior-\\texttt{{{tier}}} & {lbl} & {co['drafting']['calls']} & "
            f"{co['drafting']['out']:,} & {co['suffix']['calls']:,} & "
            f"{co['suffix']['out']:,} \\\\")
(PAPER / "generated_parity_cost_rows.tex").write_text(
    "\n".join(rows + [r"\bottomrule"]) + "\n")

bt, bl = b["tight"], b["loose"]
wt, wl = (bt["positions_with_wrong_regenerated_incumbent"],
          bl["positions_with_wrong_regenerated_incumbent"])
m = [
    rf"\newcommand{{\BatchIdentity}}{{{100*c['B_vs_S_all']['byte_identity_rate']:.1f}\%}}",
    rf"\newcommand{{\PrefixIdentity}}{{{100*c['P_vs_S']['byte_identity_rate']:.1f}\%}}",
    rf"\newcommand{{\BatchFlipShare}}{{{100*(1-c['B_vs_S_all']['workflow_outcome_agreement']):.1f}\%}}",
    rf"\newcommand{{\BatchNetPP}}{{{100*c['B_vs_S_all']['net_outcome_change']/c['B_vs_S_all']['n_pairs']:.2f}}}",
    rf"\newcommand{{\WrongIncTight}}{{{wt['n']}}}",
    rf"\newcommand{{\WrongIncLoose}}{{{wl['n']}}}",
    rf"\newcommand{{\WrongIncAssignTight}}{{{wt['assignment_final_correct_same_positions']}}}",
    rf"\newcommand{{\WrongIncAssignLoose}}{{{wl['assignment_final_correct_same_positions']}}}",
    rf"\newcommand{{\WrongIncSameTight}}{{{wt['same_policy_final_correct']}}}",
    rf"\newcommand{{\ParityRepairTight}}{{{ci(b['tight']['paired_differences_assign_minus_same_policy']['repair_rate'])}}}",
    rf"\newcommand{{\ParityRepairLoose}}{{{ci(b['loose']['paired_differences_assign_minus_same_policy']['repair_rate'])}}}",
    rf"\newcommand{{\WrongIncSameLoose}}{{{wl['same_policy_final_correct']}}}",
]
(PAPER / "generated_followup_macros.tex").write_text("\n".join(m) + "\n")
print("wrote follow-up fragments")


def split_clean_matrix() -> None:
    """Main-text table shows the uncensored cells; math/tight moves to the
    appendix, where its blocked-path counts are discussed."""
    src = (PAPER / "generated_clean_matrix_rows.tex").read_text().splitlines()
    main, censored, in_censored = [], [], False
    for line in src:
        if line.startswith("math/tight"):
            in_censored = True
        elif line.startswith("math/loose"):
            in_censored = False
        if line.strip() == r"\bottomrule":
            continue
        (censored if in_censored else main).append(line)
    while main and main[-1].startswith(r"\addlinespace"):
        main.pop()
    while censored and censored[-1].startswith(r"\addlinespace"):
        censored.pop()
    (PAPER / "generated_clean_matrix_main_rows.tex").write_text(
        "\n".join(main + [r"\bottomrule"]) + "\n")
    (PAPER / "generated_clean_matrix_censored_rows.tex").write_text(
        "\n".join(censored + [r"\bottomrule"]) + "\n")
    print("split clean matrix:", len(main), "main rows,", len(censored), "censored")
