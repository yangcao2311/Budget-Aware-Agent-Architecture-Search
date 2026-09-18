"""The revert control must be a determined replay, not an estimate."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "experiments/revert_control_20260917.json"


def load():
    return json.load(open(REPORT))["cells"]


def test_replay_is_reproducible():
    """Rerunning the analysis must reproduce the committed report byte for byte."""
    before = REPORT.read_bytes()
    subprocess.run([sys.executable, "scripts/analyze_revert_control.py"],
                   cwd=ROOT, check=True, capture_output=True)
    assert REPORT.read_bytes() == before


def test_empty_traces_already_return_the_incumbent():
    for cell, d in load().items():
        assert d["empty_trace_matches_incumbent"], cell


def test_every_changed_return_is_a_forfeited_repair():
    """Assignment breakage is zero, so revert can only lose, never gain."""
    for cell, d in load().items():
        assert d["actual"]["breakage_count"] == 0, cell
        assert d["revert"]["breakage_count"] == 0, cell
        lost = d["actual"]["repair_count"] - d["revert"]["repair_count"]
        assert lost == d["returns_changed_by_revert"], cell
        assert d["revert"]["delta"] <= d["actual"]["delta"], cell


def test_pairs_and_denominators_are_complete():
    for cell, d in load().items():
        assert d["pairs"] == 450, cell
        assert d["reference_correct"] + d["reference_wrong"] == 450, cell
        assert d["accepted"] + d["no_acceptance"] == 450, cell


def test_macros_match_the_report():
    cells = load()
    txt = (ROOT / "paper/generated_revert_macros.tex").read_text()
    assert f"{{{sum(d['pairs'] for d in cells.values()):,}}}" in txt
    assert "{%d}" % sum(d["returns_changed_by_revert"] for d in cells.values()) in txt
