# Repair, Breakage, and Blocked Execution

Code accompanying the paper *Repair, Breakage, and Blocked Execution: A Path-Level
Regression Audit of Verify–Refine Workflows*. The repository intentionally
contains the reusable workflow implementation and analysis code only; paper
sources, figures, raw logs, frozen data files, and credentials are kept out of
the public code package.

## Layout

- `hbws/` — workflow DSL, static validator, reservation ledger, runner,
  verifiers, protocol, and search utilities.
- `scripts/` — the analyses that produce every generated table and macro in the
  paper, plus zero-cost reanalysis tools.
- `tests/` — offline regression tests over the analysis code.
- `PREREGISTRATION.md` — the Part-I/Part-II preregistration, frozen at the
  `prereg-freeze-partI` and `prereg-freeze-partII` tags.
- `data/SHA256SUMS` — checksum manifest binding the frozen splits.
- `LOCAL_QWEN_CONTROL.md` — hardware, serving configuration and commands for
  the local Qwen2.5-Coder-7B-Instruct deterministic-serving control.
- `requirements.txt` — pinned Python dependencies.
- `.env.example` — empty configuration template; no credentials are stored.

## Instrumentation

Each executed position is logged as one JSON row: the returned `solution`, the
external-grader `success`, and a `trace` of node records (`generate`, `verify`,
`refine`) carrying a cumulative budget snapshot and the remaining budget
fraction. Verifier verdicts are **not** stored as an explicit field; they are
reconstructed from the trace shape — a `verify` followed by a `refine` is a
rejection, a trajectory ending on `verify` is an acceptance. Execution is
serial per task and acceptance terminates the graph, so no accept-then-rewrite
interleaving is possible. Rejected intermediate candidates are not retained, so
candidate-level verifier fidelity is recoverable only where the incumbent's
correctness is known by construction (the assignment arm).

The reservation ledger refuses a call whose worst-case cost does not fit the
remaining per-task cap; a refused call appears as a terminated trace, not as an
error.

## Reproducing the paper's numbers

**The per-task execution records are not in this repository.** The analysis
entry points below read them from `experiments/`, so a fresh clone will not
reproduce the tables on its own. The anonymized supplementary package that
accompanies the paper ships derived records — every field the analyses use,
with the model outputs replaced by their SHA-256 digests — and those scripts do
run against it.

These scripts read existing per-task result rows and never issue provider
requests. With the records in place, run from the repository root:

```bash
python scripts/analyze_signal_paths.py
python scripts/analyze_blocked_paths.py
python scripts/analyze_provenance_followup.py && python scripts/render_followup_tables.py
python scripts/analyze_code_parity.py
python scripts/analyze_revert_control.py && python scripts/render_revert_tables.py
python scripts/analyze_clean_matrix_byte_identity.py
python scripts/analyze_provenance_content.py
python scripts/analyze_effect_precision.py && python scripts/render_effect_precision_tables.py
python scripts/audit_final_auxiliary_evidence.py
```

Earlier confirmatory and sensitivity entry points:

```bash
python scripts/audit_claims.py
python scripts/confirm_partI.py
python scripts/confirm_partII.py
python scripts/false_rejection.py
python scripts/best_of_3_zero_cost.py
python scripts/kimi_sensitivity.py
python scripts/provenance_causal_analysis.py
```

`scripts/audit_submission_ready.py` checks that the paper sources and the
generated fragments agree; `python -m pytest tests/` runs the offline tests.

The experiment runner and workflow library are separated from any credential
management; put local values in an untracked `.env` file when reproducing an
authorized run.

## Safety

Do not commit API keys, raw private logs, paper artifacts, or generated build
files. The repository is kept code-only so that the paper and its local
research data remain separate from the public code package.
