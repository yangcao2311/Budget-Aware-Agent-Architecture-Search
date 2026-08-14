# Repair, Breakage, and Reference Preservation

Code accompanying the paper *Repair, Breakage, and Reference Preservation in
Verify--Refine Workflows*. The repository intentionally contains the reusable
workflow implementation and analysis code only; paper sources, figures, raw
logs, frozen data, and credentials are kept out of the public code package.

## Layout

- `hbws/` — workflow DSL, static validator, reservation ledger, runner,
  verifiers, protocol, and search utilities.
- `scripts/` — paper analyses and zero-cost reanalysis tools, including the
  Kimi effective-$n$/CI and failure-as-wrong sensitivity analysis.
- `requirements.txt` — pinned Python dependencies.
- `.env.example` — empty configuration template; no credentials are stored.

## Local analysis entry points

Run these commands from the repository root after installing the dependencies
and providing the private experiment logs locally:

```bash
python scripts/audit_claims.py
python scripts/confirm_partI.py
python scripts/confirm_partII.py
python scripts/false_rejection.py
python scripts/best_of_3_zero_cost.py
python scripts/kimi_sensitivity.py
python scripts/provenance_causal_analysis.py
```

The analysis scripts are offline: they read existing per-task results and do
not issue model or provider requests. The experiment runner and workflow
library are likewise separated from any credential management; put local
values in an untracked `.env` file when reproducing an authorized run.

## Safety

Do not commit API keys, raw private logs, paper artifacts, or generated build
files. The repository is kept code-only so that the paper and its local
research data remain separate from the public code package.
