# When Verification Helps: Repair, Breakage, and Hard Budgets in Agent Workflows

Anonymous submission. This repository reproduces every number, table and figure
in the paper from raw per-task logs.

## What is here

```
hbws/            library: DSL + validator, reservation ledger, graph runner,
                 verifiers, evaluation protocol, search
scripts/         one entry point per experiment or analysis (see below)
data/            frozen splits + SHA256SUMS (generated once, never regenerated)
experiments/     raw per-task results, budget ledgers, search registries
paper/           main.tex, figures, compiled PDF
PREREGISTRATION.md   claims, frozen before execution, with an append-only
                     deviation log recording every outcome including failures
```

## Quick start (no API key needed)

Everything below recomputes the paper from logs already in `experiments/`.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python scripts/audit_claims.py        # all 190 paper numbers vs raw logs
.venv/bin/python scripts/confirm_partI.py       # C1-C5 verdicts (C4 unsupported as stated)
.venv/bin/python scripts/confirm_partII.py      # P1-P5 verdicts (three unsupported)
.venv/bin/python scripts/false_rejection.py     # the bound b <= P(false reject), 10 conditions
.venv/bin/python scripts/regen_leak.py          # regeneration-vs-reuse leak, from stored seeds
.venv/bin/python scripts/score_ood_predictions.py   # the locked OOD prediction (it failed)
.venv/bin/python scripts/sensitivity_ledger.py  # robustness to the ledger defect
.venv/bin/python scripts/voi_model.py           # decomposition identity + transfer tests
.venv/bin/python scripts/make_figures.py        # regenerate all figures
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

`audit_claims.py` exits non-zero if any number in `paper/main.tex` disagrees
with the logs. It is the check to run first.

## Safety harness (no API key needed)

```bash
.venv/bin/python scripts/fuzz_safety.py 3000
```
Generates 3,000 random legal workflows against random budget profiles with a
mocked model and asserts no per-task cap is ever exceeded in any dimension.

## Reproducing from scratch (needs an API key)

Set the model endpoint in `.env` (see `.env.example`; the repository contains
no credentials). Costs are the logical figures from Appendix F.

```bash
.venv/bin/python -c "from hbws.data import prepare_all, prepare_logic; prepare_all(); prepare_logic()"
.venv/bin/python scripts/build_ood_visible_tests.py

# Part I: confirmatory structures on the frozen test split
for s in 0 1 2; do
  .venv/bin/python scripts/run_envelope.py --split test --n 150 --seed $s \
     --families code --structures direct verify_refine_3 incumbent_refine \
     --tiers tight unseen loose
  .venv/bin/python scripts/run_envelope.py --split test --n 150 --seed $s \
     --families math --structures cot verify_refine_3 incumbent_refine_cot \
     --tiers tight unseen loose
done

# verifier degradation (Figure 2, left)
for s in 0 1 2; do for m in 0.5 0.0; do
  .venv/bin/python scripts/run_envelope.py --split test --n 150 --seed $s \
     --families code --structures direct verify_refine_3 --tiers loose --mask-tests $m
done; done

# Part II: Protocol A/B
.venv/bin/python scripts/run_protocol.py --family code --seed 0 --cap 60
.venv/bin/python scripts/eval_policy.py --search-run A_hbws_code_s0 --family code \
   --splits test --tiers tight unseen loose

# external validity + the prospective third domain
.venv/bin/python scripts/run_envelope.py --split ood --n 100 --seed 0 ...
.venv/bin/python scripts/run_envelope.py --split logic_prospective --families logic --n 120 --seed 0 ...

# apply the corrected math grading to stored responses (no inference)
.venv/bin/python scripts/apply_regrade.py
```

Note that `run_envelope.py` enables the response cache. Runs that must not share
first drafts across arms have to disable it; see Appendix G of the paper.

## Data

| split | source | n | role |
|---|---|---|---|
| `code_{dev,val,test}` | MBPP+ | 120/40/150 | search / selection / confirmation |
| `code_ood` | HumanEval+ | 100 | out-of-domain (no visible tests) |
| `code_ood_visible` | HumanEval+ | 68 | out-of-domain with tests recovered from docstrings |
| `math_{dev,val,test}` | MATH L4–5, 4 subjects | 120/40/150 | search / selection / confirmation |
| `math_ood` | MATH, 2 held-out subjects | 100 | out-of-domain |
| `logic_prospective` | BIG-Bench Hard, 4 subtasks | 120 | prospective validation, never used elsewhere |

Splits were generated once with seed `20260804` and are pinned by
`data/SHA256SUMS`. `hbws/data.py` refuses to overwrite an existing split.

## Preregistration

`PREREGISTRATION.md` contains the claims and the append-only outcome log. The
commits that froze each set of claims are tagged:

| tag | what it froze |
|---|---|
| `prereg-freeze-partI` | C1–C5, before any frozen test execution |
| `prereg-freeze-partII` | P1–P5, before any searched policy saw val/test |
| `ood-predictions-locked` | the out-of-domain transfer prediction (which failed) |
| `prospective-locked` | PV1–PV4 on a third domain, before any execution there |

Outcomes are recorded verbatim, including one refuted Part-I claim, three
unsupported Part-II claims, and the failed transfer prediction with the
design confound that caused it.

## Known defects, and where they are quantified

- **A preregistered claim does not hold.** C4 asserted that adding a verifier
  gate eliminates breakage. Auditing the implementation showed both compared
  workflows already had that gate --- their control flow is identical edge for
  edge --- so the comparison never manipulated the registered variable. The
  paper marks C4 unsupported as preregistered and states the
  reference-preservation account separately, labelled post hoc.
- **Response caching was on for the structure-library runs**, including the
  frozen test split, contrary to an earlier version of the paper. Appendix G
  audits what this did and did not affect: execution seeds were not collapsed
  (the cache key includes the seed, and 102-150 of 150 tasks per cell differ
  across seeds), and cache hits still consume the full budget (the reservation
  precedes the lookup). It did couple the baseline and reference-preserving
  arms into a shared first draft, which the paper treats as part of the
  treatment definition.
- The frozen test runs predate an exact tokeniser; 4.7% of executions settled
  above their reservation and were scored as failures.
  `scripts/sensitivity_ledger.py` recomputes every headline contrast on the
  defect-free subset.
- The math grader's symbolic-equivalence branch never fired because sympy was
  absent from the environment that produced those runs, misgrading 2.6% of math
  answers. Every stored response was re-graded at no inference cost; both
  gradings are in the data (`success` and `success_symbolic`).
- Two search bugs invalidated our first Part-II runs (a racing rule whose fixed
  slack was smaller than the confidence-interval gap between rungs, and the
  token estimator above). Fixes are in `hbws/search.py` and `hbws/llm.py`.

## Requirements

Python 3.12; see `requirements.txt`. CPU only, no GPU anywhere.

`sympy` (with `antlr4-python3-runtime`) is **required, not optional**: the math
grader falls back to symbolic equivalence, and that branch fails silently if the
library is absent -- which is exactly the defect described in Appendix F of the
paper. Verify it is importable before re-grading anything:

```bash
.venv/bin/python -c "from sympy.parsing.latex import parse_latex; print('ok')"
```
