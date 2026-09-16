# Provenance follow-up handoff (2026-09-15/16)

Two independent, complete post-hoc controlled follow-ups, run in an
isolated working directory/branch (`provenance-followup-20260915`) off
`server-qwen-math-tight-20260914`. Neither modifies the paper, Figure 2,
the frozen splits, any preregistration, or any existing source file or
result. No paid API was called; the GLM calls used the existing free
`glm-4-flash-250414` endpoint, and only after the operator explicitly
confirmed no concurrent Mac GLM-4.7 task.

## Part A: Qwen serving-configuration intervention (reviewer B1)

**Question**: same model weights, same physical GPU, same request -- does
changing vLLM concurrency/batching or prefix caching change output bytes?

**Setup**: `Qwen/Qwen2.5-Coder-7B-Instruct` (same HF revision/shard hashes
as the archived `qwen25coder7b_server_math_tight_20260914` campaign),
same GPU (`CUDA_VISIBLE_DEVICES=1`), same torch/vllm/python versions --
verified byte-for-byte in
`experiments/qwen_serving_regime_20260915_environment_diff.json`
(`all_match: true`). Only the same-policy suffix
(`hbws.dsl.wf_incumbent_refine_cot`) was rerun, 150 tasks x seeds 0/1/2,
temperature 0, seed forwarded, cache disabled, nonbinding task-level caps
(same sizing as the archived causal campaign). Four conditions, run in
fixed order, one vLLM instance at a time:

| condition | max-num-seqs | prefix caching | workers | rows | errors | over_budget |
|---|---|---|---|---|---|---|
| S (serial reference) | 1 | off | 1 | 450 | 0 | 0 |
| B (batched) | 16 | off | 8 | 450 | 0→0 (6 recovered) | 0 |
| P (prefix cache) | 1 | on | 1 | 450 | 0 | 0 |
| S2 (repeat of S) | 1 | off | 1 | 450 | 0 | 0 |

- B's 6 execution errors (`ImportError: cannot import name 'AutoTokenizer'`,
  a one-time concurrent-first-import race in the NEW
  `qwen_exact_reservation.py` module under 8 worker threads, not a vLLM or
  model defect) were recovered by rerunning only those 6 exact (task,
  seed) rows serially against the same condition-B server, then fixing the
  root cause (pre-warming the tokenizer import before spawning workers) so
  it cannot recur. See `PROVENANCE_FOLLOWUP` commit history / this file's
  "known limitations" for the caveat that these 6 rows ran at client-side
  concurrency 1 against a server still configured for max-num-seqs=16.
- The reservation fix (`scripts/qwen_exact_reservation.py`, Qwen's own
  tokenizer + chat template instead of tiktoken) eliminated the archived
  campaign's `budget_exceeded:settle_overrun` class of row entirely: 0
  occurrences across all 1800 new rows (vs. 14/1350 in the archived
  campaign, which is left untouched).

**Result (`experiments/qwen_serving_regime_20260915_analysis.json`)**:

| condition | first-draft byte-identical to S | first-accepted byte-identical to S | same correctness label as S |
|---|---|---|---|
| S2 (repeat of S) | 100.0% (450/450) | 100.0% (300/300) | 100.0% |
| B (batched) | 50.0% (225/450) | 58.8% (164/279) | 91.8% |
| P (prefix cache) | 54.2% (244/450) | 62.9% (176/280) | 93.1% |

S2's perfect 100% identity validates the measurement method itself (byte
comparison, call-log extraction, condition isolation) before trusting the
B/P numbers. B was confirmed to have genuinely overlapping requests during
execution (`vllm:num_requests_running` observed at 7-8 concurrently via
`/metrics`, saved in
`experiments/qwen_serving_regime_20260915_B_metrics_snapshot.txt`) -- this
is not merely "max-num-seqs was set." P was confirmed to have genuine
prefix-cache activity during execution: vLLM's own periodic engine log
(`experiments/qwen_serving_regime_20260915_vllm_P.log`, 2187 stat lines
logged every ~10s) reports `Prefix cache hit rate` climbing from 0% at
startup to 39.5% by the end of the 450-row run, as the shared chat-
template prefix across requests is increasingly reused -- this is not
merely "prefix caching was enabled."

**This is a real, non-null result, not a null result to report and move
on from**: on identical hardware, weights, and request, both batched
serving and prefix caching change roughly half of all first-draft output
bytes relative to strictly serial single-sequence serving, and these are
not merely cosmetic -- 6.9-8.2% of task/seed positions flip which side of
the correctness line they land on. This is attributed only to the
concretely manipulated serving regime (batching, prefix caching); no claim
is made about which internal vLLM/CUDA mechanism (kernel selection,
floating-point non-associativity in batched attention/GEMM, KV-cache reuse
numerics) is responsible, nor that this explains the original cross-provider
GLM finding -- that would be an unobserved-mechanism inference this
follow-up does not support.

## Part B (replaced): GLM-4-Flash budget-parity assign-vs-same-policy control

The originally-planned "GLM clean-matrix first-draft backfill" was
superseded by the operator with this design before any Zhipu call was
made. **Not a replacement for or claim about the original clean matrix.**

**Question**: does separating drafting cost from the shared verify/refine
suffix budget change the assign-vs-same-policy repair/breakage asymmetry
seen under the original tiered total-task caps?

**Setup**: reused the existing frozen 150-task math roster and the
COMPLETE stored `cot_math_tight`/`cot_math_loose` reference outputs
(transferred by the operator as
`glm_budget_parity_references_20260916.tar.gz`, checksum-verified
byte-for-byte against the package's own `SHA256SUMS`, including an
independent confirmation that the transferred `data/math_test.jsonl`
matches the already-frozen hash
`f520b14219177c3fb5123bd80fcd0f61019b767dbb3f0ce8fbf1e6b1449a01c4`). Two
arms only (`hbws.dsl.wf_assign_refine`, `hbws.dsl.wf_incumbent_refine_cot`),
same verifier/refinement graph/prompts/temperature/per-call output caps as
every other campaign that has used them. **Not** the old
`BUDGET_TIERS` total-task caps -- those are exactly the confound this
follow-up isolates.

**Budget-parity mechanism**: a single ledger still meters the whole task
(no change to `hbws/ledger.py` or `hbws/runner.py`); `same_policy`'s total
cap is `SUFFIX_CAPS` (8 calls, 128k in-tokens, 8x1536 out-tokens -- the
prescribed 1 verify + up to 3x(refine, verify) suffix, sized generously)
plus exactly one extra call's worst-case cost (1 call, 1536 out-tokens --
the "g" node's own worst case). `assign`'s cap is `SUFFIX_CAPS` unchanged
(assign's own node is free). After `same_policy`'s draft settles, both
arms have byte-identical remaining ledger headroom for the suffix --
proven directly in
`tests/test_glm_budget_parity_noninvasive.py::test_budget_parity_arithmetic`.
Verified nonbinding in a 10-task preflight before the real roster ran (0
`over_budget`); confirmed nonbinding across the full 1800-row roster too
(0 `over_budget`, 0 execution errors).

Input-token reservation uses `THUDM/glm-4-9b-chat`'s tokenizer (the
closest openly-available same-family proxy for the undisclosed
`glm-4-flash-250414` tokenizer) plus an explicit safety margin -- see
`scripts/glm_exact_reservation.py`'s docstring. A first version of this
estimator had a bug (took `len()` of the tokenizer's returned
`BatchEncoding` -- 3 dict keys -- instead of the real token count),
caught immediately by the mandated 10-task preflight (10/10
`budget_exceeded` on the very first attempt) rather than during the real
run; fixed and covered by a regression test
(`test_exact_reservation_extracts_real_token_ids_not_dict_key_count`)
before any further Zhipu call was made.

**Concurrency**: workers=3 hit the account's rate limit (GLM error code
1302, "您的账户已达到速率限制"; 16/43 attempts were `RateLimitError`
retries, 2/10 preflight tasks permanently failed after exhausting
retries). workers=2 completed the same 10-task preflight with 0 retryable
errors and 0 failures -- adopted for the full 1800-row run; it also
completed with 0 execution errors throughout. (A second API key was made
available for rate-limit fallback via `scripts/glm_key_rotation.py`; it
was not needed since workers=2 never hit the limit.)

**Result (`experiments/glm_budget_parity_20260916_analysis.json`)**, tight
and loose reported separately (not a single-factor tight-vs-loose
comparison -- both use identical suffix caps):

| tier | arm | accuracy | repair (of ref-wrong) | breakage (of ref-correct) | total llm_calls | total tokens |
|---|---|---|---|---|---|---|
| tight | assign | 0.8000 | 17.5% (18/103) | 1.4% (5/347) | 1093 | 1,443,434 |
| tight | same_policy | 0.7600 | 21.4% (22/103) | **7.8%** (27/347) | 1573 (+480 drafting) | 1,858,048 (+414,614 drafting) |
| loose | assign | 0.7867 | 16.8% (18/107) | 2.0% (7/343) | 1113 | 1,473,451 |
| loose | same_policy | 0.7911 | 26.2% (28/107) | **4.4%** (15/343) | 1504 (+391 drafting) | 1,728,744 (+255,293 drafting) |

**Interpretation, stated plainly and without cherry-picking**: even after
removing the suffix-budget confound (both arms get an identical, verified-
sufficient suffix budget), `same_policy` still breaks more previously-
correct tasks than `assign` in both tiers (tight: 5.4x; loose: 2.1x), and
also repairs more previously-wrong tasks in both tiers. Net accuracy delta
vs. `assign` is tier-dependent (tight: -0.04; loose: +0.0044) -- not
claimed to generalize. **The original repair/breakage asymmetry was not
solely a budget-reachability artifact**: reusing the reference verbatim
vs. independently regenerating a first draft produces a real behavioral
difference downstream, even with matched suffix budgets. All rare/zero-
event CIs are marked `ci_is_degenerate` where the percentile bootstrap
would otherwise look reassuringly tight without being informative (none
of the reported breakage/repair events were rare enough to trigger this
flag here -- all denominators/CIs are reported in full in the analysis
JSON regardless). No task was filtered by correctness or by arm outcome.

## Validation performed (both parts)

- `git rev-parse HEAD` in the source clone matched
  `332fdc3fdd18c29706a627ae20ba5585506709c3` (branch tip at clone time,
  i.e. the prior Qwen math/tight campaign's own last commit) before any
  new code was written.
- `python -m pytest tests/` -- 15/15 passed throughout, including 5 new
  mock/regression tests
  (`test_qwen_exact_reservation_noninvasive.py`,
  `test_glm_budget_parity_noninvasive.py`) proving the new
  estimator/logger modules change neither request text, sampling
  parameters, call order, nor budget accounting versus the plain,
  unmodified `hbws` code path -- only the ledger admission math (Qwen: a
  tighter/more accurate reservation; GLM: the same-vs-assign suffix
  parity).
- Completeness: 1800/1800 rows for Part A (12 result files x 150 unique
  task ids), 1800/1800 rows for Part B (12 result files x 150 unique task
  ids), 0 rows with an `error:`-prefixed status in either part, 0 missing
  task/seed/condition or task/seed/tier/arm keys.
- Full SHA-256 inventory of every manifest, roster, result, summary, log,
  call-log, and analysis file:
  `experiments/provenance_followup_20260915_sha256_inventory.txt`.

## New files (this follow-up only; no existing file modified)

- `scripts/force_cuda_platform.py` is NOT part of this follow-up -- it is
  a venv-local site-packages plugin from the prior session, reused as-is
  (see the archived campaign's handoff for its purpose: bypassing a
  host NVML/driver version mismatch that does not affect CUDA compute).
- Qwen serving-regime: `scripts/run_qwen_serving_regime.py`,
  `scripts/build_serving_regime_roster.py`,
  `scripts/check_serving_regime_environment_match.py`,
  `scripts/qwen_exact_reservation.py`, `scripts/qwen_call_logger.py`,
  `scripts/analyze_qwen_serving_regime.py`,
  `tests/test_qwen_exact_reservation_noninvasive.py`.
- GLM budget-parity: `scripts/run_glm_budget_parity.py`,
  `scripts/run_glm_budget_parity_all_cells.sh`,
  `scripts/build_glm_budget_parity_roster.py`,
  `scripts/glm_exact_reservation.py`, `scripts/glm_call_logger.py`,
  `scripts/glm_key_rotation.py`, `scripts/analyze_glm_budget_parity.py`,
  `tests/test_glm_budget_parity_noninvasive.py`.

## Known limitations / unresolved items

- The 6 recovered condition-B rows (Qwen, seed 0) ran at client-side
  concurrency 1 (recovery pass) against a server still configured for
  max-num-seqs=16 -- server-side batching configuration was unchanged, but
  these 6 rows individually did not have other concurrent requests
  in flight at the moment they ran, unlike the rest of condition B's
  roster.
- `glm_exact_reservation.py` uses `THUDM/glm-4-9b-chat`'s tokenizer as a
  same-family proxy for the undisclosed `glm-4-flash-250414` tokenizer,
  not an exact match; the fixed +64-token safety margin absorbed this
  gap cleanly (0 `over_budget` across 1800 rows) but is not claimed to be
  byte-exact.
- Part A attributes the B/P byte differences only to the serving regime
  actually manipulated (batching, prefix caching); it does not identify
  which internal vLLM/CUDA mechanism causes them, and does not claim this
  explains the original cross-provider (GLM) finding that motivated
  reviewer B1's question.
- The GLM clean-matrix first-draft backfill (original Part B) remains
  un-run; it was explicitly superseded by the budget-parity design above
  before any Zhipu call was made, per the operator's instruction.
