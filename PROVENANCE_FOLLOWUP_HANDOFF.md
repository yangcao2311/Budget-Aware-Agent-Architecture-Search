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

## Part C: GLM-4-Flash code-domain budget-parity replication (2026-09-17)

**This is a code-domain REPLICATION of the completed math budget-parity
follow-up (Part B), using the identical design -- not a cross-domain
replication claim.** Two cohorts (`prior-tight`, `prior-loose`, named
after their inherited frozen reference directories -- not a budget-tier
manipulation), two arms (assign, same_policy), 150 tasks x seeds 0/1/2
each, 1800 rows total.

**Inputs**: `experiments/glm4flash_repaired_matrix_clean_20260910_
envelope_test/direct_code_{tight,loose}/results_seed{0,1,2}.jsonl` and
`data/code_test.jsonl`, transferred by the operator as
`glm_budget_parity_code_references_20260916.tar.gz` (package SHA256
`bb0f06c655d2e688ad7b3acfd3015b871af0631ccfbcd7e841250bed61df4d62`,
confirmed before extraction), all 7 files verified against the package's
own `SHA256SUMS` -- all `OK`, none regenerated.

**Design differences from math, both frozen before any call and both
required by the domain itself, not chosen post-hoc**:
- `hbws.dsl.wf_incumbent_refine` (prompt_id `solve_direct`, the code
  family's own "direct" first-draft policy) replaces math's
  `wf_incumbent_refine_cot` for the same_policy arm.
- `DRAFT_EXTRA_OUT_TOKENS = 1024` (code's own "g" node cap), not math's
  1536 -- verified in
  `tests/test_glm_budget_parity_code_noninvasive.py::test_budget_parity_arithmetic_matches_code_draft_cost`.
- Code's verify node reserves a TOOL call (running the task's own visible
  `feedback_tests` in the sandbox), never an LLM call -- so `SUFFIX_CAPS`
  (reused verbatim from the math version) is used only by up to 4 refine
  calls in this domain (discovered while writing the mock test: the
  `v -> r` edge has no loop cap, only `r -> v` does, at `max_iter=3`, so
  a verifier that never passes drives 4 refines, not 3 -- confirmed by
  direct trace inspection and covered by
  `test_same_policy_worst_case_refine_loop_never_gated_under_suffix_caps`
  / `test_assign_worst_case_refine_loop_never_gated_under_suffix_caps`,
  both of which also assert this worst case is never suffix-gated).

**Directional prediction, frozen in the manifest before any real GLM
call** (verbatim, per operator instruction): the code-domain assignment
advantage was expected to be smaller than math's, possibly
indistinguishable, because (a) code's verifier runs the task's own tests
rather than a gold-free self-consistency check, so incorrect regenerated
drafts are rejected far more reliably before acceptance, and (b) the
triggering event (a non-byte-identical first-accepted draft) is itself
~4x rarer in code (94.9%/93.0% byte-identical) than math (32.6%/23.8%).

**Run**: workers=2 throughout (no rate-limit retries observed -- code
verification runs locally in the sandbox, so far fewer GLM calls per row
than math). All 1800 rows `completed`, 0 execution errors, 0
`over_budget`. Wall time: ~32 minutes total for all 4 cells (vs. ~9-10
hours for the math version's 1800 rows) -- code's verify cost is a local
sandbox run, not an additional GLM call.

**Result (`experiments/glm_budget_parity_code_20260917_analysis.json`)**:

| cohort | arm | accuracy | repair (of ref-wrong) | breakage (of ref-correct) | counterfactual blocked under original tier caps |
|---|---|---|---|---|---|
| tight | assign | 0.7244 | 10.1% (14/138) | **0.0%** (0/312) | 11/450 |
| tight | same_policy | 0.7111 | 8.0% (11/138) | **1.0%** (3/312) | 71/450 |
| loose | assign | 0.7244 | 8.8% (12/136) | **0.0%** (0/314) | 0/450 |
| loose | same_policy | 0.7333 | 13.2% (18/136) | **0.6%** (2/314) | 0/450 |

Accuracy delta (assign - same_policy): tight +0.0133, loose -0.0089 --
opposite signs, both tiny.

**The prediction held**: the code-domain assign-vs-same_policy asymmetry
is far smaller than math's (breakage differs by 0-3 events out of
312-314 reference-correct positions per cohort, vs. math's 20-22-event
gaps out of ~347) and is not even consistent in direction between
cohorts (tight slightly favors assign, loose slightly favors
same_policy) -- reported as indistinguishable, exactly as the frozen
prediction anticipated, in both cohorts, without adding conditions,
swapping metrics, or omitting either cohort. `ci_is_degenerate` is not
triggered for any of these cells' breakage/repair rates (each has >2
events), but the intervals are wide relative to the tiny point estimates
and are reported as post-hoc descriptive, not as evidence the difference
is exactly zero.

**Counterfactual original-tier-caps audit** (never affects the actual
run; simulated post-hoc from each row's own `trace` plus its raw call
log, replayed against a fresh shadow ledger using the pre-budget-parity
`BUDGET_TIERS[cohort]` caps): under the ORIGINAL tight-tier caps
(`max_llm_calls=4`), same_policy's extra draft call combined with its
refine calls would have been gated at 71/450 positions (15.8%) versus
assign's 11/450 (2.4%) -- a real, substantial admission-gating asymmetry
that the original design's confound would have produced in this domain
too. Under the original loose-tier caps, neither arm would have been
gated at all (0/450 each) -- loose's caps were already generous enough
for code's shorter LLM-call footprint. This confirms the original
confound was real specifically for code/tight, even though the
budget-parity-corrected behavioral difference (breakage/repair) turns out
to be small once that confound is removed.

**Secondary: stratified by same_policy's first-draft outcome** (did the
independently regenerated first draft pass the first verify?):

| cohort | first draft passed: n, final accuracy | first draft rejected: n, final accuracy |
|---|---|---|
| tight | 352, 87.2% | 98, 13.3% |
| loose | 357, 87.4% | 93, 19.4% |

As expected, final accuracy is overwhelmingly determined by whether the
independently regenerated first draft itself passed verification --
consistent with code's task-test verifier being decisive rather than
gold-free-heuristic-permissive.

**New files** (code-domain only; none of Parts A/B's files or the frozen
math source touched): `scripts/run_glm_budget_parity_code.py`,
`run_glm_budget_parity_code_all_cells.sh`,
`build_glm_budget_parity_code_roster.py`, `analyze_glm_budget_parity_code.py`,
`tests/test_glm_budget_parity_code_noninvasive.py`. Reuses
`glm_exact_reservation.py` / `glm_call_logger.py` / `glm_key_rotation.py`
unchanged from Part B.

**Validation**: `python -m pytest tests/` 18/18 passed (3 new for this
part). 1800/1800 rows, 0 `error:`-prefixed statuses, 8 result files x 150
unique task ids. Full SHA-256 inventory:
`experiments/glm_budget_parity_code_20260917_sha256_inventory.txt`.

**Known limitations**: `ORIGINAL_TIER_CAPS` in the counterfactual audit
reserves each replayed call using `glm_exact_reservation`'s same-family
proxy estimator (recomputed from the actual logged request messages), not
the exact reservation the real run made at execution time -- a
faithful-but-approximate replay, not a byte-exact reproduction of what an
original-caps run would have measured live.

## Part D: Factorial budget ablation -- call cap x output-token cap (2026-09-17)

**Question**: the paper's tight/loose budget profiles move the LLM-call
cap and the per-task output-token cap together. Which one (if either)
actually drives the breakage effect, and is there an interaction?

**Design**: 2x2 grid decomposing `hbws.ledger.BUDGET_TIERS`'s tight/loose
profiles into their two named axes, with the three auxiliary dimensions
(`max_in_tokens`, `max_tool_calls`, `max_wall_sec`, `max_usd`) bundled
with whichever axis they naturally track (count-of-actions with the call
cap; token volume with the token cap) -- see
`scripts/run_factorial_budget_ablation.py`'s module docstring for the
exact bundling and rationale.

| cell | call cap | token cap | == existing tier? |
|---|---|---|---|
| A | tight (4) | tight (2000) | `BUDGET_TIERS["tight"]` exactly |
| B | tight (4) | loose (4000) | -- |
| C | loose (8) | tight (2000) | -- |
| D | loose (8) | loose (4000) | `BUDGET_TIERS["loose"]` exactly |

**Corner-reproduction check**: `CELLS["A"] == BUDGET_TIERS["tight"]` and
`CELLS["D"] == BUDGET_TIERS["loose"]` are asserted at import time in
`run_factorial_budget_ablation.py` (the process will not run at all if
this fails) and re-confirmed independently in the analysis JSON and by
`tests/test_factorial_budget_ablation_noninvasive.py::test_corner_cells_equal_budget_tiers_exactly`.
**Both hold exactly.**

**Sizing**: per the operator's frozen power simulation, the clean/normal
verifier's baseline breakage (~0.02) gives 9-27% power to detect even a
50% relative reduction -- reproducing the uninformative null already seen
in the code budget-parity replication (Part C). The fully-masked verifier
(mask fraction 0.0, the existing, unmodified `scripts/run_envelope.py::
mask_tests` mechanism) was specified to have baseline breakage ~0.235,
giving 84-100% power. **Domain: code. Verifier: fully masked. Workflow:
`hbws.dsl.wf_incumbent_refine`, self-drafting arm only** (not the
assign-vs-same_policy budget-parity design -- no separate drafting/suffix
accounting here, by design).

A mechanical fact discovered while sizing the mock tests, worth recording
explicitly: `hbws.verify.run_code_tests` returns `(False, "no tests are
available...")` -- never a vacuous pass -- when `feedback_tests` is empty
(mask 0.0). So under this condition, verify **never** passes, for any
candidate, regardless of correctness. Combined with the suffix graph's
`v -> r` edge having no loop cap (only `r -> v` does, at `max_iter=3`),
every task deterministically drives exactly 4 refines (5 LLM calls: g +
r1..r4) before the graph's own termination condition is reached, with no
final re-verify after the 4th refine. Tight's `max_llm_calls=4` cannot
reach this natural termination (it allows only g + 3 refines before the
4th call is refused), so **every task under a tight call cap is expected
to hit `reserve_rejected` deterministically** -- confirmed empirically
(cells A and B: 150/150 `reserve_rejected` in every seed).

**Per-iteration logging** (`scripts/glm_iteration_logger.py`): for every
LLM call, records the reservation vector computed before the call (using
the SAME installed estimator hbws.llm.chat would use internally, so it is
not a re-derivation error), whether it was refused, the full request, the
exact response text, and usage. `scripts/reconstruct_iterations.py`
combines this with each row's own `trace` to produce, per task/seed/cell,
a full per-iteration record (candidate text, masked-verifier verdict,
correctness under the REAL held-out `grading_tests` -- graded offline,
never seen by the runner or the masked verifier) -- 1800 rows worth of
per-iteration data now sit in
`experiments/factorial_budget_ablation_20260917_{A,B,C,D}_iterations.jsonl`,
making any tighter/looser cap question replayable without another GLM
call.

**Run**: workers=2, all 4 cells, 0 execution errors, 0 `over_budget`
(genuine settle-overrun defects) across all 1800 rows.
`reserve_rejected` counts: A=450/450, B=450/450, C=36/450, D=0/450 (all
in the expected direction -- call cap dominates; token cap alone produces
a small nonzero residual at cell C that the 10-task preflight, by chance,
did not surface, underscoring why the full 150-task roster matters).

**Result (`experiments/factorial_budget_ablation_20260917_analysis.json`)**:

| cell | breakage rate | breakage events/denominator | repair rate |
|---|---|---|---|
| A (tight, tight) | 6.75% | 21/311 | 8.63% (12/139) |
| B (tight, loose) | 6.71% | 21/313 | 7.30% (10/137) |
| C (loose, tight) | 6.73% | 21/312 | 10.14% (14/138) |
| D (loose, loose) | 6.71% | 21/313 | 5.84% (8/137) |

Main effects on breakage (task-clustered paired bootstrap, 10000 draws,
post-hoc descriptive): call-cap main effect ~0 (95% CI [-0.012, 0.013]),
token-cap main effect ~0 (95% CI [-0.011, 0.012]), interaction ~0 (95% CI
[-0.027, 0.027]). Verified this is not a bug (not the same 21 task/seed
pairs breaking in every cell -- e.g. A and D's breakage sets overlap only
10/21): each cell's 150-task x 3-seed first draft is byte-identical
across all four cells (same seed, same temperature-0 prompt, cap-
independent), so which tasks start "reference-correct" is necessarily
identical across cells, but WHICH of those end up broken differs
cell-to-cell -- the totals landing on the same count (21) across all four
cells is a genuine numeric coincidence in this particular roster, not an
artifact of the analysis.

**Power reality check, reported honestly rather than silently absorbed**:
the frozen simulation assumed baseline breakage 0.235 for this condition;
the observed pooled breakage rate here is ~0.067, roughly a third of
that. The achieved power to detect a 30-50% relative main effect is
correspondingly lower than the planned 84-100%. **The near-zero main
effects above must be read against this weaker-than-planned power** --
this run does not have the statistical power originally budgeted for,
and the null should not be reported as strong evidence that call cap and
token cap have no true effect on breakage; it is a genuine null at the
power this run actually achieved.

**Terminal-path categories**: `pass` = 0 in every cell (mechanically
expected under full masking, as derived above); 100% of completed and
gated rows fall in the `reject` category (verify always rejects); no
`no_verify` or `verify_gated` rows occurred (tool-call caps, generous
relative to the 4 verifies ever attempted, were never the binding
constraint in this design).

**New files**: `scripts/run_factorial_budget_ablation.py`,
`run_factorial_budget_ablation_all_cells.sh`,
`build_factorial_masked_code_roster.py`, `glm_iteration_logger.py`,
`reconstruct_iterations.py`, `analyze_factorial_budget_ablation.py`,
`tests/test_factorial_budget_ablation_noninvasive.py`. Reuses
`glm_exact_reservation.py` unchanged from Parts B/C.

**Validation**: `python -m pytest tests/` 21/21 passed (3 new for this
part, including a corner-reproduction assertion and a reservation-
refusal-is-logged regression test). 1800/1800 rows across all 4 cells, 0
`error:`-prefixed statuses (only `completed`/`reserve_rejected`, both
legitimate outcomes of this design). Full SHA-256 inventory:
`experiments/factorial_budget_ablation_20260917_sha256_inventory.txt`.

**Known limitations**: `glm_exact_reservation`'s same-family proxy
tokenizer (not GLM-4-Flash's exact tokenizer) is used for the logged
`reservation_request` vectors, same caveat as Parts B/C. The
reconstruction's per-iteration `verdict` field is inferred from trace
node order (generic logic, works for any mask fraction), not hardcoded
to "always rejects" -- confirmed to independently reproduce the "always
rejects under full mask" mechanical fact rather than assuming it.
