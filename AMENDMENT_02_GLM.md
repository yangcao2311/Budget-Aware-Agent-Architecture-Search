# AMENDMENT 02 — GLM-5.3-Flash provenance replication

Status: **FROZEN BEFORE ANY GLM EXPERIMENTAL CALL**. Written on 2026-09-01
after GLM-5.3-Flash became available to the project. A one-message API
connectivity check and short workflow smoke tests are implementation checks,
not observations included in the analysis. No condition may be removed or
added in response to its measured accuracy, repair, or breakage.

This is a post-hoc external replication, not part of the original
preregistration and not a new confirmatory claim about the GPT-4o study.

## 1. Question and scope

Does the directly observed relationship between incumbent provenance and
workflow regression recur in a third model family, GLM-5.3-Flash, when the
three downstream workflows are held fixed?

The study does **not** test whether GLM is better than GPT-4o or Kimi, whether
verify--refine is compute-optimal, or whether effect magnitudes transfer.
Model-to-model accuracy and dollar-cost comparisons are out of scope.

## 2. Frozen model and execution configuration

- Provider/model: Z.AI general API, `glm-5.3-flash`.
- Reasoning effort: `low`, fixed for every model call. Thinking tokens count
  as output tokens whenever returned in provider usage.
- The existing family-specific prompts, node temperatures, output-token caps,
  verifier implementations, refinement prompts, frozen task order, and
  `loose` reservation caps are unchanged.
- Families: code and mathematics.
- Tasks: the first 150 tasks of each existing frozen test split.
- Replicates: indices 0, 1, and 2. If the endpoint rejects or documents no
  support for `seed`, the parameter is omitted for every GLM call and these
  remain independent execution-replicate indices, not claims of seeded
  determinism. This compatibility fact is recorded before the full run.
- Cross-arm response caching is disabled. The explicit-reuse arm uses direct
  assignment of stored baseline text and therefore requires no cache.
- Provider errors are retained in an append-only attempt log. Primary results
  use the matched task-by-replicate intersection across the GLM baseline and
  all three arms. A failure-as-wrong sensitivity analysis is also reported.

## 3. GLM reference baseline

The reference population must be defined by GLM, not inherited from GPT-4o.
Before the causal arms, run the family baseline (`direct` for code, `cot` for
math) on all 150 tasks and all three replicates, with cache disabled. Store the
literal output and held-out correctness for every task-by-replicate pair.

The primary conditioning population is tasks on which the GLM baseline is
correct in all three completed replicates, matching the original analysis.
As a sensitivity analysis, also report pair-level results conditional on the
baseline being correct in that same replicate.

## 4. Three-arm intervention

All arms use the same verifier and refine-on-rejection downstream graph and
differ only in the incumbent's source:

1. **Explicit reuse:** assign the stored GLM baseline output byte-for-byte.
2. **Same-policy regeneration:** make a fresh call with the baseline family's
   prompt, temperature, and replicate index.
3. **Different-policy regeneration:** make a fresh call with the vanilla
   workflow's first-draft policy (`solve_cot`, temperature 0.7).

Each arm is run on both families, all 150 tasks, and all three replicates.

## 5. Readouts and interpretation fixed in advance

For each family and arm report:

- baseline accuracy and number of baseline-correct tasks;
- repair, total breakage, and net accuracy change;
- acceptance-path regression: final output is wrong after an accepting
  verifier and no refinement node is visited;
- rejection/refinement-path regression: a verifier rejects a GLM reference-
  correct incumbent, refinement is visited, and the final output is wrong;
- task-clustered percentile-bootstrap 95% intervals, averaging replicates
  within task before resampling;
- effective matched sample size, provider-error count, calls, tokens, and
  actual provider-reported dollar spend using the price in force at run time.

The structural statement that explicit reuse makes acceptance-path regression
impossible follows from the workflow graph and byte-identical assignment; its
observed value is an implementation audit, not an empirical discovery.

The cross-model directional replication is called **fully supported** only if
explicit reuse has lower total breakage than both regeneration arms in both
families. It is **partially supported** if this holds in some but not all four
within-family comparisons, and **not supported** if it holds in none. No
monotonic ordering between same-policy and different-policy regeneration is
predicted. Regardless of this label, every cell and both sensitivity analyses
are reported.

## 6. Stopping and retry rules

Before the full study, run one API connectivity call followed by one code and
one math task through each relevant call type. Full execution starts only if
responses contain a gradeable final answer, usage fields are present, and the
reservation estimator does not settle-overrun.

Full-run concurrency is at most 2. Retry only provider-side 429/5xx/timeouts,
with fixed backoff 10/30/90 seconds and at most three attempts. Do not retry a
completed, reserve-rejected, or normally returned incorrect answer. If the
provider remains unavailable, report the study as incomplete rather than
changing model, endpoint, reasoning effort, prompts, tasks, or outcome rules.

