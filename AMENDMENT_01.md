# AMENDMENT 01 — equal-realized-compute baselines, factorial attribution, budget-dimension scan

Status: **FROZEN** at the commit tagged `amendment-01`. Written and committed
before any of the runs it specifies were executed. This amendment does not
alter, retract or reinterpret anything in `PREREGISTRATION.md`; it adds three
studies prompted by review, and records their design in advance for the same
reason the original claims were recorded in advance.

Nothing here is contingent on results already seen from these three studies,
because none exist at the time of writing.

## 0. Why these three

Review identified three respects in which the submitted paper's claims outrun
its evidence.

1. Enforcing a common budget **cap** does not equalise **realized** compute.
   The reported vanilla workflow spends about $5.5\times$ the baseline's
   dollars. The paper therefore cannot, as written, address whether structure
   improves the compute--accuracy frontier against strong equal-cost
   alternatives.
2. The vanilla and protected workflows differ in **three** respects, not one:
   first-draft prompt (`solve_cot` vs `solve_direct`), first-draft temperature
   ($0.7$ vs $0$), and the overwrite rule. Attribution of reduced breakage to
   incumbent protection is therefore confounded. (Verified from the prompt
   registry: `solve_cot` prepends "First reason step by step about edge cases
   and the algorithm, then" to the `solve_direct` template.)
3. Budget profiles vary dollars, call caps and token caps together, and the
   realized dollar spend ($0.002$--$0.015$) sits far below the dollar cap
   ($0.10$--$0.25$), so the dollar axis is not the binding constraint and the
   phrase "budget entry fee" is not identified.

## 1. Study A — equal-realized-compute frontier (primary)

**Question.** Under matched *realized* inference cost, does verifier-gated
structure sit on or above the accuracy--cost frontier traced by structure-free
alternatives?

**Methods compared**, all on the same tasks, model and grader:

| arm | description |
|---|---|
| single call | baseline prompt, one call |
| long single call | same, output cap raised to the tier's full output budget |
| best-of-$k$ | $k$ independent samples, first-passing selection (code: visible tests; math/logic: self-consistency majority) |
| self-consistency-$k$ | $k$ samples, majority vote on the normalised answer |
| verify-only cascade | one draft, verifier runs, no refinement (measures the cost of verification alone) |
| vanilla verify--refine | as in the paper |
| protected verify--refine | as in the paper |

**Cost accounting** includes input and output tokens, verifier LLM calls, tool
executions, and every failed or rejected call. Latency is reported but not
matched.

**Matching procedure.** We do not tune $k$ to a single matched point. On the
**dev split only**, we measure the realized cost distribution of each arm and
select $k$ values placing arms at three preregistered cost levels — the
realized median cost of the protected workflow at `tight`, at `unseen`, and at
`loose`. Those $k$ values are then **locked** and carried unchanged to the
frozen test split. No configuration is chosen using test-split accuracy.

**Primary readout.** The accuracy--cost Pareto frontier per family, with
per-arm mean cost, cost quantiles, and stratified paired bootstrap intervals
on accuracy. **Secondary:** repair and breakage for every arm, so the
decomposition is reported for structure-free methods too.

**Preregistered interpretations.**
- If a structure-free arm dominates the protected workflow at matched realized
  cost in a family, we will say so, and the paper's contribution in that family
  narrows to the repair--breakage diagnostic rather than any claim of workflow
  superiority.
- If the protected workflow is on the frontier, we will claim only that, at the
  cost levels tested, on this model and these families.

## 2. Study B — factorial attribution of incumbent protection

**Question.** Is the reduction in breakage attributable to the overwrite rule
rather than to first-draft temperature?

**Design.** $2\times2$: first-draft temperature $\{0,\,0.7\}$ $\times$
overwrite rule $\{$unconditional, verifier-gated$\}$. The first-draft **prompt
is held fixed** at the family baseline's prompt (`solve_direct` for code,
`solve_cot` for math) in all four arms; verifier, refiner prompt, maximum
iterations, budget profile and stopping rule are identical throughout. Prompt
is therefore removed as a variable rather than crossed; a prompt ablation is
deferred to the dev split and is explicitly not part of this confirmatory
design.

**Draft sharing.** Within a temperature level, all arms issue an identical
first call (same prompt, temperature, seed, token cap). Runs are executed with
the response cache enabled so that the two arms at a given temperature receive
a **byte-identical incumbent**, making the comparison exactly paired rather
than merely identically configured. Cache-key equality is asserted at run time.

**Readout.** Breakage, repair and net effect for each of the four cells, at
`tight` and `loose`, both families, three execution seeds. The attribution
claim is supported only if the overwrite-rule main effect on breakage is
present at both temperature levels.

## 3. Study C — which budget dimension binds

**Question.** Is there an entry threshold, and in which dimension?

**Design.** Starting from the `loose` profile, scan one dimension at a time
holding the others at `loose`: LLM-call cap $\in\{1,2,4,6,8\}$; output-token
cap $\in\{500,1000,2000,4000\}$; dollar cap $\in\{0.02,0.05,0.10,0.25\}$.
Code family, protected and vanilla workflows.

**Readout.** For every cell, the distribution of admission failures by the
dimension that caused them (call, input-token, output-token, tool, dollar,
or simultaneous), reported as a distribution over rejected admissions rather
than as a single "binding dimension" label. The phrase "entry fee" survives
only for a dimension in which a single-dimension scan produces a crossover;
otherwise the finding is reported as a crossover across composite profiles.

## 4. Statistics

As in `PREREGISTRATION.md` §4: task as the unit, stratified paired bootstrap
over $10{,}000$ resamples, seeds averaged within task before resampling, Holm
correction within each of the three studies treated as its own family.

**Zero-event rates.** Breakage rates of $0.000$ are reported with a one-sided
$95\%$ upper bound computed by task-clustered bootstrap, not by a naive rule
of three: the three execution seeds of a task are not independent observations
and must not be counted as such. The denominator is stated explicitly as
baseline-correct **tasks**.

## 5. Status of corrective reruns

If the frozen test split is re-executed with the corrected token estimator and
grader, those runs are reported as **corrective replication**, never as the
original preregistered confirmation, and the original runs, the defects, the
commits that fixed them, and which numbers come from which execution all remain
in the paper and the deviation log. Removing that history is not an option: it
is evidence about how the results were produced.

## 6. Claim adjustments already committed (independent of these studies)

These are corrections of statements that are wrong or unidentified on the
existing evidence, and are made now rather than being contingent on new runs:

- The claim that the two compared workflows have a single structural difference
  is false and is withdrawn.
- "Breakage is eliminated by construction" is withdrawn. The defensible
  statement is that incumbent protection removes breakage caused by
  unconditional overwriting, and that residual breakage requires a false
  rejection first, so $b \le \Pr(\text{reject} \mid \text{incumbent correct})$
  — an upper bound, not an equality.
- The dollar-labelled budget axis is withdrawn in favour of named profiles with
  their full budget vectors given, pending Study C.
- "An independent optimiser rediscovered the principle" is withdrawn in favour
  of a statement about a search procedure operating inside our own prespecified
  design space and objective.
