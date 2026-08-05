# PREREGISTRATION — Repair/Breakage Decomposition of Agent Workflow Structure, and HBWS (v1.0)

Status for PART I: **FROZEN** at the commit tagged `prereg-freeze-partI`
(2026-08-05). The Part-I confirmatory hypotheses in §2A below were written
BEFORE any frozen-test-set or OOD execution; all Part-I numbers seen so far
come from the dev split and are exploratory. Changes after this tag are
allowed only as dated deviation records in §9 — never edits in place.

Status for PART II (HBWS search): DRAFT until `prereg-freeze-partII`
(target 2026-08-17, before the first full search run).

## 0. Confirmatory claims and how they will be tested (Part I)

Every claim below is tested ONCE on the frozen test split
(`data/{code,math}_test.jsonl`, 150 tasks each) with 3 execution seeds,
seed-averaged per task, stratified paired bootstrap (10,000 resamples),
Holm-corrected within the Part-I family. Dev-set results never appear as
confirmatory evidence in the paper.

## 1. Frozen design decisions (already binding)

- Data splits: `data/SHA256SUMS` (commit of 2026-08-04). MBPP+ 120/40/150 +
  HumanEval+ OOD 100; MATH L4–5, 4 in-domain subjects 120/40/150 + 2 held-out
  subjects OOD 100. Split seed 20260804. Never regenerated.
- Budget profiles: E1 $0.02 / E2 $0.05 / tight $0.10 / unseen $0.15 / loose
  $0.25 as defined in `hbws/ledger.py BUDGET_TIERS`. `unseen` is final-test
  only: no search, selection, thresholding, or tuning code path may read it.
- Reservation ledger: reserve→call→settle; wall clock enforced by continuous
  metering + per-call deadlines, not reservation. Policy-attributable
  violation is defined as any settle-overrun or cap excess; reservation
  rejection is NOT a violation.
- Tier clamp rule: each node's per-call output cap is clamped to the tier's
  total output cap; vote nodes divide the tier output cap k-ways (floor of
  64) so their all-k upfront reservation is well-defined
  (`run_envelope.clamp_to_tier`). Applied identically to every structure and
  method. (k-way division added 2026-08-04 after the v0 screen showed votes
  spuriously infeasible at every tier — an artifact, not a finding.)
- Verifier taxonomy: code = oracle environment feedback (visible asserts
  only; extended tests exist only in the external grader); math =
  non-oracle structured self-check (k independent gold-free re-derivations,
  majority agreement); final grading deterministic for both.
- Envelope structure library: the 8 structures in `hbws/dsl.py ENVELOPE_LIB`
  under their frozen names.
- Search budgets: {tight, loose} only. Policies read continuous remaining
  fractions and reserve-feasibility, never tier IDs.

## 2A. CONFIRMATORY hypotheses — Part I (FROZEN 2026-08-05)

Reference baseline per family = the strongest single-call structure:
`direct` for code, `cot` for math (fixed by dev results, stated here before
any test-set run). Contrast structure = `verify_refine_3` (vanilla) and
`incumbent_refine` / `incumbent_refine_cot` (incumbent-protecting).

- **C1 (net-zero of vanilla structure).** At `loose`, the success difference
  vanilla − baseline has a 95% CI containing 0 in BOTH families, while the
  same comparison consumes ≥2× the dollars per task. Directional prediction:
  |difference| ≤ 0.05.
- **C2 (budget floor).** At `tight`, vanilla − baseline is significantly
  negative in the code family (95% CI upper bound < 0).
- **C3 (verifier-signal dose-response).** In the code family at `loose`,
  vanilla − baseline is monotone non-increasing across visible-test masking
  1.0 → 0.5 → 0.0, and is significantly negative at mask 0.0 (CI < 0).
- **C4 (incumbent protection).** In BOTH families and at ALL of
  {tight, unseen, loose}: (i) breakage rate of the incumbent-protecting
  structure on baseline-solved tasks ≤ 0.02, and (ii) success is
  non-inferior to the baseline at δ = 0.02 (one-sided 95% LCB ≥ −0.02).
  Additionally, in the code family the difference is significantly positive
  (CI > 0) at ≥2 of the 3 tiers.
- **C5 (repair is verifier-bounded).** Repair rate (fraction of
  baseline-failed tasks solved by the incumbent-protecting structure) is
  significantly higher in the code family (oracle tests) than in the math
  family (non-oracle self-check); math repair rate 95% CI includes 0.

Definitions fixed here: breakage rate = mean over tasks the baseline solves
(seed-averaged success = 1.0) of (1 − structure success); repair rate = mean
over tasks the baseline never solves (0.0) of structure success.

Failure handling: any claim whose CI does not meet the stated criterion is
reported as not supported, with the observed interval. No claim is restated
post hoc to fit the data.

## 2B. Superseded exploratory hypotheses (dev only, kept for the record)

- H-E1 (≥2 stable crossovers): NOT supported at full resolution (dev).
- H-E2 (degradation narrows the advantage): supported on dev; the
  confirmatory version is C3.
- H-E3 (reconciliation of the two literatures): reformulated as the
  repair/breakage decomposition (C1–C5); the reconciliation is now argued
  from the decomposition rather than from crossover positions.

## 3. Hypotheses — Part II (HBWS)

- H1a: HBWS AUBPC > Human Strong Workflow and > Random Search (paired
  bootstrap, Holm within the H1 family).
- H1b: Full HBWS > No-CB-objective ablation.
- H2a (non-inferiority, intersection–union): for BOTH tight and loose,
  one-sided 95% LCB of S(HBWS,b) − S(StaticA,b) ≥ −δ, δ = 0.03. Both must
  pass before any non-inferiority claim.
- H2b: at least one seen tier with Holm-corrected two-sided 95% CI lower
  bound > 0. Interpreted only if H2a passes.
- H2c (unseen $0.15): HBWS vs. the pre-selected static transfer control
  (choice rule: whichever of {tight-policy run at $0.15, loose-policy
  hard-capped at $0.15} scores higher on validation, selected once). Report
  differences with CI; confirmatory non-inferiority uses the same δ.
- H3a: median search dollars to reach the deployment-quality threshold
  (TBD after pilot, frozen at prereg-freeze) ≤ 60% of full-fidelity
  evolution, with final AUBPC non-inferior (δ = 3pt).
- H3b: under matched total search cap (Protocol B), HBWS AUBPC > Static
  Evolution Search with the preregistered S/2+S/2 split.
- H3c: amortization N* = ⌈(C_search − C_design)/Δc⌉ reported under 10/50/90%
  deployment mixtures; if Δc ≤ 0 or quality non-inferiority fails, N* = ∞.

## 4. Statistics

- Unit = task; stratified paired bootstrap, 10,000 resamples (code: by
  difficulty/source block; math: by subject). Execution seeds averaged
  within task before resampling. ≥3 execution seeds per frozen candidate;
  3 independent search seeds for HBWS / Static / Random (all reported).
- Holm correction within each hypothesis family (H1, H2, H3). Part I trend
  tests corrected within H-E family. OOD and mechanism analyses are
  exploratory unless listed above.
- Effect sizes (success-rate difference, $/success difference) always
  reported alongside intervals.
- Search-phase candidate selection uses conservative LCBs (empirical-
  Bernstein if implemented by prereg-freeze, else Hoeffding — recorded
  here, not switched afterwards). Confirmatory CIs come only from frozen
  test sets.

## 5. Feasibility gates (search-phase, TBD after pilot)

- Deployability floors q_{d,b}: frozen after the Direct/CoT baseline runs,
  before any search starts. Placeholder: TBD.
- Fidelity ladder: F1 = dev[:24]×1 seed, F2 = dev[:64]×2, F3 = dev[:120]×3.

## 6. Case selection (anti-cherry-picking)

Failure taxonomy fixed in advance: wrong-verify-signal, premature budget
exhaustion, over-conservative reservation, non-converging loop, homogeneous
vote error, aggregation-overwrites-correct-incumbent, OOD routing failure,
external API/sandbox failure. Cases: 3 successes (one each from
low-budget-early-stop, high-budget-verify-correct, unseen-budget-smooth-
fallback; nearest to cluster center) + 3 failures (top-3 most frequent
classes, representative by frequency). Full anonymized traces published.

## 7. Cost controls

Global hard cap $2,000 (80% warn / 90% freeze optional items / 100% break).
Price table: Azure GPT-4o, $2.50/M in, $10.00/M out (verify + archive by
prereg-freeze). Reduction order if pilot P95 projects >$1,800: Tier-3
optionals → secondary ablations → ID test 150→100. Never cut: core
baselines, 3 search seeds, $0.15 unseen evaluation.

## 8. Deviations from v1.0/v2.0 plans (recorded before freeze)

- 2026-08-04: one-time authorized data resplit 80/50/150 → 120/40/150 and
  MATH L3–5 → L4–5, before any real search run (v4.0 §5.1).
- 2026-08-04: "AgentEvo" citation supplied by an external review agent was
  checked (DOI unindexed, name unfindable) and treated as unverifiable;
  nearest real neighbors EvoFlow/EvoAgentX/FlowEvo added to related work.
- 2026-08-05: confirmatory dev-set results recorded BEFORE freeze: H-E1 as
  originally stated fails at full resolution; H-E2 confirmed (monotone,
  no-signal harm CI<0). Envelope library extended with incumbent_refine
  and incumbent_refine_cot (dated additions), motivated by the
  repair/breakage decomposition. Part-I confirmatory hypotheses for the
  FROZEN TEST SET will be restated at freeze around: (a) net-zero of
  vanilla structures, (b) budget floor harm, (c) signal-degradation harm,
  (d) incumbent-protection dominance. Dev-set numbers are exploratory and
  will not be reported as confirmatory.

## 9. Post-freeze deviation log

(append-only)

- **2026-08-05 — Part-I confirmatory run executed once on the frozen test
  split (150 tasks/family, 3 execution seeds). Verdicts, recorded verbatim
  with no post-hoc restatement:**
  - C1 SUPPORTED. Vanilla verify-refine vs baseline @ loose: code
    −0.033 [−0.087, +0.020] at 5.5× the cost; math +0.016 [−0.016, +0.047]
    at 2.2× the cost. Net-zero quality for multiplied spend, both families.
  - C2 SUPPORTED. Code @ tight: −0.104 [−0.162, −0.047].
  - C3 SUPPORTED. Code @ loose across visible-test masking 1.0 → 0.5 → 0.0:
    −0.033, −0.033, −0.118 [−0.178, −0.058]; monotone and significantly
    harmful with no signal.
  - C4 SUPPORTED. Incumbent protection: breakage 0.000 in every family ×
    tier cell; code significant at 3/3 tiers (+0.013, +0.020, +0.020);
    math non-inferior at all tiers and significant at loose (+0.011
    [+0.002, +0.022]).
  - **C5 NOT SUPPORTED.** Predicted: oracle-verified repair strictly
    exceeds non-oracle repair, with the math CI containing 0. Observed:
    code repair 0.060 [0.000, 0.137] (n=39), math repair 0.036
    [0.009, 0.072] (n=37). The intervals overlap and the math interval
    EXCLUDES zero — non-oracle self-check does produce real repair, and the
    oracle/non-oracle gap is not resolvable at this sample size. The dev
    split had shown math repair = 0.000; that was not reproduced. No
    claim about verifier-type-bounded repair will be made in the paper
    beyond the observed intervals.

- **2026-08-05 (Part II, still pre-freeze) — search-space defect found and
  fixed by measurement, not by outcome-peeking.** A budget-contingent policy
  sees normalized remaining budget, which equals 1.0 in every tier at task
  start; tiers differ only in how fast a call consumes the budget. Probe
  (`scripts/probe_budget_signal.py`, zero API cost) measured the min
  remaining fraction after generate: tight 0.616, unseen 0.744, loose 0.808.
  The mutation operators' threshold grid was [0.3, 0.4, 0.5, 0.6] — every
  value fires identically in all tiers, so the policy class provably could
  not express tier-dependent behaviour, which is the premise of Part II.
  Grid extended to [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]; the
  values 0.65-0.8 separate tiers. Searches run before this fix
  (A_hbws_code_s0 and its static counterparts) are superseded and will be
  reported, if at all, only as a search-space ablation.
