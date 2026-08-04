# PREREGISTRATION — Budget–Structure Envelope & HBWS (DRAFT v0.9)

Status: DRAFT. Freezes upon the commit tagged `prereg-freeze` (target: 2026-08-10,
after the cost pilot fills the TBD thresholds). After that tag, changes are
allowed only as dated deviation records appended to §9 — never edits in place.

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
  total output cap (`run_envelope.clamp_to_tier`); applied identically to
  every structure and method.
- Verifier taxonomy: code = oracle environment feedback (visible asserts
  only; extended tests exist only in the external grader); math =
  non-oracle structured self-check (k independent gold-free re-derivations,
  majority agreement); final grading deterministic for both.
- Envelope structure library: the 8 structures in `hbws/dsl.py ENVELOPE_LIB`
  under their frozen names.
- Search budgets: {tight, loose} only. Policies read continuous remaining
  fractions and reserve-feasibility, never tier IDs.

## 2. Hypotheses — Part I (envelope)

- H-E1: In ≥1 task family, the best structure changes ≥2 times across the
  5-tier budget axis, and the crossover tier positions are stable (±1 tier)
  across 3 execution seeds.
- H-E2: Degrading verifier reliability (code: visible-assert masking
  1.0/0.5/0.25/0.0; math: critic k = 3/2/1) monotonically narrows the
  budget range where verify-refine structures are optimal (ordered bootstrap
  trend test).
- H-E3: Under this single protocol, low tiers reproduce the single-agent-
  dominant finding and high tiers with reliable verifiers reproduce the
  structure-dominant finding (each with 95% CI excluding 0).

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

## 9. Post-freeze deviation log

(append-only; empty until prereg-freeze)
