# PROSPECTIVE VALIDATION — predictions locked before any execution

Domain: BIG-Bench Hard (logical deduction, shuffled-object tracking, date
understanding), 120 tasks, multiple-choice, exact-match grading.
This domain has NEVER been touched by any search, tuning, selection, or
exploratory run in this project. Verifier: non-oracle self-consistency
check (independent re-derivation, gold-free), i.e. the same regime as the
math family.

Structures: `direct`, `cot`, `verify_refine_3`, `incumbent_refine`
(protected draft = solve_direct), `incumbent_refine_cot` (protected draft =
solve_cot). Tiers: tight ($0.10) and loose ($0.25). 3 execution seeds.

## What the framework predicts

The framework's transferable claim is STRUCTURAL, not quantitative: for
incumbent-protecting structures the breakage rate is driven to ~0 wherever
the verifier carries signal, independently of domain. Repair magnitude is
NOT predicted — the r parameter has been shown not to transfer.

- **PV1.** Breakage rate of each incumbent-protecting structure, measured
  against its own first-draft baseline, is <= 0.02 at BOTH tiers.
- **PV2.** Breakage rate of vanilla `verify_refine_3` is materially higher
  than the incumbent-protecting structures at both tiers (> 0.02, and at
  least 3x the incumbent value).
- **PV3.** The success delta of each incumbent-protecting structure vs its
  own baseline is >= -0.01 at both tiers (never materially harmful).
- **PV4.** No prediction is made about repair rate, about the absolute
  success levels, or about which of `direct`/`cot` is the stronger
  baseline in this domain.

## Falsification

PV1 fails if any incumbent breakage cell exceeds 0.02.
PV2 fails if vanilla breakage is <= 0.02 or below 3x the incumbent value.
PV3 fails if any incumbent delta is below -0.01.
Failures are reported as observed; no prediction will be restated.

Locked: 2026-08-05, before the first BBH execution.
