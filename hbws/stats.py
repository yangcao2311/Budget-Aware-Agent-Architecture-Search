"""Paired bootstrap CIs for method comparisons (pre-registered protocol §6).

Both result lists must be aligned per task (same task order, same split).
"""
from __future__ import annotations

import math
import random


def binomial_exact_upper(
    events: int,
    trials: int,
    *,
    alpha: float = 0.05,
) -> float:
    """One-sided Clopper--Pearson upper limit for a binomial proportion.

    The returned value ``u`` solves ``Pr[Binomial(trials, u) <= events] =
    alpha`` when ``events < trials``.  This construction requires a binomial
    sampling model; unlike :func:`cluster_ratio_upper`, it does not cover
    fractional within-task masses or a fixed-stratum design without an
    additional argument connecting that design to a common Bernoulli rate.
    """
    if (not isinstance(events, int) or not isinstance(trials, int) or
            trials <= 0 or not 0 <= events <= trials):
        raise ValueError("expected integer counts satisfying 0 <= events <= trials")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    if events == trials:
        return 1.0

    def cdf(probability: float) -> float:
        if probability <= 0.0:
            return 1.0
        if probability >= 1.0:
            return 0.0
        log_p = math.log(probability)
        log_q = math.log1p(-probability)
        terms = [
            math.lgamma(trials + 1) - math.lgamma(index + 1) -
            math.lgamma(trials - index + 1) + index * log_p +
            (trials - index) * log_q
            for index in range(events + 1)
        ]
        maximum = max(terms)
        return math.exp(maximum) * sum(math.exp(term - maximum) for term in terms)

    lower = events / trials
    upper = 1.0
    for _ in range(80):
        midpoint = (lower + upper) / 2.0
        if cdf(midpoint) > alpha:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def cluster_ratio_upper(
    numerators: list[float],
    denominators: list[float],
    *,
    alpha: float = 0.05,
) -> float:
    """One-sided bound for ``E[N_t] / E[D_t]`` from independent task clusters.

    Each task contributes bounded sufficient statistics satisfying
    ``0 <= N_t <= D_t <= 1``.  Replicates or seeds may be dependent within a
    task because they are averaged before this function is called.  For
    ``theta = E[N_t] / E[D_t]``, Hoeffding's inequality applied to
    ``Z_t = N_t - theta D_t`` (whose range has width one) gives

        U = min(1, (mean(N_t) + sqrt(log(1/alpha)/(2n))) / mean(D_t)).

    Thus ``Pr(theta <= U) >= 1-alpha`` for i.i.d. representative task
    clusters.  Returning one when no denominator mass is observed makes the
    procedure abstain rather than manufacture a certificate.
    """
    if len(numerators) != len(denominators) or not numerators:
        raise ValueError("numerator and denominator must be non-empty and aligned")
    for num, den in zip(numerators, denominators):
        if not (0.0 <= num <= den <= 1.0):
            raise ValueError(f"expected 0 <= numerator <= denominator <= 1; got {num}, {den}")
    n = len(numerators)
    mean_den = sum(denominators) / n
    if mean_den == 0.0:
        return 1.0
    mean_num = sum(numerators) / n
    radius = math.sqrt(math.log(1.0 / alpha) / (2.0 * n))
    return min(1.0, (mean_num + radius) / mean_den)


def paired_bootstrap_diff(a: list[bool], b: list[bool], *, n_boot: int = 10000,
                          seed: int = 0) -> dict:
    """95% CI on success-rate difference (a - b), paired by task."""
    assert len(a) == len(b) and a
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(a[i] for i in idx) / n - sum(b[i] for i in idx) / n)
    diffs.sort()
    point = sum(a) / n - sum(b) / n
    lo, hi = diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot)]
    return {"diff": round(point, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "significant": lo > 0 or hi < 0}


def holm_correction(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm-Bonferroni: name -> reject_null."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    reject, still = {}, True
    for i, (name, p) in enumerate(items):
        still = still and p < alpha / (m - i)
        reject[name] = still
    return reject
