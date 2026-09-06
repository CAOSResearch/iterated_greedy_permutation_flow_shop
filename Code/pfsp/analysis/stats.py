"""Non-parametric statistics for the operator comparison (EXP-007).

This module provides the small, dependency-free statistical machinery used to
compare the destruction operators (random / idle-greedy / idle-rcl) on the same
instances: the **Friedman test** (an omnibus rank test for related samples) and
the **Nemenyi post-hoc** critical difference (Demšar, 2006, *Statistical
Comparisons of Classifiers over Multiple Data Sets*). It also exposes the average
ranks the Friedman test is built on.

The design constraint of the project is Python + NumPy only (no SciPy), so the
chi-square survival function needed for the Friedman p-value is implemented here
with the regularized upper incomplete gamma function (a standard, well-defined
computation, not a fabricated value). Only the makespan-derived quality metric
feeds these tests; nothing here invents experimental data — the caller supplies
the measured per-instance values.

All functions are pure and deterministic, so they are unit-testable in isolation
on synthetic inputs (no benchmark data required).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "FriedmanResult",
    "average_ranks",
    "chi_square_sf",
    "friedman_test",
    "nemenyi_critical_difference",
]


def chi_square_sf(x: float, df: int) -> float:
    """Return the chi-square survival function ``P(X > x)`` for ``df`` degrees.

    Equivalent to ``1 - CDF``; used to turn a Friedman statistic into a p-value
    without SciPy. Computed as the regularized upper incomplete gamma
    ``Q(df / 2, x / 2)``.

    Parameters
    ----------
    x:
        The chi-square statistic (``x >= 0``).
    df:
        Degrees of freedom (a positive integer).

    Returns
    -------
    float
        The upper-tail probability ``P(X > x)`` in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``df`` is not a positive integer or ``x`` is negative.
    """
    if df < 1:
        raise ValueError(f"df must be a positive integer, got {df}.")
    if x < 0:
        raise ValueError(f"x must be non-negative, got {x}.")
    if x == 0:
        return 1.0
    return _gammaincc(df / 2.0, x / 2.0)


def _gammaincc(s: float, x: float) -> float:
    """Return the regularized upper incomplete gamma ``Q(s, x)``.

    Uses the series expansion for the lower function when ``x < s + 1`` and the
    Lentz continued fraction for the upper function otherwise (the standard
    "Numerical Recipes" split), which keeps both branches numerically stable.
    """
    if x < 0 or s <= 0:
        raise ValueError("s must be positive and x non-negative.")
    if x == 0:
        return 1.0
    if x < s + 1.0:
        # Lower series P(s, x); Q = 1 - P.
        return 1.0 - _gamma_series(s, x)
    # Upper continued fraction Q(s, x) directly.
    return _gamma_continued_fraction(s, x)


def _gamma_series(
    s: float, x: float, *, max_iter: int = 1000, eps: float = 1e-14
) -> float:
    """Regularized lower incomplete gamma ``P(s, x)`` via its power series."""
    term = 1.0 / s
    total = term
    for n in range(1, max_iter):
        term *= x / (s + n)
        total += term
        if abs(term) < abs(total) * eps:
            break
    return total * math.exp(-x + s * math.log(x) - math.lgamma(s))


def _gamma_continued_fraction(
    s: float, x: float, *, max_iter: int = 1000, eps: float = 1e-14
) -> float:
    """Regularized upper incomplete gamma ``Q(s, x)`` via a Lentz continued fraction."""
    tiny = 1e-300
    b = x + 1.0 - s
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, max_iter):
        an = -i * (i - s)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return math.exp(-x + s * math.log(x) - math.lgamma(s)) * h


def average_ranks(data: list[list[float]]) -> list[float]:
    """Return the average rank of each treatment across the blocks.

    ``data`` is a list of blocks (one per instance); each block is a list of the
    ``k`` treatments' scores in a fixed treatment order (lower is better, e.g.
    makespan or RPD). Within each block the scores are ranked ascending (rank 1 =
    best); tied scores share the average of the ranks they span. The returned
    list holds, per treatment, the mean of its per-block ranks.

    Parameters
    ----------
    data:
        A non-empty list of blocks, each a list of ``k >= 1`` numeric scores, all
        of the same length ``k``.

    Returns
    -------
    list[float]
        The ``k`` average ranks, in the treatment order of the input.

    Raises
    ------
    ValueError
        If ``data`` is empty or its blocks are empty or of unequal length.
    """
    if not data:
        raise ValueError("data must contain at least one block.")
    k = len(data[0])
    if k == 0:
        raise ValueError("each block must contain at least one treatment.")
    if any(len(block) != k for block in data):
        raise ValueError("all blocks must have the same number of treatments.")

    rank_sums = [0.0] * k
    for block in data:
        for treatment, rank in enumerate(_ranks_ascending(block)):
            rank_sums[treatment] += rank
    n_blocks = len(data)
    return [rank_sum / n_blocks for rank_sum in rank_sums]


def _ranks_ascending(scores: list[float]) -> list[float]:
    """Return the fractional (tie-averaged) ranks of ``scores``, rank 1 = lowest."""
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        # Extend the tie group over positions with an equal score.
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        # Average rank (1-based) shared by the tied positions.
        average = (i + j) / 2.0 + 1.0
        for position in order[i : j + 1]:
            ranks[position] = average
        i = j + 1
    return ranks


@dataclass(frozen=True)
class FriedmanResult:
    """Outcome of a Friedman test.

    Attributes
    ----------
    statistic:
        The Friedman chi-square statistic.
    dof:
        Degrees of freedom (``k - 1``).
    p_value:
        Upper-tail p-value from the chi-square approximation.
    average_ranks:
        The average rank per treatment (input order), the basis of the test.
    n_blocks:
        Number of blocks (instances) compared.
    k:
        Number of treatments (operators) compared.
    """

    statistic: float
    dof: int
    p_value: float
    average_ranks: list[float]
    n_blocks: int
    k: int


def friedman_test(data: list[list[float]]) -> FriedmanResult:
    """Run the Friedman test on ``data`` (blocks x treatments, lower is better).

    The Friedman statistic is
    ``chi2 = 12 * N / (k * (k + 1)) * sum_j (R_j - (k + 1) / 2) ** 2``
    where ``N`` is the number of blocks, ``k`` the number of treatments and
    ``R_j`` the average rank of treatment ``j``. Under the null hypothesis (all
    treatments equivalent) it is approximately chi-square distributed with
    ``k - 1`` degrees of freedom.

    Parameters
    ----------
    data:
        A non-empty list of blocks, each a list of ``k >= 2`` scores in a fixed
        treatment order (lower = better). Every block must have length ``k``.

    Returns
    -------
    FriedmanResult
        The statistic, degrees of freedom, p-value and average ranks.

    Raises
    ------
    ValueError
        If there are fewer than two treatments, or the blocks are empty or of
        unequal length.
    """
    ranks = average_ranks(data)
    k = len(ranks)
    if k < 2:
        raise ValueError("the Friedman test needs at least two treatments.")
    n_blocks = len(data)

    grand_mean = (k + 1) / 2.0
    ss = sum((rank - grand_mean) ** 2 for rank in ranks)
    statistic = 12.0 * n_blocks / (k * (k + 1)) * ss
    dof = k - 1
    p_value = chi_square_sf(statistic, dof)
    return FriedmanResult(
        statistic=statistic,
        dof=dof,
        p_value=p_value,
        average_ranks=ranks,
        n_blocks=n_blocks,
        k=k,
    )


#: Critical values ``q_alpha`` of the Nemenyi post-hoc test (studentized range
#: statistic divided by ``sqrt(2)``), indexed by the number of treatments ``k``.
#: Tabulated from Demšar (2006), Table 5, for the significance levels supported
#: here. Only small ``k`` (the operators compared in this study) are provided.
_NEMENYI_Q: dict[float, dict[int, float]] = {
    0.05: {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031},
    0.10: {2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459, 6: 2.589, 7: 2.693, 8: 2.780},
}


def nemenyi_critical_difference(k: int, n_blocks: int, alpha: float = 0.05) -> float:
    """Return the Nemenyi critical difference (CD) for average-rank comparison.

    Two treatments differ significantly (at level ``alpha``) when the absolute
    difference of their average ranks exceeds the CD
    ``q_alpha * sqrt(k * (k + 1) / (6 * N))`` (Demšar, 2006). This is the
    standard post-hoc after a significant Friedman test.

    Parameters
    ----------
    k:
        Number of treatments (must have a tabulated critical value; here 2..8).
    n_blocks:
        Number of blocks (instances).
    alpha:
        Significance level; ``0.05`` or ``0.10`` are supported.

    Returns
    -------
    float
        The critical difference for the average ranks.

    Raises
    ------
    ValueError
        If ``alpha`` is unsupported, ``k`` has no tabulated value, or
        ``n_blocks`` is not positive.
    """
    if alpha not in _NEMENYI_Q:
        raise ValueError(f"Unsupported alpha {alpha}; supported: {sorted(_NEMENYI_Q)}.")
    q_by_k = _NEMENYI_Q[alpha]
    if k not in q_by_k:
        raise ValueError(
            f"No tabulated Nemenyi q for k={k}; supported k: {sorted(q_by_k)}."
        )
    if n_blocks < 1:
        raise ValueError(f"n_blocks must be positive, got {n_blocks}.")
    q_alpha = q_by_k[k]
    return q_alpha * math.sqrt(k * (k + 1) / (6.0 * n_blocks))
