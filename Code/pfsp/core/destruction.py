"""Destruction operators and idle-time scoring for the Iterated Greedy algorithm.

The random destruction operator (:func:`destruct`) selects ``d`` distinct jobs
uniformly at random from a permutation and removes them, returning the remaining
partial sequence (preserving relative order) and the removed jobs in selection
order. It operates on a copy and does not mutate the input permutation. This is
the destruction phase of the classic IG algorithm by Ruiz & Stuetzle (2005). All
randomness is drawn from a seeded NumPy generator provided by
:func:`pfsp.repro.seeds.make_rng`, so the operation is fully reproducible under
the same seed.

The module also provides the building blocks of the guided (intelligent)
destruction: :func:`idle_scores`, the per-job idle-time score, and
:func:`build_rcl`, the Restricted Candidate List built from those scores. The
idle score is derived from the completion-time matrix already computed for the
makespan (see :func:`pfsp.core.makespan.completion_matrix`), so it is
essentially free at ``O(n*m)`` and runs no separate makespan recurrence. Idle
time is used purely as a selection signal, never as a second optimized
objective.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from pfsp.core.makespan import completion_matrix

# A destruction operator: removes ``d`` jobs from a permutation and returns the
# remaining partial sequence (relative order preserved) and the ordered list of
# removed jobs. Guided operators fix the processing times (and Alpha) in a
# closure so they match this call site.
DestructFn = Callable[
    [list[int], int, np.random.Generator],
    "tuple[list[int], list[int]]",
]


def idle_scores(
    permutation: NDArray[np.int64] | list[int],
    processing_times: NDArray[np.int64],
) -> NDArray[np.int64]:
    """Compute the per-position idle-time score of a permutation.

    The score of the job at processing position ``k`` is the *front delay* it
    induces, accumulated across machines::

        idle(k) = sum over machines i >= 2 of max(0, C[i-1][k] - C[i][k-1])

    where ``C`` is the completion-time matrix (see
    :func:`pfsp.core.makespan.completion_matrix`) and ``C[i][k-1]`` for the
    first position (``k = 0``) is taken as ``0`` by convention. With that
    convention the first position accumulates the *ramp-up* idle
    ``idle(0) = sum over machines i >= 2 of C[i-1][0]`` (the time the downstream
    machines sit idle while the very first job climbs through the shop), which
    tends to be large. The guided destruction mitigates this dominance with an
    RCL plus uniform sampling.

    The score is a pure function of the completion-time matrix that already
    underlies the makespan, so it costs ``O(n*m)`` and runs no separate makespan
    recurrence. It is deterministic and never mutates its inputs.

    Parameters
    ----------
    permutation:
        Sequence of the ``n`` job indices defining the processing order. Must be
        a valid permutation of ``0..n-1``. Not mutated.
    processing_times:
        Processing-time matrix of shape ``(m, n)``. Not mutated.

    Returns
    -------
    numpy.ndarray
        A non-negative ``int64`` array of shape ``(n,)`` whose entry ``k`` is
        the idle score of the job at position ``k`` (position-indexed, not
        job-indexed).

    Raises
    ------
    ValueError
        Propagated from :func:`completion_matrix` when the processing-time
        matrix or the permutation is invalid.
    """
    completion = completion_matrix(processing_times, permutation)

    # Vectorized: max(0, C[i-1][k] - C[i][k-1]) for machines i = 1..m-1
    # (0-indexed), all positions k, summed over machines.
    # - Predecessor completion times C[i-1][k] are the rows 0..m-2 (``C[:-1]``).
    # - Previous-position completion times C[i][k-1] are the rows 1..m-1 shifted
    #   one column right, with the k = 0 column set to 0 (ramp-up convention,
    #   ``C[i][-1] = 0``).
    previous_position = np.zeros_like(completion)
    previous_position[:, 1:] = completion[:, :-1]

    front_delay = np.maximum(0, completion[:-1, :] - previous_position[1:, :])
    scores: NDArray[np.int64] = front_delay.sum(axis=0)
    return scores


def build_rcl(idle_scores: np.ndarray, alpha: float, d: int) -> list[int]:
    """Build the Restricted Candidate List (RCL) of positions to destroy.

    The RCL collects the positions with the highest Idle_Score, so that the
    guided destruction operator concentrates removals on the jobs that induce
    the most front delay while still leaving room for diversification via
    uniform sampling.

    Parameters
    ----------
    idle_scores:
        Per-position Idle_Score array of shape ``(n,)`` (non-negative). The
        value at index ``k`` is the front delay induced by the job currently at
        position ``k``. Not mutated.
    alpha:
        Fraction that sets the RCL size (``0 < alpha <= 1``). Validation is the
        caller's responsibility (``iterated_greedy`` / ``IGConfig`` validate it
        before the run starts).
    d:
        Destruction_Size, the number of jobs that will be sampled from the RCL.
        The RCL is guaranteed to contain at least ``d`` positions.

    Returns
    -------
    list[int]
        The RCL as a list of ``size = max(d, ceil(alpha * n))`` distinct
        positions, ordered by descending Idle_Score. Ties are broken
        deterministically by lowest position index, so the RCL is reproducible.

    Notes
    -----
    The size rule ``max(d, ceil(alpha * n))`` guarantees ``len(rcl) >= d`` even
    when ``alpha * n < d`` (small instances or small Alpha), so that ``d`` jobs
    can always be sampled without replacement. Sorting uses a stable argsort on
    the negated scores: stability makes equal-score positions keep their natural
    ascending-index order, which realizes the lowest-index tie-break.
    """
    n = len(idle_scores)
    size = max(d, math.ceil(alpha * n))

    # Descending Idle_Score, with a stable sort so that equal scores keep their
    # ascending position-index order (deterministic lowest-index tie-break).
    ranked = np.argsort(-idle_scores, kind="stable")

    return ranked[:size].tolist()


def destruct(
    permutation: list[int],
    destruction_size: int,
    rng: np.random.Generator,
) -> tuple[list[int], list[int]]:
    """Remove ``destruction_size`` distinct jobs chosen uniformly at random.

    Parameters
    ----------
    permutation:
        Current permutation of job indices. Not mutated.
    destruction_size:
        Number of jobs (``d``) to extract. Must satisfy ``1 <= d <= n - 1``
        but validation is the caller's responsibility (``iterated_greedy``
        validates before calling).
    rng:
        A seeded NumPy random generator (from ``Seed_Manager.make_rng``).

    Returns
    -------
    tuple[list[int], list[int]]
        A pair ``(partial_sequence, removed_jobs)`` where:
        - ``partial_sequence`` contains the ``n - d`` non-removed jobs in their
          original relative order.
        - ``removed_jobs`` contains the ``d`` extracted jobs in the order they
          were selected by the generator (deterministic under seed).

    Notes
    -----
    Selection is performed over **positions** (not job values) using
    ``rng.choice(n, size=d, replace=False)``. The removed jobs are returned in
    selection order — the order in which ``rng.choice`` picks them — which is
    deterministic for a given generator state. The partial sequence preserves
    the relative order of the remaining jobs.
    """
    n = len(permutation)

    # Select d positions uniformly at random (without replacement).
    selected_positions = rng.choice(n, size=destruction_size, replace=False)

    # Removed jobs in selection order.
    removed_jobs = [permutation[pos] for pos in selected_positions]

    # Build the partial sequence: jobs at non-selected positions, preserving
    # their relative order.
    removed_set = set(selected_positions.tolist())
    partial_sequence = [permutation[i] for i in range(n) if i not in removed_set]

    return partial_sequence, removed_jobs


def _partial_without(permutation: list[int], removed_positions: set[int]) -> list[int]:
    """Return the jobs at non-removed positions, preserving relative order."""
    return [job for pos, job in enumerate(permutation) if pos not in removed_positions]


def make_idle_rcl_destruct(
    processing_times: NDArray[np.int64],
    alpha: float,
) -> DestructFn:
    """Build the guided idle-RCL destruction operator O2 (the star contribution).

    The returned operator scores every position by the front delay it induces
    (:func:`idle_scores`), builds the Restricted Candidate List of the
    highest-idle positions (:func:`build_rcl`) and samples ``d`` of them
    **uniformly at random without replacement** with the seeded generator. This
    keeps the structural guidance of the idle signal while the uniform draw over
    the RCL adds the diversification that a purely greedy removal lacks
    (GRASP-style).

    The factory fixes the ``processing_times`` and ``alpha`` in a closure so the
    returned operator matches the classic destruction call site
    ``destruct_fn(permutation, d, rng)`` and can be swapped in without touching
    the IG loop.

    Parameters
    ----------
    processing_times:
        Processing-time matrix of shape ``(m, n)``. Captured by the closure and
        never mutated.
    alpha:
        Fraction that sets the RCL size (``0 < alpha <= 1``). Validation is the
        caller's responsibility (``IGConfig`` / ``iterated_greedy``).

    Returns
    -------
    DestructFn
        A ``destruct_fn(permutation, d, rng) -> (partial, removed)`` operator.
        ``partial`` holds the ``n - d`` non-removed jobs in their original
        relative order; ``removed`` holds the ``d`` sampled jobs in draw order.
        The operator never mutates its input permutation and is reproducible
        under a fixed generator state.

    Notes
    -----
    Idle time is used purely as a selection signal here, never as a second
    optimized objective. Only makespan is optimized by the IG.
    """

    def idle_rcl_destruct(
        permutation: list[int],
        destruction_size: int,
        rng: np.random.Generator,
    ) -> tuple[list[int], list[int]]:
        scores = idle_scores(permutation, processing_times)
        rcl = build_rcl(scores, alpha, destruction_size)

        # Uniform sampling without replacement from the RCL positions.
        selected_positions = rng.choice(rcl, size=destruction_size, replace=False)

        removed_jobs = [permutation[pos] for pos in selected_positions]
        partial_sequence = _partial_without(
            permutation, set(selected_positions.tolist())
        )
        return partial_sequence, removed_jobs

    return idle_rcl_destruct


def make_idle_greedy_destruct(
    processing_times: NDArray[np.int64],
) -> DestructFn:
    """Build the greedy idle destruction operator O1 (for the ablation).

    The returned operator removes the ``d`` positions with the highest
    Idle_Score, breaking ties deterministically by lowest position index. It is
    a pure greedy removal with no randomness in the selection; the ``rng``
    argument is kept only to match the destruction call site
    ``destruct_fn(permutation, d, rng)``. This operator exists to isolate, in
    the ablation, the effect of the RCL plus uniform sampling (O2) against a
    purely greedy choice, and is never the default operator.

    Parameters
    ----------
    processing_times:
        Processing-time matrix of shape ``(m, n)``. Captured by the closure and
        never mutated.

    Returns
    -------
    DestructFn
        A ``destruct_fn(permutation, d, rng) -> (partial, removed)`` operator.
        ``partial`` holds the ``n - d`` non-removed jobs in their original
        relative order; ``removed`` holds the ``d`` highest-idle jobs ordered by
        descending Idle_Score (ties by lowest position index). The operator
        never mutates its input permutation and is deterministic.

    Notes
    -----
    Idle time is used purely as a selection signal here, never as a second
    optimized objective. Only makespan is optimized by the IG.
    """

    def idle_greedy_destruct(
        permutation: list[int],
        destruction_size: int,
        rng: np.random.Generator,  # noqa: ARG001 - kept for call-site compatibility
    ) -> tuple[list[int], list[int]]:
        scores = idle_scores(permutation, processing_times)

        # Top-d positions by descending Idle_Score; the stable argsort on the
        # negated scores makes equal scores keep their ascending-index order
        # (deterministic lowest-index tie-break).
        ranked = np.argsort(-scores, kind="stable")
        selected_positions = ranked[:destruction_size]

        removed_jobs = [permutation[pos] for pos in selected_positions]
        partial_sequence = _partial_without(
            permutation, set(selected_positions.tolist())
        )
        return partial_sequence, removed_jobs

    return idle_greedy_destruct


__all__ = [
    "build_rcl",
    "destruct",
    "idle_scores",
    "make_idle_greedy_destruct",
    "make_idle_rcl_destruct",
]
