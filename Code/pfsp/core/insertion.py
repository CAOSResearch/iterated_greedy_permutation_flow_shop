"""Shared insertion helper for NEH and the Iterated Greedy construction phase.

This module extracts the mechanics of inserting a job into a partial sequence at
the position that minimizes the makespan. Both the NEH heuristic and the IG
reconstruction/local-search operators share this logic (Req 4.4), avoiding
reimplementation.

Two implementations are provided, with the **same signature, return value and
tie-breaking** (lowest insertion index on ties):

* :func:`best_insertion` — the **accelerated** evaluation (Taillard, 1990). Using
  precomputed head (earliest completion) and tail matrices, it evaluates *all*
  ``L + 1`` insertion positions of one job into a length-``L`` sequence in
  ``O(L * m)`` instead of recomputing a full ``O(L * m)`` makespan per position
  (``O(L^2 * m)`` overall). This is the version used by NEH, the IG
  reconstruction and the local search, and it is what makes the IG tractable on
  large instances.
* :func:`best_insertion_naive` — the straightforward version that recomputes the
  makespan from scratch at every candidate position. Kept as the reference
  implementation for the equivalence tests and for the speedup study (EXP-005):
  the two must return identical results on every input.

Both go through / are consistent with :func:`pfsp.core.makespan.makespan`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pfsp.core.makespan import makespan


def best_insertion(
    sequence: list[int],
    job: int,
    processing_times: NDArray[np.int64],
) -> tuple[list[int], int]:
    """Insert ``job`` at the lowest-makespan position (Taillard-accelerated).

    Evaluates every insertion position with the head/tail acceleration of
    Taillard (1990): the earliest completion times of the current sequence
    (*head*) and the tail contribution of its suffix are precomputed once in
    ``O(L * m)``; the makespan of inserting ``job`` at each of the ``L + 1``
    positions is then obtained in ``O(L * m)`` total. Ties are broken by the
    lowest insertion index, identical to :func:`best_insertion_naive`.

    Parameters
    ----------
    sequence:
        Current partial permutation (may be empty). Not mutated.
    job:
        Job index to insert.
    processing_times:
        Full processing-time matrix of shape ``(m, n)`` for all ``n`` jobs.

    Returns
    -------
    tuple[list[int], int]
        A pair ``(new_sequence, makespan_value)`` with ``job`` inserted at the
        best position and the makespan of that sequence.

    Notes
    -----
    The returned ``makespan_value`` is exactly
    ``makespan(processing_times, new_sequence)`` by construction (the head/tail
    recurrence is the same completion-time recurrence). The equivalence with
    :func:`best_insertion_naive` (same sequence and makespan for every input,
    including tie-breaking) is enforced by the property-based tests.
    """
    p = np.asarray(processing_times)
    m = p.shape[0]
    pk = p[:, job].astype(np.int64)  # processing times of the job to insert, (m,)

    length = len(sequence)
    if length == 0:
        # Single job: makespan is the cumulative sum of its times over machines.
        return [job], int(np.cumsum(pk)[-1])

    ordered = p[:, sequence].astype(np.int64, copy=True)  # (m, L)

    # Head: e[i][r] = completion time of the r-th sequence job on machine i.
    head = _completion_matrix(ordered)  # (m, L)
    # Tail: q[i][r] = duration from starting the r-th job on machine i until the
    # whole suffix r..L-1 finishes on the last machine.
    tail = _completion_matrix(ordered[::-1, ::-1])[::-1, ::-1]  # (m, L)

    # Predecessor head for each insertion position r in 0..L: column r holds the
    # head completion of the job at index r-1 (zeros at position 0: no predecessor).
    pred_head = np.zeros((m, length + 1), dtype=np.int64)
    pred_head[:, 1:] = head
    # Suffix tail for each position r in 0..L: column r holds the tail of jobs
    # from index r onward (zeros at position L: nothing after the inserted job).
    suffix_tail = np.zeros((m, length + 1), dtype=np.int64)
    suffix_tail[:, :length] = tail

    # Completion time of the inserted job at every position, per machine.
    completion = np.empty((m, length + 1), dtype=np.int64)
    completion[0, :] = pred_head[0, :] + pk[0]
    for i in range(1, m):
        completion[i, :] = np.maximum(completion[i - 1, :], pred_head[i, :]) + pk[i]

    # Candidate makespan at every position, then pick the lowest index minimum.
    candidates = np.max(completion + suffix_tail, axis=0)  # (L+1,)
    position = int(np.argmin(candidates))  # first minimum -> lowest index
    new_sequence = sequence[:position] + [job] + sequence[position:]
    return new_sequence, int(candidates[position])


def _completion_matrix(ordered: NDArray[np.int64]) -> NDArray[np.int64]:
    """Return the completion-time matrix ``e`` of an ordered ``(m, L)`` block.

    ``e[i][r]`` is the completion time of the ``r``-th job on machine ``i`` under
    the flow-shop recurrence ``C(i, k) = max(C(i-1, k), C(i, k-1)) + p(i, k)``.
    Vectorized over jobs with the same ``cumsum`` + ``maximum.accumulate`` trick
    as :func:`pfsp.core.makespan.makespan` (the outer loop runs over the ``m``
    machines). The input is not mutated.
    """
    m = ordered.shape[0]
    completion = np.empty_like(ordered)
    previous_row = np.cumsum(ordered[0])
    completion[0] = previous_row
    for i in range(1, m):
        row_sum = np.cumsum(ordered[i])
        shifted_sum = row_sum - ordered[i]  # s(k-1), with s(-1) = 0
        previous_row = row_sum + np.maximum.accumulate(previous_row - shifted_sum)
        completion[i] = previous_row
    return completion


def best_insertion_naive(
    sequence: list[int],
    job: int,
    processing_times: NDArray[np.int64],
) -> tuple[list[int], int]:
    """Insert ``job`` at the lowest-makespan position (reference, no acceleration).

    Recomputes the makespan from scratch at each of the ``L + 1`` candidate
    positions (``O(L^2 * m)`` overall). Kept as the reference implementation for
    the equivalence tests and the EXP-005 speedup study; :func:`best_insertion`
    must return identical results. Ties are broken by the lowest insertion index
    (ascending scan with strict ``<``).

    Parameters
    ----------
    sequence:
        Current partial permutation (may be empty). Not mutated.
    job:
        Job index to insert.
    processing_times:
        Full processing-time matrix of shape ``(m, n)`` for all ``n`` jobs.

    Returns
    -------
    tuple[list[int], int]
        A pair ``(new_sequence, makespan_value)``.
    """
    best_seq: list[int] | None = None
    best_makespan: int | None = None

    for position in range(len(sequence) + 1):
        candidate = sequence[:position] + [job] + sequence[position:]
        candidate_makespan = _partial_makespan(processing_times, candidate)
        if best_makespan is None or candidate_makespan < best_makespan:
            best_makespan = candidate_makespan
            best_seq = candidate

    # Always set: at least one position exists (the empty sequence has position 0).
    assert best_seq is not None
    assert best_makespan is not None
    return best_seq, best_makespan


def _partial_makespan(processing_times: NDArray[np.int64], sequence: list[int]) -> int:
    """Makespan of an ordered partial job list via the makespan calculator.

    The full matrix is restricted to the columns of ``sequence`` (in order) and
    evaluated with the identity permutation, so that
    ``processing_times[:, sequence]`` already encodes the job order.
    """
    submatrix = processing_times[:, sequence]
    return makespan(submatrix, list(range(len(sequence))))
