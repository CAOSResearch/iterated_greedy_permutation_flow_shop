"""Vectorized makespan (``C_max``) computation for the PFSP.

The makespan of a permutation is computed with the standard completion-time
recurrence (see ``KB/03_fundamentos/pfsp-formal.md``)::

    C(i, k) = max(C(i-1, k), C(i, k-1)) + p(i, pi_k)

with ``C(1, 1) = p(1, pi_1)``, ``C(i, 1) = C(i-1, 1) + p(i, pi_1)`` and
``C(1, k) = C(1, k-1) + p(1, pi_k)``. The makespan is ``C(m, n)``.

The implementation is vectorized over jobs with NumPy: for each machine row the
completion times are obtained by combining the previous machine row with the
running prefix sum of the current row's processing times via ``np.cumsum`` and
``np.maximum``. The outer loop runs over the ``m`` machines (``m`` is typically
much smaller than ``n``). No Numba is used.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def completion_matrix(
    processing_times: NDArray[np.int64],
    permutation: Sequence[int],
) -> NDArray[np.int64]:
    """Compute the full completion-time matrix ``C`` of a permutation.

    This is the single implementation of the completion-time recurrence used
    across the package: :func:`makespan` returns ``C[-1, -1]`` and the
    idle-score computation consumes the whole matrix, so the recurrence is
    written once here.

    Parameters
    ----------
    processing_times:
        Processing-time matrix of shape ``(m, n)`` where
        ``processing_times[i][j]`` is the processing time of job ``j`` on
        machine ``i``. The matrix is never mutated: columns are reordered on a
        copy.
    permutation:
        Sequence of the ``n`` job indices defining the processing order. Must be
        a valid permutation of ``0..n-1`` (each index exactly once).

    Returns
    -------
    numpy.ndarray
        The completion-time matrix ``C`` of shape ``(m, n)`` and dtype
        ``int64``, where ``C[i][k]`` is the completion time of the job at
        processing position ``k`` on machine ``i``. The makespan is
        ``C[m - 1][n - 1]``.

    Raises
    ------
    ValueError
        If ``processing_times`` is not a 2-D array of shape ``(m, n)`` with
        ``m >= 1`` and ``n >= 1``; if the permutation length does not equal
        ``n``; or if the permutation is not a valid permutation of ``0..n-1``.
        In every error case the input matrix is left unchanged and no matrix is
        returned.
    """
    p = np.asarray(processing_times)

    # --- Validate the processing-time matrix shape (Req 4.7) ---
    if p.ndim != 2:
        raise ValueError(
            "processing_times must be a 2-D array of shape (m, n); "
            f"got an array with {p.ndim} dimension(s) and shape {p.shape}."
        )
    m, n = p.shape
    if m < 1 or n < 1:
        raise ValueError(
            "processing_times must have shape (m, n) with m >= 1 and n >= 1; "
            f"got shape {p.shape}."
        )

    # --- Validate the permutation (Reqs 4.5, 4.6) ---
    perm = np.asarray(permutation)
    if perm.ndim != 1:
        raise ValueError(
            "permutation must be a one-dimensional sequence of job indices; "
            f"got an array with {perm.ndim} dimension(s)."
        )
    if perm.shape[0] != n:
        raise ValueError(
            "permutation length does not match the number of jobs: "
            f"supplied length {perm.shape[0]}, expected n = {n}."
        )
    if not np.issubdtype(perm.dtype, np.integer):
        raise ValueError(
            "permutation must contain integer job indices; " f"got dtype {perm.dtype}."
        )
    # A length-n array is a valid permutation of 0..n-1 iff sorting it yields
    # exactly arange(n). Identify the first offending index for a descriptive
    # error (out-of-range value or duplicate / missing index).
    expected = np.arange(n)
    if not np.array_equal(np.sort(perm), expected):
        out_of_range = perm[(perm < 0) | (perm >= n)]
        if out_of_range.size > 0:
            raise ValueError(
                "permutation is not a valid permutation of 0..n-1: "
                f"index {int(out_of_range[0])} is out of range for n = {n}."
            )
        counts = np.bincount(perm, minlength=n)
        duplicated = np.nonzero(counts > 1)[0]
        raise ValueError(
            "permutation is not a valid permutation of 0..n-1: "
            f"index {int(duplicated[0])} appears more than once."
        )

    # --- Vectorized completion-time recurrence (Reqs 4.1, 4.2, 4.4, 4.8) ---
    # Reorder columns on a copy so the input matrix is never mutated.
    ordered = p[:, perm].astype(np.int64, copy=True)

    # The row recurrence C(i, k) = max(C(i-1, k), C(i, k-1)) + p(i, pi_k) is
    # vectorized over jobs as follows. For machine row ``i`` let ``q`` be its
    # processing times, ``s = cumsum(q)`` and ``a`` the previous machine row.
    # Writing ``D(k) = C(i, k) - s(k)`` turns the recurrence into a cumulative
    # maximum: ``D(k) = max(D(k-1), a(k) - s(k-1))``, so
    # ``C(i, :) = cumsum(q) + maximum.accumulate(a - (cumsum(q) - q))``.
    # The first machine has no predecessor, so its row is just the prefix sum.
    completion = np.empty((m, n), dtype=np.int64)
    completion[0] = np.cumsum(ordered[0])
    for i in range(1, m):
        row_sum = np.cumsum(ordered[i])
        shifted_sum = row_sum - ordered[i]  # s(k-1), with s(-1) = 0
        completion[i] = row_sum + np.maximum.accumulate(completion[i - 1] - shifted_sum)

    return completion


def makespan(
    processing_times: NDArray[np.int64],
    permutation: Sequence[int],
) -> int:
    """Compute the makespan ``C_max`` of a permutation.

    Thin wrapper over :func:`completion_matrix`: the makespan is the completion
    time of the last job on the last machine, ``C[m - 1][n - 1]``. The
    completion-time recurrence lives in :func:`completion_matrix` so it is
    implemented exactly once.

    Parameters
    ----------
    processing_times:
        Processing-time matrix of shape ``(m, n)`` where
        ``processing_times[i][j]`` is the processing time of job ``j`` on
        machine ``i``. The matrix is never mutated.
    permutation:
        Sequence of the ``n`` job indices defining the processing order. Must be
        a valid permutation of ``0..n-1`` (each index exactly once).

    Returns
    -------
    int
        The makespan ``C_max`` as a single non-negative integer.

    Raises
    ------
    ValueError
        Propagated from :func:`completion_matrix` when the processing-time
        matrix or the permutation is invalid. In every error case the input
        matrix is left unchanged and no makespan is returned.
    """
    completion = completion_matrix(processing_times, permutation)
    return int(completion[-1, -1])
