"""Tests for the completion-time matrix (``completion_matrix``).

These tests cover Requisito 2.1: ``makespan.py`` exposes a ``completion_matrix``
function returning the full ``(m, n)`` completion-time matrix ``C``, and
``makespan()`` is refactored to reuse it (the recurrence implemented once). The
manual matrices below are derived with the recurrence
``C(i, k) = max(C(i-1, k), C(i, k-1)) + p(i, pi_k)`` (see
``KB/03_fundamentos/pfsp-formal.md``) and mirror the small instances used in
``test_makespan.py`` so that ``C[-1, -1]`` matches the makespan.
"""

from __future__ import annotations

import numpy as np

from pfsp.core.makespan import completion_matrix, makespan


def test_completion_matrix_manual_2x3_identity() -> None:
    """2 machines x 3 jobs, identity permutation -> hand-computed matrix.

    Matrix ``p`` (rows = machines, columns = jobs)::

        machine 0: [3, 1, 4]
        machine 1: [2, 5, 1]

    Completion times for permutation ``[0, 1, 2]``::

        C(0,*) = [3,  4,  8]
        C(1,*) = [5, 10, 11]
    """
    processing_times = np.array([[3, 1, 4], [2, 5, 1]], dtype=np.int64)
    expected = np.array([[3, 4, 8], [5, 10, 11]], dtype=np.int64)

    result = completion_matrix(processing_times, [0, 1, 2])

    assert result.shape == (2, 3)
    assert np.array_equal(result, expected)


def test_completion_matrix_manual_3x4_identity() -> None:
    """3 machines x 4 jobs, identity permutation -> hand-computed matrix.

    Matrix ``p``::

        machine 0: [5, 5, 3, 6]
        machine 1: [4, 3, 6, 2]
        machine 2: [3, 6, 2, 5]

    Completion times for permutation ``[0, 1, 2, 3]``::

        C(0,*) = [ 5, 10, 13, 19]
        C(1,*) = [ 9, 13, 19, 21]
        C(2,*) = [12, 19, 21, 26]
    """
    processing_times = np.array(
        [[5, 5, 3, 6], [4, 3, 6, 2], [3, 6, 2, 5]],
        dtype=np.int64,
    )
    expected = np.array(
        [[5, 10, 13, 19], [9, 13, 19, 21], [12, 19, 21, 26]],
        dtype=np.int64,
    )

    result = completion_matrix(processing_times, [0, 1, 2, 3])

    assert result.shape == (3, 4)
    assert np.array_equal(result, expected)


def test_completion_matrix_last_entry_equals_makespan() -> None:
    """``C[-1, -1]`` must equal the makespan for the same permutation (Req 2.1).

    Checked on both the identity and a non-identity permutation of the 2x3
    instance so that the wiring holds for any reordering.
    """
    processing_times = np.array([[3, 1, 4], [2, 5, 1]], dtype=np.int64)

    for permutation in ([0, 1, 2], [2, 0, 1], [1, 2, 0]):
        matrix = completion_matrix(processing_times, permutation)
        assert int(matrix[-1, -1]) == makespan(processing_times, permutation)


def test_completion_matrix_does_not_mutate_input() -> None:
    """``completion_matrix`` reorders columns on a copy; the input is unchanged."""
    processing_times = np.array([[5, 5, 3, 6], [4, 3, 6, 2]], dtype=np.int64)
    original = np.copy(processing_times)

    completion_matrix(processing_times, [3, 1, 2, 0])

    assert np.array_equal(processing_times, original)
