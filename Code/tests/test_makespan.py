"""Tests for the vectorized makespan (``C_max``) computation.

These tests cover Requisitos 6.1 and 6.7:

* 6.1 — Verify the Makespan_Calculator against small instances (at most 5 jobs
  and at most 3 machines) whose makespan is computed by hand and asserted as a
  fixed integer literal.
* 6.7 — Verify that an invalid Permutation (invalid permutation of the job
  indices, or wrong length) raises a descriptive ``ValueError`` and leaves the
  input Processing_Time_Matrix unchanged.

The manual values below are derived with the completion-time recurrence
``C(i, k) = max(C(i-1, k), C(i, k-1)) + p(i, pi_k)`` (see
``KB/03_fundamentos/pfsp-formal.md``).
"""

from __future__ import annotations

import numpy as np
import pytest

from pfsp.core.makespan import makespan

# --- Requisito 6.1: manual makespan on small instances ---------------------


def test_makespan_manual_2x3_identity_permutation() -> None:
    """2 machines x 3 jobs, identity permutation -> C_max = 11 (computed by hand).

    Matrix ``p`` (rows = machines, columns = jobs)::

        machine 0: [3, 1, 4]
        machine 1: [2, 5, 1]

    Completion times for permutation ``[0, 1, 2]``::

        C(0,0)=3  C(0,1)=4   C(0,2)=8
        C(1,0)=5  C(1,1)=10  C(1,2)=11

    so the makespan is ``C(1, 2) = 11``.
    """
    processing_times = np.array([[3, 1, 4], [2, 5, 1]], dtype=np.int64)

    assert makespan(processing_times, [0, 1, 2]) == 11


def test_makespan_manual_2x3_non_identity_permutation() -> None:
    """Same 2x3 matrix, permutation ``[2, 0, 1]`` -> C_max = 14 (by hand).

    Processing times in processing order (job 2, job 0, job 1)::

        machine 0: [4, 3, 1]
        machine 1: [1, 2, 5]

    Completion times::

        C(0,0)=4  C(0,1)=7  C(0,2)=8
        C(1,0)=5  C(1,1)=9  C(1,2)=14

    so the makespan is ``C(1, 2) = 14``.
    """
    processing_times = np.array([[3, 1, 4], [2, 5, 1]], dtype=np.int64)

    assert makespan(processing_times, [2, 0, 1]) == 14


def test_makespan_manual_3x4_identity_permutation() -> None:
    """3 machines x 4 jobs, identity permutation -> C_max = 26 (by hand).

    Matrix ``p`` (rows = machines, columns = jobs)::

        machine 0: [5, 5, 3, 6]
        machine 1: [4, 3, 6, 2]
        machine 2: [3, 6, 2, 5]

    Completion times for permutation ``[0, 1, 2, 3]``::

        C(0,*) = [ 5, 10, 13, 19]
        C(1,*) = [ 9, 13, 19, 21]
        C(2,*) = [12, 19, 21, 26]

    so the makespan is ``C(2, 3) = 26``.
    """
    processing_times = np.array(
        [[5, 5, 3, 6], [4, 3, 6, 2], [3, 6, 2, 5]],
        dtype=np.int64,
    )

    assert makespan(processing_times, [0, 1, 2, 3]) == 26


def test_makespan_single_machine_single_job() -> None:
    """Degenerate 1x1 instance: the makespan is the only processing time."""
    processing_times = np.array([[7]], dtype=np.int64)

    assert makespan(processing_times, [0]) == 7


# --- Requisito 6.7: invalid permutation raises and does not mutate ---------


def test_duplicate_permutation_raises_and_does_not_mutate() -> None:
    """A permutation with a repeated index raises ``ValueError`` and leaves
    the input matrix unchanged."""
    processing_times = np.array([[3, 1, 4], [2, 5, 1]], dtype=np.int64)
    original = np.copy(processing_times)

    with pytest.raises(ValueError, match="more than once"):
        makespan(processing_times, [0, 0, 1])

    assert np.array_equal(processing_times, original)


def test_out_of_range_permutation_raises_and_does_not_mutate() -> None:
    """A permutation with an out-of-range index raises ``ValueError`` and leaves
    the input matrix unchanged."""
    processing_times = np.array([[3, 1, 4], [2, 5, 1]], dtype=np.int64)
    original = np.copy(processing_times)

    with pytest.raises(ValueError, match="out of range"):
        makespan(processing_times, [0, 1, 3])

    assert np.array_equal(processing_times, original)


def test_wrong_length_permutation_raises_and_does_not_mutate() -> None:
    """A permutation whose length differs from ``n`` raises ``ValueError``
    reporting both lengths and leaves the input matrix unchanged."""
    processing_times = np.array([[3, 1, 4], [2, 5, 1]], dtype=np.int64)
    original = np.copy(processing_times)

    with pytest.raises(ValueError, match="supplied length 2, expected n = 3"):
        makespan(processing_times, [0, 1])

    assert np.array_equal(processing_times, original)
