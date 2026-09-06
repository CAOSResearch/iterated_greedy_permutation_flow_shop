"""Tests for the per-position idle-time score (``idle_scores``).

These tests cover Requisitos 2.2-2.5 and 12.1:

* 2.2 / 2.3 — The score is derived from the completion-time matrix and equals
  ``sum over machines i >= 2 of max(0, C[i-1][k] - C[i][k-1])`` with
  ``C[i][-1] = 0`` for the first position (ramp-up convention). The first
  position convention is anchored by a hand-computed value.
* 2.4 — Non-negativity and input immutability.
* 2.5 — Determinism (no randomness in the score).

The manual idle values below are derived from the completion-time matrices used
in ``test_completion_matrix.py``.
"""

from __future__ import annotations

import numpy as np

from pfsp.core.destruction import idle_scores


def test_idle_scores_manual_2x3_identity() -> None:
    """2 machines x 3 jobs -> idle scores ``[3, 0, 0]`` (by hand).

    Completion matrix ``C = [[3, 4, 8], [5, 10, 11]]``. With a single downstream
    machine (i = 1, 0-indexed):

        idle(0) = C[0][0]                 = 3   (ramp-up, C[1][-1] = 0)
        idle(1) = max(0, C[0][1] - C[1][0]) = max(0, 4 - 5)  = 0
        idle(2) = max(0, C[0][2] - C[1][1]) = max(0, 8 - 10) = 0
    """
    processing_times = np.array([[3, 1, 4], [2, 5, 1]], dtype=np.int64)

    result = idle_scores([0, 1, 2], processing_times)

    assert result.shape == (3,)
    assert np.array_equal(result, np.array([3, 0, 0], dtype=np.int64))


def test_idle_scores_manual_3x4_identity_including_position_zero() -> None:
    """3 machines x 4 jobs -> idle scores ``[14, 2, 0, 0]`` (by hand).

    Completion matrix::

        C(0,*) = [ 5, 10, 13, 19]
        C(1,*) = [ 9, 13, 19, 21]
        C(2,*) = [12, 19, 21, 26]

    Summing the front delay over the two downstream machines (i = 1, 2):

        idle(0) = C[0][0] + C[1][0]                       = 5 + 9 = 14
        idle(1) = max(0, 10 - 9)  + max(0, 13 - 12)        = 1 + 1 = 2
        idle(2) = max(0, 13 - 13) + max(0, 19 - 19)        = 0
        idle(3) = max(0, 19 - 19) + max(0, 21 - 21)        = 0

    The ``idle(0) = 14`` value pins down the first-position (ramp-up)
    convention ``C[i][-1] = 0``.
    """
    processing_times = np.array(
        [[5, 5, 3, 6], [4, 3, 6, 2], [3, 6, 2, 5]],
        dtype=np.int64,
    )

    result = idle_scores([0, 1, 2, 3], processing_times)

    assert np.array_equal(result, np.array([14, 2, 0, 0], dtype=np.int64))


def test_idle_scores_non_negative() -> None:
    """Every per-position idle score is non-negative (Req 2.4)."""
    rng = np.random.default_rng(20260716)
    processing_times = rng.integers(1, 99, size=(6, 9), dtype=np.int64)
    permutation = rng.permutation(9)

    result = idle_scores(permutation, processing_times)

    assert np.all(result >= 0)


def test_idle_scores_does_not_mutate_inputs() -> None:
    """Neither the processing matrix nor the permutation is mutated (Req 2.4)."""
    processing_times = np.array([[5, 5, 3, 6], [4, 3, 6, 2]], dtype=np.int64)
    permutation = np.array([2, 0, 3, 1], dtype=np.int64)
    matrix_before = np.copy(processing_times)
    perm_before = np.copy(permutation)

    idle_scores(permutation, processing_times)

    assert np.array_equal(processing_times, matrix_before)
    assert np.array_equal(permutation, perm_before)


def test_idle_scores_deterministic() -> None:
    """Computing the score twice yields identical values (Req 2.5)."""
    rng = np.random.default_rng(7)
    processing_times = rng.integers(1, 50, size=(4, 7), dtype=np.int64)
    permutation = rng.permutation(7)

    first = idle_scores(permutation, processing_times)
    second = idle_scores(permutation, processing_times)

    assert np.array_equal(first, second)


def test_idle_scores_single_machine_is_all_zero() -> None:
    """With a single machine there is no downstream front delay: all zeros.

    The sum runs over machines ``i >= 2``; a 1-machine shop has none, so every
    position scores 0.
    """
    processing_times = np.array([[4, 7, 2, 9]], dtype=np.int64)

    result = idle_scores([0, 1, 2, 3], processing_times)

    assert np.array_equal(result, np.zeros(4, dtype=np.int64))
