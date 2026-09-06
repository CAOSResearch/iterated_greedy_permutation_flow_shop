"""Tests for the guided destruction operators O2 (idle-RCL) and O1 (idle-greedy).

Covers (spec destruccion-inteligente):

* O2 ``make_idle_rcl_destruct`` (Req 4.1-4.4, 12.3, 13.6):
  - ``partial`` has ``n - d`` jobs and ``removed`` has ``d`` jobs (Req 4.1).
  - The union of ``partial`` and ``removed`` equals the original job set.
  - Removed jobs are sampled **only from the RCL** (Req 4.2, 13.6).
  - The relative order of the non-removed jobs is preserved (Req 4.3).
  - The input permutation is not mutated (Req 4.3).
  - The operator is reproducible under a fixed generator state (Req 4.4).

* O1 ``make_idle_greedy_destruct`` (Req 5.1-5.3, 12.4):
  - Removes the ``d`` highest-Idle_Score jobs, breaking ties by lowest position
    index (Req 5.1, 5.2).
  - Preserves relative order and does not mutate the input (Req 5.3).

The 3x4 instance below reuses the hand-computed idle scores ``[14, 2, 0, 0]``
from ``test_idle_scores.py`` so the greedy selection can be reasoned about by
hand, including the tie between the two zero-score positions.
"""

from __future__ import annotations

import numpy as np
import pytest

from pfsp.core.destruction import (
    build_rcl,
    idle_scores,
    make_idle_greedy_destruct,
    make_idle_rcl_destruct,
)
from pfsp.repro.seeds import make_rng

# Reusable instances -----------------------------------------------------------

# 3 machines x 4 jobs; identity permutation -> idle scores [14, 2, 0, 0].
PROC_3X4 = np.array(
    [[5, 5, 3, 6], [4, 3, 6, 2], [3, 6, 2, 5]],
    dtype=np.int64,
)

# A larger seeded instance for the sampling / partition properties.
_RNG = np.random.default_rng(20260716)
PROC_5X8 = _RNG.integers(1, 99, size=(5, 8), dtype=np.int64)


class TestIdleRclContract:
    """O2 satisfies the Phase 2 destruction contract (Req 4.1)."""

    @pytest.mark.parametrize(
        "n,d,alpha",
        [
            (8, 2, 0.5),
            (8, 4, 0.10),
            (8, 1, 0.30),
            (8, 7, 1.0),
        ],
    )
    def test_partition_sizes_and_union(self, n: int, d: int, alpha: float) -> None:
        """partial (n-d) + removed (d) union equals the original job set."""
        permutation = list(range(n))
        operator = make_idle_rcl_destruct(PROC_5X8, alpha)
        rng = make_rng(11)

        partial, removed = operator(permutation, d, rng)

        assert len(partial) == n - d
        assert len(removed) == d
        assert len(set(removed)) == d
        assert sorted(partial + removed) == sorted(permutation)


class TestIdleRclSamplesFromRcl:
    """O2 removes only jobs that belong to the RCL (Req 4.2, 13.6)."""

    def test_removed_are_rcl_members(self) -> None:
        """Every removed job sits at an RCL position, across many seeds."""
        permutation = [3, 1, 6, 0, 7, 2, 5, 4]
        d = 2
        alpha = 0.5  # ceil(0.5 * 8) = 4 -> RCL size max(2, 4) = 4 > d

        scores = idle_scores(permutation, PROC_5X8)
        rcl_positions = build_rcl(scores, alpha, d)
        rcl_jobs = {permutation[pos] for pos in rcl_positions}
        # Sanity: the RCL is a strict superset of what a single draw removes.
        assert len(rcl_jobs) == 4

        operator = make_idle_rcl_destruct(PROC_5X8, alpha)
        for seed in range(25):
            rng = make_rng(seed)
            _, removed = operator(permutation, d, rng)
            assert set(removed) <= rcl_jobs


class TestIdleRclRelativeOrder:
    """O2 preserves the relative order of the non-removed jobs (Req 4.3)."""

    def test_partial_preserves_relative_order(self) -> None:
        """Non-removed jobs keep their input order in the partial sequence."""
        permutation = [3, 1, 6, 0, 7, 2, 5, 4]
        operator = make_idle_rcl_destruct(PROC_5X8, 0.5)

        for seed in range(15):
            rng = make_rng(seed)
            partial, removed = operator(permutation, 3, rng)

            removed_set = set(removed)
            expected = [job for job in permutation if job not in removed_set]
            assert partial == expected


class TestIdleRclImmutability:
    """O2 does not mutate its inputs (Req 4.3)."""

    def test_input_permutation_not_mutated(self) -> None:
        """The input permutation list is unchanged after destruction."""
        permutation = [3, 1, 6, 0, 7, 2, 5, 4]
        original = permutation.copy()
        operator = make_idle_rcl_destruct(PROC_5X8, 0.5)

        operator(permutation, 3, make_rng(5))

        assert permutation == original

    def test_processing_matrix_not_mutated(self) -> None:
        """The captured processing-time matrix is unchanged after destruction."""
        matrix = PROC_5X8.copy()
        operator = make_idle_rcl_destruct(matrix, 0.5)

        operator([3, 1, 6, 0, 7, 2, 5, 4], 3, make_rng(5))

        assert np.array_equal(matrix, PROC_5X8)


class TestIdleRclReproducibility:
    """O2 is reproducible under a fixed generator state (Req 4.4)."""

    def test_same_seed_same_result(self) -> None:
        """Two identically-seeded runs return identical partial and removed."""
        permutation = [3, 1, 6, 0, 7, 2, 5, 4]
        operator = make_idle_rcl_destruct(PROC_5X8, 0.5)

        partial1, removed1 = operator(permutation, 3, make_rng(2025))
        partial2, removed2 = operator(permutation, 3, make_rng(2025))

        assert partial1 == partial2
        assert removed1 == removed2

    def test_different_seed_may_differ(self) -> None:
        """Different seeds select different jobs (with high probability)."""
        permutation = list(range(8))
        # Full-RCL O2 is equivalent to uniform random destruction, so distinct
        # seeds almost surely pick different subsets.
        operator = make_idle_rcl_destruct(PROC_5X8, 1.0)

        _, removed1 = operator(permutation, 3, make_rng(1))
        _, removed2 = operator(permutation, 3, make_rng(2))

        assert set(removed1) != set(removed2)


class TestIdleGreedy:
    """O1 removes the ``d`` highest-idle jobs with index tie-break (Req 5.1-5.3)."""

    def test_removes_top_d_by_idle(self) -> None:
        """Top-2 idle positions of the 3x4 instance are jobs 0 and 1."""
        # idle scores are [14, 2, 0, 0] for the identity permutation.
        operator = make_idle_greedy_destruct(PROC_3X4)

        partial, removed = operator([0, 1, 2, 3], 2, make_rng(0))

        assert removed == [0, 1]  # ordered by descending idle (14, 2)
        assert partial == [2, 3]

    def test_tie_break_by_lowest_index(self) -> None:
        """The boundary tie between the two zero-idle positions picks index 2."""
        # idle scores [14, 2, 0, 0]; top-3 must be positions 0, 1, 2 (not 3).
        operator = make_idle_greedy_destruct(PROC_3X4)

        partial, removed = operator([0, 1, 2, 3], 3, make_rng(0))

        assert removed == [0, 1, 2]
        assert partial == [3]

    def test_greedy_is_deterministic_regardless_of_seed(self) -> None:
        """O1 selection ignores the generator (pure greedy)."""
        operator = make_idle_greedy_destruct(PROC_3X4)

        _, removed_a = operator([0, 1, 2, 3], 2, make_rng(1))
        _, removed_b = operator([0, 1, 2, 3], 2, make_rng(999))

        assert removed_a == removed_b

    def test_preserves_relative_order_non_identity(self) -> None:
        """Partial keeps the input order for a non-identity permutation."""
        permutation = [2, 0, 3, 1]
        operator = make_idle_greedy_destruct(PROC_3X4)

        partial, removed = operator(permutation, 2, make_rng(0))

        removed_set = set(removed)
        expected = [job for job in permutation if job not in removed_set]
        assert partial == expected

    def test_input_not_mutated(self) -> None:
        """O1 does not mutate the input permutation or the processing matrix."""
        permutation = [2, 0, 3, 1]
        original = permutation.copy()
        matrix = PROC_3X4.copy()
        operator = make_idle_greedy_destruct(matrix)

        operator(permutation, 2, make_rng(0))

        assert permutation == original
        assert np.array_equal(matrix, PROC_3X4)

    def test_partition_round_trip(self) -> None:
        """partial (n-d) + removed (d) is a partition of the original set."""
        permutation = [3, 1, 6, 0, 7, 2, 5, 4]
        operator = make_idle_greedy_destruct(PROC_5X8)

        partial, removed = operator(permutation, 3, make_rng(0))

        assert len(partial) == 5
        assert len(removed) == 3
        assert sorted(partial + removed) == sorted(permutation)
