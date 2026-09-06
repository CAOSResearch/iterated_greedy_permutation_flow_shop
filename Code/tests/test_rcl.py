"""Tests for the Restricted Candidate List builder (``build_rcl``).

Covers (spec destruccion-inteligente, Req 3.1, 3.2, 3.3, 7.4, 12.2):
- RCL size equals ``max(d, ceil(alpha * n))`` (Req 3.1, 7.4).
- Every RCL member is among the highest-Idle_Score positions (Req 3.1).
- Deterministic tie-breaking by lowest position index (Req 3.3).
- ``RCL >= d`` even when ``alpha * n < d`` (Req 3.2, 7.4).
- Determinism and input immutability.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pfsp.core.destruction import build_rcl


class TestRclSize:
    """The RCL size follows ``max(d, ceil(alpha * n))`` (Req 3.1, 7.4)."""

    @pytest.mark.parametrize(
        "n,alpha,d",
        [
            (10, 0.10, 4),  # ceil(1.0) = 1 -> max(4, 1) = 4
            (10, 0.50, 2),  # ceil(5.0) = 5 -> max(2, 5) = 5
            (20, 0.30, 4),  # ceil(6.0) = 6 -> max(4, 6) = 6
            (7, 0.20, 3),  # ceil(1.4) = 2 -> max(3, 2) = 3
            (100, 1.00, 4),  # ceil(100) = 100 -> max(4, 100) = 100
        ],
    )
    def test_size_rule(self, n: int, alpha: float, d: int) -> None:
        """RCL length equals max(d, ceil(alpha * n))."""
        idle = np.arange(n, dtype=float)
        rcl = build_rcl(idle, alpha, d)

        expected_size = max(d, math.ceil(alpha * n))
        assert len(rcl) == expected_size

    def test_members_are_distinct_positions(self) -> None:
        """The RCL holds distinct, valid position indices."""
        idle = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0], dtype=float)
        rcl = build_rcl(idle, 0.5, 2)

        assert len(rcl) == len(set(rcl))
        assert all(0 <= pos < len(idle) for pos in rcl)


class TestRclHighestIdle:
    """Every RCL member is among the highest-Idle_Score positions (Req 3.1)."""

    def test_members_are_top_scored(self) -> None:
        """No excluded position has a higher score than an included one."""
        idle = np.array([10.0, 2.0, 8.0, 5.0, 1.0, 9.0, 3.0], dtype=float)
        d = 2
        alpha = 0.4  # ceil(2.8) = 3 -> size = max(2, 3) = 3

        rcl = build_rcl(idle, alpha, d)

        in_rcl = set(rcl)
        min_score_in = min(idle[pos] for pos in rcl)
        max_score_out = max(
            (idle[pos] for pos in range(len(idle)) if pos not in in_rcl),
            default=-np.inf,
        )
        # Every included score is >= every excluded score.
        assert min_score_in >= max_score_out

    def test_ordered_by_descending_idle(self) -> None:
        """The RCL is returned in descending Idle_Score order."""
        idle = np.array([1.0, 5.0, 3.0, 9.0, 2.0, 7.0], dtype=float)
        rcl = build_rcl(idle, 1.0, 1)  # full ranking

        scores = [idle[pos] for pos in rcl]
        assert scores == sorted(scores, reverse=True)
        # Highest score (9.0 at index 3) comes first.
        assert rcl[0] == 3


class TestRclTieBreaking:
    """Ties resolved deterministically by lowest position index (Req 3.3)."""

    def test_tie_break_lowest_index(self) -> None:
        """Equal-score positions are ordered by ascending index."""
        # All equal scores: the ranking must be 0, 1, 2, ... in index order.
        idle = np.zeros(6, dtype=float)
        rcl = build_rcl(idle, 1.0, 1)

        assert rcl == [0, 1, 2, 3, 4, 5]

    def test_tie_break_at_boundary(self) -> None:
        """A tie straddling the RCL boundary picks the lower index."""
        # Positions 1 and 3 both score 5.0; with size 2 (top score 9.0 at 2,
        # then the first 5.0), the lower index (1) must be chosen over 3.
        idle = np.array([1.0, 5.0, 9.0, 5.0], dtype=float)
        rcl = build_rcl(idle, 0.5, 1)  # ceil(2.0) = 2 -> size 2

        assert rcl == [2, 1]

    def test_deterministic_across_calls(self) -> None:
        """Repeated calls return the identical RCL (determinism)."""
        idle = np.array([4.0, 4.0, 4.0, 1.0, 4.0], dtype=float)
        first = build_rcl(idle, 0.6, 2)
        second = build_rcl(idle, 0.6, 2)

        assert first == second


class TestRclMinimumSize:
    """RCL always has at least ``d`` members (Req 3.2, 7.4)."""

    @pytest.mark.parametrize(
        "n,alpha,d",
        [
            (10, 0.05, 4),  # ceil(0.5) = 1 < d -> size = d = 4
            (10, 0.10, 5),  # ceil(1.0) = 1 < d -> size = d = 5
            (5, 0.10, 4),  # ceil(0.5) = 1 < d -> size = d = 4
            (8, 0.01, 3),  # ceil(0.08) = 1 < d -> size = d = 3
        ],
    )
    def test_rcl_at_least_d_when_alpha_small(
        self, n: int, alpha: float, d: int
    ) -> None:
        """When alpha * n < d, the RCL still has exactly d members."""
        idle = np.arange(n, dtype=float)
        rcl = build_rcl(idle, alpha, d)

        assert len(rcl) >= d
        assert len(rcl) == d


class TestRclImmutability:
    """The input Idle_Score array is not mutated."""

    def test_input_not_mutated(self) -> None:
        """build_rcl leaves the idle_scores array unchanged."""
        idle = np.array([3.0, 1.0, 4.0, 1.0, 5.0], dtype=float)
        original = idle.copy()

        build_rcl(idle, 0.4, 2)

        assert np.array_equal(idle, original)
