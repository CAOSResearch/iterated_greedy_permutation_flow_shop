"""Tests for the non-parametric statistics helper (pfsp.analysis.stats).

Covers the pure, dependency-free statistical machinery for the EXP-007 operator
comparison: average ranks (with tie handling), the Friedman statistic (checked
against a hand computation), the chi-square survival function (checked against
its closed form for two degrees of freedom, ``sf(x) = exp(-x / 2)``), and the
Nemenyi critical difference.
"""

from __future__ import annotations

import math

import pytest

from pfsp.analysis.stats import (
    average_ranks,
    chi_square_sf,
    friedman_test,
    nemenyi_critical_difference,
)


class TestChiSquareSf:
    def test_zero_statistic_is_one(self) -> None:
        assert chi_square_sf(0.0, 3) == 1.0

    def test_df2_matches_closed_form(self) -> None:
        # For df = 2 the chi-square survival function is exactly exp(-x / 2).
        for x in (0.5, 1.0, 2.5, 5.0, 9.21):
            assert chi_square_sf(x, 2) == pytest.approx(math.exp(-x / 2.0), rel=1e-9)

    def test_df1_known_tail(self) -> None:
        # chi2(df=1) at x = 3.841459 has an upper tail of ~0.05 (the 95% point).
        assert chi_square_sf(3.841459, 1) == pytest.approx(0.05, abs=1e-4)

    def test_monotone_decreasing(self) -> None:
        assert chi_square_sf(1.0, 3) > chi_square_sf(5.0, 3)

    def test_invalid_arguments(self) -> None:
        with pytest.raises(ValueError):
            chi_square_sf(1.0, 0)
        with pytest.raises(ValueError):
            chi_square_sf(-1.0, 2)


class TestAverageRanks:
    def test_simple_ranks_lower_is_better(self) -> None:
        # Two blocks, three treatments; lower score = rank 1.
        data = [[10.0, 20.0, 30.0], [5.0, 15.0, 25.0]]
        assert average_ranks(data) == [1.0, 2.0, 3.0]

    def test_ties_share_average_rank(self) -> None:
        # One block with a tie between the two best treatments: ranks 1 and 2
        # are averaged to 1.5 each; the worst gets rank 3.
        data = [[5.0, 5.0, 9.0]]
        assert average_ranks(data) == [1.5, 1.5, 3.0]

    def test_rejects_empty_and_ragged(self) -> None:
        with pytest.raises(ValueError):
            average_ranks([])
        with pytest.raises(ValueError):
            average_ranks([[1.0, 2.0], [1.0]])


class TestFriedman:
    def test_statistic_matches_hand_computation(self) -> None:
        # Operator B strictly best, A middle, C worst on every block:
        # average ranks are A=2, B=1, C=3 over N=4 blocks, k=3.
        # chi2 = 12N / (k(k+1)) * sum (R_j - (k+1)/2)^2
        #      = 12*4 / (3*4) * ((2-2)^2 + (1-2)^2 + (3-2)^2)
        #      = 4 * (0 + 1 + 1) = 8.0
        data = [
            [20.0, 10.0, 30.0],
            [21.0, 11.0, 31.0],
            [22.0, 12.0, 32.0],
            [23.0, 13.0, 33.0],
        ]
        result = friedman_test(data)
        assert result.average_ranks == [2.0, 1.0, 3.0]
        assert result.statistic == pytest.approx(8.0, rel=1e-12)
        assert result.dof == 2
        assert result.k == 3
        assert result.n_blocks == 4
        # df = 2 closed form: p = exp(-chi2 / 2) = exp(-4).
        assert result.p_value == pytest.approx(math.exp(-4.0), rel=1e-9)

    def test_all_equal_gives_zero_statistic(self) -> None:
        data = [[5.0, 5.0, 5.0], [7.0, 7.0, 7.0]]
        result = friedman_test(data)
        assert result.statistic == pytest.approx(0.0)
        assert result.p_value == pytest.approx(1.0)

    def test_needs_two_treatments(self) -> None:
        with pytest.raises(ValueError):
            friedman_test([[1.0], [2.0]])


class TestNemenyi:
    def test_known_value_k3(self) -> None:
        # CD = q_alpha * sqrt(k(k+1) / (6N)); k=3, N=10, q_0.05=2.343.
        expected = 2.343 * math.sqrt(3 * 4 / (6.0 * 10))
        assert nemenyi_critical_difference(3, 10, 0.05) == pytest.approx(expected)

    def test_smaller_alpha_gives_larger_cd(self) -> None:
        assert nemenyi_critical_difference(3, 10, 0.05) > nemenyi_critical_difference(
            3, 10, 0.10
        )

    def test_more_blocks_shrinks_cd(self) -> None:
        assert nemenyi_critical_difference(3, 100, 0.05) < nemenyi_critical_difference(
            3, 10, 0.05
        )

    def test_invalid_arguments(self) -> None:
        with pytest.raises(ValueError):
            nemenyi_critical_difference(3, 10, 0.01)  # unsupported alpha
        with pytest.raises(ValueError):
            nemenyi_critical_difference(99, 10, 0.05)  # no tabulated q
        with pytest.raises(ValueError):
            nemenyi_critical_difference(3, 0, 0.05)  # non-positive blocks
