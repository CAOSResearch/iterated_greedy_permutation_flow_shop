"""Tests for the random destruction operator (``destruct``).

Covers:
- Partition property: ``partial`` has ``n - d`` elements and ``removed`` has
  ``d`` elements whose union is the original set (Req 3.1).
- Relative order preservation: non-removed jobs keep their original relative
  order in the partial sequence (Req 3.2).
- Input immutability: the input permutation is NOT mutated (Req 3.4).
- Reproducibility: same seeded rng produces the same selection (Req 3.3).
"""

from __future__ import annotations

import pytest

from pfsp.core.destruction import destruct
from pfsp.repro.seeds import make_rng


class TestDestructPartitionProperty:
    """Verify that partial + removed form a partition of the original set."""

    @pytest.mark.parametrize(
        "n,d",
        [
            (5, 1),
            (5, 4),
            (10, 3),
            (10, 5),
            (20, 4),
        ],
    )
    def test_union_equals_original(self, n: int, d: int) -> None:
        """partial (n-d) + removed (d) union == original job set."""
        permutation = list(range(n))
        rng = make_rng(42)

        partial, removed = destruct(permutation, d, rng)

        assert len(partial) == n - d
        assert len(removed) == d
        assert sorted(partial + removed) == sorted(permutation)

    def test_removed_are_distinct(self) -> None:
        """All removed jobs are distinct (no duplicates)."""
        permutation = list(range(15))
        rng = make_rng(7)

        _, removed = destruct(permutation, 5, rng)

        assert len(removed) == len(set(removed))


class TestDestructRelativeOrder:
    """Verify relative order of non-removed jobs is preserved."""

    def test_partial_preserves_relative_order(self) -> None:
        """Non-removed jobs appear in the same order as in the input."""
        permutation = [4, 2, 7, 1, 9, 3, 8, 0, 6, 5]
        rng = make_rng(123)
        d = 4

        partial, removed = destruct(permutation, d, rng)

        # Build expected order: filter the original permutation keeping only
        # jobs that are NOT in removed.
        removed_set = set(removed)
        expected_order = [j for j in permutation if j not in removed_set]

        assert partial == expected_order

    def test_relative_order_various_seeds(self) -> None:
        """Relative order holds across different seeds."""
        permutation = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

        for seed in range(10):
            rng = make_rng(seed)
            partial, removed = destruct(permutation, 3, rng)

            removed_set = set(removed)
            expected_order = [j for j in permutation if j not in removed_set]
            assert partial == expected_order


class TestDestructInputImmutability:
    """Verify the input permutation is not mutated (Req 3.4)."""

    def test_input_not_mutated(self) -> None:
        """The original list is unchanged after destruct."""
        permutation = [3, 1, 4, 1, 5, 9, 2, 6]
        original_copy = permutation.copy()
        rng = make_rng(99)

        destruct(permutation, 3, rng)

        assert permutation == original_copy

    def test_returned_partial_is_independent(self) -> None:
        """Modifying the returned partial does not affect the original."""
        permutation = [0, 1, 2, 3, 4]
        rng = make_rng(0)

        partial, _ = destruct(permutation, 2, rng)
        partial.append(999)

        assert 999 not in permutation


class TestDestructReproducibility:
    """Verify that same rng seed produces the same selection (Req 3.3)."""

    def test_same_seed_same_result(self) -> None:
        """Two calls with identically-seeded rng return the same output."""
        permutation = [5, 3, 8, 1, 7, 0, 4, 2, 6, 9]
        d = 4

        rng1 = make_rng(2025)
        partial1, removed1 = destruct(permutation, d, rng1)

        rng2 = make_rng(2025)
        partial2, removed2 = destruct(permutation, d, rng2)

        assert partial1 == partial2
        assert removed1 == removed2

    def test_different_seed_different_result(self) -> None:
        """Different seeds yield different selections (with high probability)."""
        permutation = list(range(20))
        d = 4

        rng1 = make_rng(1)
        _, removed1 = destruct(permutation, d, rng1)

        rng2 = make_rng(2)
        _, removed2 = destruct(permutation, d, rng2)

        # With 20 jobs and d=4, the probability that two random seeds pick the
        # same 4 positions is negligible.
        assert removed1 != removed2
