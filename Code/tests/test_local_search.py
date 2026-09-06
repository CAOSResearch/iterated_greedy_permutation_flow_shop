"""Tests for the insertion-based local search operator.

Covers:
- Makespan of result ≤ makespan of input (Req 8.2).
- Result is a valid permutation (each index exactly once).
- Determinism: same seed → same result (Reqs 8.4, 9.1).
- Input permutation not mutated.
"""

from __future__ import annotations

import numpy as np
import pytest

from pfsp.core.local_search import local_search
from pfsp.core.makespan import makespan
from pfsp.repro.seeds import make_rng


@pytest.fixture()
def small_instance() -> np.ndarray:
    """A small 3-machine × 5-job instance for testing."""
    return np.array(
        [
            [5, 9, 8, 10, 1],
            [9, 3, 10, 1, 8],
            [9, 4, 5, 8, 6],
        ],
        dtype=np.int64,
    )


@pytest.fixture()
def medium_instance() -> np.ndarray:
    """A medium 4-machine × 8-job instance generated with a fixed seed."""
    rng = np.random.default_rng(12345)
    return rng.integers(1, 100, size=(4, 8), dtype=np.int64)


class TestLocalSearchMakespanImprovement:
    """The local search must not worsen the makespan."""

    def test_result_makespan_leq_input(self, small_instance: np.ndarray) -> None:
        """Makespan of the result is ≤ makespan of the input."""
        permutation = [4, 3, 2, 1, 0]  # Likely suboptimal order
        rng = make_rng(42)
        input_makespan = makespan(small_instance, permutation)

        result_perm, result_makespan = local_search(permutation, small_instance, rng)

        assert result_makespan <= input_makespan

    def test_result_makespan_leq_input_medium(
        self, medium_instance: np.ndarray
    ) -> None:
        """Makespan improvement on a medium instance."""
        permutation = list(range(7, -1, -1))  # Reversed order
        rng = make_rng(99)
        input_makespan = makespan(medium_instance, permutation)

        _, result_makespan = local_search(permutation, medium_instance, rng)

        assert result_makespan <= input_makespan

    @pytest.mark.parametrize("seed", [0, 7, 42, 100, 999])
    def test_never_worsens_across_seeds(
        self, small_instance: np.ndarray, seed: int
    ) -> None:
        """Across multiple seeds, the result never worsens the input."""
        permutation = [2, 0, 4, 1, 3]
        rng = make_rng(seed)
        input_makespan = makespan(small_instance, permutation)

        _, result_makespan = local_search(permutation, small_instance, rng)

        assert result_makespan <= input_makespan


class TestLocalSearchValidPermutation:
    """The result must be a valid permutation (each job index exactly once)."""

    def test_valid_permutation_small(self, small_instance: np.ndarray) -> None:
        """Result contains each job index 0..n-1 exactly once."""
        permutation = [0, 1, 2, 3, 4]
        rng = make_rng(7)

        result_perm, _ = local_search(permutation, small_instance, rng)

        assert sorted(result_perm) == list(range(5))

    def test_valid_permutation_medium(self, medium_instance: np.ndarray) -> None:
        """Result on a medium instance is a valid permutation."""
        permutation = list(range(8))
        rng = make_rng(33)

        result_perm, _ = local_search(permutation, medium_instance, rng)

        assert sorted(result_perm) == list(range(8))


class TestLocalSearchDeterminism:
    """Same seed must produce the same result (Reqs 8.4, 9.1)."""

    def test_same_seed_same_result(self, small_instance: np.ndarray) -> None:
        """Two calls with the same seed produce identical results."""
        permutation = [3, 1, 0, 4, 2]

        rng1 = make_rng(2025)
        result1 = local_search(permutation, small_instance, rng1)

        rng2 = make_rng(2025)
        result2 = local_search(permutation, small_instance, rng2)

        assert result1 == result2

    def test_same_seed_same_result_medium(self, medium_instance: np.ndarray) -> None:
        """Determinism holds on a medium instance."""
        permutation = [5, 2, 7, 0, 3, 6, 1, 4]

        rng1 = make_rng(777)
        result1 = local_search(permutation, medium_instance, rng1)

        rng2 = make_rng(777)
        result2 = local_search(permutation, medium_instance, rng2)

        assert result1 == result2

    def test_different_seed_may_differ(self, small_instance: np.ndarray) -> None:
        """Different seeds can produce different scan orders (not guaranteed)."""
        permutation = [4, 3, 2, 1, 0]

        rng1 = make_rng(1)
        result1 = local_search(permutation, small_instance, rng1)

        rng2 = make_rng(2)
        result2 = local_search(permutation, small_instance, rng2)

        # Both should be valid and non-worsening; they might or might not differ
        # depending on whether different scan orders lead to different local optima.
        assert sorted(result1[0]) == list(range(5))
        assert sorted(result2[0]) == list(range(5))


class TestLocalSearchInputImmutability:
    """The input permutation must not be mutated."""

    def test_input_not_mutated(self, small_instance: np.ndarray) -> None:
        """The original permutation list is unchanged after local_search."""
        permutation = [2, 4, 1, 3, 0]
        original_copy = permutation.copy()
        rng = make_rng(55)

        local_search(permutation, small_instance, rng)

        assert permutation == original_copy

    def test_input_not_mutated_medium(self, medium_instance: np.ndarray) -> None:
        """Input immutability on a medium instance."""
        permutation = [7, 6, 5, 4, 3, 2, 1, 0]
        original_copy = permutation.copy()
        rng = make_rng(11)

        local_search(permutation, medium_instance, rng)

        assert permutation == original_copy


class TestLocalSearchMakespanConsistency:
    """The reported makespan matches the actual makespan of the result."""

    def test_reported_makespan_is_correct(self, small_instance: np.ndarray) -> None:
        """The returned makespan equals makespan(result_permutation)."""
        permutation = [1, 0, 3, 2, 4]
        rng = make_rng(42)

        result_perm, result_makespan = local_search(permutation, small_instance, rng)

        assert result_makespan == makespan(small_instance, result_perm)

    def test_reported_makespan_is_correct_medium(
        self, medium_instance: np.ndarray
    ) -> None:
        """Makespan consistency on a medium instance."""
        permutation = list(range(8))
        rng = make_rng(0)

        result_perm, result_makespan = local_search(permutation, medium_instance, rng)

        assert result_makespan == makespan(medium_instance, result_perm)
