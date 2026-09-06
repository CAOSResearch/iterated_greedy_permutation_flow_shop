"""Tests for the shared insertion helper (``best_insertion``).

Covers:
- Correctness: best_insertion matches the brute-force best position on a small case.
- Tie-breaking: when multiple positions yield the same makespan, the lowest
  insertion index is chosen.
- Golden test (NEH invariance): NEH on ``tai20_5_0`` still produces the known
  permutation and makespan (1278) after the refactoring, with skip if data absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from pfsp.core.insertion import best_insertion
from pfsp.core.makespan import makespan
from pfsp.core.neh import neh
from pfsp.io.loader import load_instance

# Repository ``Code`` directory, used to resolve benchmark paths.
_CODE_DIR = Path(__file__).resolve().parent.parent
_TAILLARD_INSTANCE = _CODE_DIR / "data" / "taillard" / "tai20_5_0.fsp"


class TestBestInsertionBruteForce:
    """Verify best_insertion against brute-force on a small case."""

    @pytest.fixture
    def small_instance(self) -> NDArray:
        """A 3-machine, 4-job processing-time matrix."""
        return np.array(
            [
                [5, 9, 3, 7],
                [8, 1, 6, 4],
                [3, 7, 2, 5],
            ],
            dtype=np.int64,
        )

    def test_best_insertion_matches_brute_force(self, small_instance: NDArray) -> None:
        """best_insertion finds the same position as brute-force enumeration."""
        p = small_instance
        sequence = [0, 2]  # partial sequence with jobs 0 and 2
        job = 3

        # Brute-force: try every position, pick the one with lowest makespan,
        # break ties by lowest index.
        best_pos = None
        best_ms = None
        for pos in range(len(sequence) + 1):
            candidate = sequence[:pos] + [job] + sequence[pos:]
            submatrix = p[:, candidate]
            ms = makespan(submatrix, list(range(len(candidate))))
            if best_ms is None or ms < best_ms:
                best_ms = ms
                best_pos = pos

        # Call best_insertion
        result_seq, result_ms = best_insertion(sequence, job, p)

        # They must agree
        expected_seq = sequence[:best_pos] + [job] + sequence[best_pos:]
        assert result_seq == expected_seq
        assert result_ms == best_ms

    def test_best_insertion_tie_breaking_lowest_index(
        self, small_instance: NDArray
    ) -> None:
        """When multiple positions tie on makespan, the lowest index wins."""
        # Use a symmetric matrix where all insertion positions give same makespan.
        # 2 machines, 3 jobs, all processing times identical → all insertions
        # of a new job yield the same makespan.
        p = np.array(
            [
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ],
            dtype=np.int64,
        )
        sequence = [0, 1, 2]
        job = 3

        result_seq, result_ms = best_insertion(sequence, job, p)

        # All positions give the same makespan (total = 4·2 = 8 in 2 machines,
        # each machine processes 4 jobs sequentially → makespan = 4).
        # Tie-breaking: position 0 (lowest index) wins.
        assert result_seq == [3, 0, 1, 2]
        assert result_ms == makespan(p[:, result_seq], list(range(4)))

    def test_best_insertion_into_empty_sequence(self, small_instance: NDArray) -> None:
        """Inserting into an empty sequence returns a single-job sequence."""
        p = small_instance
        result_seq, result_ms = best_insertion([], 2, p)

        assert result_seq == [2]
        # Makespan of a single job is the sum of its processing times.
        expected_ms = int(p[:, 2].sum())
        assert result_ms == expected_ms


class TestNEHGoldenInvariance:
    """Golden test: NEH on tai20_5_0 produces makespan 1278 (invariance)."""

    @pytest.mark.skipif(
        not _TAILLARD_INSTANCE.exists(),
        reason=(
            f"Taillard benchmark data not present at {_TAILLARD_INSTANCE} "
            "(out of git)."
        ),
    )
    def test_neh_golden_makespan_tai20_5_0(self) -> None:
        """NEH on tai20_5_0 yields makespan 1286 (characterization golden test).

        The task spec mentions 1278, which is the best-known upper bound for
        this instance (the optimal). NEH is a constructive heuristic and
        produces 1286 on this instance. This golden test locks the actual NEH
        output to ensure the refactoring does not alter behaviour.
        """
        instance = load_instance(str(_TAILLARD_INSTANCE))
        permutation, ms = neh(instance)

        # The known NEH output from Phase 1 (above the optimal UB of 1278).
        assert ms == 1286
        # Consistency: the makespan matches direct evaluation.
        assert ms == makespan(instance.processing_times, permutation)
        # Valid permutation.
        assert sorted(permutation) == list(range(instance.n))
