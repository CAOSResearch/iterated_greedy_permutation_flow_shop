"""Equivalence tests: accelerated vs naive best_insertion (and its consequences).

The Taillard-accelerated :func:`pfsp.core.insertion.best_insertion` must return
**exactly** the same result as the reference :func:`best_insertion_naive` for
every input — same permutation and same makespan, including tie-breaking. Since
best_insertion is the single insertion primitive shared by NEH, the IG
reconstruction and the local search, this equivalence guarantees that the whole
pipeline (and therefore the EXP-003 vs EXP-005 comparison) is bit-identical, and
that the acceleration is a pure speedup.

Following the project convention (see ``test_properties.py``), randomness comes
only from the seeded ``Seed_Manager`` over a small, varied ``(seed, shape)`` grid
— no new dependencies (no ``hypothesis``).
"""

from __future__ import annotations

import numpy as np

from pfsp.core.insertion import best_insertion, best_insertion_naive
from pfsp.core.makespan import makespan
from pfsp.core.neh import neh
from pfsp.instance import Instance
from pfsp.repro.seeds import make_rng

_SEEDS: tuple[int, ...] = (0, 1, 7, 42, 123, 2024)
_SHAPES: tuple[tuple[int, int], ...] = (
    (1, 1),
    (5, 1),
    (1, 4),
    (2, 5),
    (5, 4),
    (10, 6),
    (20, 8),
)
_MAX_PROCESSING_TIME = 99


def _random_processing_times(seed: int, n: int, m: int) -> np.ndarray:
    rng = make_rng(seed)
    return rng.integers(0, _MAX_PROCESSING_TIME + 1, size=(m, n), dtype=np.int64)


def test_best_insertion_matches_naive_over_grid() -> None:
    """Accelerated and naive best_insertion agree over many seeded inputs.

    For each instance, sweep every prefix length ``k`` of a random permutation:
    the sequence is the first ``k`` jobs and the inserted job is the ``k``-th, so
    the job is guaranteed not to be in the sequence. Both implementations must
    return the same permutation and the same makespan, and the reported makespan
    must equal ``makespan`` of the returned permutation.
    """
    for seed in _SEEDS:
        for n, m in _SHAPES:
            p = _random_processing_times(seed, n, m)
            perm = make_rng(seed + 1).permutation(n).tolist()
            for k in range(n):  # k = 0 (empty seq) .. n-1
                sequence = perm[:k]
                job = perm[k]

                seq_fast, ms_fast = best_insertion(sequence, job, p)
                seq_naive, ms_naive = best_insertion_naive(sequence, job, p)

                assert seq_fast == seq_naive, (
                    f"seq mismatch seed={seed} shape=({n},{m}) k={k}: "
                    f"{seq_fast} != {seq_naive}"
                )
                assert ms_fast == ms_naive, (
                    f"makespan mismatch seed={seed} shape=({n},{m}) k={k}: "
                    f"{ms_fast} != {ms_naive}"
                )
                # Reported makespan is consistent with the calculator (partial
                # sequence: restrict the matrix columns and use identity order).
                assert ms_fast == makespan(p[:, seq_fast], list(range(len(seq_fast))))


def test_best_insertion_does_not_mutate_inputs() -> None:
    """Neither implementation mutates the input sequence or matrix."""
    p = _random_processing_times(42, 6, 3)
    p_copy = p.copy()
    sequence = [2, 0, 4]
    seq_snapshot = list(sequence)

    best_insertion(sequence, 1, p)
    assert sequence == seq_snapshot
    assert np.array_equal(p, p_copy)


def test_best_insertion_tie_break_lowest_index() -> None:
    """On tied makespans, both implementations pick the lowest insertion index.

    With an all-equal single-machine matrix, every insertion position yields the
    same makespan, so the tie-break rule (lowest index) must place the job at
    position 0.
    """
    p = np.ones((1, 5), dtype=np.int64)  # single machine, identical times
    sequence = [0, 1, 2]
    job = 3

    seq_fast, ms_fast = best_insertion(sequence, job, p)
    seq_naive, ms_naive = best_insertion_naive(sequence, job, p)

    assert seq_fast == seq_naive == [3, 0, 1, 2]  # inserted at position 0
    assert ms_fast == ms_naive


def test_empty_sequence_single_job() -> None:
    """Inserting into an empty sequence returns the single-job makespan."""
    p = _random_processing_times(7, 5, 4)
    seq_fast, ms_fast = best_insertion([], 2, p)
    assert seq_fast == [2]
    # Single job makespan = sum of its processing times over machines.
    assert ms_fast == int(p[:, 2].sum())
    assert ms_fast == best_insertion_naive([], 2, p)[1]


def _neh_naive(p: np.ndarray) -> tuple[list[int], int]:
    """NEH built with the naive insertion primitive (reference for equivalence)."""
    n = p.shape[1]
    total = np.asarray(p).sum(axis=0)
    ordering = sorted(range(n), key=lambda job: (-int(total[job]), job))
    sequence: list[int] = []
    for job in ordering:
        sequence, _ = best_insertion_naive(sequence, job, p)
    return sequence, makespan(p, sequence)


def test_neh_accelerated_matches_naive_construction() -> None:
    """The accelerated NEH equals an NEH built with the naive insertion.

    This lifts the primitive-level equivalence to the full constructive
    heuristic: same permutation and makespan over the seeded grid.
    """
    for seed in _SEEDS:
        for n, m in _SHAPES:
            p = _random_processing_times(seed, n, m)
            instance = Instance(
                processing_times=p,
                n=n,
                m=m,
                source_benchmark="synthetic",
                source_path=f"synthetic/eq_{seed}_{n}_{m}",
            )
            perm_fast, ms_fast = neh(instance)
            perm_naive, ms_naive = _neh_naive(p)
            assert perm_fast == perm_naive
            assert ms_fast == ms_naive
