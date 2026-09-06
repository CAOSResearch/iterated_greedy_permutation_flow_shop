"""Property-based tests over seeded random instances (Correctness Properties).

These tests reinforce the example-based suite of Requisito 6 by checking the
invariants of the design's *Correctness Properties* section over many seeded
random instances. Randomness is drawn exclusively through the project's
``Seed_Manager`` (``pfsp.repro.seeds.make_rng``) with fixed seeds, so the suite
is fully reproducible and introduces no new dependencies (no ``hypothesis``).

Properties covered:

* **P1** — NEH determinism: ``neh`` run twice on the same instance returns an
  identical permutation and makespan. (Req 5.6)
* **P2** — makespan<->NEH consistency: the makespan reported by ``neh`` equals
  ``makespan(processing_times, returned_permutation)``. (Reqs 5.5, 6.5)
* **P3** — input immutability: ``makespan`` and ``neh`` do not mutate the
  processing-time matrix, and the matrix exposed by an ``Instance`` is
  non-writeable. (Reqs 4.4, 4.5, 4.6, 6.7)
* **P4** — permutation round-trip: reordering the matrix columns by ``pi`` and
  then by ``pi^-1`` reproduces the original matrix. (Reqs 6.4, 7.4)
* **P6** — no fabrication: a VRF-style instance has ``seed``/``UB``/``LB`` unset,
  and ``compute_rpd(makespan, None)`` is ``None``. (Reqs 7.3, 10.5)
* **P7** — cheap theoretical lower bounds: a makespan is always greater than or
  equal to the maximum per-machine row sum and to the maximum per-job column
  sum. Checked for both ``makespan`` on an arbitrary permutation and the NEH
  makespan. (Req 5.7)

The size grid and seed list below are kept small but varied: every
``(seed, n, m)`` combination yields a different random instance, so each property
is exercised across a range of shapes (including degenerate ``1xN`` / ``Nx1``
cases) without being slow.
"""

from __future__ import annotations

import numpy as np
import pytest

from pfsp.core.makespan import makespan
from pfsp.core.neh import neh
from pfsp.instance import Instance
from pfsp.repro.seeds import make_rng
from pfsp.results.record import compute_rpd

# Seeds and instance shapes to sweep. Every (seed, (n, m)) pair builds a distinct
# seeded random instance, so the properties are checked across many inputs.
_SEEDS: tuple[int, ...] = (0, 1, 7, 42, 123, 2024)
_SHAPES: tuple[tuple[int, int], ...] = (
    (1, 1),  # degenerate: single job, single machine
    (5, 1),  # single machine
    (1, 3),  # single job
    (2, 3),
    (5, 3),
    (10, 5),
    (20, 5),
)

# Maximum processing time drawn for synthetic matrices (inclusive lower bound 0
# is allowed so the non-negativity edge is exercised).
_MAX_PROCESSING_TIME = 99


def _random_processing_times(seed: int, n: int, m: int) -> np.ndarray:
    """Build a random ``(m, n)`` int64 processing-time matrix from a seed.

    Randomness comes solely from the project's ``Seed_Manager`` so the matrix is
    reproducible. Values are non-negative integers in ``[0, _MAX_PROCESSING_TIME]``.
    """
    rng = make_rng(seed)
    return rng.integers(0, _MAX_PROCESSING_TIME + 1, size=(m, n), dtype=np.int64)


def _make_instance(
    seed: int,
    n: int,
    m: int,
    source_benchmark: str = "taillard",
    source_path: str = "synthetic/random.fsp",
) -> Instance:
    """Build a synthetic ``Instance`` from a seeded random matrix."""
    processing_times = _random_processing_times(seed, n, m)
    return Instance(
        processing_times=processing_times,
        n=n,
        m=m,
        source_benchmark=source_benchmark,
        source_path=source_path,
    )


# A reusable parametrization across the (seed, shape) grid.
_GRID = [(seed, n, m) for seed in _SEEDS for (n, m) in _SHAPES]


# --- Property 1: NEH determinism (Req 5.6) ---------------------------------


@pytest.mark.parametrize("seed, n, m", _GRID)
def test_p1_neh_is_deterministic(seed: int, n: int, m: int) -> None:
    """NEH returns an identical permutation and makespan across runs.

    **Validates: Requirements 5.6**
    """
    instance = _make_instance(seed, n, m)

    first_permutation, first_makespan = neh(instance)
    second_permutation, second_makespan = neh(instance)

    assert first_permutation == second_permutation
    assert first_makespan == second_makespan


# --- Property 2: makespan<->NEH consistency (Reqs 5.5, 6.5) ----------------


@pytest.mark.parametrize("seed, n, m", _GRID)
def test_p2_neh_makespan_matches_calculator(seed: int, n: int, m: int) -> None:
    """The makespan reported by NEH equals ``makespan`` on its permutation.

    **Validates: Requirements 5.5, 6.5**
    """
    instance = _make_instance(seed, n, m)

    permutation, reported_makespan = neh(instance)

    # The permutation is a valid permutation of 0..n-1.
    assert len(permutation) == n
    assert sorted(permutation) == list(range(n))
    # NEH keeps no separate bookkeeping: its makespan is exactly the recurrence.
    assert reported_makespan == makespan(instance.processing_times, permutation)


# --- Property 3: input immutability (Reqs 4.4, 4.5, 4.6, 6.7) --------------


@pytest.mark.parametrize("seed, n, m", _GRID)
def test_p3_makespan_does_not_mutate_matrix(seed: int, n: int, m: int) -> None:
    """``makespan`` works on a reordered copy and never mutates its input.

    **Validates: Requirements 4.4, 4.5, 4.6**
    """
    # Use a writeable standalone matrix (not wrapped in an Instance) so any
    # mutation by ``makespan`` would be observable.
    processing_times = _random_processing_times(seed, n, m)
    before = np.copy(processing_times)

    rng = make_rng(seed + 1)
    permutation = list(rng.permutation(n))
    makespan(processing_times, permutation)

    assert np.array_equal(processing_times, before)


@pytest.mark.parametrize("seed, n, m", _GRID)
def test_p3_neh_does_not_mutate_matrix(seed: int, n: int, m: int) -> None:
    """``neh`` reads but never mutates the instance matrix.

    **Validates: Requirements 4.4, 6.7**
    """
    processing_times = _random_processing_times(seed, n, m)
    before = np.copy(processing_times)
    instance = Instance(
        processing_times=processing_times,
        n=n,
        m=m,
        source_benchmark="taillard",
        source_path="synthetic/random.fsp",
    )

    neh(instance)

    assert np.array_equal(instance.processing_times, before)


@pytest.mark.parametrize("seed, n, m", _GRID)
def test_p3_instance_matrix_is_non_writeable(seed: int, n: int, m: int) -> None:
    """The processing-time matrix of an ``Instance`` cannot be written to.

    **Validates: Requirements 4.4, 4.5, 4.6, 6.7**
    """
    instance = _make_instance(seed, n, m)

    assert instance.processing_times.flags.writeable is False
    with pytest.raises(ValueError):
        instance.processing_times[0, 0] = 1234


# --- Property 4: permutation round-trip (Reqs 6.4, 7.4) --------------------


@pytest.mark.parametrize("seed, n, m", _GRID)
def test_p4_permutation_round_trip(seed: int, n: int, m: int) -> None:
    """Reordering columns by ``pi`` then by ``pi^-1`` reproduces the matrix.

    **Validates: Requirements 6.4**
    """
    processing_times = _random_processing_times(seed, n, m)

    rng = make_rng(seed + 2)
    pi = rng.permutation(n)
    inverse_pi = np.argsort(pi)

    # NumPy advanced indexing returns copies, so the original is never mutated.
    reordered = processing_times[:, pi]
    restored = reordered[:, inverse_pi]

    assert np.array_equal(restored, processing_times)


# --- Property 6: no fabrication of data (Reqs 7.3, 10.5) -------------------


@pytest.mark.parametrize("seed, n, m", _GRID)
def test_p6_vrf_instance_has_unset_metadata(seed: int, n: int, m: int) -> None:
    """A VRF-style instance leaves ``seed``/``UB``/``LB`` unset (no fabrication).

    **Validates: Requirements 7.3**
    """
    instance = _make_instance(
        seed,
        n,
        m,
        source_benchmark="vrf",
        source_path="synthetic/VFR_random_Gap.txt",
    )

    assert instance.seed is None
    assert instance.upper_bound is None
    assert instance.lower_bound is None


@pytest.mark.parametrize("seed, n, m", _GRID)
def test_p6_rpd_is_none_without_reference(seed: int, n: int, m: int) -> None:
    """``compute_rpd`` returns ``None`` when no reference is available.

    A makespan is computed for a real permutation; with no reference value the
    RPD must not be fabricated.

    **Validates: Requirements 10.5**
    """
    instance = _make_instance(
        seed,
        n,
        m,
        source_benchmark="vrf",
        source_path="synthetic/VFR_random_Gap.txt",
    )

    _, instance_makespan = neh(instance)

    assert compute_rpd(instance_makespan, None) is None


# --- Property 7: cheap theoretical lower bounds (Req 5.7) ------------------


def _max_machine_row_sum(processing_times: np.ndarray) -> int:
    """Largest total processing time on any single machine (a row sum)."""
    return int(np.asarray(processing_times).sum(axis=1).max())


def _max_job_column_sum(processing_times: np.ndarray) -> int:
    """Largest total processing time of any single job (a column sum)."""
    return int(np.asarray(processing_times).sum(axis=0).max())


@pytest.mark.parametrize("seed, n, m", _GRID)
def test_p7_arbitrary_permutation_respects_lower_bounds(
    seed: int, n: int, m: int
) -> None:
    """Any permutation's makespan meets both cheap theoretical lower bounds.

    The makespan can never be smaller than the busiest machine's total work
    (a row sum) nor than the longest job's total work (a column sum), since the
    last machine must process every job and every job must traverse every
    machine.

    **Validates: Requirements 5.7**
    """
    processing_times = _random_processing_times(seed, n, m)

    rng = make_rng(seed + 3)
    permutation = list(rng.permutation(n))
    value = makespan(processing_times, permutation)

    assert value >= _max_machine_row_sum(processing_times)
    assert value >= _max_job_column_sum(processing_times)


@pytest.mark.parametrize("seed, n, m", _GRID)
def test_p7_neh_makespan_respects_lower_bounds(seed: int, n: int, m: int) -> None:
    """The NEH makespan meets both cheap theoretical lower bounds.

    **Validates: Requirements 5.7**
    """
    instance = _make_instance(seed, n, m)

    _, neh_makespan = neh(instance)

    assert neh_makespan >= _max_machine_row_sum(instance.processing_times)
    assert neh_makespan >= _max_job_column_sum(instance.processing_times)
