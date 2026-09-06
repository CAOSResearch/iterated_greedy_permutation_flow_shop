"""Property-based tests for the guided destruction (Correctness Properties).

These tests verify invariant properties of the guided-destruction operators and
of the Iterated Greedy running them, over seeded random instances. They provide
guarantees that are defensible before the thesis tribunal. Randomness is drawn
exclusively through the project's ``Seed_Manager`` (``pfsp.repro.seeds.make_rng``)
with fixed seeds, so the suite is fully reproducible and introduces no new
dependencies (no ``hypothesis``): the input space is explored with seeded loops
over a parametrized grid of instances, seeds, and operators.

This file mirrors ``test_properties_ig.py`` (Phase 2) and extends its property
set to the guided operators ``idle-greedy`` (O1) and ``idle-rcl`` (O2), plus the
lower-level building blocks ``idle_scores`` and ``build_rcl``.

Properties covered (Requisito 13):

* **P1** — Determinism under seed: same operator, Alpha, seed, parameters, and
  Instance yields identical permutation and makespan. (Req 13.1)
* **P2** — Monotonicity vs NEH: with any destruction operator, IG makespan is
  ``<=`` the NEH makespan on the same Instance. (Req 13.2)
* **P3** — Valid permutation: Best_Solution contains each job index ``0..n-1``
  exactly once, for every operator. (Req 13.3)
* **P4** — Makespan consistency: reported best makespan equals
  ``makespan(best_permutation)``, for every operator. (Req 13.4)
* **P5** — Input immutability: neither ``idle_scores`` nor the destruction
  operators (nor the IG) mutate the Processing_Time_Matrix. (Req 13.5)
* **P6** — RCL membership: every job removed by ``idle-rcl`` (O2) belongs to the
  RCL for that state. (Req 13.6)
* **P7** — Lower bound respected: with any operator, best makespan is ``>=`` the
  trivial lower bound (max machine load, max job load). (Req 13.7)
* **P8** — Idle_Score non-negativity: every per-position Idle_Score is ``>= 0``
  over seeded random instances. (Req 13.8)
* **P9** — Removed-set partition: ``idle-rcl`` and ``idle-greedy`` return ``d``
  removed jobs and ``n - d`` partial jobs that are disjoint and whose union is
  exactly the original job set (round-trip). (Req 13.9)
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
from pfsp.core.ig import IGConfig, StoppingCriterion, iterated_greedy
from pfsp.core.makespan import makespan
from pfsp.core.neh import neh
from pfsp.instance import Instance
from pfsp.repro.seeds import make_rng

# ---------------------------------------------------------------------------
# Instance / permutation generation helpers (all seeded via Seed_Manager)
# ---------------------------------------------------------------------------

# Seeds and shapes for the property grid. Every (seed, (n, m)) builds a unique
# seeded random instance. Shapes are kept small for fast tests but varied enough
# to cover different regimes.
_SEEDS: tuple[int, ...] = (0, 7, 42, 123, 2024)
_SHAPES: tuple[tuple[int, int], ...] = (
    (5, 2),
    (8, 3),
    (10, 5),
    (12, 4),
    (15, 5),
)

_MAX_PROCESSING_TIME = 99

# The three destruction operators exercised by the IG-level properties. The
# guided Alpha is only used by ``idle-rcl``; the others ignore it.
_OPERATORS: tuple[str, ...] = ("random", "idle-greedy", "idle-rcl")

# Alpha used for the guided ``idle-rcl`` operator in the IG-level tests.
_ALPHA = 0.30

# Destruction size shared across the IG-level tests (valid since min n is 5).
_D = 3

# A modest iteration budget to keep tests fast while still exercising the loop.
_FAST_STOP = StoppingCriterion(kind="iterations", value=20)


def _random_processing_times(seed: int, n: int, m: int) -> np.ndarray:
    """Build a random (m, n) int64 processing-time matrix from a seed."""
    rng = make_rng(seed)
    return rng.integers(1, _MAX_PROCESSING_TIME + 1, size=(m, n), dtype=np.int64)


def _make_instance(seed: int, n: int, m: int) -> Instance:
    """Build a synthetic Instance from a seeded random matrix."""
    processing_times = _random_processing_times(seed, n, m)
    return Instance(
        processing_times=processing_times,
        n=n,
        m=m,
        source_benchmark="taillard",
        source_path="synthetic/random.fsp",
    )


def _seeded_permutation(seed: int, n: int) -> list[int]:
    """Build a seeded random permutation of ``0..n-1`` via the Seed_Manager."""
    return make_rng(seed).permutation(n).tolist()


def _config(operator: str) -> IGConfig:
    """IGConfig for the given operator with the shared test parameters."""
    return IGConfig(
        destruction_size=_D,
        destruction=operator,
        alpha=_ALPHA,
        local_search=True,
        stop=_FAST_STOP,
    )


# ---------------------------------------------------------------------------
# Lower-bound helpers (trivial lower bounds, mirroring the Phase 2 property set)
# ---------------------------------------------------------------------------


def _max_machine_load(processing_times: np.ndarray) -> int:
    """Max row sum: the busiest machine must process all its jobs sequentially."""
    return int(np.asarray(processing_times).sum(axis=1).max())


def _max_job_load(processing_times: np.ndarray) -> int:
    """Max column sum: the longest job traverses all machines without overlap."""
    return int(np.asarray(processing_times).sum(axis=0).max())


# Parametrize grids.
_GRID = [(seed, n, m) for seed in _SEEDS for (n, m) in _SHAPES]
_GRID_OPS = [(op, seed, n, m) for op in _OPERATORS for (seed, n, m) in _GRID]


# ---------------------------------------------------------------------------
# Property 1: Determinism under seed (Req 13.1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("operator, seed, n, m", _GRID_OPS)
def test_p1_determinism(operator: str, seed: int, n: int, m: int) -> None:
    """Same operator + Alpha + seed + config + instance -> identical result.

    **Validates: Requirements 13.1**
    """
    instance = _make_instance(seed, n, m)
    config = _config(operator)

    perm1, ms1 = iterated_greedy(instance, seed=seed, config=config)
    perm2, ms2 = iterated_greedy(instance, seed=seed, config=config)

    assert perm1 == perm2
    assert ms1 == ms2


# ---------------------------------------------------------------------------
# Property 2: Monotonicity vs NEH (Req 13.2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("operator, seed, n, m", _GRID_OPS)
def test_p2_monotonicity_vs_neh(operator: str, seed: int, n: int, m: int) -> None:
    """With any operator, best makespan is <= the NEH makespan.

    The IG starts from NEH and only updates the best on strict improvement,
    so it can never be worse regardless of the destruction operator.

    **Validates: Requirements 13.2**
    """
    instance = _make_instance(seed, n, m)

    _, neh_makespan = neh(instance)
    _, ig_makespan = iterated_greedy(instance, seed=seed, config=_config(operator))

    assert ig_makespan <= neh_makespan


# ---------------------------------------------------------------------------
# Property 3: Valid permutation (Req 13.3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("operator, seed, n, m", _GRID_OPS)
def test_p3_valid_permutation(operator: str, seed: int, n: int, m: int) -> None:
    """Best_Solution contains each job index 0..n-1 exactly once, per operator.

    **Validates: Requirements 13.3**
    """
    instance = _make_instance(seed, n, m)

    perm, _ = iterated_greedy(instance, seed=seed, config=_config(operator))

    assert sorted(perm) == list(range(n))


# ---------------------------------------------------------------------------
# Property 4: Makespan consistency (Req 13.4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("operator, seed, n, m", _GRID_OPS)
def test_p4_makespan_consistency(operator: str, seed: int, n: int, m: int) -> None:
    """Reported best makespan equals makespan(best_permutation), per operator.

    **Validates: Requirements 13.4**
    """
    instance = _make_instance(seed, n, m)

    perm, reported_ms = iterated_greedy(instance, seed=seed, config=_config(operator))
    computed_ms = makespan(instance.processing_times, perm)

    assert reported_ms == computed_ms


# ---------------------------------------------------------------------------
# Property 5: Input immutability (Req 13.5)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed, n, m", _GRID)
def test_p5_idle_scores_do_not_mutate_matrix(seed: int, n: int, m: int) -> None:
    """``idle_scores`` never mutates the Processing_Time_Matrix.

    **Validates: Requirements 13.5**
    """
    pt = _random_processing_times(seed, n, m)
    before = np.copy(pt)
    permutation = _seeded_permutation(seed, n)

    idle_scores(permutation, pt)

    assert np.array_equal(pt, before)


@pytest.mark.parametrize("seed, n, m", _GRID)
def test_p5_operators_do_not_mutate_matrix(seed: int, n: int, m: int) -> None:
    """Neither guided operator mutates the captured Processing_Time_Matrix.

    **Validates: Requirements 13.5**
    """
    pt = _random_processing_times(seed, n, m)
    before = np.copy(pt)
    permutation = _seeded_permutation(seed, n)

    make_idle_rcl_destruct(pt, _ALPHA)(permutation, _D, make_rng(seed))
    make_idle_greedy_destruct(pt)(permutation, _D, make_rng(seed))

    assert np.array_equal(pt, before)


@pytest.mark.parametrize("operator, seed, n, m", _GRID_OPS)
def test_p5_ig_does_not_mutate_matrix(operator: str, seed: int, n: int, m: int) -> None:
    """Running the IG with any operator leaves the matrix content unchanged.

    **Validates: Requirements 13.5**
    """
    pt = _random_processing_times(seed, n, m)
    before = np.copy(pt)
    instance = Instance(
        processing_times=pt,
        n=n,
        m=m,
        source_benchmark="taillard",
        source_path="synthetic/random.fsp",
    )

    iterated_greedy(instance, seed=seed, config=_config(operator))

    assert np.array_equal(instance.processing_times, before)


# ---------------------------------------------------------------------------
# Property 6: RCL membership for O2 (Req 13.6)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed, n, m", _GRID)
def test_p6_o2_removes_only_rcl_members(seed: int, n: int, m: int) -> None:
    """Every job removed by ``idle-rcl`` (O2) sits at an RCL position.

    ``d = 2`` with ``alpha = 0.5`` makes the RCL a strict superset of a single
    draw for the shapes in the grid (``max(2, ceil(0.5 n)) >= 3 > 2``), so this
    is a non-trivial membership check rather than "RCL is the whole sequence".

    **Validates: Requirements 13.6**
    """
    pt = _random_processing_times(seed, n, m)
    d = 2
    alpha = 0.5

    for perm_seed in range(seed, seed + 8):
        permutation = _seeded_permutation(perm_seed, n)
        scores = idle_scores(permutation, pt)
        rcl_positions = build_rcl(scores, alpha, d)
        rcl_jobs = {permutation[pos] for pos in rcl_positions}

        _, removed = make_idle_rcl_destruct(pt, alpha)(
            permutation, d, make_rng(perm_seed)
        )

        assert set(removed) <= rcl_jobs


# ---------------------------------------------------------------------------
# Property 7: Lower bound respected (Req 13.7)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("operator, seed, n, m", _GRID_OPS)
def test_p7_lower_bounds(operator: str, seed: int, n: int, m: int) -> None:
    """Best makespan >= max(machine-load) and >= max(job-load), per operator.

    **Validates: Requirements 13.7**
    """
    instance = _make_instance(seed, n, m)

    _, ig_makespan = iterated_greedy(instance, seed=seed, config=_config(operator))

    assert ig_makespan >= _max_machine_load(instance.processing_times)
    assert ig_makespan >= _max_job_load(instance.processing_times)


# ---------------------------------------------------------------------------
# Property 8: Idle_Score non-negativity (Req 13.8)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed, n, m", _GRID)
def test_p8_idle_scores_non_negative(seed: int, n: int, m: int) -> None:
    """Every per-position Idle_Score is >= 0 over seeded random instances.

    **Validates: Requirements 13.8**
    """
    pt = _random_processing_times(seed, n, m)

    for perm_seed in range(seed, seed + 5):
        permutation = _seeded_permutation(perm_seed, n)
        scores = idle_scores(permutation, pt)

        assert scores.shape == (n,)
        assert np.all(scores >= 0)


# ---------------------------------------------------------------------------
# Property 9: Removed-set partition for O2 and O1 (Req 13.9)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("guided", ["idle-rcl", "idle-greedy"])
@pytest.mark.parametrize("seed, n, m", _GRID)
def test_p9_removed_set_partition(guided: str, seed: int, n: int, m: int) -> None:
    """The guided operators partition the original job set (round-trip).

    ``removed`` has ``d`` jobs, ``partial`` has ``n - d`` jobs, they are
    disjoint, and their union is exactly the original permutation's job set.

    **Validates: Requirements 13.9**
    """
    pt = _random_processing_times(seed, n, m)
    permutation = _seeded_permutation(seed, n)

    if guided == "idle-rcl":
        operator = make_idle_rcl_destruct(pt, _ALPHA)
    else:
        operator = make_idle_greedy_destruct(pt)

    partial, removed = operator(permutation, _D, make_rng(seed))

    assert len(removed) == _D
    assert len(partial) == n - _D
    assert set(partial).isdisjoint(set(removed))
    assert sorted(partial + removed) == sorted(permutation)
