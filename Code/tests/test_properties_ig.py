"""Property-based tests for the Iterated Greedy (Correctness Properties).

These tests verify invariant properties of the IG algorithm over seeded random
instances, providing guarantees that are defensible before the thesis tribunal.
Randomness is drawn exclusively through the project's ``Seed_Manager``
(``pfsp.repro.seeds.make_rng``) with fixed seeds, so the suite is fully
reproducible and introduces no new dependencies (no ``hypothesis``).

Properties covered (Requisito 13):

* **P1** — Determinism under seed: same Instance, same IGConfig, same seed
  yields identical permutation and makespan (LS on and off). (Reqs 9.1, 8.5, 13.1)
* **P2** — Monotonicity vs NEH: IG makespan <= NEH makespan on the same instance.
  (Reqs 7.6, 13.2)
* **P3** — Valid permutation: Best_Solution contains each job index 0..n-1
  exactly once. (Reqs 7.4, 13.3)
* **P4** — Makespan consistency: reported makespan equals
  ``makespan(best_permutation)``. (Reqs 7.5, 13.4)
* **P5** — Lower bound respected: makespan >= max(machine-load) and >=
  max(job-load). (Reqs 7.7, 13.5)
* **P6** — Input immutability: IG does not mutate the Instance's
  Processing_Time_Matrix. (Req 13.6)
* **P7** — No-fabrication of RPD: when no reference value is available, RPD is
  None/unset. (Reqs 10.4, 13.7)
"""

from __future__ import annotations

import numpy as np
import pytest

from pfsp.core.ig import IGConfig, StoppingCriterion, iterated_greedy
from pfsp.core.makespan import makespan
from pfsp.core.neh import neh
from pfsp.instance import Instance
from pfsp.repro.seeds import make_rng
from pfsp.results.record import compute_rpd

# ---------------------------------------------------------------------------
# Instance generation helpers
# ---------------------------------------------------------------------------

# Seeds and shapes for the property grid. Every (seed, (n, m)) builds a unique
# seeded random instance. Shapes are kept small for fast tests but varied enough
# to cover different regimes.
_SEEDS: tuple[int, ...] = (0, 7, 42, 99, 123, 2024, 31415, 65535, 99999, 100000)
_SHAPES: tuple[tuple[int, int], ...] = (
    (5, 2),
    (8, 3),
    (10, 5),
    (12, 4),
    (15, 5),
    (20, 5),
)

_MAX_PROCESSING_TIME = 99

# A modest iteration budget to keep tests fast while still exercising the loop.
_FAST_STOP = StoppingCriterion(kind="iterations", value=20)


def _random_processing_times(seed: int, n: int, m: int) -> np.ndarray:
    """Build a random (m, n) int64 processing-time matrix from a seed."""
    rng = make_rng(seed)
    return rng.integers(1, _MAX_PROCESSING_TIME + 1, size=(m, n), dtype=np.int64)


def _make_instance(
    seed: int,
    n: int,
    m: int,
    source_benchmark: str = "taillard",
    source_path: str = "synthetic/random.fsp",
) -> Instance:
    """Build a synthetic Instance from a seeded random matrix."""
    processing_times = _random_processing_times(seed, n, m)
    return Instance(
        processing_times=processing_times,
        n=n,
        m=m,
        source_benchmark=source_benchmark,
        source_path=source_path,
    )


# Parametrize grid: (seed, n, m) combinations.
_GRID = [(seed, n, m) for seed in _SEEDS for (n, m) in _SHAPES]


# ---------------------------------------------------------------------------
# Lower-bound helpers
# ---------------------------------------------------------------------------


def _max_machine_load(processing_times: np.ndarray) -> int:
    """Max row sum: the busiest machine must process all its jobs sequentially."""
    return int(np.asarray(processing_times).sum(axis=1).max())


def _max_job_load(processing_times: np.ndarray) -> int:
    """Max column sum: the longest job traverses all machines without overlap."""
    return int(np.asarray(processing_times).sum(axis=0).max())


# ---------------------------------------------------------------------------
# Property 1: Determinism under seed (Reqs 9.1, 8.5, 13.1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed, n, m", _GRID)
def test_p1_determinism_ls_on(seed: int, n: int, m: int) -> None:
    """IG with LS ON: same seed + config + instance -> identical result.

    **Validates: Requirements 9.1, 8.5, 13.1**
    """
    instance = _make_instance(seed, n, m)
    config = IGConfig(local_search=True, stop=_FAST_STOP)

    perm1, ms1 = iterated_greedy(instance, seed=seed, config=config)
    perm2, ms2 = iterated_greedy(instance, seed=seed, config=config)

    assert perm1 == perm2
    assert ms1 == ms2


@pytest.mark.parametrize("seed, n, m", _GRID)
def test_p1_determinism_ls_off(seed: int, n: int, m: int) -> None:
    """IG with LS OFF: same seed + config + instance -> identical result.

    **Validates: Requirements 9.1, 8.5, 13.1**
    """
    instance = _make_instance(seed, n, m)
    config = IGConfig(local_search=False, stop=_FAST_STOP)

    perm1, ms1 = iterated_greedy(instance, seed=seed, config=config)
    perm2, ms2 = iterated_greedy(instance, seed=seed, config=config)

    assert perm1 == perm2
    assert ms1 == ms2


# ---------------------------------------------------------------------------
# Property 2: Monotonicity vs NEH (Reqs 7.6, 13.2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed, n, m", _GRID)
def test_p2_monotonicity_vs_neh(seed: int, n: int, m: int) -> None:
    """IG makespan is <= NEH makespan on the same instance.

    The IG starts from NEH and only updates the best on strict improvement,
    so it can never be worse.

    **Validates: Requirements 7.6, 13.2**
    """
    instance = _make_instance(seed, n, m)
    config = IGConfig(local_search=True, stop=_FAST_STOP)

    _, neh_makespan = neh(instance)
    _, ig_makespan = iterated_greedy(instance, seed=seed, config=config)

    assert ig_makespan <= neh_makespan


# ---------------------------------------------------------------------------
# Property 3: Valid permutation (Reqs 7.4, 13.3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed, n, m", _GRID)
def test_p3_valid_permutation(seed: int, n: int, m: int) -> None:
    """Best_Solution contains each job index 0..n-1 exactly once.

    **Validates: Requirements 7.4, 13.3**
    """
    instance = _make_instance(seed, n, m)
    config = IGConfig(local_search=True, stop=_FAST_STOP)

    perm, _ = iterated_greedy(instance, seed=seed, config=config)

    assert sorted(perm) == list(range(n))


# ---------------------------------------------------------------------------
# Property 4: Makespan consistency (Reqs 7.5, 13.4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed, n, m", _GRID)
def test_p4_makespan_consistency(seed: int, n: int, m: int) -> None:
    """Reported makespan equals makespan(best_permutation).

    **Validates: Requirements 7.5, 13.4**
    """
    instance = _make_instance(seed, n, m)
    config = IGConfig(local_search=True, stop=_FAST_STOP)

    perm, reported_ms = iterated_greedy(instance, seed=seed, config=config)
    computed_ms = makespan(instance.processing_times, perm)

    assert reported_ms == computed_ms


# ---------------------------------------------------------------------------
# Property 5: Lower bound respected (Reqs 7.7, 13.5)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed, n, m", _GRID)
def test_p5_lower_bounds(seed: int, n: int, m: int) -> None:
    """Makespan >= max(machine-load) and >= max(job-load) (cheap lower bounds).

    **Validates: Requirements 7.7, 13.5**
    """
    instance = _make_instance(seed, n, m)
    config = IGConfig(local_search=True, stop=_FAST_STOP)

    _, ig_makespan = iterated_greedy(instance, seed=seed, config=config)

    assert ig_makespan >= _max_machine_load(instance.processing_times)
    assert ig_makespan >= _max_job_load(instance.processing_times)


# ---------------------------------------------------------------------------
# Property 6: Input immutability (Req 13.6)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed, n, m", _GRID)
def test_p6_input_immutability(seed: int, n: int, m: int) -> None:
    """IG does not mutate the Instance's Processing_Time_Matrix.

    The matrix is frozen (writeable=False) by the Instance contract, and
    operators work on copies. This test verifies the content is unchanged.

    **Validates: Requirements 13.6**
    """
    # Build a writeable copy so any sneaky mutation would be detectable.
    pt = _random_processing_times(seed, n, m)
    before = np.copy(pt)
    instance = Instance(
        processing_times=pt,
        n=n,
        m=m,
        source_benchmark="taillard",
        source_path="synthetic/random.fsp",
    )
    config = IGConfig(local_search=True, stop=_FAST_STOP)

    iterated_greedy(instance, seed=seed, config=config)

    # Content unchanged (even though Instance sets writeable=False, verify data).
    assert np.array_equal(instance.processing_times, before)


# ---------------------------------------------------------------------------
# Property 7: No-fabrication of RPD (Reqs 10.4, 13.7)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed, n, m", _GRID)
def test_p7_no_rpd_fabrication(seed: int, n: int, m: int) -> None:
    """When no reference value is available, RPD is None (never fabricated).

    **Validates: Requirements 10.4, 13.7**
    """
    instance = _make_instance(
        seed,
        n,
        m,
        source_benchmark="vrf",
        source_path="synthetic/VFR_random_Gap.txt",
    )
    config = IGConfig(local_search=True, stop=_FAST_STOP)

    _, ig_makespan = iterated_greedy(instance, seed=seed, config=config)

    # No reference is available for a synthetic VRF instance not in the CSV.
    rpd = compute_rpd(ig_makespan, None)
    assert rpd is None
