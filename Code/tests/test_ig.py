"""Tests for the iterated_greedy main loop.

Covers:
- Valid permutation and makespan consistency on a small synthetic instance.
- Monotonicity: best makespan <= NEH makespan.
- On Taillard tai20_5_0 (skip if absent): valid permutation, makespan
  consistency, and lower-bound sanity.
- Parameter validation cuts before executing.

Spec: iterated-greedy, task 8.
Requirements: 2.1, 2.2, 2.3, 5.5, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 8.3, 9.1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pfsp.core.ig import (
    IGConfig,
    StoppingCriterion,
    iterated_greedy,
)
from pfsp.core.makespan import makespan
from pfsp.core.neh import neh
from pfsp.instance import Instance
from pfsp.io.taillard import read_taillard

# ---------------------------------------------------------------------------
# Paths and skip conditions
# ---------------------------------------------------------------------------

_CODE_DIR = Path(__file__).resolve().parent.parent
_TAILLARD_INSTANCE = _CODE_DIR / "data" / "taillard" / "tai20_5_0.fsp"

_skip_without_taillard = pytest.mark.skipif(
    not _TAILLARD_INSTANCE.exists(),
    reason="Taillard benchmark data not present (data/taillard/tai20_5_0.fsp)",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_synthetic_instance(n: int = 8, m: int = 3, seed: int = 42) -> Instance:
    """Create a small synthetic PFSP instance for testing."""
    rng = np.random.default_rng(seed)
    processing_times = rng.integers(1, 100, size=(m, n), dtype=np.int64)
    return Instance(
        processing_times=processing_times,
        n=n,
        m=m,
        source_benchmark="synthetic",
        source_path="synthetic/test_instance.fsp",
    )


# ---------------------------------------------------------------------------
# Test 1: Valid permutation and makespan consistency (synthetic)
# ---------------------------------------------------------------------------


class TestIGSynthetic:
    """IG on a small synthetic instance returns valid output."""

    def test_valid_permutation(self) -> None:
        """The returned permutation contains each job index exactly once."""
        instance = _make_synthetic_instance()
        config = IGConfig(
            destruction_size=2,
            stop=StoppingCriterion(kind="iterations", value=50),
        )
        perm, ms = iterated_greedy(instance, seed=123, config=config)

        assert sorted(perm) == list(range(instance.n))
        assert len(perm) == instance.n

    def test_makespan_consistency(self) -> None:
        """The reported makespan equals makespan(perm)."""
        instance = _make_synthetic_instance()
        config = IGConfig(
            destruction_size=2,
            stop=StoppingCriterion(kind="iterations", value=50),
        )
        perm, ms = iterated_greedy(instance, seed=123, config=config)

        computed = makespan(instance.processing_times, perm)
        assert ms == computed


# ---------------------------------------------------------------------------
# Test 2: Monotonicity — best makespan <= NEH makespan
# ---------------------------------------------------------------------------


class TestMonotonicity:
    """The IG best solution is at least as good as NEH."""

    def test_monotonicity_synthetic(self) -> None:
        """best makespan <= NEH makespan on a synthetic instance."""
        instance = _make_synthetic_instance()
        config = IGConfig(
            destruction_size=2,
            stop=StoppingCriterion(kind="iterations", value=100),
        )
        _, neh_makespan = neh(instance)
        _, ig_makespan = iterated_greedy(instance, seed=7, config=config)

        assert ig_makespan <= neh_makespan

    def test_monotonicity_no_ls(self) -> None:
        """Monotonicity holds even without local search."""
        instance = _make_synthetic_instance()
        config = IGConfig(
            destruction_size=2,
            local_search=False,
            stop=StoppingCriterion(kind="iterations", value=100),
        )
        _, neh_makespan = neh(instance)
        _, ig_makespan = iterated_greedy(instance, seed=7, config=config)

        assert ig_makespan <= neh_makespan


# ---------------------------------------------------------------------------
# Test 3: Taillard instance (skip if absent)
# ---------------------------------------------------------------------------


class TestIGTaillard:
    """IG on tai20_5_0 (small real instance)."""

    @_skip_without_taillard
    def test_valid_permutation_taillard(self) -> None:
        """Returned permutation is a valid permutation of 0..n-1."""
        instance = read_taillard(str(_TAILLARD_INSTANCE))
        config = IGConfig(
            destruction_size=4,
            stop=StoppingCriterion(kind="iterations", value=50),
        )
        perm, ms = iterated_greedy(instance, seed=42, config=config)

        assert sorted(perm) == list(range(instance.n))
        assert len(perm) == instance.n

    @_skip_without_taillard
    def test_makespan_consistency_taillard(self) -> None:
        """The reported makespan equals makespan(perm) on Taillard data."""
        instance = read_taillard(str(_TAILLARD_INSTANCE))
        config = IGConfig(
            destruction_size=4,
            stop=StoppingCriterion(kind="iterations", value=50),
        )
        perm, ms = iterated_greedy(instance, seed=42, config=config)

        computed = makespan(instance.processing_times, perm)
        assert ms == computed

    @_skip_without_taillard
    def test_lower_bound_respected_taillard(self) -> None:
        """If the instance has a lower bound, best makespan >= LB."""
        instance = read_taillard(str(_TAILLARD_INSTANCE))
        config = IGConfig(
            destruction_size=4,
            stop=StoppingCriterion(kind="iterations", value=50),
        )
        _, ms = iterated_greedy(instance, seed=42, config=config)

        if instance.lower_bound is not None:
            assert ms >= instance.lower_bound

    @_skip_without_taillard
    def test_monotonicity_taillard(self) -> None:
        """IG makespan <= NEH makespan on the real Taillard instance."""
        instance = read_taillard(str(_TAILLARD_INSTANCE))
        config = IGConfig(
            destruction_size=4,
            stop=StoppingCriterion(kind="iterations", value=100),
        )
        _, neh_makespan = neh(instance)
        _, ig_makespan = iterated_greedy(instance, seed=42, config=config)

        assert ig_makespan <= neh_makespan


# ---------------------------------------------------------------------------
# Test 4: Validation of invalid parameters cuts before executing
# ---------------------------------------------------------------------------


class TestValidationCutsEarly:
    """Invalid config raises ValueError without running the loop."""

    def test_d_zero_raises(self) -> None:
        """d=0 -> ValueError without executing."""
        instance = _make_synthetic_instance()
        config = IGConfig(
            destruction_size=0,
            stop=StoppingCriterion(kind="iterations", value=10),
        )
        with pytest.raises(ValueError, match="destruction_size"):
            iterated_greedy(instance, seed=1, config=config)

    def test_d_too_large_raises(self) -> None:
        """d >= n -> ValueError without executing."""
        instance = _make_synthetic_instance(n=5)
        config = IGConfig(
            destruction_size=5,
            stop=StoppingCriterion(kind="iterations", value=10),
        )
        with pytest.raises(ValueError, match="destruction_size"):
            iterated_greedy(instance, seed=1, config=config)

    def test_negative_tp_raises(self) -> None:
        """Tp < 0 -> ValueError without executing."""
        instance = _make_synthetic_instance()
        config = IGConfig(
            temperature_factor=-0.1,
            stop=StoppingCriterion(kind="iterations", value=10),
        )
        with pytest.raises(ValueError, match="temperature_factor"):
            iterated_greedy(instance, seed=1, config=config)

    def test_invalid_stop_kind_raises(self) -> None:
        """Unrecognized stop kind -> ValueError."""
        instance = _make_synthetic_instance()
        stop = StoppingCriterion(kind="unknown", value=10)  # type: ignore[arg-type]
        config = IGConfig(stop=stop)
        with pytest.raises(ValueError, match="Unrecognized"):
            iterated_greedy(instance, seed=1, config=config)

    def test_stop_value_zero_raises(self) -> None:
        """stop.value = 0 -> ValueError."""
        instance = _make_synthetic_instance()
        stop = StoppingCriterion(kind="iterations", value=0)
        config = IGConfig(stop=stop)
        with pytest.raises(ValueError, match="must be positive"):
            iterated_greedy(instance, seed=1, config=config)


# ---------------------------------------------------------------------------
# Test 5: Determinism under seed (Req 9.1)
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Same seed + same config + same instance -> identical results."""

    def test_determinism_with_ls(self) -> None:
        """Two runs with the same seed yield identical results (LS on)."""
        instance = _make_synthetic_instance()
        config = IGConfig(
            destruction_size=2,
            stop=StoppingCriterion(kind="iterations", value=30),
        )
        perm1, ms1 = iterated_greedy(instance, seed=999, config=config)
        perm2, ms2 = iterated_greedy(instance, seed=999, config=config)

        assert perm1 == perm2
        assert ms1 == ms2

    def test_determinism_without_ls(self) -> None:
        """Two runs with the same seed yield identical results (LS off)."""
        instance = _make_synthetic_instance()
        config = IGConfig(
            destruction_size=2,
            local_search=False,
            stop=StoppingCriterion(kind="iterations", value=30),
        )
        perm1, ms1 = iterated_greedy(instance, seed=999, config=config)
        perm2, ms2 = iterated_greedy(instance, seed=999, config=config)

        assert perm1 == perm2
        assert ms1 == ms2
