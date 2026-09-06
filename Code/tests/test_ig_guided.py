"""Tests for the guided-destruction integration into the Iterated Greedy.

Covers (spec destruccion-inteligente, task 4):

* Config validation (Req 6.1, 6.2, 7.1, 7.2, 12.6):
  - ``IGConfig`` exposes ``destruction`` (default ``"random"``) and ``alpha``
    (default ``0.10``).
  - An unrecognized operator name or an out-of-range ``alpha`` raises a
    descriptive ``ValueError`` before the run starts.

* Baseline regression (Req 1.6, 12.5):
  - With ``destruction="random"`` the IG reproduces a fixed golden
    ``(permutation, makespan)`` on a small known instance and seed, and matches
    the default-config run (the guided path does not disturb the rng
    consumption of the random baseline).

* Guided happy path with O1/O2 (Req 6.4, 6.5, 12.9):
  - Valid permutation, makespan consistent with the Makespan_Calculator,
    monotonicity versus NEH, determinism under a fixed seed, and
    ``makespan >= LB`` on a small Taillard instance when a lower bound is
    available.
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

# A fixed, hand-written instance (not rng-generated) so the golden regression
# value does not depend on the NumPy random-stream version.
_GOLDEN_PROC = np.array(
    [
        [5, 9, 3, 7, 2, 8, 4, 6],
        [4, 3, 6, 2, 9, 5, 7, 1],
        [3, 6, 2, 5, 4, 7, 8, 2],
    ],
    dtype=np.int64,
)


def _golden_instance() -> Instance:
    """The small known instance used for the baseline golden regression."""
    return Instance(
        processing_times=_GOLDEN_PROC.copy(),
        n=8,
        m=3,
        source_benchmark="synthetic",
        source_path="synthetic/golden.fsp",
    )


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
# Test 1: Config defaults and validation
# ---------------------------------------------------------------------------


class TestGuidedConfigDefaults:
    """IGConfig exposes the guided-destruction knobs with the right defaults."""

    def test_defaults(self) -> None:
        """destruction defaults to 'random' and alpha to 0.10 (Req 6.1, 7.1)."""
        config = IGConfig()
        assert config.destruction == "random"
        assert config.alpha == 0.10


class TestGuidedValidationCutsEarly:
    """Invalid operator / alpha raise ValueError without running (Req 6.2, 7.2)."""

    def test_unknown_operator_raises(self) -> None:
        """An unrecognized operator name is rejected before execution."""
        instance = _make_synthetic_instance()
        config = IGConfig(
            destruction="idle-random",  # typo / not a valid operator
            stop=StoppingCriterion(kind="iterations", value=10),
        )
        with pytest.raises(ValueError, match="Unrecognized destruction operator"):
            iterated_greedy(instance, seed=1, config=config)

    @pytest.mark.parametrize("bad_alpha", [0.0, -0.1, 1.5, 2.0])
    def test_alpha_out_of_range_raises(self, bad_alpha: float) -> None:
        """alpha outside (0, 1] is rejected before execution."""
        instance = _make_synthetic_instance()
        config = IGConfig(
            destruction="idle-rcl",
            alpha=bad_alpha,
            stop=StoppingCriterion(kind="iterations", value=10),
        )
        with pytest.raises(ValueError, match="alpha"):
            iterated_greedy(instance, seed=1, config=config)

    def test_alpha_reports_supplied_value_and_range(self) -> None:
        """The alpha error message reports the supplied value and the range."""
        instance = _make_synthetic_instance()
        config = IGConfig(
            destruction="idle-rcl",
            alpha=3.0,
            stop=StoppingCriterion(kind="iterations", value=10),
        )
        with pytest.raises(ValueError, match=r"0 < alpha <= 1"):
            iterated_greedy(instance, seed=1, config=config)


# ---------------------------------------------------------------------------
# Test 2: Baseline regression (Req 1.6, 12.5)
# ---------------------------------------------------------------------------


class TestRandomBaselineIntact:
    """destruction='random' reproduces the Phase 2 IG behavior exactly."""

    # Golden (permutation, makespan) captured for the _golden_instance with
    # seed 20260716, d=3, LS on, 200 iterations. This anchors the classic-IG
    # baseline: the guided path must not alter the random operator's results.
    _GOLDEN_PERM = [6, 5, 3, 4, 0, 1, 2, 7]
    _GOLDEN_MAKESPAN = 48

    def _config(self, **overrides: object) -> IGConfig:
        params: dict[str, object] = {
            "destruction_size": 3,
            "stop": StoppingCriterion(kind="iterations", value=200),
        }
        params.update(overrides)
        return IGConfig(**params)  # type: ignore[arg-type]

    def test_golden_permutation_and_makespan(self) -> None:
        """Explicit random operator matches the fixed golden result."""
        instance = _golden_instance()
        config = self._config(destruction="random")
        perm, ms = iterated_greedy(instance, seed=20260716, config=config)

        assert perm == self._GOLDEN_PERM
        assert ms == self._GOLDEN_MAKESPAN

    def test_default_equals_explicit_random(self) -> None:
        """Omitting 'destruction' (default) equals passing 'random' explicitly."""
        instance = _golden_instance()
        perm_default, ms_default = iterated_greedy(
            instance, seed=20260716, config=self._config()
        )
        perm_random, ms_random = iterated_greedy(
            instance, seed=20260716, config=self._config(destruction="random")
        )

        assert perm_default == perm_random
        assert ms_default == ms_random

    def test_makespan_consistent_with_calculator(self) -> None:
        """The golden makespan equals makespan(golden_perm)."""
        instance = _golden_instance()
        assert (
            makespan(instance.processing_times, self._GOLDEN_PERM)
            == self._GOLDEN_MAKESPAN
        )


# ---------------------------------------------------------------------------
# Test 3: Guided happy path (O1 idle-greedy and O2 idle-rcl)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("operator", ["idle-greedy", "idle-rcl"])
class TestGuidedHappyPathSynthetic:
    """The IG with a guided operator behaves correctly on a synthetic instance."""

    def _config(self, operator: str) -> IGConfig:
        return IGConfig(
            destruction_size=2,
            destruction=operator,
            alpha=0.30,
            stop=StoppingCriterion(kind="iterations", value=100),
        )

    def test_valid_permutation(self, operator: str) -> None:
        """The returned permutation contains each job index exactly once."""
        instance = _make_synthetic_instance()
        perm, _ = iterated_greedy(instance, seed=123, config=self._config(operator))

        assert sorted(perm) == list(range(instance.n))

    def test_makespan_consistency(self, operator: str) -> None:
        """The reported makespan equals makespan(perm)."""
        instance = _make_synthetic_instance()
        perm, ms = iterated_greedy(instance, seed=123, config=self._config(operator))

        assert ms == makespan(instance.processing_times, perm)

    def test_monotonicity_vs_neh(self, operator: str) -> None:
        """Best makespan <= NEH makespan for every guided operator (Req 6.5)."""
        instance = _make_synthetic_instance()
        _, neh_makespan = neh(instance)
        _, ig_makespan = iterated_greedy(
            instance, seed=7, config=self._config(operator)
        )

        assert ig_makespan <= neh_makespan

    def test_determinism(self, operator: str) -> None:
        """Same operator + alpha + seed + instance -> identical result (Req 6.4)."""
        instance = _make_synthetic_instance()
        perm1, ms1 = iterated_greedy(instance, seed=999, config=self._config(operator))
        perm2, ms2 = iterated_greedy(instance, seed=999, config=self._config(operator))

        assert perm1 == perm2
        assert ms1 == ms2


# ---------------------------------------------------------------------------
# Test 4: Guided happy path on a small Taillard instance (skip if absent)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("operator", ["idle-greedy", "idle-rcl"])
class TestGuidedHappyPathTaillard:
    """The IG with a guided operator on tai20_5_0 (small real instance)."""

    def _config(self, operator: str) -> IGConfig:
        return IGConfig(
            destruction_size=4,
            destruction=operator,
            alpha=0.10,
            stop=StoppingCriterion(kind="iterations", value=50),
        )

    @_skip_without_taillard
    def test_valid_permutation_and_makespan(self, operator: str) -> None:
        """Valid permutation and makespan consistency on Taillard data."""
        instance = read_taillard(str(_TAILLARD_INSTANCE))
        perm, ms = iterated_greedy(instance, seed=42, config=self._config(operator))

        assert sorted(perm) == list(range(instance.n))
        assert ms == makespan(instance.processing_times, perm)

    @_skip_without_taillard
    def test_monotonicity_vs_neh(self, operator: str) -> None:
        """IG makespan <= NEH makespan on the real Taillard instance (Req 6.5)."""
        instance = read_taillard(str(_TAILLARD_INSTANCE))
        _, neh_makespan = neh(instance)
        _, ig_makespan = iterated_greedy(
            instance, seed=42, config=self._config(operator)
        )

        assert ig_makespan <= neh_makespan

    @_skip_without_taillard
    def test_lower_bound_respected(self, operator: str) -> None:
        """If the instance has a lower bound, best makespan >= LB (Req 6.5)."""
        instance = read_taillard(str(_TAILLARD_INSTANCE))
        _, ms = iterated_greedy(instance, seed=42, config=self._config(operator))

        if instance.lower_bound is not None:
            assert ms >= instance.lower_bound
