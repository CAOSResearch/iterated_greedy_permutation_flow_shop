"""Tests for IGConfig, StoppingCriterion, and validate_config.

Verifies default values, valid configurations, and descriptive ValueError
for invalid parameters (destruction_size, temperature_factor, stopping
criterion).

Spec: iterated-greedy, task 7.
Requirements: 6.1, 6.2, 6.5, 6.6, 3.5, 5.6, 8.1.
"""

from __future__ import annotations

import pytest

from pfsp.core.ig import (
    DEFAULT_MAX_EVALUATIONS,
    IGConfig,
    StoppingCriterion,
    validate_config,
)

# ---------------------------------------------------------------------------
# Test 1: Default values are correct
# ---------------------------------------------------------------------------


class TestDefaults:
    """Verify that the default IGConfig matches the design specification."""

    def test_stopping_criterion_defaults(self) -> None:
        stop = StoppingCriterion()
        assert stop.kind == "evaluations"
        assert stop.value == DEFAULT_MAX_EVALUATIONS

    def test_ig_config_defaults(self) -> None:
        config = IGConfig()
        assert config.destruction_size == 4
        assert config.temperature_factor == 0.4
        assert config.local_search is True
        assert config.stop.kind == "evaluations"
        assert config.stop.value == DEFAULT_MAX_EVALUATIONS


# ---------------------------------------------------------------------------
# Test 8: Valid config passes without error
# ---------------------------------------------------------------------------


class TestValidConfig:
    """A valid configuration must not raise any error."""

    def test_valid_default_config(self) -> None:
        config = IGConfig()
        # n=10 jobs; d=4 is in [1, 9]
        validate_config(config, n=10)

    def test_valid_custom_config(self) -> None:
        config = IGConfig(
            destruction_size=2,
            temperature_factor=0.0,
            local_search=False,
            stop=StoppingCriterion(kind="iterations", value=100),
        )
        validate_config(config, n=5)

    def test_valid_time_stop(self) -> None:
        config = IGConfig(
            destruction_size=1,
            temperature_factor=1.5,
            stop=StoppingCriterion(kind="time", value=30.0),
        )
        validate_config(config, n=20)


# ---------------------------------------------------------------------------
# Tests 2-7: Invalid parameters raise ValueError
# ---------------------------------------------------------------------------


class TestDestructionSizeValidation:
    """destruction_size (d) must be an integer in [1, n-1]."""

    def test_d_zero_raises(self) -> None:
        """Test 2: d=0 -> ValueError."""
        config = IGConfig(destruction_size=0)
        with pytest.raises(ValueError, match="destruction_size"):
            validate_config(config, n=10)

    def test_d_equal_n_raises(self) -> None:
        """Test 3: d=n -> ValueError (d must be < n)."""
        config = IGConfig(destruction_size=10)
        with pytest.raises(ValueError, match="destruction_size"):
            validate_config(config, n=10)

    def test_d_greater_than_n_raises(self) -> None:
        config = IGConfig(destruction_size=15)
        with pytest.raises(ValueError, match="destruction_size"):
            validate_config(config, n=10)

    def test_d_negative_raises(self) -> None:
        config = IGConfig(destruction_size=-1)
        with pytest.raises(ValueError, match="destruction_size"):
            validate_config(config, n=10)

    def test_d_float_raises(self) -> None:
        """Test 4: d not int (e.g. float) -> ValueError."""
        # Force a float value bypassing type hints (runtime check)
        config = IGConfig(destruction_size=4.0)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="must be an integer"):
            validate_config(config, n=10)


class TestTemperatureFactorValidation:
    """temperature_factor (Tp) must be >= 0."""

    def test_tp_negative_raises(self) -> None:
        """Test 5: Tp < 0 -> ValueError."""
        config = IGConfig(temperature_factor=-0.1)
        with pytest.raises(ValueError, match="temperature_factor"):
            validate_config(config, n=10)

    def test_tp_zero_valid(self) -> None:
        """Tp=0 is valid (disables probabilistic acceptance)."""
        config = IGConfig(temperature_factor=0.0)
        validate_config(config, n=10)


class TestStoppingCriterionValidation:
    """Stopping criterion must have a recognized kind and positive value."""

    def test_invalid_stop_kind_raises(self) -> None:
        """Test 6: Invalid stop kind -> ValueError."""
        # Force an invalid kind bypassing Literal type hints
        stop = StoppingCriterion(kind="invalid_kind", value=100)  # type: ignore[arg-type]
        config = IGConfig(stop=stop)
        with pytest.raises(ValueError, match="Unrecognized stopping criterion kind"):
            validate_config(config, n=10)

    def test_stop_value_zero_raises(self) -> None:
        """Test 7: stop.value <= 0 -> ValueError."""
        stop = StoppingCriterion(kind="evaluations", value=0)
        config = IGConfig(stop=stop)
        with pytest.raises(ValueError, match="must be positive"):
            validate_config(config, n=10)

    def test_stop_value_negative_raises(self) -> None:
        stop = StoppingCriterion(kind="iterations", value=-10)
        config = IGConfig(stop=stop)
        with pytest.raises(ValueError, match="must be positive"):
            validate_config(config, n=10)
