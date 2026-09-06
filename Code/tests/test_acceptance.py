"""Tests for the SA acceptance criterion and temperature computation.

Validates Requirements 5.1, 5.2, 5.3, 5.4.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pfsp.core.ig import accept, compute_temperature


class TestAccept:
    """Tests for the accept() function."""

    def test_accepts_improvement(self) -> None:
        """Accept when candidate makespan is strictly less than current."""
        rng = np.random.default_rng(42)
        result = accept(
            candidate_makespan=100,
            current_makespan=110,
            temperature=1.0,
            rng=rng,
        )
        assert result is True

    def test_accepts_equal(self) -> None:
        """Accept when candidate makespan equals current makespan."""
        rng = np.random.default_rng(42)
        result = accept(
            candidate_makespan=100,
            current_makespan=100,
            temperature=1.0,
            rng=rng,
        )
        assert result is True

    def test_rejects_worsening_with_zero_temperature(self) -> None:
        """With T=0, reject any worsening candidate."""
        rng = np.random.default_rng(42)
        result = accept(
            candidate_makespan=120,
            current_makespan=100,
            temperature=0.0,
            rng=rng,
        )
        assert result is False

    def test_worsening_with_positive_temperature_matches_formula(self) -> None:
        """With T>0 and a seeded rng, the decision matches exp(-delta/T).

        We create a fresh rng, draw the random value ourselves to know what
        accept() will compare against, then reset and call accept().
        """
        seed = 12345
        candidate_makespan = 150
        current_makespan = 100
        temperature = 50.0

        delta = candidate_makespan - current_makespan
        probability = math.exp(-delta / temperature)

        # Draw from a fresh rng to know the random value that accept() will use
        rng_check = np.random.default_rng(seed)
        random_draw = rng_check.random()
        expected = random_draw <= probability

        # Now call accept with an identically-seeded rng
        rng = np.random.default_rng(seed)
        result = accept(
            candidate_makespan=candidate_makespan,
            current_makespan=current_makespan,
            temperature=temperature,
            rng=rng,
        )

        assert result is expected

    def test_worsening_with_positive_temperature_deterministic(self) -> None:
        """Two calls with the same rng seed produce the same decision."""
        seed = 99
        kwargs = {
            "candidate_makespan": 200,
            "current_makespan": 180,
            "temperature": 10.0,
        }

        result1 = accept(**kwargs, rng=np.random.default_rng(seed))
        result2 = accept(**kwargs, rng=np.random.default_rng(seed))

        assert result1 == result2


class TestComputeTemperature:
    """Tests for the compute_temperature() function."""

    def test_matches_manual_calculation(self) -> None:
        """Temperature matches T = Tp * sum(p) / (n * m * 10) on a small case."""
        # 2 machines, 3 jobs
        processing_times = np.array([[10, 20, 30], [5, 15, 25]], dtype=np.int64)
        temperature_factor = 0.4

        # Manual: sum = 10+20+30+5+15+25 = 105; n=3, m=2
        # T = 0.4 * 105 / (3 * 2 * 10) = 42.0 / 60 = 0.7
        expected = 0.4 * 105 / (3 * 2 * 10)

        result = compute_temperature(processing_times, temperature_factor)

        assert result == pytest.approx(expected)

    def test_zero_temperature_factor(self) -> None:
        """With Tp=0, the temperature is 0."""
        processing_times = np.array([[10, 20], [5, 15]], dtype=np.int64)
        result = compute_temperature(processing_times, temperature_factor=0.0)
        assert result == 0.0

    def test_single_job_single_machine(self) -> None:
        """Temperature on a trivial 1x1 instance."""
        processing_times = np.array([[7]], dtype=np.int64)
        temperature_factor = 1.0

        # T = 1.0 * 7 / (1 * 1 * 10) = 0.7
        expected = 1.0 * 7 / (1 * 1 * 10)

        result = compute_temperature(processing_times, temperature_factor)

        assert result == pytest.approx(expected)
