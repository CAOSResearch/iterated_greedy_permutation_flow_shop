"""Tests for the Seed_Manager (``pfsp.repro.seeds.make_rng``).

These tests cover Requisitos 1.7 and 1.8:

* 1.7 — When invoked two or more times with the same integer seed, the
  Seed_Manager returns generators that produce identical sequences of random
  values (reproducibility). As a cheap sanity check, distinct seeds yield
  different sequences.
* 1.8 — When a value that is not a non-negative integer is supplied as the
  seed, the Seed_Manager raises a descriptive ``ValueError`` and does not return
  a generator.
"""

from __future__ import annotations

import numpy as np
import pytest

from pfsp.repro.seeds import make_rng

# --- Requisito 1.7: same seed -> identical sequences -----------------------


def test_same_seed_produces_identical_float_sequences() -> None:
    """Two generators built from the same seed produce identical ``random``
    sequences."""
    rng_a = make_rng(12345)
    rng_b = make_rng(12345)

    assert np.array_equal(rng_a.random(10), rng_b.random(10))


def test_same_seed_produces_identical_integer_sequences() -> None:
    """Two generators built from the same seed produce identical ``integers``
    sequences."""
    rng_a = make_rng(2025)
    rng_b = make_rng(2025)

    np.testing.assert_array_equal(
        rng_a.integers(0, 1000, size=20),
        rng_b.integers(0, 1000, size=20),
    )


def test_seed_zero_is_accepted_and_reproducible() -> None:
    """Zero is a valid non-negative integer seed and is reproducible."""
    rng_a = make_rng(0)
    rng_b = make_rng(0)

    assert np.array_equal(rng_a.random(5), rng_b.random(5))


def test_numpy_integer_seed_is_accepted_and_reproducible() -> None:
    """A NumPy integer seed is accepted and behaves like the equivalent int."""
    rng_a = make_rng(np.int64(777))
    rng_b = make_rng(777)

    assert np.array_equal(rng_a.random(5), rng_b.random(5))


def test_different_seeds_produce_different_sequences() -> None:
    """Distinct seeds yield different sequences (cheap sanity check)."""
    rng_a = make_rng(1)
    rng_b = make_rng(2)

    assert not np.array_equal(rng_a.random(10), rng_b.random(10))


# --- Requisito 1.8: invalid seed raises and returns no generator -----------


@pytest.mark.parametrize(
    "invalid_seed",
    [
        -1,  # negative integer
        1.5,  # non-integer float
        "3",  # string
        None,  # missing value
        True,  # bool (subclass of int, rejected on purpose)
    ],
)
def test_invalid_seed_raises_value_error(invalid_seed: object) -> None:
    """Each invalid seed raises a descriptive ``ValueError``; no generator is
    returned because the exception propagates out of ``make_rng``."""
    with pytest.raises(ValueError, match="seed"):
        make_rng(invalid_seed)  # type: ignore[arg-type]
