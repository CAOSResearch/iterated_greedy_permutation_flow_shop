"""Seed management for reproducible stochastic processes.

This module centralizes the creation of NumPy random generators so that every
process that consumes randomness receives an explicit seed instead of relying on
an implicit global random state. NEH is deterministic and does not use this, but
the component is established here for the Iterated Greedy of later phases.
"""

from __future__ import annotations

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    """Create a NumPy random generator initialized with an explicit seed.

    Parameters
    ----------
    seed:
        A non-negative integer seed. The same seed always yields a generator
        that produces an identical sequence of random values, which is what
        makes stochastic runs reproducible.

    Returns
    -------
    numpy.random.Generator
        A generator created with ``numpy.random.default_rng(seed)``.

    Raises
    ------
    ValueError
        If ``seed`` is not a non-negative integer. Booleans are rejected even
        though ``bool`` is a subclass of ``int`` in Python, because a boolean
        seed is almost certainly a programming error rather than an intended
        value. No generator is returned when the seed is invalid.

    Notes
    -----
    The seed must be supplied explicitly: there is no implicit default. This
    enforces the project rule that any process consuming randomness declares its
    seed, keeping experiments reproducible.
    """
    # ``bool`` is a subclass of ``int``; exclude it explicitly so that ``True``
    # or ``False`` are not silently accepted as the seeds 1 or 0.
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError(
            f"seed must be a non-negative integer, got {seed!r} "
            f"of type {type(seed).__name__}"
        )
    if seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed}")
    return np.random.default_rng(seed)


__all__ = ["make_rng"]
