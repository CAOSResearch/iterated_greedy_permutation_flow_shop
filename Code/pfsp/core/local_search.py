"""Insertion-based local search for the Iterated Greedy algorithm.

Implements the local search operator described by Ruiz & Stützle (2005): in each
pass, jobs are scanned in a random order drawn from the RNG; each job is extracted
from its current position and reinserted at the position that minimizes the
makespan (using :func:`~pfsp.core.insertion.best_insertion`). Passes repeat until
a full pass produces no improvement (local optimum for the insertion neighborhood).

The function is deterministic under seed (the random scan order comes from `rng`)
and does not mutate the input permutation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pfsp.core.insertion import best_insertion
from pfsp.core.makespan import makespan


def local_search(
    permutation: list[int],
    processing_times: NDArray[np.int64],
    rng: np.random.Generator,
) -> tuple[list[int], int]:
    """Insertion-based local search until a full pass yields no improvement.

    Parameters
    ----------
    permutation:
        Initial job permutation (list of job indices, 0-based). Not mutated.
    processing_times:
        Full processing-time matrix of shape ``(m, n)`` for all ``n`` jobs.
    rng:
        NumPy random generator (seeded externally) used to determine the random
        job scan order in each pass.

    Returns
    -------
    tuple[list[int], int]
        A pair ``(improved_permutation, makespan_value)`` where the permutation
        is a local optimum with respect to the insertion neighborhood and the
        makespan is its corresponding ``C_max`` value.

    Notes
    -----
    - Reuses :func:`~pfsp.core.insertion.best_insertion` and
      :func:`~pfsp.core.makespan.makespan`; does not reimplement their logic
      (Req 1.1).
    - Deterministic under seed: the random scan order is drawn from ``rng``
      via ``rng.permutation(n)`` (Reqs 8.4, 9.1).
    - Does NOT mutate the input ``permutation`` (works on a copy).
    """
    current = list(permutation)
    current_makespan = makespan(processing_times, current)

    improved = True
    while improved:
        improved = False
        n = len(current)
        scan_order = rng.permutation(n)

        for idx in scan_order:
            job = current[idx]
            # Extract the job from the current sequence.
            sequence_without = [j for j in current if j != job]
            # Reinsert at the best position.
            new_sequence, new_makespan = best_insertion(
                sequence_without, job, processing_times
            )
            if new_makespan < current_makespan:
                current = new_sequence
                current_makespan = new_makespan
                improved = True

    return current, current_makespan
