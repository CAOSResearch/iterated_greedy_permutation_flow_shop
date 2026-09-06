"""NEH constructive heuristic (baseline) for the PFSP.

This module implements the classic Nawaz-Enscore-Ham (NEH) constructive
heuristic. NEH builds a permutation in two stages:

1. **Initial ordering.** Jobs are sorted by non-increasing total processing
   time across all machines. Ties between jobs of equal total processing time
   are broken by ascending job index.
2. **Incremental construction.** Jobs are inserted one at a time, in the initial
   ordering, into the position of the current partial sequence that yields the
   lowest makespan. Ties between insertion positions of equal makespan are
   broken by choosing the lowest insertion index.

Both tie-breaking rules are fixed, which makes the heuristic deterministic: the
same instance always yields the same permutation and makespan. NEH uses no
randomness.

Every makespan evaluation goes through :func:`pfsp.core.makespan.makespan`, so
the reported value is consistent with the completion-time recurrence. To
evaluate the makespan of a partial sequence ``seq`` of job indices, the
processing-time matrix is restricted to those columns
(``processing_times[:, seq]``) and evaluated with the identity permutation
``range(len(seq))``; this never mutates the instance matrix (which is read-only
anyway).
"""

from __future__ import annotations

import numpy as np

from pfsp.core.insertion import best_insertion
from pfsp.core.makespan import makespan
from pfsp.instance import Instance


def neh(instance: Instance) -> tuple[list[int], int]:
    """Run the NEH constructive heuristic on an instance.

    Parameters
    ----------
    instance:
        The PFSP instance to solve. Its ``processing_times`` matrix of shape
        ``(m, n)`` is read but never mutated.

    Returns
    -------
    tuple[list[int], int]
        A pair ``(permutation, makespan)`` where ``permutation`` is a list of
        the ``n`` job indices, each appearing exactly once, and ``makespan`` is
        the makespan of that permutation computed with the makespan
        recurrence.

    Notes
    -----
    The heuristic is deterministic: repeated calls on the same instance return
    an identical permutation and makespan (Req 5.6).
    """
    p = instance.processing_times
    n = instance.n

    # --- Stage 1: initial ordering (Req 5.2) ---
    # Total processing time per job, summed across all machines.
    total_time = np.asarray(p).sum(axis=0)
    # Sort by non-increasing total time; break ties by ascending job index.
    # Sorting the (job index) keys by (-total_time, index) lexicographically
    # achieves exactly this with a stable, deterministic order.
    ordering = sorted(range(n), key=lambda job: (-int(total_time[job]), job))

    # --- Stage 2: incremental construction (Reqs 5.1, 5.3, 5.4, 5.5) ---
    # Uses the shared best_insertion helper (Req 4.4) which inserts the job at
    # the position with the lowest makespan, breaking ties by lowest index.
    sequence: list[int] = []
    for job in ordering:
        sequence, _ = best_insertion(sequence, job, p)

    # --- Final makespan, consistent with the recurrence (Reqs 5.1, 5.5) ---
    final_makespan = makespan(p, sequence)
    return sequence, final_makespan
