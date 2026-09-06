"""Common instance contract for the PFSP toolkit.

This module defines the single ``Instance`` type to which both the Taillard and
VRF readers normalize their input. Keeping one contract means the algorithmic
core (makespan, NEH and, later, the Iterated Greedy) does not depend on which
benchmark produced an instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePath

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Instance:
    """A PFSP instance normalized to the internal ``(m, n)`` convention.

    Attributes
    ----------
    processing_times:
        Processing-time matrix of shape ``(m, n)`` with integer dtype, where
        ``processing_times[i][j]`` is the processing time of job ``j`` on
        machine ``i`` (zero-based indices). The array is made read-only on
        construction to enforce input immutability.
    n:
        Number of jobs.
    m:
        Number of machines.
    source_benchmark:
        Benchmark of origin, e.g. ``"taillard"`` or ``"vrf"``.
    source_path:
        Path of the source file the instance was parsed from (kept
        relative-to-repo by callers for portability and reproducibility).
    instance_set:
        VRF instance set (``"Small"`` or ``"Large"``); ``None`` when it does not
        apply (e.g. Taillard).
    seed:
        Random seed recorded in the source file; ``None`` when the format does
        not provide it (e.g. VRF). Never fabricated.
    upper_bound:
        Best-known upper bound recorded in the source file; ``None`` when the
        format does not provide it. Never fabricated.
    lower_bound:
        Lower bound recorded in the source file; ``None`` when the format does
        not provide it. Never fabricated.
    """

    processing_times: NDArray[np.int64]
    n: int
    m: int
    source_benchmark: str
    source_path: str
    instance_set: str | None = None
    seed: int | None = None
    upper_bound: int | None = None
    lower_bound: int | None = None

    def __post_init__(self) -> None:
        # Enforce input immutability: the processing-time matrix must never be
        # mutated downstream (makespan and NEH work on reordered copies).
        # Instance is frozen, so the array reference cannot be reassigned; we
        # only need to make the underlying buffer non-writeable.
        self.processing_times.flags.writeable = False

    @property
    def identifier(self) -> str:
        """Stable, unique identifier of the instance.

        Formed by the source benchmark and the file stem of ``source_path``,
        e.g. ``"taillard/tai20_5_0"`` or ``"vrf/VFR10_5_1"``.
        """
        stem = PurePath(self.source_path).stem
        return f"{self.source_benchmark}/{stem}"
