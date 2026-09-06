"""Tests for the NEH constructive heuristic (``neh``).

Covers the NEH acceptance criterion of the Project_Skeleton test suite:

* Req 6.5: on a Taillard instance with at most 20 jobs and at most 5 machines,
  ``neh`` returns a Permutation that contains each of the ``n`` job indices
  exactly once and a makespan equal to the value returned by ``makespan`` for
  that Permutation.

The real Taillard instance ``data/taillard/tai20_5_0.fsp`` (20 jobs, 5 machines)
lives outside git, so the test is skipped when the file is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pfsp.core.makespan import makespan
from pfsp.core.neh import neh
from pfsp.io.loader import load_instance

# Repository ``Code`` directory (parent of ``tests/``), used to resolve the real
# benchmark instance path robustly regardless of the working directory.
_CODE_DIR = Path(__file__).resolve().parent.parent
_TAILLARD_INSTANCE = _CODE_DIR / "data" / "taillard" / "tai20_5_0.fsp"


@pytest.mark.skipif(
    not _TAILLARD_INSTANCE.exists(),
    reason=f"Taillard benchmark data not present at {_TAILLARD_INSTANCE} (out of git).",
)
def test_neh_returns_valid_permutation_and_consistent_makespan() -> None:
    """NEH yields a valid permutation and a consistent makespan (Req 6.5)."""
    instance = load_instance(str(_TAILLARD_INSTANCE))

    # Guard: the instance is within the required size bounds (<= 20 jobs,
    # <= 5 machines).
    assert instance.n <= 20
    assert instance.m <= 5

    permutation, reported_makespan = neh(instance)

    # The permutation contains each of the n job indices exactly once: it has
    # length n and, once sorted, equals range(n).
    assert len(permutation) == instance.n
    assert sorted(permutation) == list(range(instance.n))

    # The reported makespan equals the value computed by the makespan calculator
    # for the returned permutation (NEH keeps no separate bookkeeping).
    assert reported_makespan == makespan(instance.processing_times, permutation)


@pytest.mark.skipif(
    not _TAILLARD_INSTANCE.exists(),
    reason=f"Taillard benchmark data not present at {_TAILLARD_INSTANCE} (out of git).",
)
def test_neh_is_deterministic() -> None:
    """Repeated NEH runs on the same instance are identical (Req 5.6)."""
    instance = load_instance(str(_TAILLARD_INSTANCE))

    first_permutation, first_makespan = neh(instance)
    second_permutation, second_makespan = neh(instance)

    assert first_permutation == second_permutation
    assert first_makespan == second_makespan


def test_neh_valid_permutation_and_makespan_synthetic() -> None:
    """NEH yields a valid permutation and consistent makespan (Req 6.5).

    Self-contained variant that exercises the criterion on a small synthetic
    instance, without depending on the out-of-git benchmark data.
    """
    from pfsp.instance import Instance

    # A 3-machine, 5-job instance with distinct total processing times.
    processing_times = np.array(
        [
            [5, 9, 3, 7, 2],
            [8, 1, 6, 4, 9],
            [3, 7, 2, 5, 6],
        ],
        dtype=np.int64,
    )
    instance = Instance(
        processing_times=processing_times,
        n=5,
        m=3,
        source_benchmark="taillard",
        source_path="synthetic/tai_synthetic.fsp",
    )

    permutation, reported_makespan = neh(instance)

    assert len(permutation) == instance.n
    assert sorted(permutation) == list(range(instance.n))
    assert reported_makespan == makespan(instance.processing_times, permutation)
