"""Tests for the instance I/O subpackage (``pfsp.io``).

Single test module for the I/O layer — the Taillard reader, the VRF reader and
the format-agnostic loader — so the shared scaffolding (real-benchmark paths,
skip markers, the mandatory-field set) is defined once instead of being repeated
per file. Unlike the runner refactor there is no production duplication to
extract: the readers are already distinct modules in the package; this only
groups their tests by subpackage.

Acceptance criteria covered (Project_Skeleton test suite):

* Req 6.2 — ``read_taillard`` parses ``data/taillard/tai20_5_0.fsp`` into a
  ``(5, 20)`` matrix with its metadata.
* Req 6.3 — ``read_vrf`` parses ``data/vrf/Small/VFR10_5_1_Gap.txt`` into a
  ``(5, 10)`` matrix with ``instance_set == "Small"``.
* Req 6.6 — invalid files make the readers raise ``InstanceFormatError`` and
  return no Instance.
* Req 7.7 — Taillard and VRF load through ``load_instance`` to the same
  ``Instance`` type and contract.
* Req 6.4 — column reorder by ``pi`` then ``pi^-1`` reproduces the matrix.

Tests touching the real benchmark instances are skipped when the data (out of
git) is absent; the error and round-trip cases are self-contained.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pfsp import InstanceFormatError
from pfsp.instance import Instance
from pfsp.io.loader import load_instance
from pfsp.io.taillard import read_taillard
from pfsp.io.vrf import read_vrf

# Repository ``Code`` directory (parent of ``tests/``), used to resolve the real
# benchmark instance paths robustly regardless of the working directory.
_CODE_DIR = Path(__file__).resolve().parent.parent
_TAILLARD_INSTANCE = _CODE_DIR / "data" / "taillard" / "tai20_5_0.fsp"
_VRF_INSTANCE = _CODE_DIR / "data" / "vrf" / "Small" / "VFR10_5_1_Gap.txt"

# Mandatory fields of the Instance_Contract (Req 7.1). Both benchmarks must
# expose exactly this set regardless of their origin format.
_MANDATORY_FIELDS = {
    "processing_times",
    "n",
    "m",
    "source_benchmark",
    "source_path",
}

# Reusable skip markers for the tests that need the out-of-git benchmark data.
_skip_without_taillard = pytest.mark.skipif(
    not _TAILLARD_INSTANCE.exists(),
    reason=f"Taillard benchmark data not present at {_TAILLARD_INSTANCE} (out of git).",
)
_skip_without_vrf = pytest.mark.skipif(
    not _VRF_INSTANCE.exists(),
    reason=f"VRF benchmark data not present at {_VRF_INSTANCE} (out of git).",
)
_skip_without_both = pytest.mark.skipif(
    not _TAILLARD_INSTANCE.exists() or not _VRF_INSTANCE.exists(),
    reason=(
        f"Benchmark data not present at {_TAILLARD_INSTANCE} or "
        f"{_VRF_INSTANCE} (out of git)."
    ),
)


# --------------------------------------------------------------------------- #
# Taillard reader (pfsp/io/taillard.py)
# --------------------------------------------------------------------------- #


@_skip_without_taillard
def test_reads_real_taillard_instance() -> None:
    """``tai20_5_0.fsp`` parses to a ``(5, 20)`` matrix with its metadata (Req 6.2)."""
    instance = read_taillard(str(_TAILLARD_INSTANCE))

    # Rows = machines (m), columns = jobs (n): shape is (5, 20).
    assert instance.processing_times.shape == (5, 20)
    assert instance.processing_times.dtype == np.int64
    assert instance.n == 20
    assert instance.m == 5

    # Metadata recorded from the header line, never fabricated.
    assert instance.seed == 873654221
    assert instance.upper_bound == 1278
    assert instance.lower_bound == 1232

    assert instance.source_benchmark == "taillard"
    # Spot-check the corners of the matrix against the file content.
    assert instance.processing_times[0, 0] == 54
    assert instance.processing_times[4, 19] == 28


def test_taillard_invalid_file_raises(tmp_path: Path) -> None:
    """An invalid file makes the reader raise and return no Instance (Req 6.6)."""
    # Matrix rows of unequal length cannot form a rectangular (m, n) matrix.
    invalid_file = tmp_path / "invalid.fsp"
    invalid_file.write_text(
        "processing times :\n 1 2 3 4\n 5 6 7\n",
        encoding="utf-8",
    )

    with pytest.raises(InstanceFormatError):
        read_taillard(str(invalid_file))


def test_taillard_negative_processing_time_raises(tmp_path: Path) -> None:
    """A negative processing time is rejected with a descriptive error (Req 6.6)."""
    invalid_file = tmp_path / "negative.fsp"
    invalid_file.write_text(
        "processing times :\n 1 2 3\n 4 -5 6\n",
        encoding="utf-8",
    )

    with pytest.raises(InstanceFormatError):
        read_taillard(str(invalid_file))


# --------------------------------------------------------------------------- #
# VRF reader (pfsp/io/vrf.py)
# --------------------------------------------------------------------------- #


@_skip_without_vrf
def test_reads_real_vrf_instance() -> None:
    """``VFR10_5_1_Gap.txt`` parses to a ``(5, 10)`` matrix with metadata (Req 6.3)."""
    instance = read_vrf(str(_VRF_INSTANCE))

    # Rows = machines (m), columns = jobs (n): shape is (5, 10).
    assert instance.processing_times.shape == (5, 10)
    assert instance.processing_times.dtype == np.int64
    assert instance.n == 10
    assert instance.m == 5

    # ``instance_set`` is derived from the ``Small`` path segment.
    assert instance.instance_set == "Small"
    assert instance.source_benchmark == "vrf"

    # The VRF format provides no seed/UB/LB, so these stay unset (never faked).
    assert instance.seed is None
    assert instance.upper_bound is None
    assert instance.lower_bound is None

    # Job 0's times across machines 0..4 are the first body row of the file.
    assert list(instance.processing_times[:, 0]) == [45, 31, 54, 54, 64]


def test_vrf_invalid_file_raises(tmp_path: Path) -> None:
    """A wrong number of body rows makes the reader raise and return none (Req 6.6)."""
    # Header declares n=2 jobs and m=2 machines but only one body row follows.
    invalid_file = tmp_path / "invalid.txt"
    invalid_file.write_text("2 2\n 0 5 1 7\n", encoding="utf-8")

    with pytest.raises(InstanceFormatError):
        read_vrf(str(invalid_file))


def test_vrf_duplicate_machine_index_raises(tmp_path: Path) -> None:
    """A repeated machine index (not exactly 0..m-1) is rejected (Req 6.6)."""
    # Header n=2, m=2; both rows use machine index 0 twice instead of 0 and 1.
    invalid_file = tmp_path / "duplicate.txt"
    invalid_file.write_text("2 2\n 0 5 0 7\n 0 3 0 9\n", encoding="utf-8")

    with pytest.raises(InstanceFormatError):
        read_vrf(str(invalid_file))


def test_vrf_negative_processing_time_raises(tmp_path: Path) -> None:
    """A negative processing time is rejected with a descriptive error (Req 6.6)."""
    invalid_file = tmp_path / "negative.txt"
    invalid_file.write_text("2 2\n 0 5 1 7\n 0 3 1 -9\n", encoding="utf-8")

    with pytest.raises(InstanceFormatError):
        read_vrf(str(invalid_file))


# --------------------------------------------------------------------------- #
# Loader and contract equivalence (pfsp/io/loader.py)
# --------------------------------------------------------------------------- #


@_skip_without_both
def test_loader_equivalence_across_benchmarks() -> None:
    """Taillard and VRF load to the same type and contract (Req 7.7)."""
    taillard = load_instance(str(_TAILLARD_INSTANCE))
    vrf = load_instance(str(_VRF_INSTANCE))

    # Both are the common Instance type, regardless of source benchmark.
    assert type(taillard) is type(vrf)
    assert isinstance(taillard, Instance)
    assert isinstance(vrf, Instance)

    # Both expose the same set of mandatory field names.
    for instance in (taillard, vrf):
        names = set(vars(instance))
        assert _MANDATORY_FIELDS <= names

    # Both expose a Processing_Time_Matrix of shape (m, n).
    assert taillard.processing_times.shape == (taillard.m, taillard.n)
    assert vrf.processing_times.shape == (vrf.m, vrf.n)

    # The benchmark of origin is recorded distinctly on each.
    assert taillard.source_benchmark == "taillard"
    assert vrf.source_benchmark == "vrf"


@_skip_without_taillard
def test_column_reorder_round_trip_on_parsed_instance() -> None:
    """Reordering by ``pi`` then ``pi^-1`` reproduces the matrix (Req 6.4)."""
    instance = load_instance(str(_TAILLARD_INSTANCE))
    original = instance.processing_times

    # A non-identity permutation of the n job indices: reverse the job order.
    pi = np.arange(instance.n)[::-1]
    assert not np.array_equal(pi, np.arange(instance.n))  # guard: non-identity.

    # Inverse permutation: pi_inverse[pi[k]] == k.
    pi_inverse = np.argsort(pi)

    # Reorder columns by pi, then by its inverse. Fancy indexing returns copies,
    # so the read-only input matrix is never mutated.
    reordered = original[:, pi]
    restored = reordered[:, pi_inverse]

    assert np.array_equal(restored, original)
    # The single application by pi is a genuine reordering (sanity on the setup).
    assert not np.array_equal(reordered, original)


def test_column_reorder_round_trip_synthetic() -> None:
    """Round-trip holds for an arbitrary (m, n) matrix (Req 6.4).

    Self-contained variant that exercises the round-trip property without
    depending on the out-of-git benchmark data.
    """
    m, n = 3, 5
    original = np.arange(m * n, dtype=np.int64).reshape(m, n)

    # Non-identity permutation of the n columns.
    pi = np.array([2, 0, 4, 1, 3])
    pi_inverse = np.argsort(pi)

    restored = original[:, pi][:, pi_inverse]

    assert np.array_equal(restored, original)
