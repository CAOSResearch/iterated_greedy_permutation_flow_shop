"""Tests for the Reference_Provider (``pfsp.results.reference``).

These tests cover Requisitos 10.3 and 10.5:

* 10.3 — The Reference_Provider resolves ``ref`` from benchmark metadata: the
  recorded ``UB`` for Taillard instances, and the best-known upper bound from
  the normalized VRF bounds CSV for VRF instances.
* 10.5 — When a reference cannot be resolved, ``get_reference`` returns ``None``
  rather than fabricating a value.

The VRF and Taillard checks load real benchmark instances through the
format-agnostic loader. That data lives outside git, so each such test is
skipped when its instance file (or the bounds CSV) is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pfsp.instance import Instance
from pfsp.io.loader import load_instance
from pfsp.results.reference import get_reference

# Repository ``Code`` directory (parent of ``tests/``), used to resolve the real
# benchmark paths robustly regardless of the working directory.
_CODE_DIR = Path(__file__).resolve().parent.parent
_VRF_INSTANCE = _CODE_DIR / "data" / "vrf" / "Small" / "VFR10_5_1_Gap.txt"
_VRF_BOUNDS_CSV = _CODE_DIR / "data" / "vrf" / "best_solutions_and_bounds.csv"
_TAILLARD_INSTANCE = _CODE_DIR / "data" / "taillard" / "tai20_5_0.fsp"

#: Best-known upper bound of ``VFR10_5_1`` as recorded in the bounds CSV.
_VRF10_5_1_UPPER_BOUND = 695


# --- Requisito 10.3: VRF reference resolved from the bounds CSV ------------


@pytest.mark.skipif(
    not (_VRF_INSTANCE.exists() and _VRF_BOUNDS_CSV.exists()),
    reason=(
        f"VRF benchmark data not present at {_VRF_INSTANCE} or "
        f"{_VRF_BOUNDS_CSV} (out of git)."
    ),
)
def test_get_reference_returns_vrf_upper_bound_from_csv() -> None:
    """A known VRF instance resolves to its UB from the CSV (Reqs 10.3, 10.5)."""
    instance = load_instance(str(_VRF_INSTANCE))

    assert get_reference(instance) == _VRF10_5_1_UPPER_BOUND


# --- Requisito 10.3: Taillard reference taken from the instance UB ----------


@pytest.mark.skipif(
    not _TAILLARD_INSTANCE.exists(),
    reason=f"Taillard benchmark data not present at {_TAILLARD_INSTANCE} (out of git).",
)
def test_get_reference_returns_taillard_upper_bound() -> None:
    """A Taillard instance resolves to the UB embedded in its file (Req 10.3)."""
    instance = load_instance(str(_TAILLARD_INSTANCE))

    # ``tai20_5_0.fsp`` records an upper bound of 1278 in its header.
    assert get_reference(instance) == instance.upper_bound
    assert get_reference(instance) == 1278


# --- Requisito 10.5: an unresolvable reference is never fabricated ----------


def _make_synthetic_instance(benchmark: str, source_path: str) -> Instance:
    """Build a minimal in-memory instance for no-fabrication checks."""
    processing_times = np.array([[1, 2], [3, 4]], dtype=np.int64)
    return Instance(
        processing_times=processing_times,
        n=2,
        m=2,
        source_benchmark=benchmark,
        source_path=source_path,
    )


def test_get_reference_is_none_for_unknown_benchmark() -> None:
    """An unrecognized benchmark resolves to ``None``, not a guess (Req 10.5)."""
    instance = _make_synthetic_instance("unknown", "somewhere/mystery.txt")

    assert get_reference(instance) is None


def test_get_reference_is_none_for_taillard_without_upper_bound() -> None:
    """A Taillard instance lacking a UB stays unresolved (``None``) (Req 10.5)."""
    instance = _make_synthetic_instance("taillard", "taillard/no_bound.fsp")

    assert get_reference(instance) is None


def test_get_reference_is_none_for_vrf_not_in_csv() -> None:
    """A VRF instance absent from the bounds CSV resolves to ``None`` (Req 10.5)."""
    instance = _make_synthetic_instance("vrf", "vrf/Small/VFR_does_not_exist_Gap.txt")

    assert get_reference(instance) is None
