"""Tests for the common ``Instance`` contract.

These tests build a small synthetic instance and check the Instance_Contract:
mandatory fields are present, optional fields default to ``None``, the
``identifier`` property is derived correctly, and the processing-time matrix is
made read-only on construction (Requisitos 7.1, 7.3, 7.4).
"""

from __future__ import annotations

import numpy as np
import pytest

from pfsp.instance import Instance


def _make_synthetic_instance() -> Instance:
    """Build a minimal valid ``(m, n)`` instance (2 machines, 3 jobs)."""
    processing_times = np.array(
        [[3, 1, 4], [2, 5, 1]],
        dtype=np.int64,
    )
    return Instance(
        processing_times=processing_times,
        n=3,
        m=2,
        source_benchmark="taillard",
        source_path="data/taillard/tai20_5_0.fsp",
    )


def test_mandatory_fields_present() -> None:
    """Mandatory fields are stored with the supplied values and `(m, n)` shape."""
    instance = _make_synthetic_instance()

    assert instance.n == 3
    assert instance.m == 2
    assert instance.source_benchmark == "taillard"
    assert instance.source_path == "data/taillard/tai20_5_0.fsp"
    assert instance.processing_times.shape == (instance.m, instance.n)
    assert instance.processing_times.dtype == np.int64


def test_optional_fields_default_to_none() -> None:
    """Optional metadata fields default to ``None`` rather than fabricated values."""
    instance = _make_synthetic_instance()

    assert instance.instance_set is None
    assert instance.seed is None
    assert instance.upper_bound is None
    assert instance.lower_bound is None


def test_identifier_is_derived_from_benchmark_and_stem() -> None:
    """``identifier`` combines the benchmark with the source file stem."""
    instance = _make_synthetic_instance()

    assert instance.identifier == "taillard/tai20_5_0"


def test_identifier_with_vrf_path() -> None:
    """``identifier`` works for VRF-style paths and arbitrary directories."""
    processing_times = np.array([[1, 2], [3, 4]], dtype=np.int64)
    instance = Instance(
        processing_times=processing_times,
        n=2,
        m=2,
        source_benchmark="vrf",
        source_path="data/vrf/Small/VFR10_5_1_Gap.txt",
        instance_set="Small",
    )

    assert instance.identifier == "vrf/VFR10_5_1_Gap"
    assert instance.instance_set == "Small"


def test_processing_times_is_read_only() -> None:
    """Writing to the processing-time matrix raises an error (input immutability)."""
    instance = _make_synthetic_instance()

    assert instance.processing_times.flags.writeable is False
    with pytest.raises(ValueError):
        instance.processing_times[0, 0] = 99
