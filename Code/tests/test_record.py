"""Tests for the RPD metric and no-fabrication rule (``pfsp.results.record``).

These tests cover Requisito 10.5 (and the RPD definition of Requisito 10.4):

* 10.5 — When no reference can be resolved, the RPD is recorded as an explicit
  unset value (``None``) rather than fabricating a reference. ``compute_rpd``
  therefore returns ``None`` for a ``None`` reference, and ``ResultRecord``
  serializes a ``None`` RPD as the empty string.
* 10.4 — ``RPD = 100 * (C_max - ref) / ref`` for a real reference, with the
  zero-deviation and known-value cases checked against fixed literals.
"""

from __future__ import annotations

from pfsp.results.record import CSV_HEADER, ResultRecord, compute_rpd

# --- Requisito 10.5: a missing reference is never fabricated ---------------


def test_compute_rpd_returns_none_without_reference() -> None:
    """A ``None`` reference yields ``None``, never a fabricated number (Req 10.5)."""
    assert compute_rpd(1278, None) is None


def test_compute_rpd_returns_none_for_zero_reference() -> None:
    """A degenerate zero reference is undefined and resolves to ``None`` (Req 10.5)."""
    assert compute_rpd(1278, 0) is None


# --- Requisito 10.4: RPD = 100 * (C_max - ref) / ref -----------------------


def test_compute_rpd_is_zero_when_makespan_matches_reference() -> None:
    """A makespan equal to the reference has zero deviation."""
    assert compute_rpd(695, 695) == 0.0


def test_compute_rpd_matches_formula_for_known_values() -> None:
    """RPD is the documented percentage for a worse-than-reference makespan."""
    # 100 * (712 - 695) / 695 = 2.4460431654676257
    assert compute_rpd(712, 695) == 100.0 * (712 - 695) / 695


def test_compute_rpd_is_negative_below_reference() -> None:
    """A makespan better than the reference yields a negative RPD."""
    assert compute_rpd(680, 695) == 100.0 * (680 - 695) / 695


# --- Requisito 10.5: an unset RPD is serialized as empty, not fabricated ---


def test_record_row_renders_none_rpd_as_empty_string() -> None:
    """A record with no resolvable reference writes an empty ``rpd`` cell (Req 10.5)."""
    record = ResultRecord(
        instance_id="vrf/VFR10_5_1",
        instance_set="Small",
        n=10,
        m=5,
        algorithm="NEH",
        seed=None,
        makespan=712,
        runtime_s=0.0009,
        rpd=None,
        manifest_id="run-20260607-153000-ab12",
        timestamp="2026-06-07T15:30:00Z",
    )

    row = record.to_row()
    rpd_cell = row[CSV_HEADER.index("rpd")]

    assert rpd_cell == ""


def test_record_row_renders_real_rpd() -> None:
    """A resolved RPD is written through to the corresponding CSV cell."""
    rpd = compute_rpd(712, 695)
    record = ResultRecord(
        instance_id="vrf/VFR10_5_1",
        instance_set="Small",
        n=10,
        m=5,
        algorithm="NEH",
        seed=None,
        makespan=712,
        runtime_s=0.0009,
        rpd=rpd,
        manifest_id="run-20260607-153000-ab12",
        timestamp="2026-06-07T15:30:00Z",
    )

    row = record.to_row()
    rpd_cell = row[CSV_HEADER.index("rpd")]

    assert rpd_cell == repr(rpd)
