"""Tests for the Aggregator (``pfsp.results.aggregate``).

These tests cover the summary-derivation requirements of the results framework:

* 10.6 / 10.10 — Summary tables are **derived** from the persisted raw records
  and regenerated from them, never hand-edited.
* 10.9 — The summary groups by ``(algorithm, instance_set, n, m)`` and reports at
  least the mean, standard deviation, minimum and maximum of the RPD per group
  (plus the mean makespan).
* 10.11 — Records from more than one algorithm coexist in the same schema.

The approach: build a synthetic raw CSV (the exact format written by
:mod:`pfsp.results.record`), run the aggregator, and assert that the grouping is
correct and the derived statistics match a hand computation.
"""

from __future__ import annotations

import csv
import statistics

from pfsp.results.aggregate import (
    SUMMARY_HEADER,
    SummaryRow,
    summarize,
    write_summary,
)
from pfsp.results.record import CSV_HEADER


def _write_raw_csv(path, rows: list[dict[str, str]]) -> None:
    """Write ``rows`` to ``path`` using the fixed raw-record header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_HEADER))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _raw_row(
    *,
    instance_id: str,
    instance_set: str,
    n: int,
    m: int,
    algorithm: str,
    makespan: int,
    rpd: str,
) -> dict[str, str]:
    """Build a raw-record row dict, leaving non-relevant cells fixed/empty."""
    return {
        "instance_id": instance_id,
        "instance_set": instance_set,
        "n": str(n),
        "m": str(m),
        "algorithm": algorithm,
        "seed": "",
        "makespan": str(makespan),
        "runtime_s": "0.001",
        "rpd": rpd,
        "manifest_id": "run-20260607-000000-0000",
        "timestamp": "2026-06-07T00:00:00Z",
    }


def _by_key(rows: list[SummaryRow]) -> dict[tuple, SummaryRow]:
    return {(r.algorithm, r.instance_set, r.n, r.m): r for r in rows}


def test_summarize_groups_and_matches_manual_statistics(tmp_path) -> None:
    """Grouping is correct and RPD/makespan stats match a manual computation."""
    raw_dir = tmp_path / "raw"

    # Group A: NEH / Small / 10x5 -> three records with known RPDs.
    group_a = [
        _raw_row(
            instance_id="vrf/VFR10_5_1",
            instance_set="Small",
            n=10,
            m=5,
            algorithm="NEH",
            makespan=700,
            rpd="2.0",
        ),
        _raw_row(
            instance_id="vrf/VFR10_5_2",
            instance_set="Small",
            n=10,
            m=5,
            algorithm="NEH",
            makespan=720,
            rpd="4.0",
        ),
        _raw_row(
            instance_id="vrf/VFR10_5_3",
            instance_set="Small",
            n=10,
            m=5,
            algorithm="NEH",
            makespan=740,
            rpd="6.0",
        ),
    ]
    # Group B: NEH / Small / 20x5 -> distinct size group, single record.
    group_b = [
        _raw_row(
            instance_id="vrf/VFR20_5_1",
            instance_set="Small",
            n=20,
            m=5,
            algorithm="NEH",
            makespan=1500,
            rpd="3.0",
        ),
    ]
    _write_raw_csv(raw_dir / "NEH.csv", group_a + group_b)

    summaries = summarize(raw_dir)
    by_key = _by_key(summaries)

    # Two distinct groups were derived.
    assert len(summaries) == 2
    assert set(by_key) == {
        ("NEH", "Small", 10, 5),
        ("NEH", "Small", 20, 5),
    }

    # Group A statistics, computed by hand on RPDs [2.0, 4.0, 6.0].
    a = by_key[("NEH", "Small", 10, 5)]
    assert a.n_instances == 3
    assert a.rpd_mean == statistics.fmean([2.0, 4.0, 6.0]) == 4.0
    assert a.rpd_std == statistics.pstdev([2.0, 4.0, 6.0])
    assert a.rpd_min == 2.0
    assert a.rpd_max == 6.0
    assert a.makespan_mean == statistics.fmean([700, 720, 740]) == 720.0

    # Group B: single record -> population std is 0.0, min == max == mean.
    b = by_key[("NEH", "Small", 20, 5)]
    assert b.n_instances == 1
    assert b.rpd_mean == 3.0
    assert b.rpd_std == 0.0
    assert b.rpd_min == 3.0
    assert b.rpd_max == 3.0
    assert b.makespan_mean == 1500.0


def test_summarize_excludes_missing_rpd_but_counts_record(tmp_path) -> None:
    """Empty RPD cells are excluded from RPD stats but still counted (Req 10.9)."""
    raw_dir = tmp_path / "raw"
    rows = [
        _raw_row(
            instance_id="taillard/tai20_5_0",
            instance_set="",
            n=20,
            m=5,
            algorithm="NEH",
            makespan=1300,
            rpd="2.0",
        ),
        # Same group, but no reference was resolved -> empty rpd cell.
        _raw_row(
            instance_id="taillard/tai20_5_1",
            instance_set="",
            n=20,
            m=5,
            algorithm="NEH",
            makespan=1340,
            rpd="",
        ),
    ]
    _write_raw_csv(raw_dir / "NEH.csv", rows)

    (row,) = summarize(raw_dir)

    # The record with empty RPD is counted and feeds the mean makespan...
    assert row.instance_set is None  # empty cell maps back to None
    assert row.n_instances == 2
    assert row.makespan_mean == statistics.fmean([1300, 1340])
    # ...but only the resolvable RPD drives the RPD statistics.
    assert row.rpd_mean == 2.0
    assert row.rpd_std == 0.0
    assert row.rpd_min == 2.0
    assert row.rpd_max == 2.0


def test_summarize_leaves_rpd_stats_empty_when_no_reference(tmp_path) -> None:
    """A group with no resolvable RPD leaves the RPD stats unset, not fabricated."""
    raw_dir = tmp_path / "raw"
    rows = [
        _raw_row(
            instance_id="taillard/tai20_5_0",
            instance_set="",
            n=20,
            m=5,
            algorithm="NEH",
            makespan=1300,
            rpd="",
        ),
    ]
    _write_raw_csv(raw_dir / "NEH.csv", rows)

    (row,) = summarize(raw_dir)

    assert row.n_instances == 1
    assert row.makespan_mean == 1300.0
    assert row.rpd_mean is None
    assert row.rpd_std is None
    assert row.rpd_min is None
    assert row.rpd_max is None


def test_summarize_accommodates_multiple_algorithms(tmp_path) -> None:
    """Records from several algorithms share the schema (Req 10.11)."""
    raw_dir = tmp_path / "raw"
    _write_raw_csv(
        raw_dir / "NEH.csv",
        [
            _raw_row(
                instance_id="vrf/VFR10_5_1",
                instance_set="Small",
                n=10,
                m=5,
                algorithm="NEH",
                makespan=700,
                rpd="2.0",
            )
        ],
    )
    _write_raw_csv(
        raw_dir / "IG.csv",
        [
            _raw_row(
                instance_id="vrf/VFR10_5_1",
                instance_set="Small",
                n=10,
                m=5,
                algorithm="IG",
                makespan=680,
                rpd="1.0",
            )
        ],
    )

    summaries = summarize(raw_dir)
    by_key = _by_key(summaries)

    assert ("NEH", "Small", 10, 5) in by_key
    assert ("IG", "Small", 10, 5) in by_key
    assert by_key[("NEH", "Small", 10, 5)].rpd_mean == 2.0
    assert by_key[("IG", "Small", 10, 5)].rpd_mean == 1.0


def test_summarize_empty_when_no_raw_records(tmp_path) -> None:
    """A missing or empty raw directory yields no summary rows (no crash)."""
    assert summarize(tmp_path / "does_not_exist") == []


def test_write_summary_persists_derived_table(tmp_path) -> None:
    """``write_summary`` regenerates the table from the raw records (Req 10.6)."""
    raw_dir = tmp_path / "raw"
    _write_raw_csv(
        raw_dir / "NEH.csv",
        [
            _raw_row(
                instance_id="vrf/VFR10_5_1",
                instance_set="Small",
                n=10,
                m=5,
                algorithm="NEH",
                makespan=700,
                rpd="2.0",
            ),
            _raw_row(
                instance_id="vrf/VFR10_5_2",
                instance_set="Small",
                n=10,
                m=5,
                algorithm="NEH",
                makespan=720,
                rpd="4.0",
            ),
        ],
    )

    out_path = tmp_path / "summary" / "summary.csv"
    written = write_summary(raw_dir, out_path)

    assert written == out_path
    assert out_path.exists()

    with open(out_path, encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        data_rows = list(reader)

    assert tuple(header) == SUMMARY_HEADER
    assert len(data_rows) == 1

    record = dict(zip(SUMMARY_HEADER, data_rows[0], strict=True))
    assert record["algorithm"] == "NEH"
    assert record["instance_set"] == "Small"
    assert record["n"] == "10"
    assert record["m"] == "5"
    assert record["n_instances"] == "2"
    assert float(record["rpd_mean"]) == 3.0
    assert float(record["rpd_min"]) == 2.0
    assert float(record["rpd_max"]) == 4.0
    assert float(record["makespan_mean"]) == 710.0


def test_write_summary_is_regenerated_each_call(tmp_path) -> None:
    """Re-running after the raw records change reflects the new data, not stale rows."""
    raw_dir = tmp_path / "raw"
    out_path = tmp_path / "summary" / "summary.csv"

    _write_raw_csv(
        raw_dir / "NEH.csv",
        [
            _raw_row(
                instance_id="vrf/VFR10_5_1",
                instance_set="Small",
                n=10,
                m=5,
                algorithm="NEH",
                makespan=700,
                rpd="2.0",
            )
        ],
    )
    write_summary(raw_dir, out_path)

    # Overwrite the raw file with a different size group and regenerate.
    _write_raw_csv(
        raw_dir / "NEH.csv",
        [
            _raw_row(
                instance_id="vrf/VFR20_10_1",
                instance_set="Large",
                n=20,
                m=10,
                algorithm="NEH",
                makespan=2000,
                rpd="5.0",
            )
        ],
    )
    write_summary(raw_dir, out_path)

    with open(out_path, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["instance_set"] == "Large"
    assert rows[0]["n"] == "20"
    assert rows[0]["m"] == "10"
    assert float(rows[0]["makespan_mean"]) == 2000.0
