"""Aggregator: derived summary tables from the raw result records.

The aggregator **derives** summary tables from the per-execution raw records
written by :mod:`pfsp.results.record`; it never receives hand-edited data
(Reqs 10.6, 10.10). Its output is the raw material for the results narrative and
the statistical study of later phases.

The raw records live in ``results/raw/`` (one CSV per algorithm, with the fixed
header :data:`pfsp.results.record.CSV_HEADER`). :func:`summarize` concatenates
all those CSV files, groups the rows by ``(algorithm, instance_set, n, m)`` and
derives, per group: the number of records, the mean / population standard
deviation / minimum / maximum of the RPD, and the mean makespan (Req 10.9).
:func:`write_summary` persists the result to ``results/summary/summary.csv`` in a
machine-readable format separate from the raw records, regenerated from the raw
rows rather than from any hand-edited input (Req 10.10).

Because grouping keys on the ``algorithm`` column, records from more than one
algorithm coexist in the same summary schema, so later phases can compare NEH and
the metaheuristics without changing it (Req 10.11).

Records with an empty ``rpd`` cell (no reference could be resolved) are excluded
from the RPD statistics but still counted and still contribute to the mean
makespan; if a whole group has no resolvable RPD, its RPD statistics are left
empty rather than fabricated (consistent with the no-fabrication rule). The
standard deviation is the **population** standard deviation
(:func:`statistics.pstdev`), which is well defined for a single record (it is
``0.0``); this is the natural choice when the group is treated as the full set of
runs for that ``(algorithm, instance_set, n, m)`` cell rather than a sample.

Only the standard library is used (:mod:`csv`, :mod:`statistics`); no pandas.
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass
from pathlib import Path

from pfsp.results.record import CSV_HEADER, RAW_SUBDIR

#: Sub-directory, relative to the results directory, where the derived summary
#: table is written.
SUMMARY_SUBDIR = "summary"

#: File name of the derived summary table.
SUMMARY_FILENAME = "summary.csv"

#: Default results directory, resolved relative to this package so it is portable
#: across checkouts: ``aggregate.py`` -> ``results`` -> ``pfsp`` -> repository
#: root (the ``Code/`` directory). Mirrors :mod:`pfsp.results.record`.
_DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

#: Fixed header of the summary CSV, in column order. One row per
#: ``(algorithm, instance_set, n, m)`` group.
SUMMARY_HEADER: tuple[str, ...] = (
    "algorithm",
    "instance_set",
    "n",
    "m",
    "n_instances",
    "rpd_mean",
    "rpd_std",
    "rpd_min",
    "rpd_max",
    "makespan_mean",
)


@dataclass(frozen=True)
class SummaryRow:
    """A single derived summary row for one ``(algorithm, instance_set, n, m)``.

    Attributes
    ----------
    algorithm:
        Name of the algorithm (e.g. ``"NEH"``); the grouping key that lets
        multiple algorithms share the schema (Req 10.11).
    instance_set:
        VRF instance set (``"Small"`` or ``"Large"``); ``None`` when it does not
        apply (e.g. Taillard), matching the empty cell of the raw record.
    n:
        Number of jobs of the size group.
    m:
        Number of machines of the size group.
    n_instances:
        Number of raw records in the group (every record is counted, including
        those with an unresolved RPD).
    rpd_mean, rpd_std, rpd_min, rpd_max:
        Mean, population standard deviation, minimum and maximum of the RPD over
        the records of the group that have a resolved RPD; ``None`` for every one
        of them when no record in the group has a resolvable RPD (Req 10.9).
    makespan_mean:
        Mean makespan over all records of the group (makespan is always present).
    """

    algorithm: str
    instance_set: str | None
    n: int
    m: int
    n_instances: int
    rpd_mean: float | None
    rpd_std: float | None
    rpd_min: float | None
    rpd_max: float | None
    makespan_mean: float

    def to_row(self) -> list[str]:
        """Return the summary row as CSV cell strings in :data:`SUMMARY_HEADER` order.

        ``instance_set`` and the RPD statistics are rendered as the empty string
        when they are unset (``None``), never as a fabricated value.
        """
        return [
            self.algorithm,
            "" if self.instance_set is None else self.instance_set,
            str(self.n),
            str(self.m),
            str(self.n_instances),
            "" if self.rpd_mean is None else repr(self.rpd_mean),
            "" if self.rpd_std is None else repr(self.rpd_std),
            "" if self.rpd_min is None else repr(self.rpd_min),
            "" if self.rpd_max is None else repr(self.rpd_max),
            repr(self.makespan_mean),
        ]


def _iter_raw_rows(raw_dir: str | Path) -> list[dict[str, str]]:
    """Read and concatenate every raw CSV under ``raw_dir``.

    Each CSV is expected to carry the fixed :data:`~pfsp.results.record.CSV_HEADER`.
    Files are read with :class:`csv.DictReader`; rows from every ``*.csv`` file in
    the directory (sorted by name for determinism) are concatenated. Files that do
    not carry the raw-record columns are skipped.

    Parameters
    ----------
    raw_dir:
        Directory holding the raw per-execution CSV files (one per algorithm).

    Returns
    -------
    list[dict[str, str]]
        The concatenated rows as dictionaries keyed by column name. An empty list
        is returned when ``raw_dir`` does not exist or contains no CSV files.
    """
    directory = Path(raw_dir)
    if not directory.is_dir():
        return []

    rows: list[dict[str, str]] = []
    expected_columns = set(CSV_HEADER)
    for csv_path in sorted(directory.glob("*.csv")):
        with open(csv_path, encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or ())
            # Skip files that do not carry the raw-record contract so an unrelated
            # CSV dropped in the directory cannot corrupt the summary.
            if not expected_columns.issubset(fieldnames):
                continue
            for row in reader:
                rows.append(row)
    return rows


def _group_key(row: dict[str, str]) -> tuple[str, str, int, int]:
    """Derive the grouping key ``(algorithm, instance_set, n, m)`` from a raw row.

    ``instance_set`` is kept as the raw cell value (the empty string for records
    where it does not apply, e.g. Taillard) so it can be sorted and serialized
    consistently; it is mapped back to ``None`` when building the
    :class:`SummaryRow`.
    """
    return (
        row["algorithm"],
        row["instance_set"],
        int(row["n"]),
        int(row["m"]),
    )


def summarize(raw_dir: str | Path) -> list[SummaryRow]:
    """Derive summary rows from the raw result records under ``raw_dir``.

    All CSV files in ``raw_dir`` are concatenated and grouped by
    ``(algorithm, instance_set, n, m)``. For each group the number of records,
    the mean / population standard deviation / minimum / maximum of the RPD and
    the mean makespan are derived (Reqs 10.6, 10.9, 10.11).

    Records whose ``rpd`` cell is empty (no reference resolved) are excluded from
    the RPD statistics but still counted in ``n_instances`` and still contribute
    to ``makespan_mean``. When no record of a group has a resolvable RPD, the four
    RPD statistics are left as ``None`` rather than fabricated.

    Parameters
    ----------
    raw_dir:
        Directory holding the raw per-execution CSV files.

    Returns
    -------
    list[SummaryRow]
        One row per group, sorted by ``(algorithm, instance_set, n, m)`` for a
        deterministic, diff-friendly output.
    """
    rows = _iter_raw_rows(raw_dir)

    grouped: dict[tuple[str, str, int, int], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(_group_key(row), []).append(row)

    summaries: list[SummaryRow] = []
    for key in sorted(grouped):
        algorithm, instance_set_cell, n, m = key
        group_rows = grouped[key]

        makespans = [int(r["makespan"]) for r in group_rows]
        rpds = [float(r["rpd"]) for r in group_rows if r["rpd"] not in ("", None)]

        if rpds:
            rpd_mean: float | None = statistics.fmean(rpds)
            rpd_std: float | None = statistics.pstdev(rpds)
            rpd_min: float | None = min(rpds)
            rpd_max: float | None = max(rpds)
        else:
            rpd_mean = rpd_std = rpd_min = rpd_max = None

        summaries.append(
            SummaryRow(
                algorithm=algorithm,
                instance_set=instance_set_cell or None,
                n=n,
                m=m,
                n_instances=len(group_rows),
                rpd_mean=rpd_mean,
                rpd_std=rpd_std,
                rpd_min=rpd_min,
                rpd_max=rpd_max,
                makespan_mean=statistics.fmean(makespans),
            )
        )
    return summaries


def write_summary(raw_dir: str | Path, out_path: str | Path) -> Path:
    """Derive the summary table from ``raw_dir`` and write it to ``out_path``.

    The summary is regenerated from the raw records on every call (Req 10.10); the
    parent directory of ``out_path`` is created if needed. The file is written
    with the fixed :data:`SUMMARY_HEADER` followed by one row per group.

    Parameters
    ----------
    raw_dir:
        Directory holding the raw per-execution CSV files.
    out_path:
        Destination path of the summary CSV (typically
        ``results/summary/summary.csv``).

    Returns
    -------
    pathlib.Path
        The path the summary was written to.
    """
    summaries = summarize(raw_dir)

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(SUMMARY_HEADER)
        for summary in summaries:
            writer.writerow(summary.to_row())
    return path


def default_summary_path(results_dir: str | Path = _DEFAULT_RESULTS_DIR) -> Path:
    """Return the default summary path ``<results_dir>/summary/summary.csv``.

    This does not create the file or its directory.
    """
    return Path(results_dir) / SUMMARY_SUBDIR / SUMMARY_FILENAME


def default_raw_dir(results_dir: str | Path = _DEFAULT_RESULTS_DIR) -> Path:
    """Return the default raw-records directory ``<results_dir>/raw``.

    Mirrors the layout written by :mod:`pfsp.results.record`; this does not create
    the directory.
    """
    return Path(results_dir) / RAW_SUBDIR


__all__ = [
    "SUMMARY_SUBDIR",
    "SUMMARY_FILENAME",
    "SUMMARY_HEADER",
    "SummaryRow",
    "summarize",
    "write_summary",
    "default_summary_path",
    "default_raw_dir",
]
