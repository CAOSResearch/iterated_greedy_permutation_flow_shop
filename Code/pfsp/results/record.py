"""Per-execution result record, RPD metric and raw-results persistence.

A :class:`ResultRecord` is the machine-readable, per-execution row of the
results framework. It captures the makespan objective of a run together with the
metadata needed to compare algorithms and trace each result back to its
:class:`~pfsp.results.manifest.RunManifest` via ``manifest_id`` (Reqs 10.1,
10.8).

Records are appended, one per execution, to ``results/raw/<algorithm>.csv`` with
a fixed header (decision 2026-06-07: one file per algorithm). Every individual
record is preserved rather than only aggregates (Req 10.2); summary tables are
derived from these raw rows by the aggregator, never written by hand (Req 10.6).

In line with the project's hard constraint, only the makespan-based objective
and its derived RPD are recorded; no energy or power metrics are ever collected
(Req 10.7).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

#: Sub-directory, relative to the results directory, where raw per-execution
#: CSV records are written (one file per algorithm).
RAW_SUBDIR = "raw"

#: Fixed header of the raw result CSV, in column order. ``instance_set``, ``seed``
#: and ``rpd`` are written empty when they do not apply (no fabrication).
CSV_HEADER: tuple[str, ...] = (
    "instance_id",
    "instance_set",
    "n",
    "m",
    "algorithm",
    "seed",
    "makespan",
    "runtime_s",
    "rpd",
    "manifest_id",
    "timestamp",
)

#: Default results directory, resolved relative to this package so it is portable
#: across checkouts: ``record.py`` -> ``results`` -> ``pfsp`` -> repository root
#: (the ``Code/`` directory). Matches the ``results/`` location used elsewhere.
_DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def compute_rpd(makespan: int, reference: int | None) -> float | None:
    """Compute the Relative Percentage Deviation of a makespan.

    The RPD measures how far a makespan deviates from a best-known reference:
    ``RPD = 100 * (makespan - reference) / reference`` (Req 10.4).

    Parameters
    ----------
    makespan:
        The makespan ``C_max`` achieved by a run.
    reference:
        The best-known reference value resolved by the
        :mod:`~pfsp.results.reference` provider, or ``None`` when no reference is
        available.

    Returns
    -------
    float | None
        The RPD as a percentage, or ``None`` when ``reference`` is ``None`` so
        that a missing reference is never fabricated into a number (Req 10.5).
        ``None`` is also returned when ``reference`` is ``0``, since the RPD is
        undefined in that degenerate case (a real benchmark reference is always
        positive).
    """
    if reference is None or reference == 0:
        return None
    return 100.0 * (makespan - reference) / reference


@dataclass(frozen=True)
class ResultRecord:
    """A single per-execution result row.

    Attributes
    ----------
    instance_id:
        ``Instance_Identifier`` of the instance that was run
        (e.g. ``"taillard/tai20_5_0"``).
    instance_set:
        VRF instance set (``"Small"`` or ``"Large"``); ``None`` when it does not
        apply (e.g. Taillard). Stored on the record so the aggregator can group
        by set without cross-referencing another source.
    n:
        Number of jobs.
    m:
        Number of machines.
    algorithm:
        Name of the algorithm executed (e.g. ``"NEH"``).
    seed:
        Explicit seed used, or ``None`` for deterministic runs that consume no
        randomness. Never fabricated.
    makespan:
        The makespan ``C_max`` of the run.
    runtime_s:
        Wall-clock runtime in seconds. Informative only; the single
        non-deterministic field of the record.
    rpd:
        Relative Percentage Deviation, or ``None`` when no reference could be
        resolved (Req 10.5).
    manifest_id:
        Identifier of the :class:`~pfsp.results.manifest.RunManifest` that
        produced this record, linking the result to its execution context
        (Req 10.8).
    timestamp:
        ISO-8601 UTC timestamp of the run (typically the manifest timestamp).
    """

    instance_id: str
    instance_set: str | None
    n: int
    m: int
    algorithm: str
    seed: int | None
    makespan: int
    runtime_s: float
    rpd: float | None
    manifest_id: str
    timestamp: str = ""

    def to_row(self) -> list[str]:
        """Return the record as a list of CSV cell strings in header order.

        Optional fields that do not apply (``instance_set``, ``seed``, ``rpd``)
        are rendered as the empty string rather than a fabricated value.
        """
        return [
            self.instance_id,
            "" if self.instance_set is None else self.instance_set,
            str(self.n),
            str(self.m),
            self.algorithm,
            "" if self.seed is None else str(self.seed),
            str(self.makespan),
            repr(self.runtime_s),
            "" if self.rpd is None else repr(self.rpd),
            self.manifest_id,
            self.timestamp,
        ]


def raw_csv_path(
    algorithm: str, results_dir: str | Path = _DEFAULT_RESULTS_DIR
) -> Path:
    """Return the raw CSV path for ``algorithm`` under ``results_dir``.

    The file lives at ``<results_dir>/raw/<algorithm>.csv`` (one file per
    algorithm). This does not create the file or its directory.
    """
    return Path(results_dir) / RAW_SUBDIR / f"{algorithm}.csv"


def append_record(
    record: ResultRecord, results_dir: str | Path = _DEFAULT_RESULTS_DIR
) -> Path:
    """Append ``record`` to its algorithm's raw CSV, preserving every row.

    The record is appended to ``<results_dir>/raw/<record.algorithm>.csv``; the
    ``raw`` sub-directory is created if needed. The fixed :data:`CSV_HEADER` is
    written only once, when the file is newly created or empty, so repeated calls
    accumulate rows under a single header without ever overwriting earlier
    records (Req 10.2).

    Parameters
    ----------
    record:
        The result record to persist.
    results_dir:
        Root directory of the results tree. Defaults to the package-relative
        ``results/`` directory.

    Returns
    -------
    pathlib.Path
        The path of the CSV file the record was appended to.
    """
    path = raw_csv_path(record.algorithm, results_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write the header only for a brand-new or empty file so appends stay valid.
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(CSV_HEADER)
        writer.writerow(record.to_row())
    return path


__all__ = [
    "RAW_SUBDIR",
    "CSV_HEADER",
    "ResultRecord",
    "compute_rpd",
    "raw_csv_path",
    "append_record",
]
