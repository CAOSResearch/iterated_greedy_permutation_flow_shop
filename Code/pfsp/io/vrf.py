"""VRF (Vallada, Ruiz and Framinan, 2015) instance reader.

Parses a VRF ``VFR{n}_{m}_{k}_Gap.txt`` file into the common
:class:`~pfsp.instance.Instance` type. The VRF layout differs from Taillard in
two ways: the first line is a bare ``n m`` header (no text labels) and the body
rows represent jobs (not machines), each row listing ``2*m`` integers as
``(machine_index, time)`` pairs::

    10  5
      0  45  1  31  2  54  3  54  4  64
      0  44  1   7  2  52  3  66  4  57
      ...                                 (n rows total, rows = jobs)

The parser reads the header, parses the ``n`` body rows into ``(machine, time)``
pairs, validates that each row's machine indices are exactly ``0..m-1`` (each
once), retains the times in machine order, stacks them as an ``(n, m)`` matrix
and transposes to the internal ``(m, n)`` convention so that
``processing_times[i][j]`` is the processing time of job ``j`` on machine ``i``
(see ``KB/05_datasets/formato-instancias.md``).
"""

from __future__ import annotations

from pathlib import PurePath

import numpy as np

from pfsp import InstanceFormatError
from pfsp.instance import Instance

# A VRF header line carries exactly two integers: number of jobs and machines.
_HEADER_TOKEN_COUNT = 2

# Known VRF instance sets, used to derive ``instance_set`` from the file path.
_INSTANCE_SETS = ("Small", "Large")


def _parse_integer_tokens(line: str) -> list[int] | None:
    """Return the integers on ``line`` or ``None`` if it is not fully numeric.

    A line is considered numeric only when it is non-empty and every
    whitespace-separated token is an integer.
    """
    tokens = line.split()
    if not tokens:
        return None
    values: list[int] = []
    for token in tokens:
        try:
            values.append(int(token))
        except ValueError:
            return None
    return values


def _derive_instance_set(path: str) -> str | None:
    """Derive the VRF instance set (``"Small"``/``"Large"``) from ``path``.

    Returns the matching path segment when present (e.g. ``data/vrf/Small/...``),
    otherwise ``None``.
    """
    parts = PurePath(path).parts
    for instance_set in _INSTANCE_SETS:
        if instance_set in parts:
            return instance_set
    return None


def read_vrf(path: str) -> Instance:
    """Parse a VRF ``VFR{n}_{m}_{k}_Gap.txt`` file into an :class:`Instance`.

    Parameters
    ----------
    path:
        Path to the VRF instance file. A single instance per file is expected.

    Returns
    -------
    Instance
        Instance whose ``processing_times`` matrix has shape ``(m, n)`` with
        ``processing_times[i][j]`` the processing time of job ``j`` on machine
        ``i``. ``instance_set`` is derived from the path when possible; the VRF
        format does not provide ``seed``, ``upper_bound`` or ``lower_bound``, so
        these are left as ``None``.

    Raises
    ------
    InstanceFormatError
        If the file cannot be read, is empty, has a missing or non-numeric
        header, contains a non-integer token, does not have exactly ``n`` body
        rows of ``2*m`` integers, has machine indices that are not exactly
        ``0..m-1`` (each once), or contains a negative processing time. The
        message identifies the failed validation and the offending row. No
        partial instance is returned on failure.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            raw_lines = handle.readlines()
    except OSError as error:
        raise InstanceFormatError(
            f"Could not read VRF file '{path}': {error}."
        ) from error

    # Keep only the non-empty lines, preserving order. The first is the header,
    # the rest are job rows.
    numeric_lines = [
        values
        for line in raw_lines
        if (values := _parse_integer_tokens(line)) is not None
    ]

    if not numeric_lines:
        raise InstanceFormatError(
            f"VRF file '{path}' is empty or contains no numeric data to parse."
        )

    # Header: exactly two integers "n m" (Req 3.2).
    header = numeric_lines[0]
    if len(header) != _HEADER_TOKEN_COUNT:
        raise InstanceFormatError(
            f"VRF file '{path}' has a malformed header {header}; expected two "
            f"integers 'n m'."
        )
    n, m = header
    if n < 1 or m < 1:
        raise InstanceFormatError(
            f"VRF file '{path}' declares non-positive dimensions n={n}, m={m}; "
            f"both must be at least 1."
        )

    body = numeric_lines[1:]

    # The body must contain exactly n job rows (Req 3.6).
    if len(body) != n:
        raise InstanceFormatError(
            f"VRF file '{path}' declares n={n} jobs but contains {len(body)} "
            f"body row(s)."
        )

    expected_tokens = 2 * m
    rows: list[list[int]] = []
    for row_index, row in enumerate(body):
        # Each job row must have exactly 2*m integers (Req 3.6).
        if len(row) != expected_tokens:
            raise InstanceFormatError(
                f"VRF file '{path}': body row {row_index} (job {row_index}) has "
                f"{len(row)} integer(s); expected {expected_tokens} "
                f"(2*m machine-time pairs)."
            )

        machine_indices = row[0::2]
        times = row[1::2]

        # Machine indices must be exactly 0..m-1, each appearing once (Req 3.6).
        if sorted(machine_indices) != list(range(m)):
            raise InstanceFormatError(
                f"VRF file '{path}': body row {row_index} (job {row_index}) has "
                f"machine indices {machine_indices}; expected each of 0..{m - 1} "
                f"exactly once."
            )

        # Retain the times ordered by machine index so that column i of the row
        # corresponds to machine i, regardless of the order of the pairs.
        times_by_machine = [0] * m
        for machine, time in zip(machine_indices, times, strict=True):
            # Processing times must be non-negative integers (Req 3.6).
            if time < 0:
                raise InstanceFormatError(
                    f"VRF file '{path}': body row {row_index} (job {row_index}) "
                    f"has a negative processing time ({time}) on machine "
                    f"{machine}; all processing times must be non-negative."
                )
            times_by_machine[machine] = time
        rows.append(times_by_machine)

    # Stack as (n, m) (rows = jobs) then transpose to the internal (m, n)
    # convention (rows = machines) so p[i][j] = time of job j on machine i.
    matrix = np.array(rows, dtype=np.int64).T
    # ``.T`` returns a view sharing the source buffer; copy so the Instance owns
    # a contiguous, independently writeable-then-frozen array.
    matrix = np.ascontiguousarray(matrix)

    return Instance(
        processing_times=matrix,
        n=n,
        m=m,
        source_benchmark="vrf",
        source_path=path,
        instance_set=_derive_instance_set(path),
        seed=None,
        upper_bound=None,
        lower_bound=None,
    )
