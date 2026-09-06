"""Taillard (1993) instance reader.

Parses a Taillard ``.fsp`` file into the common :class:`~pfsp.instance.Instance`
type. The canonical Taillard layout interleaves descriptive text labels with the
numeric content::

    number of jobs, number of machines, initial seed, upper bound and lower bound :
              20           5   873654221        1278        1232
    processing times :
     54 83 15 71 ... (n integers, machine 1)
     ...             (m rows total, rows = machines)

Some redistributions strip the text labels and present only the dimensions and
the matrix (see ``KB/05_datasets/formato-instancias.md``). The parser is
therefore robust: it does not assume a fixed line layout. It classifies each
line as numeric (all whitespace-separated tokens are integers) or non-numeric,
ignores the non-numeric lines, optionally reads an ``n m seed UB LB`` metadata
line, and stacks the remaining equal-length numeric lines into the ``(m, n)``
matrix (rows = machines, which already matches the internal convention).
"""

from __future__ import annotations

import numpy as np

from pfsp import InstanceFormatError
from pfsp.instance import Instance

# A Taillard metadata line carries exactly these five integers: number of jobs,
# number of machines, initial seed, upper bound and lower bound.
_METADATA_TOKEN_COUNT = 5


def _parse_integer_tokens(line: str) -> list[int] | None:
    """Return the integers on ``line`` or ``None`` if it is not fully numeric.

    A line is considered numeric only when it is non-empty and every
    whitespace-separated token is an integer (Req 2.4).
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


def read_taillard(path: str) -> Instance:
    """Parse a Taillard ``.fsp`` file into an :class:`Instance`.

    Parameters
    ----------
    path:
        Path to the Taillard instance file. A single instance per file is
        expected.

    Returns
    -------
    Instance
        Instance whose ``processing_times`` matrix has shape ``(m, n)`` with
        ``processing_times[i][j]`` the processing time of job ``j`` on machine
        ``i``. When the metadata line is present, ``seed``, ``upper_bound`` and
        ``lower_bound`` are recorded; otherwise they are ``None`` and ``n``/``m``
        are inferred from the matrix shape.

    Raises
    ------
    InstanceFormatError
        If the file cannot be read, contains no numeric matrix, has rows of
        unequal length, does not match the declared dimensions, or contains a
        negative processing time. No partial instance is returned on failure.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            raw_lines = handle.readlines()
    except OSError as error:
        raise InstanceFormatError(
            f"Could not read Taillard file '{path}': {error}."
        ) from error

    # Classify every line as a list of integers (numeric) or None (non-numeric,
    # e.g. the text labels), preserving order (Req 2.4).
    numeric_lines = [
        values
        for line in raw_lines
        if (values := _parse_integer_tokens(line)) is not None
    ]

    if not numeric_lines:
        raise InstanceFormatError(
            f"Taillard file '{path}' contains no numeric data to parse."
        )

    # Detect the optional metadata line "n m seed UB LB": a leading numeric line
    # of exactly five integers that is not part of the matrix block. The matrix
    # rows all share the same length (n), so when the first numeric line has 5
    # tokens but the following rows do not, it is the metadata line (Req 2.3).
    # When every line has 5 tokens it is a 5-job matrix without labels (Req 2.5).
    declared_n: int | None = None
    declared_m: int | None = None
    seed: int | None = None
    upper_bound: int | None = None
    lower_bound: int | None = None

    matrix_lines = numeric_lines
    first = numeric_lines[0]
    if (
        len(first) == _METADATA_TOKEN_COUNT
        and len(numeric_lines) > 1
        and len(numeric_lines[1]) != _METADATA_TOKEN_COUNT
    ):
        declared_n, declared_m, seed, upper_bound, lower_bound = first
        matrix_lines = numeric_lines[1:]

    # The matrix block is the set of remaining numeric lines; they must all have
    # the same length (the number of jobs n) to stack into a rectangular matrix.
    row_lengths = {len(row) for row in matrix_lines}
    if len(row_lengths) != 1:
        raise InstanceFormatError(
            f"Taillard file '{path}' has matrix rows of unequal length "
            f"{sorted(row_lengths)}; cannot build a rectangular (m, n) matrix."
        )

    matrix = np.array(matrix_lines, dtype=np.int64)
    m, n = matrix.shape

    # Validate dimensions against the declared header when present (Reqs 2.6,
    # 2.7). Rows = machines (m), columns = jobs (n).
    if declared_m is not None and declared_m != m:
        raise InstanceFormatError(
            f"Taillard file '{path}': declared number of machines m = "
            f"{declared_m} does not match the {m} matrix row(s) found."
        )
    if declared_n is not None and declared_n != n:
        raise InstanceFormatError(
            f"Taillard file '{path}': declared number of jobs n = "
            f"{declared_n} does not match the {n} column(s) per row found."
        )

    # All processing times must be non-negative integers (Reqs 2.6, 2.7).
    if matrix.min() < 0:
        negative = int(matrix.min())
        raise InstanceFormatError(
            f"Taillard file '{path}' contains a negative processing time "
            f"({negative}); all processing times must be non-negative integers."
        )

    return Instance(
        processing_times=matrix,
        n=n,
        m=m,
        source_benchmark="taillard",
        source_path=path,
        seed=seed,
        upper_bound=upper_bound,
        lower_bound=lower_bound,
    )
