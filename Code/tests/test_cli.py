"""Smoke tests for the ``pfsp-neh`` command-line entry point.

These tests exercise the CLI end to end on a tiny synthetic Taillard instance
written to a temporary directory, so they are robust whether or not the real
(out-of-git) benchmark data is present. They verify that the CLI:

* loads the instance, runs NEH, and prints the makespan to stdout (Req 1.14);
* writes the raw result record and the run manifest to the results directory
  supplied on the command line (Reqs 1.12, 8.1, 8.2, 10.2).

The format detector keys Taillard files off the ``.fsp`` extension, so the
fixture is written with that extension and a valid Taillard header + matrix.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pfsp.cli import main
from pfsp.results.record import CSV_HEADER, raw_csv_path

# A tiny, valid Taillard instance: 4 jobs, 3 machines. The first (text) line is a
# label the reader ignores; the header line carries "n m seed UB LB"; the matrix
# has m=3 rows (machines) and n=4 columns (jobs).
_TINY_TAILLARD = """jobs machines seed upper-bound lower-bound :
           4           3   123456789         100          80
processing times :
 5 9 3 7
 8 1 6 4
 3 7 2 5
"""


@pytest.fixture()
def tiny_instance(tmp_path: Path) -> Path:
    """Write a tiny valid Taillard ``.fsp`` instance and return its path."""
    path = tmp_path / "tai4_3_0.fsp"
    path.write_text(_TINY_TAILLARD, encoding="utf-8")
    return path


def test_cli_prints_makespan_and_writes_outputs(
    tiny_instance: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI prints the makespan and creates the output files (Req 1.14)."""
    results_dir = tmp_path / "results"

    exit_code = main(
        [
            "--instance",
            str(tiny_instance),
            "--results-dir",
            str(results_dir),
        ]
    )

    # The run succeeds.
    assert exit_code == 0

    captured = capsys.readouterr()
    # The makespan and the permutation are reported to stdout (Req 1.14).
    assert "makespan:" in captured.out
    assert "permutation:" in captured.out

    # The raw result record CSV exists, with the fixed header and one data row.
    csv_path = raw_csv_path("NEH", results_dir)
    assert csv_path.exists()
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == ",".join(CSV_HEADER)
    assert len(lines) == 2  # header + one execution record
    assert lines[1].startswith("taillard/tai4_3_0,")

    # Exactly one run manifest was written, as inspectable JSON linking back to
    # the result record's manifest_id.
    manifest_files = list((results_dir / "manifests").glob("*.json"))
    assert len(manifest_files) == 1
    manifest = json.loads(manifest_files[0].read_text(encoding="utf-8"))
    assert manifest["instance_id"] == "taillard/tai4_3_0"
    assert manifest["algorithm"] == "NEH"
    # The record row references the same manifest id.
    assert manifest["manifest_id"] in lines[1]


def test_cli_returns_error_for_unloadable_instance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing/unrecognizable instance yields a non-zero exit code."""
    missing = tmp_path / "does_not_exist.fsp"

    exit_code = main(["--instance", str(missing)])

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "error" in captured.err.lower()
