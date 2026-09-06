"""Smoke tests for the ``pfsp-ig`` command-line entry point.

These tests exercise the CLI end to end on a tiny synthetic Taillard instance
written to a temporary directory, so they are robust whether or not the real
(out-of-git) benchmark data is present. They verify that the CLI:

* loads the instance, runs IG, and prints the makespan to stdout (Req 11.1);
* writes the raw result record (with seed and algorithm columns) and the run
  manifest (with IG parameters) to the results directory supplied on the
  command line (Reqs 8.6, 9.2, 11.2);
* rejects missing required arguments and invalid --stop format with a non-zero
  exit code (Req 11.4).

All tests use ``tmp_path``; never write to ``Code/results/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pfsp.cli_ig import main
from pfsp.results.record import CSV_HEADER, raw_csv_path

# A tiny, valid Taillard instance: 4 jobs, 3 machines.
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


def test_cli_ig_exit_zero_and_prints_makespan(
    tiny_instance: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI exits 0 and prints 'makespan:' to stdout on a valid run."""
    results_dir = tmp_path / "results"

    exit_code = main(
        [
            "--instance",
            str(tiny_instance),
            "--seed",
            "42",
            "--d",
            "2",
            "--stop",
            "iterations:10",
            "--results-dir",
            str(results_dir),
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "makespan:" in captured.out
    assert "permutation:" in captured.out
    assert "algorithm: IG" in captured.out
    assert "seed: 42" in captured.out


def test_cli_ig_csv_created_with_seed_and_algorithm(
    tiny_instance: Path, tmp_path: Path
) -> None:
    """The CSV result file is created with seed and algorithm columns."""
    results_dir = tmp_path / "results"

    exit_code = main(
        [
            "--instance",
            str(tiny_instance),
            "--seed",
            "7",
            "--d",
            "2",
            "--stop",
            "iterations:5",
            "--results-dir",
            str(results_dir),
        ]
    )

    assert exit_code == 0

    csv_path = raw_csv_path("IG", results_dir)
    assert csv_path.exists()
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == ",".join(CSV_HEADER)
    assert len(lines) == 2  # header + one record

    # The record contains the seed and the algorithm label.
    row = lines[1]
    assert "taillard/tai4_3_0" in row
    assert ",IG," in row
    assert ",7," in row  # seed value


def test_cli_ig_manifest_created_with_parameters(
    tiny_instance: Path, tmp_path: Path
) -> None:
    """The manifest JSON is created with IG parameters."""
    results_dir = tmp_path / "results"

    exit_code = main(
        [
            "--instance",
            str(tiny_instance),
            "--seed",
            "99",
            "--d",
            "2",
            "--tp",
            "0.5",
            "--stop",
            "iterations:8",
            "--no-local-search",
            "--results-dir",
            str(results_dir),
        ]
    )

    assert exit_code == 0

    manifest_files = list((results_dir / "manifests").glob("*.json"))
    assert len(manifest_files) == 1
    manifest = json.loads(manifest_files[0].read_text(encoding="utf-8"))

    assert manifest["algorithm"] == "IG-noLS"
    assert manifest["seed"] == 99
    assert manifest["parameters"]["d"] == 2
    assert manifest["parameters"]["tp"] == 0.5
    assert manifest["parameters"]["stop_kind"] == "iterations"
    assert manifest["parameters"]["stop_value"] == 8.0
    assert manifest["parameters"]["local_search"] is False


def test_cli_ig_missing_required_instance(capsys: pytest.CaptureFixture[str]) -> None:
    """Missing --instance causes a non-zero exit (SystemExit from argparse)."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--seed", "1"])
    assert exc_info.value.code != 0


def test_cli_ig_missing_required_seed(
    tiny_instance: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Missing --seed causes a non-zero exit (SystemExit from argparse)."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--instance", str(tiny_instance)])
    assert exc_info.value.code != 0


def test_cli_ig_invalid_stop_format(
    tiny_instance: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Invalid --stop format causes a non-zero exit without executing."""
    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--instance",
                str(tiny_instance),
                "--seed",
                "1",
                "--stop",
                "bad_format",
            ]
        )
    assert exc_info.value.code != 0


def test_cli_ig_guided_idle_rcl_smoke(
    tiny_instance: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--destruction idle-rcl runs, reports makespan, labels and persists (Req 11.2)."""
    results_dir = tmp_path / "results"

    exit_code = main(
        [
            "--instance",
            str(tiny_instance),
            "--seed",
            "42",
            "--d",
            "2",
            "--stop",
            "iterations:10",
            "--destruction",
            "idle-rcl",
            "--alpha",
            "0.3",
            "--results-dir",
            str(results_dir),
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "makespan:" in captured.out
    # The label reflects the selected guided operator (Req 11.2).
    assert "algorithm: IG-idle-rcl" in captured.out

    # The raw CSV is routed to the operator's own file, never IG.csv (Req 8.4).
    csv_path = raw_csv_path("IG-idle-rcl", results_dir)
    assert csv_path.exists()
    assert not raw_csv_path("IG", results_dir).exists()
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == ",".join(CSV_HEADER)
    assert len(lines) == 2  # header + one record
    assert ",IG-idle-rcl," in lines[1]

    # The manifest records the destruction operator and the alpha value (Req 11.2).
    manifest_files = list((results_dir / "manifests").glob("*.json"))
    assert len(manifest_files) == 1
    manifest = json.loads(manifest_files[0].read_text(encoding="utf-8"))
    assert manifest["algorithm"] == "IG-idle-rcl"
    assert manifest["parameters"]["destruction"] == "idle-rcl"
    assert manifest["parameters"]["alpha"] == 0.3


def test_cli_ig_default_destruction_is_random(
    tiny_instance: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without --destruction the run stays the random baseline labelled 'IG'."""
    results_dir = tmp_path / "results"

    exit_code = main(
        [
            "--instance",
            str(tiny_instance),
            "--seed",
            "42",
            "--d",
            "2",
            "--stop",
            "iterations:10",
            "--results-dir",
            str(results_dir),
        ]
    )

    assert exit_code == 0
    assert "algorithm: IG\n" in capsys.readouterr().out
    assert raw_csv_path("IG", results_dir).exists()

    manifest_files = list((results_dir / "manifests").glob("*.json"))
    manifest = json.loads(manifest_files[0].read_text(encoding="utf-8"))
    assert manifest["parameters"]["destruction"] == "random"
    assert manifest["parameters"]["alpha"] == 0.10


def test_cli_ig_invalid_alpha(
    tiny_instance: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An out-of-range --alpha exits non-zero without executing (Req 11.3)."""
    results_dir = tmp_path / "results"

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--instance",
                str(tiny_instance),
                "--seed",
                "1",
                "--destruction",
                "idle-rcl",
                "--alpha",
                "1.5",
                "--results-dir",
                str(results_dir),
            ]
        )

    assert exc_info.value.code != 0
    # No run was executed: no results were written.
    assert not results_dir.exists()


def test_cli_ig_invalid_destruction_operator(
    tiny_instance: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unknown --destruction operator exits non-zero without executing (Req 11.3)."""
    results_dir = tmp_path / "results"

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--instance",
                str(tiny_instance),
                "--seed",
                "1",
                "--destruction",
                "not-an-operator",
                "--results-dir",
                str(results_dir),
            ]
        )

    assert exc_info.value.code != 0
    assert not results_dir.exists()
