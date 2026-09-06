"""Tests for the reusable experiment campaign engine (``pfsp.experiments.runner``).

The campaign logic shared by every runner lives in the package, so it is tested
here once with normal imports (no path-loading of scripts). Focus areas:

* the **checkpoint/resume** behaviour required by the project steering
  (``codigo-python.md``): incremental persistence, idempotent resume (skip
  completed, no duplicate rows), ``fresh`` that drops only this campaign's rows,
  and ``limit`` for staged runs;
* the sanity strategies (:func:`lower_bound_sanity`, :func:`best_known_sanity`)
  and the resulting exit codes;
* :func:`discover_instances`.

The thin CLI scripts in ``Code/scripts/`` only wire argparse to this engine, so
they carry no logic of their own to test. All tests use small synthetic instances
in a temporary tree and run quietly (``verbose=False``).
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import pytest

from pfsp.experiments.runner import (
    EXIT_ANOMALY,
    EXIT_NO_INSTANCES,
    EXIT_OK,
    best_known_sanity,
    configure_logging,
    discover_instances,
    lower_bound_sanity,
    run_campaign,
)


@pytest.fixture(autouse=True)
def _reset_experiments_logger():
    """Isolate the experiments logger between tests (handlers, propagate, level)."""
    yield
    base = logging.getLogger("pfsp.experiments")
    for handler in list(base.handlers):
        base.removeHandler(handler)
        handler.close()
    base.propagate = True
    base.setLevel(logging.NOTSET)


# --------------------------------------------------------------------------- #
# Synthetic instance writers and shared helpers
# --------------------------------------------------------------------------- #


def _write_vrf_instance(path: Path, n: int, m: int) -> None:
    """Write a tiny valid VRF instance (header ``n m`` + n job rows of pairs)."""
    lines = [f"{n} {m}"]
    for job in range(n):
        pairs: list[str] = []
        for machine in range(m):
            pairs.extend([str(machine), str((job + machine) % 7 + 1)])
        lines.append(" ".join(pairs))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_taillard_instance(
    path: Path, n: int, m: int, *, seed: int = 1, ub: int = 9999, lb: int = 1
) -> None:
    """Write a tiny valid Taillard ``.fsp`` instance (canonical labelled format).

    ``n`` is kept != 5 so the 5-integer metadata line is unambiguously
    distinguished from the matrix rows by the reader.
    """
    lines = [
        "number of jobs, number of machines, initial seed, "
        "upper bound and lower bound :",
        f"   {n}   {m}   {seed}   {ub}   {lb}",
        "processing times :",
    ]
    for machine in range(m):
        lines.append(" ".join(str((job + machine) % 7 + 1) for job in range(n)))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_vrf_dir(tmp_path: Path, count: int, n: int = 4, m: int = 2) -> Path:
    """Create a VRF data dir with ``count`` synthetic Small instances."""
    small = tmp_path / "data" / "Small"
    small.mkdir(parents=True)
    for k in range(1, count + 1):
        _write_vrf_instance(small / f"VFR{n}_{m}_{k}_Gap.txt", n, m)
    return tmp_path / "data"


def _make_taillard_dir(
    tmp_path: Path, count: int, *, n: int = 6, m: int = 3, lb: int = 1
) -> Path:
    """Create a Taillard data dir with ``count`` synthetic ``.fsp`` instances."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    for k in range(count):
        _write_taillard_instance(data_dir / f"tai{n}_{m}_{k}.fsp", n, m, seed=k, lb=lb)
    return data_dir


def _raw_csv(results_dir: Path) -> Path:
    return results_dir / "raw" / "NEH.csv"


def _count_rows(csv_path: Path, id_prefix: str) -> int:
    if not csv_path.exists():
        return 0
    with open(csv_path, encoding="utf-8", newline="") as handle:
        return sum(
            1
            for row in csv.DictReader(handle)
            if (row.get("instance_id") or "").startswith(id_prefix)
        )


def _run_vrf(data_dir: Path, results_dir: Path, **kwargs) -> int:
    instances = discover_instances(data_dir, "*.txt", subdirs=("Small",))
    return run_campaign(
        instances,
        results_dir,
        id_prefix="vrf/",
        sanity=best_known_sanity,
        verbose=False,
        **kwargs,
    )


def _run_taillard(data_dir: Path, results_dir: Path, **kwargs) -> int:
    instances = discover_instances(data_dir, "*.fsp")
    return run_campaign(
        instances,
        results_dir,
        id_prefix="taillard/",
        sanity=lower_bound_sanity,
        verbose=False,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# discover_instances
# --------------------------------------------------------------------------- #


def test_discover_instances_flat_and_subdirs(tmp_path: Path) -> None:
    """discover_instances matches flat patterns and per-subdir patterns, sorted."""
    taillard = _make_taillard_dir(tmp_path / "t", count=3)
    flat = discover_instances(taillard, "*.fsp")
    assert len(flat) == 3
    assert flat == sorted(flat)

    vrf = _make_vrf_dir(tmp_path / "v", count=2)
    nested = discover_instances(vrf, "*.txt", subdirs=("Small",))
    assert len(nested) == 2


# --------------------------------------------------------------------------- #
# Checkpoint / resume behaviour (exercised through the VRF campaign)
# --------------------------------------------------------------------------- #


def test_runs_all_and_persists_records(tmp_path: Path) -> None:
    """A first pass processes every instance and derives the summary."""
    data_dir = _make_vrf_dir(tmp_path, count=3)
    results_dir = tmp_path / "results"

    code = _run_vrf(data_dir, results_dir)

    assert code == EXIT_OK
    assert _count_rows(_raw_csv(results_dir), "vrf/") == 3
    assert (results_dir / "summary" / "summary.csv").exists()


def test_resume_skips_completed_without_duplicating(tmp_path: Path) -> None:
    """Re-running resumes: completed instances skipped, no rows duplicated."""
    data_dir = _make_vrf_dir(tmp_path, count=3)
    results_dir = tmp_path / "results"

    _run_vrf(data_dir, results_dir)
    assert _count_rows(_raw_csv(results_dir), "vrf/") == 3

    code = _run_vrf(data_dir, results_dir)
    assert code == EXIT_OK
    assert _count_rows(_raw_csv(results_dir), "vrf/") == 3


def test_limit_enables_staged_runs(tmp_path: Path) -> None:
    """``limit`` processes a slice; a later run completes the rest."""
    data_dir = _make_vrf_dir(tmp_path, count=3)
    results_dir = tmp_path / "results"

    _run_vrf(data_dir, results_dir, limit=2)
    assert _count_rows(_raw_csv(results_dir), "vrf/") == 2

    _run_vrf(data_dir, results_dir)
    assert _count_rows(_raw_csv(results_dir), "vrf/") == 3


def test_fresh_drops_only_this_campaign_rows(tmp_path: Path) -> None:
    """``fresh`` removes only the campaign's own rows, keeping other campaigns'."""
    data_dir = _make_vrf_dir(tmp_path, count=2)
    results_dir = tmp_path / "results"
    csv_path = _raw_csv(results_dir)
    csv_path.parent.mkdir(parents=True)

    # Seed a Taillard row (a different campaign sharing the same CSV).
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
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
            ]
        )
        writer.writerow(
            [
                "taillard/tai20_5_0",
                "",
                "20",
                "5",
                "NEH",
                "",
                "1278",
                "0.001",
                "0.63",
                "run-x",
                "2026-06-07T00:00:00Z",
            ]
        )

    _run_vrf(data_dir, results_dir, fresh=True)

    assert _count_rows(csv_path, "vrf/") == 2
    assert _count_rows(csv_path, "taillard/") == 1


def test_no_instances_returns_error_code(tmp_path: Path) -> None:
    """An empty selection yields the 'no instances' exit code."""
    code = run_campaign([], tmp_path / "results", id_prefix="vrf/", verbose=False)
    assert code == EXIT_NO_INSTANCES


# --------------------------------------------------------------------------- #
# Sanity strategies and exit codes
# --------------------------------------------------------------------------- #


def test_taillard_campaign_runs_clean(tmp_path: Path) -> None:
    """The Taillard campaign runs, persists rows and passes the LB sanity."""
    data_dir = _make_taillard_dir(tmp_path, count=3, lb=1)
    results_dir = tmp_path / "results"

    code = _run_taillard(data_dir, results_dir)

    assert code == EXIT_OK
    assert _count_rows(_raw_csv(results_dir), "taillard/") == 3


def test_lower_bound_violation_returns_anomaly(tmp_path: Path) -> None:
    """An impossible LB (> any makespan) trips the lower-bound sanity strategy."""
    data_dir = _make_taillard_dir(tmp_path, count=1, lb=10**9)
    results_dir = tmp_path / "results"

    code = _run_taillard(data_dir, results_dir)

    assert code == EXIT_ANOMALY


# --------------------------------------------------------------------------- #
# Progress logging
# --------------------------------------------------------------------------- #


def test_progress_is_logged(tmp_path: Path, caplog) -> None:
    """With verbose, the engine logs per-instance start/done progress lines."""
    data_dir = _make_vrf_dir(tmp_path, count=2)
    results_dir = tmp_path / "results"

    with caplog.at_level(logging.INFO, logger="pfsp.experiments"):
        run_campaign(
            discover_instances(data_dir, "*.txt", subdirs=("Small",)),
            results_dir,
            id_prefix="vrf/",
            sanity=best_known_sanity,
            verbose=True,
        )

    messages = [record.getMessage() for record in caplog.records]
    assert any("[1/2] start" in m for m in messages)
    assert any("[2/2] done" in m and "elapsed=" in m and "eta=" in m for m in messages)


def test_configure_logging_creates_log_file(tmp_path: Path) -> None:
    """configure_logging attaches handlers and returns a persistent log path."""
    results_dir = tmp_path / "results"

    log_path = configure_logging(results_dir, campaign="vrf-small")

    assert log_path.parent == results_dir / "logs"
    assert log_path.exists()
    assert log_path.suffix == ".log"

    # A logged record reaches the file handler.
    logging.getLogger("pfsp.experiments.runner").info("hello-log")
    assert "hello-log" in log_path.read_text(encoding="utf-8")


def test_watchdog_warns_on_slow_instance(tmp_path: Path, caplog) -> None:
    """The watchdog warns (without killing) when an instance exceeds the budget."""
    import time as _time

    from pfsp.core.neh import neh

    data_dir = _make_vrf_dir(tmp_path, count=1)
    results_dir = tmp_path / "results"

    def slow_solver(instance):
        _time.sleep(0.15)
        return neh(instance)

    with caplog.at_level(logging.WARNING, logger="pfsp.experiments"):
        code = run_campaign(
            discover_instances(data_dir, "*.txt", subdirs=("Small",)),
            results_dir,
            id_prefix="vrf/",
            solver=slow_solver,
            warn_after_s=0.05,
            verbose=True,
        )

    # The run completes normally (watchdog never kills) and a warning is emitted.
    assert code == EXIT_OK
    assert any("watchdog" in r.getMessage().lower() for r in caplog.records)
