"""Tests for the multi-seed stochastic extension of the campaign runner.

Validates that :func:`~pfsp.experiments.runner.run_campaign` correctly handles
multi-seed stochastic campaigns (IG-style): one Replication per (instance, seed)
pair, resume keyed by ``(instance_id, seed)``, ``--fresh`` scoped to the
campaign, and summary regeneration.

All tests use ``tmp_path`` and synthetic instances; they never touch
``Code/results/``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from pfsp.experiments.runner import (
    EXIT_OK,
    discover_instances,
    run_campaign,
)
from pfsp.instance import Instance

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _write_vrf_instance(path: Path, n: int = 4, m: int = 2) -> None:
    """Write a tiny valid VRF instance (header ``n m`` + n job rows of pairs)."""
    lines = [f"{n} {m}"]
    for job in range(n):
        pairs: list[str] = []
        for machine in range(m):
            pairs.extend([str(machine), str((job + machine) % 7 + 1)])
        lines.append(" ".join(pairs))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_instances(tmp_path: Path, count: int) -> Path:
    """Create ``count`` VRF-like synthetic instances under a Small subdir."""
    small = tmp_path / "data" / "Small"
    small.mkdir(parents=True)
    for k in range(1, count + 1):
        _write_vrf_instance(small / f"VFR4_2_{k}_Gap.txt")
    return tmp_path / "data"


def _fake_stochastic_solver(instance: Instance, seed: int) -> tuple[list[int], int]:
    """A trivial stochastic solver: identity permutation, makespan = seed + n."""
    perm = list(range(instance.n))
    # Use a simple deterministic formula based on seed so results vary per seed
    makespan = instance.n * 10 + seed
    return perm, makespan


def _raw_csv(results_dir: Path, algorithm: str = "IG") -> Path:
    return results_dir / "raw" / f"{algorithm}.csv"


def _count_rows(csv_path: Path, id_prefix: str) -> int:
    """Count CSV rows whose instance_id starts with id_prefix."""
    if not csv_path.exists():
        return 0
    with open(csv_path, encoding="utf-8", newline="") as handle:
        return sum(
            1
            for row in csv.DictReader(handle)
            if (row.get("instance_id") or "").startswith(id_prefix)
        )


def _read_seeds(csv_path: Path, id_prefix: str) -> list[int]:
    """Read seeds from all rows matching id_prefix."""
    if not csv_path.exists():
        return []
    seeds: list[int] = []
    with open(csv_path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if (row.get("instance_id") or "").startswith(id_prefix):
                seed_str = (row.get("seed") or "").strip()
                if seed_str:
                    seeds.append(int(seed_str))
    return seeds


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


class TestMultiSeedCampaign:
    """Mini-campaign: 2 instances x 2 seeds -> 4 records."""

    def test_produces_correct_number_of_records(self, tmp_path: Path) -> None:
        """2 instances x 2 seeds produces exactly 4 result records."""
        data_dir = _make_instances(tmp_path, count=2)
        results_dir = tmp_path / "results"
        instances = discover_instances(data_dir, "*.txt", subdirs=("Small",))

        code = run_campaign(
            instances,
            results_dir,
            id_prefix="vrf/",
            algorithm="IG",
            stochastic_solver=_fake_stochastic_solver,
            seeds=[42, 99],
            verbose=False,
        )

        assert code == EXIT_OK
        assert _count_rows(_raw_csv(results_dir), "vrf/") == 4

    def test_records_have_correct_seeds(self, tmp_path: Path) -> None:
        """Each record carries the correct seed value."""
        data_dir = _make_instances(tmp_path, count=2)
        results_dir = tmp_path / "results"
        instances = discover_instances(data_dir, "*.txt", subdirs=("Small",))

        run_campaign(
            instances,
            results_dir,
            id_prefix="vrf/",
            algorithm="IG",
            stochastic_solver=_fake_stochastic_solver,
            seeds=[42, 99],
            verbose=False,
        )

        seeds = _read_seeds(_raw_csv(results_dir), "vrf/")
        # 2 instances x 2 seeds: each seed appears exactly twice
        assert sorted(seeds) == [42, 42, 99, 99]

    def test_summary_is_regenerated(self, tmp_path: Path) -> None:
        """After a multi-seed campaign the summary table exists."""
        data_dir = _make_instances(tmp_path, count=2)
        results_dir = tmp_path / "results"
        instances = discover_instances(data_dir, "*.txt", subdirs=("Small",))

        run_campaign(
            instances,
            results_dir,
            id_prefix="vrf/",
            algorithm="IG",
            stochastic_solver=_fake_stochastic_solver,
            seeds=[42, 99],
            verbose=False,
        )

        assert (results_dir / "summary" / "summary.csv").exists()


class TestResumeMultiSeed:
    """Resume does not duplicate rows keyed by (instance_id, seed)."""

    def test_resume_does_not_duplicate(self, tmp_path: Path) -> None:
        """Re-running a completed multi-seed campaign adds no new rows."""
        data_dir = _make_instances(tmp_path, count=2)
        results_dir = tmp_path / "results"
        instances = discover_instances(data_dir, "*.txt", subdirs=("Small",))

        run_campaign(
            instances,
            results_dir,
            id_prefix="vrf/",
            algorithm="IG",
            stochastic_solver=_fake_stochastic_solver,
            seeds=[42, 99],
            verbose=False,
        )
        assert _count_rows(_raw_csv(results_dir), "vrf/") == 4

        # Re-run: no new rows
        code = run_campaign(
            instances,
            results_dir,
            id_prefix="vrf/",
            algorithm="IG",
            stochastic_solver=_fake_stochastic_solver,
            seeds=[42, 99],
            verbose=False,
        )
        assert code == EXIT_OK
        assert _count_rows(_raw_csv(results_dir), "vrf/") == 4

    def test_partial_resume_completes(self, tmp_path: Path) -> None:
        """A partially completed campaign resumes only the missing units."""
        data_dir = _make_instances(tmp_path, count=2)
        results_dir = tmp_path / "results"
        instances = discover_instances(data_dir, "*.txt", subdirs=("Small",))

        # Run with limit=2 (only 2 of the 4 replications)
        run_campaign(
            instances,
            results_dir,
            id_prefix="vrf/",
            algorithm="IG",
            stochastic_solver=_fake_stochastic_solver,
            seeds=[42, 99],
            limit=2,
            verbose=False,
        )
        assert _count_rows(_raw_csv(results_dir), "vrf/") == 2

        # Resume: completes the remaining 2
        run_campaign(
            instances,
            results_dir,
            id_prefix="vrf/",
            algorithm="IG",
            stochastic_solver=_fake_stochastic_solver,
            seeds=[42, 99],
            verbose=False,
        )
        assert _count_rows(_raw_csv(results_dir), "vrf/") == 4


class TestFreshMultiSeed:
    """--fresh removes only this campaign's rows."""

    def test_fresh_removes_only_own_rows(self, tmp_path: Path) -> None:
        """fresh=True removes only the campaign's rows, keeping others."""
        data_dir = _make_instances(tmp_path, count=2)
        results_dir = tmp_path / "results"
        csv_path = _raw_csv(results_dir)
        csv_path.parent.mkdir(parents=True)

        # Pre-seed with a NEH row from a different campaign file
        # (same CSV for illustration; in reality IG and NEH are different files,
        # but fresh is scoped by id_prefix within one CSV)
        neh_csv = results_dir / "raw" / "IG.csv"
        neh_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(neh_csv, "w", encoding="utf-8", newline="") as handle:
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
            # A row from a different campaign prefix
            writer.writerow(
                [
                    "taillard/tai20_5_0",
                    "",
                    "20",
                    "5",
                    "IG",
                    "42",
                    "1278",
                    "0.001",
                    "0.63",
                    "run-x",
                    "2026-06-07T00:00:00Z",
                ]
            )

        instances = discover_instances(data_dir, "*.txt", subdirs=("Small",))

        # Run with fresh: this should drop any pre-existing "vrf/" rows and add new
        code = run_campaign(
            instances,
            results_dir,
            id_prefix="vrf/",
            algorithm="IG",
            stochastic_solver=_fake_stochastic_solver,
            seeds=[42, 99],
            fresh=True,
            verbose=False,
        )

        assert code == EXIT_OK
        # 4 new vrf rows
        assert _count_rows(csv_path, "vrf/") == 4
        # Taillard row preserved
        assert _count_rows(csv_path, "taillard/") == 1


class TestDeterministicBackwardsCompat:
    """seeds=None preserves Phase 1 deterministic behaviour."""

    def test_seeds_none_uses_deterministic_solver(self, tmp_path: Path) -> None:
        """With seeds=None, the campaign uses the regular solver (NEH-style)."""
        data_dir = _make_instances(tmp_path, count=2)
        results_dir = tmp_path / "results"
        instances = discover_instances(data_dir, "*.txt", subdirs=("Small",))

        code = run_campaign(
            instances,
            results_dir,
            id_prefix="vrf/",
            algorithm="NEH",
            verbose=False,
        )

        assert code == EXIT_OK
        assert _count_rows(results_dir / "raw" / "NEH.csv", "vrf/") == 2

    def test_seeds_without_stochastic_solver_raises(self, tmp_path: Path) -> None:
        """Providing seeds without stochastic_solver raises ValueError."""
        data_dir = _make_instances(tmp_path, count=1)
        results_dir = tmp_path / "results"
        instances = discover_instances(data_dir, "*.txt", subdirs=("Small",))

        with pytest.raises(ValueError, match="stochastic_solver"):
            run_campaign(
                instances,
                results_dir,
                id_prefix="vrf/",
                algorithm="IG",
                seeds=[42],
                verbose=False,
            )
