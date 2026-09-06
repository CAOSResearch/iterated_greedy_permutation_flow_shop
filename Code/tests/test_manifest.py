"""Tests for the Run_Manifest and environment capture (``pfsp.results.manifest``).

These tests cover Requisitos 1.11, 8.1 and 8.2:

* 1.11 — The environment-capture routine reports the running Python version and
  the installed NumPy version.
* 8.1 — A generated Run_Manifest records the Instance_Identifier, the algorithm
  and its parameters, the seed, the captured environment, a timestamp and the
  paths of the output files, and its ``manifest_id`` follows the
  ``run-YYYYMMDD-HHMMSS-<short>`` pattern.
* 8.2 — The manifest is written in a machine-readable, human-inspectable JSON
  format alongside the raw results.
"""

from __future__ import annotations

import json
import platform
import re
from datetime import UTC, datetime
from importlib import metadata

import numpy as np

from pfsp.results.manifest import (
    MANIFESTS_SUBDIR,
    RunManifest,
    build_manifest,
    capture_environment,
    generate_manifest_id,
    write_manifest,
)

#: Canonical pattern for a ``manifest_id``: ``run-YYYYMMDD-HHMMSS-<short>``.
MANIFEST_ID_PATTERN = re.compile(r"^run-\d{8}-\d{6}-[0-9a-f]{4}$")

#: ISO-8601 UTC timestamp with a trailing ``Z`` (e.g. ``2026-06-07T15:30:00Z``).
TIMESTAMP_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

#: Fields that every serialized manifest must expose (Requisito 8.1).
REQUIRED_FIELDS = frozenset(
    {
        "manifest_id",
        "instance_id",
        "algorithm",
        "parameters",
        "seed",
        "environment",
        "timestamp",
        "outputs",
    }
)


# --- Requisito 8.1: manifest_id pattern ------------------------------------


def test_generate_manifest_id_matches_pattern() -> None:
    """``generate_manifest_id`` returns an id of the documented shape."""
    manifest_id = generate_manifest_id()

    assert MANIFEST_ID_PATTERN.match(manifest_id), manifest_id


def test_manifest_id_encodes_the_given_moment() -> None:
    """The date/time fields of the id reflect the supplied UTC instant."""
    moment = datetime(2026, 6, 7, 15, 30, 0, tzinfo=UTC)

    manifest_id = generate_manifest_id(moment)

    assert MANIFEST_ID_PATTERN.match(manifest_id), manifest_id
    assert manifest_id.startswith("run-20260607-153000-")


def test_built_manifest_id_matches_pattern() -> None:
    """A manifest assembled by ``build_manifest`` carries a conforming id."""
    manifest = build_manifest(
        instance_id="taillard/tai20_5_0",
        algorithm="NEH",
        outputs=["results/raw/NEH.csv"],
    )

    assert MANIFEST_ID_PATTERN.match(manifest.manifest_id), manifest.manifest_id


# --- Requisito 8.1: required fields present in the manifest JSON -----------


def test_manifest_json_contains_required_fields() -> None:
    """The serialized manifest exposes every required provenance field."""
    manifest = build_manifest(
        instance_id="vrf/VFR10_5_1",
        algorithm="NEH",
        outputs=["results/raw/NEH.csv"],
        seed=12345,
        parameters={"variant": "classic"},
    )

    data = json.loads(manifest.to_json())

    assert REQUIRED_FIELDS <= set(data)


def test_manifest_records_supplied_provenance_values() -> None:
    """The recorded values match exactly what the run supplied (no fabrication)."""
    manifest = build_manifest(
        instance_id="taillard/tai20_5_0",
        algorithm="NEH",
        outputs=["results/raw/NEH.csv", "results/raw/NEH.json"],
        seed=2025,
        parameters={"variant": "classic"},
    )

    data = json.loads(manifest.to_json())

    assert data["instance_id"] == "taillard/tai20_5_0"
    assert data["algorithm"] == "NEH"
    assert data["seed"] == 2025
    assert data["parameters"] == {"variant": "classic"}
    assert data["outputs"] == ["results/raw/NEH.csv", "results/raw/NEH.json"]


def test_manifest_environment_has_python_and_numpy() -> None:
    """The serialized ``environment`` block carries python and numpy versions."""
    manifest = build_manifest(
        instance_id="taillard/tai20_5_0",
        algorithm="NEH",
        outputs=["results/raw/NEH.csv"],
    )

    environment = json.loads(manifest.to_json())["environment"]

    assert "python" in environment
    assert "numpy" in environment


def test_manifest_timestamp_is_iso8601_utc() -> None:
    """The timestamp is an ISO-8601 UTC string ending in ``Z``."""
    moment = datetime(2026, 6, 7, 15, 30, 0, tzinfo=UTC)

    manifest = build_manifest(
        instance_id="taillard/tai20_5_0",
        algorithm="NEH",
        outputs=["results/raw/NEH.csv"],
        moment=moment,
    )

    timestamp = manifest.timestamp
    assert TIMESTAMP_PATTERN.match(timestamp), timestamp
    assert timestamp == "2026-06-07T15:30:00Z"


def test_seed_is_preserved_as_none_when_absent() -> None:
    """A run that consumes no randomness keeps ``seed`` as ``None``, not faked."""
    manifest = build_manifest(
        instance_id="taillard/tai20_5_0",
        algorithm="NEH",
        outputs=["results/raw/NEH.csv"],
    )

    assert json.loads(manifest.to_json())["seed"] is None


# --- Requisito 1.11: captured environment reflects the running versions ----


def test_capture_environment_reports_running_python_and_numpy() -> None:
    """The captured environment matches the versions actually in use."""
    environment = capture_environment()

    assert environment["python"] == platform.python_version()
    assert environment["numpy"] == metadata.version("numpy")
    # Sanity: numpy reports the same version through its own attribute.
    assert environment["numpy"] == np.__version__


# --- Requisito 8.2: manifest written alongside raw results -----------------


def test_write_manifest_creates_json_file_under_manifests_dir(tmp_path) -> None:
    """``write_manifest`` writes ``<results_dir>/manifests/<id>.json``."""
    manifest = build_manifest(
        instance_id="taillard/tai20_5_0",
        algorithm="NEH",
        outputs=["results/raw/NEH.csv"],
        seed=7,
    )

    path = write_manifest(manifest, tmp_path)

    assert path.exists()
    assert path.parent == tmp_path / MANIFESTS_SUBDIR
    assert path.name == f"{manifest.manifest_id}.json"


def test_written_manifest_is_valid_inspectable_json(tmp_path) -> None:
    """The persisted file is valid JSON whose content round-trips the manifest."""
    manifest = build_manifest(
        instance_id="taillard/tai20_5_0",
        algorithm="NEH",
        outputs=["results/raw/NEH.csv"],
        seed=7,
    )

    path = write_manifest(manifest, tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))

    assert REQUIRED_FIELDS <= set(data)
    assert data["manifest_id"] == manifest.manifest_id
    assert data["environment"] == manifest.environment


def test_round_trip_preserves_manifest_contents() -> None:
    """``to_dict`` and ``to_json`` agree, so the JSON is a faithful view."""
    manifest = RunManifest(
        manifest_id="run-20260607-153000-ab12",
        instance_id="taillard/tai20_5_0",
        algorithm="NEH",
        environment={"python": "3.13.0", "numpy": "2.0.0", "platform": "test"},
        timestamp="2026-06-07T15:30:00Z",
        outputs=["results/raw/NEH.csv"],
    )

    assert json.loads(manifest.to_json()) == manifest.to_dict()
