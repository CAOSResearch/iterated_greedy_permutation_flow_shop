"""Run manifest and execution-environment capture for the results framework.

A :class:`RunManifest` is the per-execution provenance artifact of the project:
it records, in human-inspectable JSON, which instance was run, with which
algorithm and parameters, under which seed and environment, when, and where the
outputs were written. Every result record references its manifest by
``manifest_id`` so that any figure or table can be traced back to the exact
execution that produced it.

In line with the project's hard constraint, the manifest carries only execution
metadata and the makespan objective; no energy or power metrics are ever
recorded.
"""

from __future__ import annotations

import json
import platform
import secrets
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

#: Prefix of every ``manifest_id`` (also the run identifier).
MANIFEST_ID_PREFIX = "run"

#: Number of hexadecimal characters in the random suffix of a ``manifest_id``.
#: The suffix only disambiguates runs that fall in the same second; it is a file
#: identifier and does not affect reproducibility of the result itself.
SHORT_SUFFIX_HEX_LEN = 4

#: Sub-directory, relative to the results directory, where manifests are written.
MANIFESTS_SUBDIR = "manifests"


def capture_environment() -> dict[str, str]:
    """Capture the execution environment relevant to reproducibility.

    Returns
    -------
    dict[str, str]
        A mapping with the running ``python`` version (via :mod:`platform`), the
        installed ``numpy`` version (via :mod:`importlib.metadata`) and a
        descriptive ``platform`` string for the host.
    """
    return {
        "python": platform.python_version(),
        "numpy": metadata.version("numpy"),
        "platform": platform.platform(),
    }


def _format_timestamp(moment: datetime) -> str:
    """Format ``moment`` as an ISO-8601 UTC timestamp with a trailing ``Z``."""
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def generate_manifest_id(
    moment: datetime | None = None,
    *,
    short_hex_len: int = SHORT_SUFFIX_HEX_LEN,
) -> str:
    """Build a manifest identifier of the form ``run-YYYYMMDD-HHMMSS-<short>``.

    Parameters
    ----------
    moment:
        The reference instant. Defaults to the current UTC time. It is converted
        to UTC before formatting so identifiers are timezone-independent.
    short_hex_len:
        Number of hexadecimal characters in the random suffix. Defaults to
        :data:`SHORT_SUFFIX_HEX_LEN`.

    Returns
    -------
    str
        The manifest identifier, e.g. ``"run-20260607-153000-ab12"``.

    Raises
    ------
    ValueError
        If ``short_hex_len`` is not a positive integer.
    """
    if short_hex_len <= 0:
        raise ValueError(f"short_hex_len must be positive, got {short_hex_len}")
    when = (moment or datetime.now(UTC)).astimezone(UTC)
    stamp = when.strftime("%Y%m%d-%H%M%S")
    # ``token_hex(n)`` yields ``2*n`` hex chars; request enough bytes and slice
    # to the requested length so odd lengths are supported.
    suffix = secrets.token_hex((short_hex_len + 1) // 2)[:short_hex_len]
    return f"{MANIFEST_ID_PREFIX}-{stamp}-{suffix}"


@dataclass(frozen=True)
class RunManifest:
    """Provenance record of a single execution.

    Attributes
    ----------
    manifest_id:
        Unique run identifier (``run-YYYYMMDD-HHMMSS-<short>``).
    instance_id:
        ``Instance_Identifier`` of the instance that was run
        (e.g. ``"taillard/tai20_5_0"``).
    algorithm:
        Name of the algorithm executed (e.g. ``"NEH"``).
    parameters:
        Algorithm parameters as a JSON-serializable mapping (empty for NEH).
    seed:
        Explicit seed used, or ``None`` for deterministic runs that consume no
        randomness. Never fabricated.
    environment:
        Captured execution environment (see :func:`capture_environment`).
    timestamp:
        ISO-8601 UTC timestamp of the run.
    outputs:
        Paths of the output files produced by the run (kept relative-to-repo by
        callers for portability).
    makespan:
        The makespan objective of the run, when known. ``None`` omits it from
        the serialized manifest. No objective other than ``C_max`` is recorded.
    """

    manifest_id: str
    instance_id: str
    algorithm: str
    parameters: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None
    environment: dict[str, str] = field(default_factory=dict)
    timestamp: str = ""
    outputs: list[str] = field(default_factory=list)
    makespan: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the manifest as an ordered, JSON-serializable dictionary."""
        data: dict[str, Any] = {
            "manifest_id": self.manifest_id,
            "instance_id": self.instance_id,
            "algorithm": self.algorithm,
            "parameters": dict(self.parameters),
            "seed": self.seed,
            "environment": dict(self.environment),
            "timestamp": self.timestamp,
            "outputs": list(self.outputs),
        }
        # The makespan objective is included only when known; energy or any other
        # objective is never recorded.
        if self.makespan is not None:
            data["makespan"] = self.makespan
        return data

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the manifest to a human-inspectable JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


def build_manifest(
    *,
    instance_id: str,
    algorithm: str,
    outputs: list[str],
    seed: int | None = None,
    parameters: dict[str, Any] | None = None,
    makespan: int | None = None,
    environment: dict[str, str] | None = None,
    moment: datetime | None = None,
    manifest_id: str | None = None,
) -> RunManifest:
    """Assemble a :class:`RunManifest` capturing the environment and timestamp.

    The ``manifest_id`` and the ``timestamp`` are derived from the same instant
    so they stay consistent. The environment is captured automatically unless an
    explicit ``environment`` mapping is supplied (useful for testing).

    Parameters
    ----------
    instance_id:
        ``Instance_Identifier`` of the instance that was run.
    algorithm:
        Name of the algorithm executed.
    outputs:
        Paths of the output files produced by the run.
    seed:
        Explicit seed used, or ``None`` when no randomness is consumed.
    parameters:
        Algorithm parameters. Defaults to an empty mapping.
    makespan:
        The makespan objective of the run, when known.
    environment:
        Pre-captured environment. Defaults to :func:`capture_environment`.
    moment:
        Reference instant for the identifier and timestamp. Defaults to now (UTC).
    manifest_id:
        Explicit identifier. Defaults to one generated from ``moment``.

    Returns
    -------
    RunManifest
        The assembled manifest.
    """
    when = (moment or datetime.now(UTC)).astimezone(UTC)
    return RunManifest(
        manifest_id=manifest_id or generate_manifest_id(when),
        instance_id=instance_id,
        algorithm=algorithm,
        parameters=dict(parameters) if parameters else {},
        seed=seed,
        environment=environment if environment is not None else capture_environment(),
        timestamp=_format_timestamp(when),
        outputs=list(outputs),
        makespan=makespan,
    )


def write_manifest(manifest: RunManifest, results_dir: str | Path) -> Path:
    """Write ``manifest`` as JSON alongside the raw results.

    The file is created at ``<results_dir>/manifests/<manifest_id>.json``; the
    ``manifests`` sub-directory is created if needed.

    Parameters
    ----------
    manifest:
        The manifest to persist.
    results_dir:
        Root directory of the run's results.

    Returns
    -------
    pathlib.Path
        The path of the written manifest file.
    """
    manifest_dir = Path(results_dir) / MANIFESTS_SUBDIR
    manifest_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_dir / f"{manifest.manifest_id}.json"
    path.write_text(manifest.to_json() + "\n", encoding="utf-8")
    return path


__all__ = [
    "MANIFEST_ID_PREFIX",
    "SHORT_SUFFIX_HEX_LEN",
    "MANIFESTS_SUBDIR",
    "RunManifest",
    "capture_environment",
    "generate_manifest_id",
    "build_manifest",
    "write_manifest",
]
