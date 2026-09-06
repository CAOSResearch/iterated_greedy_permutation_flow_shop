"""Command-line entry point for the NEH baseline (``pfsp-neh``).

This module wires the package together into a single, reproducible run: it loads
a PFSP instance by path, executes the NEH constructive heuristic, computes the
Relative Percentage Deviation (RPD) against the best-known reference, prints the
resulting permutation and makespan, and persists both a result record and a run
manifest that links them (Reqs 1.12, 1.14).

Every parameter that varies across executions (the instance path, the optional
seed and the optional results directory) is supplied through the command line
rather than hardcoded (Req 1.12). For Phase 1 the cardinality is one-to-one: a
single invocation processes one instance and produces one manifest and one
result record.

Usage
-----
::

    pfsp-neh --instance PATH [--seed INT] [--results-dir DIR]
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from pfsp import PfspError
from pfsp.core.neh import neh
from pfsp.io.loader import load_instance
from pfsp.results.manifest import build_manifest, write_manifest
from pfsp.results.record import ResultRecord, append_record, compute_rpd, raw_csv_path
from pfsp.results.reference import get_reference

#: Algorithm name recorded in result records and manifests for this entry point.
_ALGORITHM = "NEH"

#: Default results directory, resolved relative to this package so it is portable
#: across checkouts: ``cli.py`` -> ``pfsp`` -> repository root (``Code/``). Matches
#: the package-relative ``results/`` location used by the results framework.
_DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ``pfsp-neh`` entry point."""
    parser = argparse.ArgumentParser(
        prog="pfsp-neh",
        description=(
            "Load a PFSP instance, run the NEH constructive heuristic, and "
            "report the resulting permutation and makespan."
        ),
    )
    parser.add_argument(
        "--instance",
        required=True,
        metavar="PATH",
        help="Path to a Taillard '.fsp' or VRF 'VFR*' instance file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="INT",
        help=(
            "Optional integer seed recorded with the run. NEH is deterministic "
            "and consumes no randomness, so this is provenance metadata only."
        ),
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        metavar="DIR",
        help=(
            "Directory under which the raw result record and run manifest are "
            "written. Defaults to the package 'results/' directory."
        ),
    )
    return parser


def _format_permutation(permutation: Sequence[int]) -> str:
    """Render a permutation as a space-separated list of job indices."""
    return " ".join(str(job) for job in permutation)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the NEH baseline on one instance and persist its artifacts.

    Parameters
    ----------
    argv:
        Argument vector to parse. Defaults to ``sys.argv[1:]`` when ``None``.

    Returns
    -------
    int
        Process exit code: ``0`` on success, ``2`` when the instance cannot be
        loaded or parsed.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    results_dir = (
        Path(args.results_dir) if args.results_dir is not None else _DEFAULT_RESULTS_DIR
    )

    # --- Load the instance by path (Req 1.14) ---
    try:
        instance = load_instance(args.instance)
    except (PfspError, OSError) as error:
        print(
            f"error: could not load instance '{args.instance}': {error}",
            file=sys.stderr,
        )
        return 2

    # --- Run NEH, timing the wall-clock runtime around the call only ---
    start = time.perf_counter()
    permutation, makespan_value = neh(instance)
    runtime_s = time.perf_counter() - start

    # --- Compute RPD against the best-known reference (None if unresolved) ---
    reference = get_reference(instance)
    rpd = compute_rpd(makespan_value, reference)

    # --- Report the permutation and makespan to stdout (Req 1.14) ---
    print(f"instance: {instance.identifier}")
    print(f"algorithm: {_ALGORITHM}")
    print(f"permutation: {_format_permutation(permutation)}")
    print(f"makespan: {makespan_value}")
    print(f"rpd: {'n/a' if rpd is None else f'{rpd:.4f}'}")

    # --- Build the manifest first so the record can reference its id ---
    csv_path = raw_csv_path(_ALGORITHM, results_dir)
    moment = datetime.now(UTC)
    manifest = build_manifest(
        instance_id=instance.identifier,
        algorithm=_ALGORITHM,
        outputs=[str(csv_path)],
        seed=args.seed,
        makespan=makespan_value,
        moment=moment,
    )

    # --- Persist the result record, linked to the manifest (Reqs 8.3, 10.8) ---
    record = ResultRecord(
        instance_id=instance.identifier,
        instance_set=instance.instance_set,
        n=instance.n,
        m=instance.m,
        algorithm=_ALGORITHM,
        seed=args.seed,
        makespan=makespan_value,
        runtime_s=runtime_s,
        rpd=rpd,
        manifest_id=manifest.manifest_id,
        timestamp=manifest.timestamp,
    )
    append_record(record, results_dir)
    manifest_path = write_manifest(manifest, results_dir)

    print(f"record: {csv_path}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
