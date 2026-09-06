"""Command-line entry point for the Iterated Greedy algorithm (``pfsp-ig``).

This module wires the IG algorithm into a single, reproducible run: it loads a
PFSP instance by path, constructs an :class:`~pfsp.core.ig.IGConfig` from CLI
arguments, executes ``iterated_greedy``, computes and prints the RPD against the
best-known reference, and persists both a result record (with seed) and a run
manifest (with IG parameters and LS flag).

Every parameter that varies across executions (instance, seed, d, tp, stopping
criterion, local search toggle, results directory) is supplied through the
command line rather than hardcoded (Req 11.3). The algorithm label is ``IG``
when local search is active (default) or ``IG-noLS`` when ``--no-local-search``
is passed.

The destruction operator is selectable with ``--destruction`` (``random``, the
unchanged Phase 2 baseline, ``idle-greedy`` for O1, or ``idle-rcl`` for O2), and
the guided RCL size fraction with ``--alpha`` (Req 11.1). The label reflects the
selected operator (``IG`` / ``IG-idle-greedy`` / ``IG-idle-rcl``), and both the
operator and alpha are recorded in the run manifest (Req 11.2). An invalid
operator or alpha is rejected by the argument parser before any run executes
(Req 11.3).

Usage
-----
::

    pfsp-ig --instance PATH --seed INT
            [--d INT] [--tp FLOAT]
            [--stop iterations:N | evaluations:N | time:SECONDS]
            [--no-local-search]
            [--destruction random|idle-greedy|idle-rcl] [--alpha FLOAT]
            [--results-dir DIR]
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from pfsp import PfspError
from pfsp.core.ig import IGConfig, StoppingCriterion, iterated_greedy
from pfsp.io.loader import load_instance
from pfsp.results.manifest import build_manifest, write_manifest
from pfsp.results.record import ResultRecord, append_record, compute_rpd, raw_csv_path
from pfsp.results.reference import get_reference

#: Default results directory, resolved relative to this package so it is portable
#: across checkouts: ``cli_ig.py`` -> ``pfsp`` -> repository root (``Code/``).
#: Defined at module level so ``conftest.py`` can monkeypatch it for test isolation.
_DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

#: Valid destruction operator names, in the order shown in ``--help``. Mirrors
#: the operators recognized by :func:`pfsp.core.ig.validate_config`.
_DESTRUCTION_CHOICES: tuple[str, ...] = ("random", "idle-greedy", "idle-rcl")


def _parse_stop(value: str) -> StoppingCriterion:
    """Parse a ``kind:value`` string into a :class:`StoppingCriterion`.

    Parameters
    ----------
    value:
        String in the format ``"evaluations:5000"``, ``"iterations:1000"``, or
        ``"time:30.5"``.

    Returns
    -------
    StoppingCriterion
        The parsed stopping criterion.

    Raises
    ------
    argparse.ArgumentTypeError
        If the format is invalid (missing colon, unknown kind, non-numeric value,
        or non-positive value).
    """
    valid_kinds = {"evaluations", "iterations", "time"}
    parts = value.split(":", maxsplit=1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Invalid --stop format: {value!r}. Expected 'kind:value' "
            f"(e.g. 'evaluations:5000')."
        )
    kind, raw_val = parts[0].strip(), parts[1].strip()
    if kind not in valid_kinds:
        raise argparse.ArgumentTypeError(
            f"Unknown stopping criterion kind: {kind!r}. "
            f"Valid kinds: {sorted(valid_kinds)}."
        )
    try:
        num = float(raw_val)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Non-numeric stopping value: {raw_val!r} in {value!r}."
        ) from exc
    if num <= 0:
        raise argparse.ArgumentTypeError(
            f"Stopping value must be positive, got {num} in {value!r}."
        )
    return StoppingCriterion(kind=kind, value=num)  # type: ignore[arg-type]


def _parse_alpha(value: str) -> float:
    """Parse and validate the ``--alpha`` RCL-size fraction.

    Parameters
    ----------
    value:
        String to parse as a float in the half-open range ``(0, 1]``.

    Returns
    -------
    float
        The parsed alpha value.

    Raises
    ------
    argparse.ArgumentTypeError
        If the value is non-numeric or outside ``0 < alpha <= 1``, so argparse
        rejects it and no run is executed (Req 11.3).
    """
    try:
        alpha = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Non-numeric --alpha value: {value!r}."
        ) from exc
    if not (0 < alpha <= 1):
        raise argparse.ArgumentTypeError(
            f"--alpha must be in the range 0 < alpha <= 1, got {alpha}."
        )
    return alpha


def _algorithm_label(destruction: str, local_search_enabled: bool) -> str:
    """Return the ResultRecord algorithm label for the run.

    The label mirrors the Phase 1/2 scheme so results stay comparable in the
    same tables (Req 11.2): ``"IG"`` for the classic random operator,
    ``"IG-idle-greedy"`` for O1, and ``"IG-idle-rcl"`` for O2. When local search
    is disabled a ``"-noLS"`` suffix is appended, preserving the existing
    ``"IG-noLS"`` label for the random baseline.

    Parameters
    ----------
    destruction:
        Selected destruction operator (assumed already validated by argparse).
    local_search_enabled:
        Whether the insertion-based local search is active.

    Returns
    -------
    str
        The algorithm label written to the ResultRecord and used to route the
        raw CSV (one file per label).
    """
    base = {
        "random": "IG",
        "idle-greedy": "IG-idle-greedy",
        "idle-rcl": "IG-idle-rcl",
    }[destruction]
    return base if local_search_enabled else f"{base}-noLS"


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ``pfsp-ig`` entry point."""
    parser = argparse.ArgumentParser(
        prog="pfsp-ig",
        description=(
            "Load a PFSP instance, run the Iterated Greedy algorithm, and "
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
        required=True,
        type=int,
        metavar="INT",
        help="Non-negative integer seed for the random generator.",
    )
    parser.add_argument(
        "--d",
        type=int,
        default=None,
        metavar="INT",
        help="Destruction size (number of jobs to remove per iteration). Default: 4.",
    )
    parser.add_argument(
        "--tp",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Temperature factor (Tp >= 0). Default: 0.4.",
    )
    parser.add_argument(
        "--stop",
        type=_parse_stop,
        default=None,
        metavar="KIND:VALUE",
        help=(
            "Stopping criterion in 'kind:value' format. "
            "Kinds: evaluations, iterations, time. "
            "Example: 'evaluations:5000'. Default: evaluations:5000."
        ),
    )
    parser.add_argument(
        "--no-local-search",
        action="store_true",
        default=False,
        help="Disable the insertion-based local search (default: LS enabled).",
    )
    parser.add_argument(
        "--destruction",
        choices=_DESTRUCTION_CHOICES,
        default="random",
        metavar="{random,idle-greedy,idle-rcl}",
        help=(
            "Destruction operator: 'random' (classic Phase 2 baseline, the "
            "default), 'idle-greedy' (O1), or 'idle-rcl' (O2, guided idle-RCL)."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=_parse_alpha,
        default=0.10,
        metavar="FLOAT",
        help=(
            "RCL size fraction for the 'idle-rcl' operator (0 < alpha <= 1). "
            "Ignored by 'random' and 'idle-greedy'. Default: 0.10."
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
    """Run the Iterated Greedy on one instance and persist its artifacts.

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

    # --- Determine IG configuration from CLI args ---
    local_search_enabled = not args.no_local_search
    algorithm = _algorithm_label(args.destruction, local_search_enabled)

    config_kwargs: dict = {}
    if args.d is not None:
        config_kwargs["destruction_size"] = args.d
    if args.tp is not None:
        config_kwargs["temperature_factor"] = args.tp
    if args.stop is not None:
        config_kwargs["stop"] = args.stop
    config_kwargs["local_search"] = local_search_enabled
    config_kwargs["destruction"] = args.destruction
    config_kwargs["alpha"] = args.alpha

    config = IGConfig(**config_kwargs)

    # --- Load the instance by path (Req 11.4) ---
    try:
        instance = load_instance(args.instance)
    except (PfspError, OSError) as error:
        print(
            f"error: could not load instance '{args.instance}': {error}",
            file=sys.stderr,
        )
        return 2

    # --- Run IG, timing the wall-clock runtime around the call only ---
    start = time.perf_counter()
    permutation, makespan_value = iterated_greedy(instance, args.seed, config)
    runtime_s = time.perf_counter() - start

    # --- Compute RPD against the best-known reference (None if unresolved) ---
    reference = get_reference(instance)
    rpd = compute_rpd(makespan_value, reference)

    # --- Report the permutation and makespan to stdout (Req 11.1) ---
    print(f"instance: {instance.identifier}")
    print(f"algorithm: {algorithm}")
    print(f"seed: {args.seed}")
    print(f"permutation: {_format_permutation(permutation)}")
    print(f"makespan: {makespan_value}")
    print(f"rpd: {'n/a' if rpd is None else f'{rpd:.4f}'}")

    # --- Build the manifest with IG parameters ---
    csv_path = raw_csv_path(algorithm, results_dir)
    moment = datetime.now(UTC)
    manifest = build_manifest(
        instance_id=instance.identifier,
        algorithm=algorithm,
        outputs=[str(csv_path)],
        seed=args.seed,
        parameters={
            "d": config.destruction_size,
            "tp": config.temperature_factor,
            "stop_kind": config.stop.kind,
            "stop_value": config.stop.value,
            "local_search": local_search_enabled,
            "destruction": config.destruction,
            "alpha": config.alpha,
        },
        makespan=makespan_value,
        moment=moment,
    )

    # --- Persist the result record, linked to the manifest (Reqs 8.6, 9.2) ---
    record = ResultRecord(
        instance_id=instance.identifier,
        instance_set=instance.instance_set,
        n=instance.n,
        m=instance.m,
        algorithm=algorithm,
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
