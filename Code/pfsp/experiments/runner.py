"""Reusable experiment campaign runner.

Centralizes the logic shared by every benchmark campaign — the NEH baseline
today (EXP-001 Taillard, EXP-002 VRF), the Iterated Greedy and the Smart
Destruction operator later — so it is implemented and tested once instead of
being copied per script:

* instance discovery (:func:`discover_instances`),
* per-instance execution of the makespan objective via a pluggable ``solver``
  (NEH by default), reusing the ``pfsp`` core,
* per-instance result persistence (:class:`~pfsp.results.record.ResultRecord`)
  with a :class:`~pfsp.results.manifest.RunManifest`, RPD against the best-known
  reference, and the derived summary table,
* the **checkpoint/resume** behaviour required by the project steering
  (``codigo-python.md``): each result is flushed as soon as its instance
  finishes; on start the already-persisted instances are skipped; ``fresh``
  drops only this campaign's own rows; ``limit`` runs a slice, and
* **progress logging** (timestamped ``[i/N]`` start/done lines with per-instance
  runtime, elapsed and ETA) through the standard :mod:`logging` module, so long
  campaigns (e.g. VRF ``Large``) are observable. The library only emits log
  records; the CLI entry points call :func:`configure_logging` to stream them to
  the terminal and to a persistent log file under ``results/logs/``.

The thin CLI scripts in ``Code/scripts/`` only select instances and a per-record
``sanity`` strategy, then delegate to :func:`run_campaign`. Two ready-made sanity
strategies are provided: :func:`lower_bound_sanity` (Taillard, ``makespan >= LB``)
and :func:`best_known_sanity` (VRF, ``makespan > 0`` and ``RPD >= 0``).

Campaigns sharing an algorithm name write to the same ``results/raw/<algorithm>.csv``
(the aggregator groups by ``instance_set``); the per-campaign ``id_prefix``
(``"taillard/"`` / ``"vrf/"``) isolates resume and ``fresh`` to a campaign's own
rows, so they never clobber each other.
"""

from __future__ import annotations

import csv
import logging
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from pfsp.core.neh import neh
from pfsp.instance import Instance
from pfsp.io.loader import load_instance
from pfsp.results.aggregate import default_raw_dir, default_summary_path, write_summary
from pfsp.results.manifest import build_manifest, write_manifest
from pfsp.results.record import ResultRecord, append_record, compute_rpd, raw_csv_path
from pfsp.results.reference import get_reference

#: A solver maps an :class:`Instance` to a ``(permutation, makespan)`` pair.
Solver = Callable[[Instance], tuple[list[int], int]]

#: A stochastic solver maps an :class:`Instance` and a seed to a
#: ``(permutation, makespan)`` pair. Used for multi-seed campaigns (IG).
StochasticSolver = Callable[[Instance, int], tuple[list[int], int]]

#: A sanity strategy inspects a finished record (and its instance) and returns a
#: list of human-readable anomaly messages (empty when the record is sound).
SanityCheck = Callable[["ResultRecord", Instance], list[str]]

# Process exit codes returned by :func:`run_campaign`.
EXIT_OK = 0
EXIT_ANOMALY = 1
EXIT_NO_INSTANCES = 2
EXIT_INTERRUPTED = 130

#: Base logger for the experiments layer. The library emits records under it and
#: leaves output configuration to the application (the CLI entry points).
_BASE_LOGGER_NAME = "pfsp.experiments"
logger = logging.getLogger(f"{_BASE_LOGGER_NAME}.runner")


def configure_logging(results_dir: Path, campaign: str) -> Path:
    """Configure terminal + file logging for a campaign (called by entry points).

    Attaches a timestamped stream handler (stdout) and a file handler under
    ``<results_dir>/logs/<campaign>-<timestamp>.log`` to the experiments logger,
    so the campaign progress is visible live and kept on disk for long, resumable
    runs. Idempotent: existing handlers are replaced. Returns the log file path.

    This is application-level configuration; :func:`run_campaign` itself never
    configures logging (libraries should not), which keeps it quiet under tests
    and capturable with ``caplog``.
    """
    base = logging.getLogger(_BASE_LOGGER_NAME)
    base.setLevel(logging.INFO)
    for handler in list(base.handlers):
        base.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    base.addHandler(stream_handler)

    logs_dir = Path(results_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = logs_dir / f"{campaign}-{timestamp}.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    base.addHandler(file_handler)

    base.propagate = False
    return log_path


def _format_seconds(seconds: float) -> str:
    """Format a duration in seconds as a compact human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _start_watchdog(
    instance_id: str, warn_after_s: float | None
) -> threading.Timer | None:
    """Start a non-killing watchdog that warns if an instance runs too long.

    Returns a started :class:`threading.Timer` that logs a warning after
    ``warn_after_s`` seconds, or ``None`` when the watchdog is disabled. The
    caller must cancel the timer once the instance finishes. This never
    interrupts the computation: it only surfaces a possible hang or bug (a
    legitimate instance is slow, never "unsolvable" — NEH always terminates).
    """
    if warn_after_s is None:
        return None

    def _warn() -> None:
        logger.warning(
            "[watchdog] %s still running after %s — possible hang/bug or a very "
            "large instance (not killed).",
            instance_id,
            _format_seconds(warn_after_s),
        )

    timer = threading.Timer(warn_after_s, _warn)
    timer.daemon = True
    timer.start()
    return timer


def discover_instances(
    root: Path, pattern: str, subdirs: Sequence[str] | None = None
) -> list[Path]:
    """Return the sorted instance paths matching ``pattern`` under ``root``.

    When ``subdirs`` is given, ``pattern`` is matched inside each of those
    sub-directories of ``root`` (e.g. VRF's ``Small``/``Large``); otherwise it is
    matched directly under ``root`` (e.g. Taillard's flat folder). The result is
    sorted for a deterministic, reproducible processing order.
    """
    if subdirs:
        paths: list[Path] = []
        for subdir in subdirs:
            paths.extend((root / subdir).glob(pattern))
    else:
        paths = list(root.glob(pattern))
    return sorted(paths)


def instance_id_for(path: Path, id_prefix: str) -> str:
    """Return the ``Instance_Identifier`` of ``path`` without parsing the file.

    Mirrors :attr:`pfsp.instance.Instance.identifier` (``<benchmark>/<stem>``),
    so completed instances can be detected cheaply during resume.
    """
    return f"{id_prefix}{path.stem}"


def lower_bound_sanity(record: ResultRecord, instance: Instance) -> list[str]:
    """Taillard sanity: a valid schedule can never beat the lower bound ``LB``."""
    if instance.lower_bound is not None and record.makespan < instance.lower_bound:
        return [
            f"{record.instance_id}: makespan={record.makespan} "
            f"< LB={instance.lower_bound}"
        ]
    return []


def best_known_sanity(record: ResultRecord, instance: Instance) -> list[str]:
    """VRF sanity: ``makespan > 0`` and NEH should not beat the best-known RPD.

    VRF files carry no lower bound, so the invariant is a positive makespan and a
    non-negative RPD against the best-known value (a missing reference is not an
    anomaly; it is reported separately).
    """
    issues: list[str] = []
    if record.makespan <= 0:
        issues.append(f"{record.instance_id}: makespan={record.makespan} <= 0")
    if record.rpd is not None and record.rpd < 0:
        issues.append(
            f"{record.instance_id}: RPD={record.rpd:.4f}% < 0 (beats best-known)"
        )
    return issues


def _completed_ids(csv_path: Path, id_prefix: str) -> set[str]:
    """Read the persisted ``instance_id`` for ``id_prefix`` from ``csv_path``.

    Returns an empty set when the file does not exist. Rows of other campaigns
    (different prefix) are ignored, so resume is scoped to this campaign.
    """
    if not csv_path.exists():
        return set()
    completed: set[str] = set()
    with open(csv_path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            instance_id = (row.get("instance_id") or "").strip()
            if instance_id.startswith(id_prefix):
                completed.add(instance_id)
    return completed


def _completed_id_seed_pairs(csv_path: Path, id_prefix: str) -> set[tuple[str, int]]:
    """Read ``(instance_id, seed)`` pairs for ``id_prefix`` from ``csv_path``.

    Used for multi-seed campaigns where the resume key is the pair
    ``(instance_id, seed)`` rather than just ``instance_id``. Returns an empty
    set when the file does not exist. Rows with empty/missing seed are skipped
    (they belong to deterministic campaigns).
    """
    if not csv_path.exists():
        return set()
    completed: set[tuple[str, int]] = set()
    with open(csv_path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            instance_id = (row.get("instance_id") or "").strip()
            seed_str = (row.get("seed") or "").strip()
            if instance_id.startswith(id_prefix) and seed_str:
                try:
                    completed.add((instance_id, int(seed_str)))
                except ValueError:
                    pass  # Skip malformed seed values
    return completed


def _drop_rows(csv_path: Path, id_prefix: str) -> int:
    """Remove every ``id_prefix`` row from ``csv_path``, keeping all other rows.

    Used by ``fresh`` so a clean run discards only this campaign's rows, never
    those of other campaigns sharing the same CSV. Returns the rows removed.
    """
    if not csv_path.exists():
        return 0
    with open(csv_path, encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return 0
    header, body = rows[0], rows[1:]
    kept = [r for r in body if not (r and r[0].startswith(id_prefix))]
    removed = len(body) - len(kept)
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(kept)
    return removed


def _run_one(
    instance_path: Path, results_dir: Path, algorithm: str, solver: Solver
) -> tuple[ResultRecord, Instance]:
    """Run ``solver`` on one instance and persist its record and manifest.

    Persistence happens immediately (append + close), so a stop only loses the
    in-flight instance (the checkpoint unit). Returns the record and the loaded
    instance (the instance is needed by lower-bound sanity strategies).
    """
    instance = load_instance(str(instance_path))

    start = time.perf_counter()
    permutation, makespan_value = solver(instance)
    runtime_s = time.perf_counter() - start

    reference = get_reference(instance)
    rpd = compute_rpd(makespan_value, reference)

    csv_path = raw_csv_path(algorithm, results_dir)
    moment = datetime.now(UTC)
    manifest = build_manifest(
        instance_id=instance.identifier,
        algorithm=algorithm,
        outputs=[str(csv_path)],
        seed=None,
        makespan=makespan_value,
        moment=moment,
    )
    record = ResultRecord(
        instance_id=instance.identifier,
        instance_set=instance.instance_set,
        n=instance.n,
        m=instance.m,
        algorithm=algorithm,
        seed=None,
        makespan=makespan_value,
        runtime_s=runtime_s,
        rpd=rpd,
        manifest_id=manifest.manifest_id,
        timestamp=manifest.timestamp,
    )
    append_record(record, results_dir)
    write_manifest(manifest, results_dir)

    del permutation
    return record, instance


def _run_one_seeded(
    instance_path: Path,
    results_dir: Path,
    algorithm: str,
    stochastic_solver: StochasticSolver,
    seed: int,
    parameters: dict | None = None,
) -> tuple[ResultRecord, Instance]:
    """Run ``stochastic_solver`` on one instance with a specific seed.

    Like :func:`_run_one` but for multi-seed stochastic campaigns. The seed is
    recorded in both the :class:`ResultRecord` and the :class:`RunManifest`.
    """
    instance = load_instance(str(instance_path))

    start = time.perf_counter()
    permutation, makespan_value = stochastic_solver(instance, seed)
    runtime_s = time.perf_counter() - start

    reference = get_reference(instance)
    rpd = compute_rpd(makespan_value, reference)

    csv_path = raw_csv_path(algorithm, results_dir)
    moment = datetime.now(UTC)
    manifest = build_manifest(
        instance_id=instance.identifier,
        algorithm=algorithm,
        outputs=[str(csv_path)],
        seed=seed,
        parameters=parameters or {},
        makespan=makespan_value,
        moment=moment,
    )
    record = ResultRecord(
        instance_id=instance.identifier,
        instance_set=instance.instance_set,
        n=instance.n,
        m=instance.m,
        algorithm=algorithm,
        seed=seed,
        makespan=makespan_value,
        runtime_s=runtime_s,
        rpd=rpd,
        manifest_id=manifest.manifest_id,
        timestamp=manifest.timestamp,
    )
    append_record(record, results_dir)
    write_manifest(manifest, results_dir)

    del permutation
    return record, instance


def run_campaign(
    instance_paths: Sequence[Path],
    results_dir: Path,
    *,
    id_prefix: str,
    algorithm: str = "NEH",
    solver: Solver = neh,
    stochastic_solver: StochasticSolver | None = None,
    seeds: Sequence[int] | None = None,
    parameters: dict | None = None,
    sanity: SanityCheck | None = None,
    fresh: bool = False,
    limit: int | None = None,
    warn_after_s: float | None = None,
    verbose: bool = True,
) -> int:
    """Run a resumable campaign of ``algorithm`` over ``instance_paths``.

    Parameters
    ----------
    instance_paths:
        Instance files to process (already discovered/filtered by the caller).
    results_dir:
        Root of the results tree (``raw/``, ``manifests/``, ``summary/``).
    id_prefix:
        ``Instance_Identifier`` prefix of this campaign (``"taillard/"`` /
        ``"vrf/"``); scopes resume and ``fresh`` to its own rows.
    algorithm:
        Algorithm label recorded in records and manifests (default ``"NEH"``).
    solver:
        Callable mapping an instance to ``(permutation, makespan)`` (default
        :func:`~pfsp.core.neh.neh`). Used when ``seeds`` is ``None``.
    stochastic_solver:
        Callable mapping ``(instance, seed)`` to ``(permutation, makespan)``.
        Required when ``seeds`` is provided; ignored otherwise.
    seeds:
        Sequence of integer seeds for multi-seed stochastic campaigns (e.g. IG).
        When provided, each instance is run once per seed (one Replication per
        seed). When ``None`` (default), the campaign is deterministic (one run
        per instance using ``solver``).
    parameters:
        Algorithm parameters to record in the RunManifest for stochastic runs
        (e.g. ``{"d": 4, "tp": 0.4, ...}``). Ignored for deterministic runs.
    sanity:
        Optional per-record sanity strategy; anomalies set the exit code to
        :data:`EXIT_ANOMALY`.
    fresh:
        When ``True``, drop this campaign's previous rows before starting; the
        default resumes without wiping.
    limit:
        Process at most this many pending work units this pass (staged runs).
        For multi-seed campaigns, one work unit is one ``(instance, seed)`` pair.
    warn_after_s:
        If set, a watchdog logs a warning when an instance exceeds this many
        seconds (a possible hang/bug). It never kills the run; ``None`` disables
        it. The real stopping rule of a metaheuristic is its compute budget, not
        this watchdog.
    verbose:
        Emit progress log records (disabled in tests for quiet runs). Output is
        only shown when an application has configured logging (see
        :func:`configure_logging`).

    Returns
    -------
    int
        :data:`EXIT_OK`, :data:`EXIT_ANOMALY`, :data:`EXIT_NO_INSTANCES` or
        :data:`EXIT_INTERRUPTED`.

    Raises
    ------
    ValueError
        If ``seeds`` is provided but ``stochastic_solver`` is not.
    """
    if seeds is not None and stochastic_solver is None:
        raise ValueError("stochastic_solver must be provided when seeds is not None")

    if not instance_paths:
        logger.warning("No instances to run for campaign '%s'.", id_prefix)
        return EXIT_NO_INSTANCES

    csv_path = raw_csv_path(algorithm, results_dir)

    if fresh:
        removed = _drop_rows(csv_path, id_prefix)
        if verbose:
            logger.info("--fresh: removed %d previous '%s' row(s)", removed, id_prefix)

    # --- Multi-seed (stochastic) path ---
    if seeds is not None:
        return _run_campaign_multi_seed(
            instance_paths=instance_paths,
            results_dir=results_dir,
            csv_path=csv_path,
            id_prefix=id_prefix,
            algorithm=algorithm,
            stochastic_solver=stochastic_solver,  # type: ignore[arg-type]
            seeds=seeds,
            parameters=parameters,
            sanity=sanity,
            limit=limit,
            warn_after_s=warn_after_s,
            verbose=verbose,
        )

    # --- Deterministic (single-seed) path (unchanged from Phase 1) ---
    completed = _completed_ids(csv_path, id_prefix)
    pending = [
        p for p in instance_paths if instance_id_for(p, id_prefix) not in completed
    ]
    already = len(instance_paths) - len(pending)
    if limit is not None:
        pending = pending[:limit]

    if verbose:
        logger.info(
            "Campaign '%s': %d instance(s), %d already done, %d to run this pass.",
            id_prefix,
            len(instance_paths),
            already,
            len(pending),
        )
    if not pending:
        if verbose:
            logger.info(
                "Nothing pending; campaign already complete for this selection."
            )
        _finish(results_dir, csv_path, verbose)
        return EXIT_OK

    records: list[ResultRecord] = []
    anomalies: list[str] = []
    interrupted = False
    total = len(pending)
    campaign_start = time.perf_counter()
    try:
        for index, instance_path in enumerate(pending, start=1):
            if verbose:
                logger.info(
                    "[%d/%d] start %s",
                    index,
                    total,
                    instance_id_for(instance_path, id_prefix),
                )
            watchdog = _start_watchdog(
                instance_id_for(instance_path, id_prefix), warn_after_s
            )
            try:
                record, instance = _run_one(
                    instance_path, results_dir, algorithm, solver
                )
            finally:
                if watchdog is not None:
                    watchdog.cancel()
            records.append(record)
            if sanity is not None:
                anomalies.extend(sanity(record, instance))
            if verbose:
                elapsed = time.perf_counter() - campaign_start
                eta = (elapsed / index) * (total - index)
                rpd_text = "n/a" if record.rpd is None else f"{record.rpd:.3f}%"
                logger.info(
                    "[%d/%d] done %s (%dx%d) makespan=%d rpd=%s t=%.2fs "
                    "| elapsed=%s eta=%s",
                    index,
                    total,
                    record.instance_id,
                    record.n,
                    record.m,
                    record.makespan,
                    rpd_text,
                    record.runtime_s,
                    _format_seconds(elapsed),
                    _format_seconds(eta),
                )
    except KeyboardInterrupt:
        interrupted = True
        logger.warning(
            "Interrupted. Progress is saved; re-run the same command to resume."
        )

    if verbose and records:
        logger.info(
            "Ran %d instance(s) this pass in %s.",
            len(records),
            _format_seconds(time.perf_counter() - campaign_start),
        )
    _finish(results_dir, csv_path, verbose)
    if verbose:
        _report(records, anomalies)

    if interrupted:
        return EXIT_INTERRUPTED
    return EXIT_ANOMALY if anomalies else EXIT_OK


def _run_campaign_multi_seed(
    *,
    instance_paths: Sequence[Path],
    results_dir: Path,
    csv_path: Path,
    id_prefix: str,
    algorithm: str,
    stochastic_solver: StochasticSolver,
    seeds: Sequence[int],
    parameters: dict | None,
    sanity: SanityCheck | None,
    limit: int | None,
    warn_after_s: float | None,
    verbose: bool,
) -> int:
    """Inner loop for multi-seed stochastic campaigns.

    The resume key is ``(instance_id, seed)`` instead of just ``instance_id``.
    Each ``(instance, seed)`` pair is a separate work unit.
    """
    completed_pairs = _completed_id_seed_pairs(csv_path, id_prefix)

    # Build the full list of work units: (instance_path, seed)
    all_units: list[tuple[Path, int]] = [
        (p, seed) for p in instance_paths for seed in seeds
    ]
    pending = [
        (p, seed)
        for p, seed in all_units
        if (instance_id_for(p, id_prefix), seed) not in completed_pairs
    ]
    already = len(all_units) - len(pending)
    if limit is not None:
        pending = pending[:limit]

    if verbose:
        logger.info(
            "Campaign '%s': %d instance(s) × %d seed(s) = %d replication(s), "
            "%d already done, %d to run this pass.",
            id_prefix,
            len(instance_paths),
            len(seeds),
            len(all_units),
            already,
            len(pending),
        )
    if not pending:
        if verbose:
            logger.info(
                "Nothing pending; campaign already complete for this selection."
            )
        _finish(results_dir, csv_path, verbose)
        return EXIT_OK

    records: list[ResultRecord] = []
    anomalies: list[str] = []
    interrupted = False
    total = len(pending)
    campaign_start = time.perf_counter()
    try:
        for index, (instance_path, seed) in enumerate(pending, start=1):
            inst_id = instance_id_for(instance_path, id_prefix)
            if verbose:
                logger.info(
                    "[%d/%d] start %s seed=%d",
                    index,
                    total,
                    inst_id,
                    seed,
                )
            watchdog = _start_watchdog(f"{inst_id}@seed={seed}", warn_after_s)
            try:
                record, instance = _run_one_seeded(
                    instance_path,
                    results_dir,
                    algorithm,
                    stochastic_solver,
                    seed,
                    parameters,
                )
            finally:
                if watchdog is not None:
                    watchdog.cancel()
            records.append(record)
            if sanity is not None:
                anomalies.extend(sanity(record, instance))
            if verbose:
                elapsed = time.perf_counter() - campaign_start
                eta = (elapsed / index) * (total - index)
                rpd_text = "n/a" if record.rpd is None else f"{record.rpd:.3f}%"
                logger.info(
                    "[%d/%d] done %s seed=%d (%dx%d) makespan=%d rpd=%s "
                    "t=%.2fs | elapsed=%s eta=%s",
                    index,
                    total,
                    record.instance_id,
                    seed,
                    record.n,
                    record.m,
                    record.makespan,
                    rpd_text,
                    record.runtime_s,
                    _format_seconds(elapsed),
                    _format_seconds(eta),
                )
    except KeyboardInterrupt:
        interrupted = True
        logger.warning(
            "Interrupted. Progress is saved; re-run the same command to resume."
        )

    if verbose and records:
        logger.info(
            "Ran %d replication(s) this pass in %s.",
            len(records),
            _format_seconds(time.perf_counter() - campaign_start),
        )
    _finish(results_dir, csv_path, verbose)
    if verbose:
        _report(records, anomalies)

    if interrupted:
        return EXIT_INTERRUPTED
    return EXIT_ANOMALY if anomalies else EXIT_OK


def _finish(results_dir: Path, csv_path: Path, verbose: bool) -> None:
    """Regenerate the summary table from the raw records and log the paths."""
    summary_path = default_summary_path(results_dir)
    write_summary(default_raw_dir(results_dir), summary_path)
    if verbose:
        logger.info("raw records : %s", csv_path)
        logger.info("summary     : %s", summary_path)


def _report(records: Sequence[ResultRecord], anomalies: Sequence[str]) -> None:
    """Log the per-set / per-size-group RPD report for the instances run."""
    if not records:
        return

    by_group: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    all_rpds: list[float] = []
    missing_reference = 0
    for record in records:
        if record.rpd is None:
            missing_reference += 1
            continue
        by_group[(record.instance_set or "", record.n, record.m)].append(record.rpd)
        all_rpds.append(record.rpd)

    logger.info("--- Per-set / size-group RPD (vs. reference), this pass ---")
    for set_label, n, m in sorted(by_group):
        group = by_group[(set_label, n, m)]
        mean = sum(group) / len(group)
        label = f"{set_label} " if set_label else ""
        logger.info(
            "  %s%dx%d: count=%d mean=%.3f min=%.3f max=%.3f",
            label,
            n,
            m,
            len(group),
            mean,
            min(group),
            max(group),
        )

    if all_rpds:
        logger.info(
            "Overall: %d instance(s), mean RPD=%.4f%% (min=%.4f max=%.4f)",
            len(records),
            sum(all_rpds) / len(all_rpds),
            min(all_rpds),
            max(all_rpds),
        )
    if missing_reference:
        logger.info("Instances without resolvable reference: %d", missing_reference)

    if anomalies:
        logger.warning("ANOMALIES: %d", len(anomalies))
        for anomaly in anomalies:
            logger.warning("  - %s", anomaly)
    else:
        logger.info("Sanity OK: no anomaly for any instance run.")


__all__ = [
    "Solver",
    "StochasticSolver",
    "SanityCheck",
    "EXIT_OK",
    "EXIT_ANOMALY",
    "EXIT_NO_INSTANCES",
    "EXIT_INTERRUPTED",
    "configure_logging",
    "discover_instances",
    "instance_id_for",
    "lower_bound_sanity",
    "best_known_sanity",
    "run_campaign",
]
