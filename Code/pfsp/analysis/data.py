"""Load derived analysis data from the persisted results (standard library only).

These helpers read the raw per-execution records (``results/raw/<algorithm>.csv``)
and the derived summary (``results/summary/summary.csv``) into plain Python
structures ready for plotting. They never render anything and have no matplotlib
dependency, so they are fast and fully unit-testable.

Conventions of the result schema are defined in :mod:`pfsp.results.record` and
:mod:`pfsp.results.aggregate`. A row whose ``rpd`` cell is empty (no reference
resolved) is skipped for RPD series, never fabricated.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SizeGroupRPD:
    """Mean RPD of one ``(algorithm, instance_set, n, m)`` group from the summary."""

    algorithm: str
    instance_set: str | None
    n: int
    m: int
    n_instances: int
    rpd_mean: float | None


@dataclass(frozen=True)
class CalibrationSweep:
    """Aggregated EXP-003 IG budget sweep, ready for plotting and the knee analysis.

    Built by :func:`load_calibration_sweep` from
    ``results/calibration/EXP-003_budget_sweep.csv``. Every value is a mean over
    the completed replications (seeds); rows without a resolvable RPD are skipped
    for the RPD aggregates but still counted for runtime.

    Attributes
    ----------
    budgets:
        Sorted list of the evaluation budgets swept.
    instances:
        Sorted list of ``Instance_Identifier`` present in the sweep.
    rpd_by_instance:
        ``instance_id -> {budget -> mean RPD %}`` (mean over seeds).
    rpd_overall:
        ``budget -> mean RPD %`` over **all** rows at that budget.
    rpd_by_benchmark:
        ``benchmark -> {budget -> mean RPD %}`` (``"taillard"`` / ``"vrf"``);
        kept separate because their references differ (UB vs best-known).
    runtime_overall:
        ``budget -> mean runtime (s)`` over all rows at that budget.
    n_seeds:
        Number of distinct seeds found (for reporting).
    """

    budgets: list[int]
    instances: list[str]
    rpd_by_instance: dict[str, dict[int, float]]
    rpd_overall: dict[int, float]
    rpd_by_benchmark: dict[str, dict[int, float]]
    runtime_overall: dict[int, float]
    n_seeds: int

    def marginal_improvements(self) -> list[tuple[int, float, float]]:
        """Return ``(budget, mean_rpd, delta)`` rows ordered by budget.

        ``delta`` is the drop in overall mean RPD from the previous budget level
        (``prev_mean - mean``); it is ``nan`` for the first budget. A small delta
        means extra budget barely helps — the basis of the knee selection.
        """
        rows: list[tuple[int, float, float]] = []
        prev: float | None = None
        for budget in self.budgets:
            if budget not in self.rpd_overall:
                continue
            mean = self.rpd_overall[budget]
            delta = float("nan") if prev is None else prev - mean
            rows.append((budget, mean, delta))
            prev = mean
        return rows

    def knee_budget(self, threshold: float = 0.1) -> int | None:
        """Return the smallest budget whose marginal gain drops below ``threshold``.

        Implements the EXP-003 selection rule: pick the budget where the gain in
        overall mean RPD versus the previous level first falls below ``threshold``
        percentage points (~0.1%% by default). Returns ``None`` when no level
        qualifies (improvement never stabilizes within the swept range).
        """
        for budget, _mean, delta in self.marginal_improvements():
            if delta == delta and delta < threshold:  # delta==delta excludes nan
                return budget
        return None


@dataclass(frozen=True)
class SpeedupComparison:
    """EXP-005 comparison of the naive vs accelerated budget sweeps.

    Built by :func:`load_speedup_comparison` from the EXP-003 (naive) and EXP-005
    (accelerated) sweep CSVs. It pairs the two runs on every common
    ``(instance_id, seed, budget)`` triple to (a) verify the accelerated results
    are identical (a pure speedup) and (b) quantify the speedup, aggregated over
    seeds per ``(instance, budget)``.

    Attributes
    ----------
    budgets:
        Sorted budgets common to both sweeps.
    instances:
        Sorted instance identifiers common to both sweeps.
    n_by_instance / m_by_instance:
        Job/machine counts per instance (for the size-scaling plots).
    runtime_naive / runtime_accel:
        ``(instance_id, budget) -> mean runtime (s)`` over seeds, for each sweep.
    makespan_mismatches:
        List of ``(instance_id, seed, budget, makespan_naive, makespan_accel)``
        where the two implementations disagreed. **Expected to be empty**: a
        non-empty list means the acceleration changed results (a bug).
    n_pairs_compared:
        Number of ``(instance, seed, budget)`` triples compared for equivalence.
    """

    budgets: list[int]
    instances: list[str]
    n_by_instance: dict[str, int]
    m_by_instance: dict[str, int]
    runtime_naive: dict[tuple[str, int], float]
    runtime_accel: dict[tuple[str, int], float]
    makespan_mismatches: list[tuple[str, int, int, int, int]]
    n_pairs_compared: int

    @property
    def equivalent(self) -> bool:
        """True when every compared triple had an identical makespan."""
        return not self.makespan_mismatches

    def speedup(self, instance_id: str, budget: int) -> float | None:
        """Return ``runtime_naive / runtime_accel`` for a cell, or ``None``."""
        key = (instance_id, budget)
        naive = self.runtime_naive.get(key)
        accel = self.runtime_accel.get(key)
        if naive is None or accel is None or accel == 0:
            return None
        return naive / accel


def _benchmark_of(instance_id: str) -> str:
    """Return the benchmark prefix of an ``Instance_Identifier`` (before ``/``)."""
    return instance_id.split("/", 1)[0]


def load_rpd_series(
    raw_csv: str | Path, algorithm: str = "NEH"
) -> dict[str, list[float]]:
    """Return per-instance RPD lists keyed by a distribution label.

    Reads ``raw_csv`` and groups the (resolvable) RPD values of ``algorithm`` into
    labels suitable for a distribution plot: ``"Taillard"`` for Taillard rows and
    ``"VRF Small"`` / ``"VRF Large"`` (falling back to ``"VRF"`` when the set is
    unset) for VRF rows. Rows without a resolvable RPD are skipped.

    Returns
    -------
    dict[str, list[float]]
        Mapping label -> list of RPD percentages.
    """
    series: dict[str, list[float]] = defaultdict(list)
    path = Path(raw_csv)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("algorithm") != algorithm:
                continue
            rpd_cell = (row.get("rpd") or "").strip()
            if not rpd_cell:
                continue
            benchmark = _benchmark_of(row.get("instance_id", ""))
            if benchmark == "taillard":
                label = "Taillard"
            elif benchmark == "vrf":
                instance_set = (row.get("instance_set") or "").strip()
                label = f"VRF {instance_set}" if instance_set else "VRF"
            else:
                label = benchmark
            series[label].append(float(rpd_cell))
    return dict(series)


def load_rpd_by_size_group(
    summary_csv: str | Path,
    algorithm: str = "NEH",
    instance_set: str | None = None,
) -> list[SizeGroupRPD]:
    """Return per-size-group mean RPD rows from the summary, filtered and sorted.

    Reads ``summary_csv`` and returns the :class:`SizeGroupRPD` rows for
    ``algorithm``; when ``instance_set`` is given (``""`` matches Taillard's empty
    set), only that subset is returned. Rows are sorted by ``(n, m)``.
    """
    groups: list[SizeGroupRPD] = []
    path = Path(summary_csv)
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("algorithm") != algorithm:
                continue
            row_set = (row.get("instance_set") or "").strip()
            if instance_set is not None and row_set != instance_set:
                continue
            rpd_cell = (row.get("rpd_mean") or "").strip()
            groups.append(
                SizeGroupRPD(
                    algorithm=row["algorithm"],
                    instance_set=row_set or None,
                    n=int(row["n"]),
                    m=int(row["m"]),
                    n_instances=int(row["n_instances"]),
                    rpd_mean=float(rpd_cell) if rpd_cell else None,
                )
            )
    groups.sort(key=lambda g: (g.n, g.m))
    return groups


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def load_calibration_sweep(csv_path: str | Path) -> CalibrationSweep:
    """Load and aggregate the EXP-003 budget sweep CSV into a :class:`CalibrationSweep`.

    Reads ``results/calibration/EXP-003_budget_sweep.csv`` (schema written by
    ``scripts/calibrate_ig_budget.py``: ``instance_id, benchmark, n, m, seed,
    budget, makespan, neh_makespan, reference, rpd, runtime_s``) and aggregates
    the per-``(instance, budget)`` RPD over seeds, the overall and per-benchmark
    mean RPD per budget, and the mean runtime per budget. Rows without a
    resolvable ``rpd`` are skipped for the RPD aggregates (never fabricated) but
    still contribute to the runtime aggregate.

    Returns an empty sweep (all fields empty, ``n_seeds=0``) when the file does
    not exist.
    """
    path = Path(csv_path)
    empty = CalibrationSweep([], [], {}, {}, {}, {}, 0)
    if not path.exists():
        return empty

    # Accumulators.
    rpd_pairs: dict[tuple[str, int], list[float]] = defaultdict(list)
    rpd_overall_acc: dict[int, list[float]] = defaultdict(list)
    rpd_bench_acc: dict[str, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    runtime_acc: dict[int, list[float]] = defaultdict(list)
    budgets: set[int] = set()
    instances: set[str] = set()
    seeds: set[int] = set()

    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                budget = int(row["budget"])
            except (KeyError, ValueError):
                continue
            instance_id = (row.get("instance_id") or "").strip()
            benchmark = (row.get("benchmark") or "").strip() or _benchmark_of(
                instance_id
            )
            budgets.add(budget)
            if instance_id:
                instances.add(instance_id)
            seed_cell = (row.get("seed") or "").strip()
            if seed_cell:
                try:
                    seeds.add(int(seed_cell))
                except ValueError:
                    pass

            runtime_cell = (row.get("runtime_s") or "").strip()
            if runtime_cell:
                runtime_acc[budget].append(float(runtime_cell))

            rpd_cell = (row.get("rpd") or "").strip()
            if not rpd_cell:
                continue
            rpd = float(rpd_cell)
            rpd_pairs[(instance_id, budget)].append(rpd)
            rpd_overall_acc[budget].append(rpd)
            rpd_bench_acc[benchmark][budget].append(rpd)

    ordered_budgets = sorted(budgets)
    rpd_by_instance: dict[str, dict[int, float]] = defaultdict(dict)
    for (instance_id, budget), values in rpd_pairs.items():
        rpd_by_instance[instance_id][budget] = _mean(values)

    return CalibrationSweep(
        budgets=ordered_budgets,
        instances=sorted(instances),
        rpd_by_instance=dict(rpd_by_instance),
        rpd_overall={b: _mean(v) for b, v in rpd_overall_acc.items()},
        rpd_by_benchmark={
            bench: {b: _mean(v) for b, v in per_b.items()}
            for bench, per_b in rpd_bench_acc.items()
        },
        runtime_overall={b: _mean(v) for b, v in runtime_acc.items()},
        n_seeds=len(seeds),
    )


def _read_sweep_rows(
    path: Path,
) -> dict[tuple[str, int, int], tuple[int, float, int, int]]:
    """Read a sweep CSV into ``(instance_id, seed, budget) -> (makespan, rt, n, m)``.

    Returns an empty mapping when the file does not exist. Rows with a missing or
    malformed seed/budget/makespan are skipped (they cannot be paired reliably).
    """
    rows: dict[tuple[str, int, int], tuple[int, float, int, int]] = {}
    if not path.exists():
        return rows
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            instance_id = (row.get("instance_id") or "").strip()
            try:
                seed = int((row.get("seed") or "").strip())
                budget = int((row.get("budget") or "").strip())
                makespan_value = int((row.get("makespan") or "").strip())
                n = int((row.get("n") or "").strip())
                m = int((row.get("m") or "").strip())
            except ValueError:
                continue
            runtime_cell = (row.get("runtime_s") or "").strip()
            runtime = float(runtime_cell) if runtime_cell else float("nan")
            rows[(instance_id, seed, budget)] = (makespan_value, runtime, n, m)
    return rows


def load_speedup_comparison(
    naive_csv: str | Path, accel_csv: str | Path
) -> SpeedupComparison:
    """Pair the naive (EXP-003) and accelerated (EXP-005) sweeps into a comparison.

    Reads both budget-sweep CSVs, pairs them on every common
    ``(instance_id, seed, budget)`` triple, records makespan disagreements
    (expected: none) and aggregates the mean runtime over seeds per
    ``(instance, budget)`` for each implementation. Instances/budgets are the
    intersection of the two files. Returns an empty comparison when either file
    is missing.
    """
    naive_rows = _read_sweep_rows(Path(naive_csv))
    accel_rows = _read_sweep_rows(Path(accel_csv))
    empty = SpeedupComparison([], [], {}, {}, {}, {}, [], 0)
    if not naive_rows or not accel_rows:
        return empty

    common_keys = naive_rows.keys() & accel_rows.keys()

    mismatches: list[tuple[str, int, int, int, int]] = []
    rt_naive_acc: dict[tuple[str, int], list[float]] = defaultdict(list)
    rt_accel_acc: dict[tuple[str, int], list[float]] = defaultdict(list)
    n_by_instance: dict[str, int] = {}
    m_by_instance: dict[str, int] = {}
    budgets: set[int] = set()
    instances: set[str] = set()

    for key in common_keys:
        instance_id, _seed, budget = key
        ms_naive, rt_naive, n, m = naive_rows[key]
        ms_accel, rt_accel, _n2, _m2 = accel_rows[key]
        if ms_naive != ms_accel:
            mismatches.append((instance_id, _seed, budget, ms_naive, ms_accel))
        instances.add(instance_id)
        budgets.add(budget)
        n_by_instance[instance_id] = n
        m_by_instance[instance_id] = m
        cell = (instance_id, budget)
        if rt_naive == rt_naive:  # not NaN
            rt_naive_acc[cell].append(rt_naive)
        if rt_accel == rt_accel:
            rt_accel_acc[cell].append(rt_accel)

    return SpeedupComparison(
        budgets=sorted(budgets),
        instances=sorted(instances),
        n_by_instance=n_by_instance,
        m_by_instance=m_by_instance,
        runtime_naive={k: _mean(v) for k, v in rt_naive_acc.items()},
        runtime_accel={k: _mean(v) for k, v in rt_accel_acc.items()},
        makespan_mismatches=sorted(mismatches),
        n_pairs_compared=len(common_keys),
    )


__all__ = [
    "SizeGroupRPD",
    "CalibrationSweep",
    "SpeedupComparison",
    "load_rpd_series",
    "load_rpd_by_size_group",
    "load_calibration_sweep",
    "load_speedup_comparison",
]
