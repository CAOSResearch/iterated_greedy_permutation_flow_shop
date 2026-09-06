"""Figures for the thesis results (matplotlib, dev-only dependency).

Rendering layer on top of :mod:`pfsp.analysis.data`. Uses the non-interactive
``Agg`` backend so it runs headless (CI, scripts) and writes vector PDF by
default (the thesis is LaTeX/IEEE). Colormaps are perceptually uniform
(``viridis``) so figures stay readable in grayscale and for colour-vision
deficiencies.

These functions only *consume* already-loaded data structures and a destination
path; they never read the result files themselves (that is the data layer's job)
nor mutate any global state beyond creating the output file.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend; must precede pyplot import.

import matplotlib.pyplot as plt  # noqa: E402  (backend selected above)

from pfsp.analysis.data import (  # noqa: E402
    CalibrationSweep,
    SizeGroupRPD,
    SpeedupComparison,
)

#: Preferred order of distribution labels when present.
_LABEL_ORDER = ["Taillard", "VRF Small", "VRF Large", "VRF"]


def _ordered_labels(labels: list[str]) -> list[str]:
    """Order labels by the preferred sequence, appending any extras at the end."""
    known = [label for label in _LABEL_ORDER if label in labels]
    extra = sorted(label for label in labels if label not in _LABEL_ORDER)
    return known + extra


def plot_rpd_distribution(
    series: dict[str, list[float]],
    out_path: str | Path,
    *,
    title: str = "NEH baseline: RPD distribution by benchmark",
) -> Path:
    """Box plot of the per-instance RPD distribution per benchmark/subset.

    Makes the "Taillard ceiling effect vs. VRF hardness" argument visible: each
    box summarizes the spread of NEH's RPD on that benchmark. ``series`` maps a
    label (e.g. ``"VRF Small"``) to its list of RPD percentages.

    Returns the written file path. Raises ``ValueError`` if ``series`` is empty.
    """
    if not series:
        raise ValueError("No RPD series to plot (empty input).")

    labels = _ordered_labels(list(series))
    data = [series[label] for label in labels]

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.boxplot(data, tick_labels=labels, showmeans=True)
    ax.set_ylabel("RPD vs. best-known / UB (%)")
    ax.set_xlabel("Benchmark")
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_rpd_heatmap(
    groups: list[SizeGroupRPD],
    out_path: str | Path,
    *,
    title: str = "NEH baseline: mean RPD by instance size (n x m)",
) -> Path:
    """Heat map of mean RPD over the instance-size grid ``n`` (rows) x ``m`` (cols).

    Shows the size pattern of NEH's RPD (rises with the number of machines ``m``,
    falls with the number of jobs ``n``). ``groups`` are the per-size-group rows
    from :func:`pfsp.analysis.data.load_rpd_by_size_group`. Cells without a value
    are left blank. Returns the written file path. Raises ``ValueError`` if empty.
    """
    if not groups:
        raise ValueError("No size groups to plot (empty input).")

    ns = sorted({g.n for g in groups})
    ms = sorted({g.m for g in groups})
    lookup = {(g.n, g.m): g.rpd_mean for g in groups}

    grid = [[lookup.get((n, m)) for m in ms] for n in ns]

    fig, ax = plt.subplots(figsize=(1.2 + 0.8 * len(ms), 1.2 + 0.5 * len(ns)))
    # Mask missing cells so they render blank rather than as 0.
    masked = [
        [float("nan") if value is None else value for value in row] for row in grid
    ]
    image = ax.imshow(masked, aspect="auto", cmap="viridis")

    ax.set_xticks(range(len(ms)), labels=[str(m) for m in ms])
    ax.set_yticks(range(len(ns)), labels=[str(n) for n in ns])
    ax.set_xlabel("machines (m)")
    ax.set_ylabel("jobs (n)")
    ax.set_title(title)

    # Annotate each cell with its mean RPD for precise reading.
    for i, n in enumerate(ns):
        for j, m in enumerate(ms):
            value = lookup.get((n, m))
            if value is not None:
                ax.text(
                    j,
                    i,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value < (max_value(masked) * 0.6) else "black",
                )
    fig.colorbar(image, ax=ax, label="mean RPD (%)")
    fig.tight_layout()

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def max_value(masked: list[list[float]]) -> float:
    """Return the maximum non-NaN value of a masked grid (0.0 if all NaN)."""
    values = [v for row in masked for v in row if v == v]  # v==v excludes NaN
    return max(values) if values else 0.0


def plot_calibration_curve(
    sweep: CalibrationSweep,
    out_path: str | Path,
    *,
    knee: int | None = None,
    title: str = "EXP-003: IG budget calibration — RPD vs budget",
) -> Path:
    """Line plot of mean RPD vs budget: one faint line per instance + the mean.

    Makes the calibration argument visible: each instance line shows how its RPD
    decreases as the budget grows, and the bold black ``MEAN`` line is the
    overall mean used for the knee selection. ``knee`` (if given) is marked with
    a vertical dashed line — the chosen budget. The x-axis is logarithmic because
    budgets span two orders of magnitude.

    Returns the written file path. Raises ``ValueError`` if ``sweep`` is empty.
    """
    if not sweep.budgets or not sweep.rpd_overall:
        raise ValueError("No calibration data to plot (empty sweep).")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    # Faint per-instance lines (context, not the focus).
    for instance_id in sweep.instances:
        per_budget = sweep.rpd_by_instance.get(instance_id, {})
        xs = [b for b in sweep.budgets if b in per_budget]
        ys = [per_budget[b] for b in xs]
        if xs:
            ax.plot(xs, ys, marker="o", markersize=3, alpha=0.35, linewidth=0.8)

    # Bold overall mean line.
    mx = [b for b in sweep.budgets if b in sweep.rpd_overall]
    my = [sweep.rpd_overall[b] for b in mx]
    ax.plot(mx, my, marker="s", color="black", linewidth=2.5, label="Mean (all)")

    if knee is not None:
        ax.axvline(knee, color="crimson", linestyle="--", linewidth=1.5)
        ax.annotate(
            f"chosen budget = {knee}",
            xy=(knee, max(my)),
            xytext=(6, 0),
            textcoords="offset points",
            color="crimson",
            fontsize=8,
            va="top",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Budget (IG iterations, log scale)")
    ax.set_ylabel("Mean RPD vs. reference (%)")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_calibration_by_benchmark(
    sweep: CalibrationSweep,
    out_path: str | Path,
    *,
    knee: int | None = None,
    title: str = "EXP-003: mean RPD vs budget by benchmark",
) -> Path:
    """Line plot of mean RPD vs budget split by benchmark (Taillard vs VRF).

    Taillard's RPD is vs the upper bound and VRF's vs the best-known, so their
    levels are not directly comparable; plotting them as separate lines keeps the
    comparison honest while still showing both converge. Returns the written path.
    Raises ``ValueError`` if ``sweep`` has no benchmark data.
    """
    if not sweep.rpd_by_benchmark:
        raise ValueError("No per-benchmark calibration data to plot.")

    labels = {"taillard": "Taillard (vs UB)", "vrf": "VRF (vs best-known)"}
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for benchmark in sorted(sweep.rpd_by_benchmark):
        per_budget = sweep.rpd_by_benchmark[benchmark]
        xs = [b for b in sweep.budgets if b in per_budget]
        ys = [per_budget[b] for b in xs]
        if xs:
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=2.0,
                label=labels.get(benchmark, benchmark),
            )

    if knee is not None:
        ax.axvline(
            knee,
            color="crimson",
            linestyle="--",
            linewidth=1.5,
            label=f"chosen = {knee}",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Budget (IG iterations, log scale)")
    ax.set_ylabel("Mean RPD vs. reference (%)")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_calibration_runtime(
    sweep: CalibrationSweep,
    out_path: str | Path,
    *,
    title: str = "EXP-003: mean runtime per run vs budget",
) -> Path:
    """Log-log plot of mean runtime per run vs budget (the cost side of the trade-off).

    Pairs with the RPD curve to expose the diminishing returns: runtime grows
    ~linearly with the budget while RPD flattens. Returns the written path.
    Raises ``ValueError`` if ``sweep`` has no runtime data.
    """
    if not sweep.runtime_overall:
        raise ValueError("No runtime data to plot (empty sweep).")

    xs = [b for b in sweep.budgets if b in sweep.runtime_overall]
    ys = [sweep.runtime_overall[b] for b in xs]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(xs, ys, marker="o", color="crimson", linewidth=2.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Budget (IG iterations, log scale)")
    ax.set_ylabel("Mean runtime per run (s, log scale)")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    fig.tight_layout()

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_calibration_heatmap(
    sweep: CalibrationSweep,
    out_path: str | Path,
    *,
    title: str = "EXP-003: mean RPD by instance x budget",
) -> Path:
    """Heat map of mean RPD over the ``instance`` (rows) x ``budget`` (cols) grid.

    Complements the line plot: shows at a glance which instances drive the
    remaining RPD at high budgets (the hard ``*x20`` ones) and which saturate
    immediately. Instances are ordered by their RPD at the smallest budget
    (hardest on top). Returns the written path. Raises ``ValueError`` if empty.
    """
    if not sweep.instances or not sweep.budgets:
        raise ValueError("No calibration data to plot (empty sweep).")

    first_budget = sweep.budgets[0]

    def _hardness(instance_id: str) -> float:
        return sweep.rpd_by_instance.get(instance_id, {}).get(first_budget, 0.0)

    instances = sorted(sweep.instances, key=_hardness, reverse=True)
    budgets = sweep.budgets

    grid = [
        [sweep.rpd_by_instance.get(inst, {}).get(b, float("nan")) for b in budgets]
        for inst in instances
    ]

    fig, ax = plt.subplots(
        figsize=(1.5 + 0.7 * len(budgets), 1.5 + 0.4 * len(instances))
    )
    image = ax.imshow(grid, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(budgets)), labels=[str(b) for b in budgets])
    ax.set_yticks(
        range(len(instances)),
        labels=[i.split("/", 1)[-1] for i in instances],
    )
    ax.set_xlabel("Budget (IG iterations)")
    ax.set_ylabel("Instance")
    ax.set_title(title)

    vmax = max_value(grid)
    for i in range(len(instances)):
        for j in range(len(budgets)):
            value = grid[i][j]
            if value == value:  # not NaN
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if value < vmax * 0.6 else "black",
                )
    fig.colorbar(image, ax=ax, label="mean RPD (%)")
    fig.tight_layout()

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_speedup_vs_size(
    comparison: SpeedupComparison,
    out_path: str | Path,
    *,
    budget: int,
    title: str | None = None,
) -> Path:
    """Scatter of the naive/accelerated speedup factor vs. instance job count ``n``.

    Each point is one instance at the given ``budget`` (runtime averaged over
    seeds); the y-value is ``runtime_naive / runtime_accel``. Points are labelled
    by ``n`` and coloured by machine count ``m`` to show that the speedup grows
    with instance size — the empirical justification for the acceleration.
    Returns the written path. Raises ``ValueError`` if there is nothing to plot.
    """
    xs: list[int] = []
    ys: list[float] = []
    ms: list[int] = []
    for instance_id in comparison.instances:
        speedup = comparison.speedup(instance_id, budget)
        if speedup is None:
            continue
        xs.append(comparison.n_by_instance[instance_id])
        ys.append(speedup)
        ms.append(comparison.m_by_instance[instance_id])
    if not xs:
        raise ValueError(f"No speedup data to plot at budget={budget}.")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    scatter = ax.scatter(xs, ys, c=ms, cmap="viridis", s=60, edgecolor="black")
    ax.set_xlabel("Jobs (n)")
    ax.set_ylabel("Speedup  (runtime naive / accelerated)")
    ax.set_title(title or f"Speedup vs. instance size (budget = {budget})")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.axhline(1.0, color="grey", linewidth=0.8)
    fig.colorbar(scatter, ax=ax, label="Machines (m)")
    fig.tight_layout()

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_runtime_scaling(
    comparison: SpeedupComparison,
    out_path: str | Path,
    *,
    budget: int,
    title: str | None = None,
) -> Path:
    """Log-log runtime vs. ``n`` for both implementations at a fixed ``budget``.

    Plots two series (naive and accelerated) of mean runtime per run against the
    job count ``n``, on log-log axes so the different scaling slopes
    (steeper for the naive ``O(n^3 m)`` local-search pass vs the accelerated
    ``O(n^2 m)``) are visible as diverging lines. Returns the written path.
    Raises ``ValueError`` if there is nothing to plot.
    """
    points: list[tuple[int, float, float]] = []
    for instance_id in comparison.instances:
        key = (instance_id, budget)
        naive = comparison.runtime_naive.get(key)
        accel = comparison.runtime_accel.get(key)
        if naive is None or accel is None:
            continue
        points.append((comparison.n_by_instance[instance_id], naive, accel))
    if not points:
        raise ValueError(f"No runtime data to plot at budget={budget}.")

    points.sort()
    xs = [n for n, _, _ in points]
    naive_ys = [rn for _, rn, _ in points]
    accel_ys = [ra for _, _, ra in points]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(xs, naive_ys, marker="o", label="Naive (recompute)", color="crimson")
    ax.plot(xs, accel_ys, marker="s", label="Accelerated (Taillard)", color="steelblue")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Jobs (n, log scale)")
    ax.set_ylabel("Mean runtime per run (s, log scale)")
    ax.set_title(title or f"Runtime scaling: naive vs. accelerated (budget = {budget})")
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


__all__ = [
    "plot_rpd_distribution",
    "plot_rpd_heatmap",
    "plot_calibration_curve",
    "plot_calibration_by_benchmark",
    "plot_calibration_runtime",
    "plot_calibration_heatmap",
    "plot_speedup_vs_size",
    "plot_runtime_scaling",
]
