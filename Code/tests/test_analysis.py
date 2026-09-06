"""Tests for the analysis layer (``pfsp.analysis``).

The data loaders (standard library) are tested directly; the matplotlib plotting
functions get a headless smoke test that is skipped when matplotlib is absent
(it is a dev-only dependency), so the suite still passes on a clean clone.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from pfsp.analysis.data import (
    SizeGroupRPD,
    load_rpd_by_size_group,
    load_rpd_series,
)

_RAW_HEADER = [
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


def _write_raw(path: Path, rows: list[list[str]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(_RAW_HEADER)
        writer.writerows(rows)


def _write_summary(path: Path, rows: list[list[str]]) -> None:
    header = [
        "algorithm",
        "instance_set",
        "n",
        "m",
        "n_instances",
        "rpd_mean",
        "rpd_std",
        "rpd_min",
        "rpd_max",
        "makespan_mean",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def test_load_rpd_series_groups_and_skips_unresolved(tmp_path: Path) -> None:
    """RPD series are labelled by benchmark/subset; empty-RPD rows are skipped."""
    raw = tmp_path / "NEH.csv"
    _write_raw(
        raw,
        [
            [
                "taillard/tai20_5_0",
                "",
                "20",
                "5",
                "NEH",
                "",
                "1278",
                "0.0",
                "0.63",
                "r",
                "t",
            ],
            [
                "vrf/VFR10_5_1_Gap",
                "Small",
                "10",
                "5",
                "NEH",
                "",
                "712",
                "0.0",
                "2.45",
                "r",
                "t",
            ],
            [
                "vrf/VFR100_20_1_Gap",
                "Large",
                "100",
                "20",
                "NEH",
                "",
                "5000",
                "0.0",
                "3.10",
                "r",
                "t",
            ],
            # Unresolved RPD (empty) must be skipped, not counted as 0.
            [
                "vrf/VFR10_10_1_Gap",
                "Small",
                "10",
                "10",
                "NEH",
                "",
                "1149",
                "0.0",
                "",
                "r",
                "t",
            ],
            # A different algorithm must be ignored under the NEH filter.
            [
                "vrf/VFR10_5_2_Gap",
                "Small",
                "10",
                "5",
                "IG",
                "1",
                "700",
                "0.0",
                "1.0",
                "r",
                "t",
            ],
        ],
    )

    series = load_rpd_series(raw)

    assert series == {
        "Taillard": [0.63],
        "VRF Small": [2.45],
        "VRF Large": [3.10],
    }


def test_load_rpd_series_missing_file_returns_empty(tmp_path: Path) -> None:
    """A missing raw file yields an empty mapping (no error)."""
    assert load_rpd_series(tmp_path / "absent.csv") == {}


def test_load_rpd_by_size_group_filters_sorts_and_parses(tmp_path: Path) -> None:
    """Size groups are filtered by set, sorted by (n, m); empty mean -> None."""
    summary = tmp_path / "summary.csv"
    _write_summary(
        summary,
        [
            ["NEH", "Small", "20", "5", "10", "3.25", "1.0", "0.4", "7.2", "1261.1"],
            ["NEH", "Small", "10", "5", "10", "2.18", "0.8", "0.6", "3.4", "712.4"],
            ["NEH", "Large", "100", "20", "10", "4.28", "0.5", "3.3", "5.0", "5000.0"],
            ["NEH", "Small", "10", "10", "10", "", "", "", "", "1149.0"],  # no mean
            ["IG", "Small", "10", "5", "10", "1.0", "0.1", "0.5", "1.5", "700.0"],
        ],
    )

    small = load_rpd_by_size_group(summary, instance_set="Small")

    # Only NEH/Small rows, sorted by (n, m): (10,5), (10,10), (20,5).
    assert [(g.n, g.m) for g in small] == [(10, 5), (10, 10), (20, 5)]
    assert small[0].rpd_mean == pytest.approx(2.18)
    assert small[1].rpd_mean is None  # empty mean is not fabricated
    assert all(g.algorithm == "NEH" and g.instance_set == "Small" for g in small)


def test_plots_smoke(tmp_path: Path) -> None:
    """Headless smoke test: the figures render to non-empty files."""
    pytest.importorskip("matplotlib")
    from pfsp.analysis.plots import plot_rpd_distribution, plot_rpd_heatmap

    series = {"Taillard": [1.0, 2.0, 3.0], "VRF Small": [3.0, 4.0, 5.0]}
    dist_path = plot_rpd_distribution(series, tmp_path / "dist.png")
    assert dist_path.exists() and dist_path.stat().st_size > 0

    groups = [
        SizeGroupRPD("NEH", "Small", 10, 5, 10, 1.3),
        SizeGroupRPD("NEH", "Small", 10, 10, 10, 4.2),
        SizeGroupRPD("NEH", "Small", 20, 5, 10, 1.5),
    ]
    heatmap_path = plot_rpd_heatmap(groups, tmp_path / "heatmap.png")
    assert heatmap_path.exists() and heatmap_path.stat().st_size > 0


def test_plots_reject_empty_input(tmp_path: Path) -> None:
    """Empty inputs raise a descriptive error rather than writing a blank figure."""
    pytest.importorskip("matplotlib")
    from pfsp.analysis.plots import plot_rpd_distribution, plot_rpd_heatmap

    with pytest.raises(ValueError):
        plot_rpd_distribution({}, tmp_path / "x.png")
    with pytest.raises(ValueError):
        plot_rpd_heatmap([], tmp_path / "y.png")


# --------------------------------------------------------------------------- #
# EXP-003 calibration sweep loader
# --------------------------------------------------------------------------- #

_CALIB_HEADER = [
    "instance_id",
    "benchmark",
    "n",
    "m",
    "seed",
    "budget",
    "makespan",
    "neh_makespan",
    "reference",
    "rpd",
    "runtime_s",
]


def _write_calibration(path: Path, rows: list[list[str]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(_CALIB_HEADER)
        writer.writerows(rows)


def _calib_row(
    instance_id: str,
    benchmark: str,
    n: int,
    m: int,
    seed: int,
    budget: int,
    rpd: str,
    runtime: str = "1.0",
) -> list[str]:
    return [
        instance_id,
        benchmark,
        str(n),
        str(m),
        str(seed),
        str(budget),
        "1000",
        "1100",
        "1000",
        rpd,
        runtime,
    ]


def test_load_calibration_missing_file_returns_empty(tmp_path: Path) -> None:
    """A missing sweep file yields an empty sweep (no error)."""
    from pfsp.analysis.data import load_calibration_sweep

    sweep = load_calibration_sweep(tmp_path / "absent.csv")
    assert sweep.budgets == []
    assert sweep.rpd_overall == {}
    assert sweep.n_seeds == 0


def test_load_calibration_aggregates_over_seeds(tmp_path: Path) -> None:
    """RPD per (instance, budget) is the mean over seeds; budgets sorted."""
    from pfsp.analysis.data import load_calibration_sweep

    csv_path = tmp_path / "sweep.csv"
    _write_calibration(
        csv_path,
        [
            _calib_row("taillard/tai20_5_0", "taillard", 20, 5, 42, 50, "1.0"),
            _calib_row("taillard/tai20_5_0", "taillard", 20, 5, 99, 50, "3.0"),
            _calib_row("taillard/tai20_5_0", "taillard", 20, 5, 42, 100, "0.0"),
            _calib_row("taillard/tai20_5_0", "taillard", 20, 5, 99, 100, "0.0"),
            _calib_row("vrf/VFR20_5_1_Gap", "vrf", 20, 5, 42, 50, "2.0"),
            _calib_row("vrf/VFR20_5_1_Gap", "vrf", 20, 5, 42, 100, "1.0"),
        ],
    )

    sweep = load_calibration_sweep(csv_path)

    assert sweep.budgets == [50, 100]
    assert sweep.instances == ["taillard/tai20_5_0", "vrf/VFR20_5_1_Gap"]
    assert sweep.n_seeds == 2
    # Mean over seeds for the Taillard instance at budget 50: (1+3)/2 = 2.0.
    assert sweep.rpd_by_instance["taillard/tai20_5_0"][50] == pytest.approx(2.0)
    assert sweep.rpd_by_instance["taillard/tai20_5_0"][100] == pytest.approx(0.0)
    # Per-benchmark means at budget 50: taillard {1,3}->2.0, vrf {2}->2.0.
    assert sweep.rpd_by_benchmark["taillard"][50] == pytest.approx(2.0)
    assert sweep.rpd_by_benchmark["vrf"][50] == pytest.approx(2.0)
    # Overall at budget 50: mean of [1,3,2] = 2.0; at 100: mean of [0,0,1] = 1/3.
    assert sweep.rpd_overall[50] == pytest.approx(2.0)
    assert sweep.rpd_overall[100] == pytest.approx(1.0 / 3.0)


def test_load_calibration_skips_unresolved_rpd_but_keeps_runtime(
    tmp_path: Path,
) -> None:
    """Empty-RPD rows are skipped for RPD aggregates but still count for runtime."""
    from pfsp.analysis.data import load_calibration_sweep

    csv_path = tmp_path / "sweep.csv"
    _write_calibration(
        csv_path,
        [
            _calib_row("vrf/VFR20_5_1_Gap", "vrf", 20, 5, 42, 50, "", runtime="5.0"),
            _calib_row("vrf/VFR20_5_1_Gap", "vrf", 20, 5, 99, 50, "2.0", runtime="7.0"),
        ],
    )

    sweep = load_calibration_sweep(csv_path)

    # Only the resolved RPD row contributes to the RPD mean.
    assert sweep.rpd_overall[50] == pytest.approx(2.0)
    # Both rows contribute to the runtime mean: (5+7)/2 = 6.0.
    assert sweep.runtime_overall[50] == pytest.approx(6.0)


def test_calibration_knee_selection(tmp_path: Path) -> None:
    """knee_budget returns the first budget whose marginal gain < threshold."""
    from pfsp.analysis.data import load_calibration_sweep

    csv_path = tmp_path / "sweep.csv"
    # One instance, one seed; RPD: 50->1.0, 100->0.5, 200->0.45, 500->0.44.
    # Deltas: 100:0.5, 200:0.05, 500:0.01. Threshold 0.1 -> knee at 200.
    _write_calibration(
        csv_path,
        [
            _calib_row("taillard/t", "taillard", 20, 5, 42, 50, "1.0"),
            _calib_row("taillard/t", "taillard", 20, 5, 42, 100, "0.5"),
            _calib_row("taillard/t", "taillard", 20, 5, 42, 200, "0.45"),
            _calib_row("taillard/t", "taillard", 20, 5, 42, 500, "0.44"),
        ],
    )

    sweep = load_calibration_sweep(csv_path)

    improvements = sweep.marginal_improvements()
    assert [b for b, _, _ in improvements] == [50, 100, 200, 500]
    assert sweep.knee_budget(threshold=0.1) == 200
    # A stricter threshold pushes the knee further right.
    assert sweep.knee_budget(threshold=0.001) is None


def test_calibration_plots_smoke(tmp_path: Path) -> None:
    """Headless smoke test for the four calibration figures."""
    pytest.importorskip("matplotlib")
    from pfsp.analysis.data import load_calibration_sweep
    from pfsp.analysis.plots import (
        plot_calibration_by_benchmark,
        plot_calibration_curve,
        plot_calibration_heatmap,
        plot_calibration_runtime,
    )

    csv_path = tmp_path / "sweep.csv"
    _write_calibration(
        csv_path,
        [
            _calib_row("taillard/t", "taillard", 20, 5, 42, 50, "1.0"),
            _calib_row("taillard/t", "taillard", 20, 5, 42, 100, "0.5"),
            _calib_row("vrf/v", "vrf", 20, 5, 42, 50, "2.0"),
            _calib_row("vrf/v", "vrf", 20, 5, 42, 100, "1.0"),
        ],
    )
    sweep = load_calibration_sweep(csv_path)
    knee = sweep.knee_budget()

    for fn, name in [
        (lambda p: plot_calibration_curve(sweep, p, knee=knee), "curve.png"),
        (lambda p: plot_calibration_by_benchmark(sweep, p, knee=knee), "bench.png"),
        (lambda p: plot_calibration_runtime(sweep, p), "rt.png"),
        (lambda p: plot_calibration_heatmap(sweep, p), "heat.png"),
    ]:
        out = fn(tmp_path / name)
        assert out.exists() and out.stat().st_size > 0


def test_calibration_plots_reject_empty(tmp_path: Path) -> None:
    """Empty sweeps raise a descriptive error rather than writing blank figures."""
    pytest.importorskip("matplotlib")
    from pfsp.analysis.data import CalibrationSweep
    from pfsp.analysis.plots import (
        plot_calibration_curve,
        plot_calibration_runtime,
    )

    empty = CalibrationSweep([], [], {}, {}, {}, {}, 0)
    with pytest.raises(ValueError):
        plot_calibration_curve(empty, tmp_path / "x.png")
    with pytest.raises(ValueError):
        plot_calibration_runtime(empty, tmp_path / "y.png")


# --------------------------------------------------------------------------- #
# EXP-005 speedup comparison loader (naive vs accelerated sweeps)
# --------------------------------------------------------------------------- #


def _write_calibration_rt(
    path: Path, rows: list[tuple[str, str, int, int, int, int, int, str]]
) -> None:
    """Write a calibration CSV from (inst, bench, n, m, seed, budget, ms, runtime)."""
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(_CALIB_HEADER)
        for inst, bench, n, m, seed, budget, ms, rt in rows:
            writer.writerow(
                [inst, bench, n, m, seed, budget, ms, ms + 100, ms, "0.0", rt]
            )


def test_load_speedup_comparison_pairs_and_speedup(tmp_path: Path) -> None:
    """Comparison pairs common triples, aggregates runtime, computes speedup."""
    from pfsp.analysis.data import load_speedup_comparison

    naive = tmp_path / "naive.csv"
    accel = tmp_path / "accel.csv"
    # One instance, 2 seeds, budget 500. Naive ~10x slower.
    _write_calibration_rt(
        naive,
        [
            ("taillard/t20", "taillard", 20, 5, 42, 500, 1278, "10.0"),
            ("taillard/t20", "taillard", 20, 5, 99, 500, 1300, "12.0"),
        ],
    )
    _write_calibration_rt(
        accel,
        [
            ("taillard/t20", "taillard", 20, 5, 42, 500, 1278, "1.0"),
            ("taillard/t20", "taillard", 20, 5, 99, 500, 1300, "1.2"),
        ],
    )

    comp = load_speedup_comparison(naive, accel)

    assert comp.equivalent  # identical makespans
    assert comp.n_pairs_compared == 2
    # Mean runtimes over seeds: naive (10+12)/2=11, accel (1+1.2)/2=1.1 -> ~10x.
    assert comp.runtime_naive[("taillard/t20", 500)] == pytest.approx(11.0)
    assert comp.runtime_accel[("taillard/t20", 500)] == pytest.approx(1.1)
    assert comp.speedup("taillard/t20", 500) == pytest.approx(10.0)
    assert comp.n_by_instance["taillard/t20"] == 20


def test_load_speedup_comparison_detects_mismatch(tmp_path: Path) -> None:
    """A differing makespan is recorded as a mismatch and breaks equivalence."""
    from pfsp.analysis.data import load_speedup_comparison

    naive = tmp_path / "naive.csv"
    accel = tmp_path / "accel.csv"
    _write_calibration_rt(
        naive, [("taillard/t", "taillard", 20, 5, 42, 500, 1278, "10.0")]
    )
    _write_calibration_rt(
        accel, [("taillard/t", "taillard", 20, 5, 42, 500, 9999, "1.0")]
    )

    comp = load_speedup_comparison(naive, accel)

    assert not comp.equivalent
    assert comp.makespan_mismatches == [("taillard/t", 42, 500, 1278, 9999)]


def test_load_speedup_comparison_missing_file(tmp_path: Path) -> None:
    """A missing sweep file yields an empty comparison (no error)."""
    from pfsp.analysis.data import load_speedup_comparison

    comp = load_speedup_comparison(tmp_path / "a.csv", tmp_path / "b.csv")
    assert comp.instances == []
    assert comp.n_pairs_compared == 0


def test_speedup_plots_smoke(tmp_path: Path) -> None:
    """Headless smoke test for the two speedup figures."""
    pytest.importorskip("matplotlib")
    from pfsp.analysis.data import load_speedup_comparison
    from pfsp.analysis.plots import plot_runtime_scaling, plot_speedup_vs_size

    naive = tmp_path / "naive.csv"
    accel = tmp_path / "accel.csv"
    _write_calibration_rt(
        naive,
        [
            ("taillard/t20", "taillard", 20, 5, 42, 500, 1278, "10.0"),
            ("taillard/t100", "taillard", 100, 10, 42, 500, 5771, "300.0"),
        ],
    )
    _write_calibration_rt(
        accel,
        [
            ("taillard/t20", "taillard", 20, 5, 42, 500, 1278, "2.0"),
            ("taillard/t100", "taillard", 100, 10, 42, 500, 5771, "10.0"),
        ],
    )
    comp = load_speedup_comparison(naive, accel)

    p1 = plot_speedup_vs_size(comp, tmp_path / "sp.png", budget=500)
    p2 = plot_runtime_scaling(comp, tmp_path / "sc.png", budget=500)
    assert p1.exists() and p1.stat().st_size > 0
    assert p2.exists() and p2.stat().st_size > 0
