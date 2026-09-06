"""Analysis layer: derived data loading and figures for the thesis.

Separated from the algorithmic core, the I/O and the experiment runners
(steering ``codigo-python.md``): this layer only *reads* the persisted results
(``results/raw`` and ``results/summary``) and turns them into figures. Data
loading (:mod:`pfsp.analysis.data`) uses the standard library and is fully
testable; rendering (:mod:`pfsp.analysis.plots`) uses matplotlib and is a
dev-only dependency, never part of the runtime.

Every figure is regenerable from the persisted results, preserving the
experiment -> result -> figure traceability required by the project.
"""

from pfsp.analysis.data import (
    CalibrationSweep,
    SizeGroupRPD,
    SpeedupComparison,
    load_calibration_sweep,
    load_rpd_by_size_group,
    load_rpd_series,
    load_speedup_comparison,
)
from pfsp.analysis.stats import (
    FriedmanResult,
    average_ranks,
    chi_square_sf,
    friedman_test,
    nemenyi_critical_difference,
)

__all__ = [
    "SizeGroupRPD",
    "CalibrationSweep",
    "SpeedupComparison",
    "load_rpd_series",
    "load_rpd_by_size_group",
    "load_calibration_sweep",
    "load_speedup_comparison",
    "FriedmanResult",
    "average_ranks",
    "chi_square_sf",
    "friedman_test",
    "nemenyi_critical_difference",
]
