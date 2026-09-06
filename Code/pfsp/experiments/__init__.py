"""Experiment campaign layer.

This subpackage hosts the reusable machinery that drives experimental campaigns
(running an algorithm over a benchmark and persisting comparable results). The
thin CLI scripts in ``Code/scripts/`` delegate to :func:`run_campaign`, so the
campaign logic — including the checkpoint/resume behaviour required by the
project steering — lives in one tested place instead of being duplicated per
script.
"""

from pfsp.experiments.runner import (
    EXIT_ANOMALY,
    EXIT_INTERRUPTED,
    EXIT_NO_INSTANCES,
    EXIT_OK,
    best_known_sanity,
    configure_logging,
    discover_instances,
    lower_bound_sanity,
    run_campaign,
)

__all__ = [
    "EXIT_OK",
    "EXIT_ANOMALY",
    "EXIT_NO_INSTANCES",
    "EXIT_INTERRUPTED",
    "run_campaign",
    "configure_logging",
    "discover_instances",
    "lower_bound_sanity",
    "best_known_sanity",
]
