"""Iterated Greedy core: operators, configuration, and main loop.

This module implements the classic Iterated Greedy algorithm (Ruiz & Stützle,
2005) for the PFSP. It contains the IG configuration, validation, operators
(reconstruction, acceptance), and the main ``iterated_greedy`` loop.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from pfsp.core.destruction import (
    DestructFn,
    destruct,
    make_idle_greedy_destruct,
    make_idle_rcl_destruct,
)
from pfsp.core.insertion import best_insertion
from pfsp.core.local_search import local_search
from pfsp.core.makespan import makespan
from pfsp.core.neh import neh
from pfsp.instance import Instance
from pfsp.repro.seeds import make_rng

# ---------------------------------------------------------------------------
# Stopping criterion and IG configuration
# ---------------------------------------------------------------------------

StopKind = Literal["iterations", "evaluations", "time"]

DEFAULT_MAX_EVALUATIONS: int = 5000
"""Modest default for reproducible quick runs (deterministic mode)."""


@dataclass(frozen=True)
class StoppingCriterion:
    """Stopping criterion for the IG loop.

    Parameters
    ----------
    kind:
        One of ``"iterations"``, ``"evaluations"``, or ``"time"``.
        Defaults to ``"evaluations"`` (deterministic and reproducible).
    value:
        Positive number indicating the budget. For ``iterations`` and
        ``evaluations`` it is an integer count; for ``time`` it is seconds.
    """

    kind: StopKind = "evaluations"
    value: float = DEFAULT_MAX_EVALUATIONS


@dataclass(frozen=True)
class IGConfig:
    """Configuration for the Iterated Greedy algorithm.

    Parameters
    ----------
    destruction_size:
        Number of jobs to remove per iteration (``d``). Must be an integer
        in ``[1, n-1]``.
    temperature_factor:
        Scaling factor for the acceptance temperature (``Tp``). Must be
        non-negative.
    local_search:
        Whether to apply insertion-based local search (enabled by default,
        as in the classic IG of Ruiz & Stützle 2005).
    stop:
        Stopping criterion (defaults to evaluations-based with
        ``DEFAULT_MAX_EVALUATIONS``).
    destruction:
        Destruction operator selector, one of ``"random"`` (classic uniform
        destruction, the default and unchanged Phase 2 baseline),
        ``"idle-greedy"`` (O1, greedy top-idle removal, for the ablation), or
        ``"idle-rcl"`` (O2, the guided idle-RCL operator, the star
        contribution). Defaults to ``"random"`` so the baseline is unchanged.
    alpha:
        Fraction that sets the RCL size for the guided ``"idle-rcl"`` operator
        (``0 < alpha <= 1``). Documented default ``0.10``. Only ``"idle-rcl"``
        uses it; ``"random"`` and ``"idle-greedy"`` ignore it.
    """

    destruction_size: int = 4
    temperature_factor: float = 0.4
    local_search: bool = True
    stop: StoppingCriterion = field(default_factory=StoppingCriterion)
    destruction: str = "random"
    alpha: float = 0.10


_VALID_STOP_KINDS: set[str] = {"iterations", "evaluations", "time"}

_VALID_DESTRUCTION: set[str] = {"random", "idle-greedy", "idle-rcl"}


def validate_config(config: IGConfig, n: int) -> None:
    """Validate IG parameters before starting the loop.

    Raises :class:`ValueError` with a descriptive message if any parameter
    is invalid.

    Parameters
    ----------
    config:
        The IG configuration to validate.
    n:
        Number of jobs in the instance (needed to bound ``destruction_size``).

    Raises
    ------
    ValueError
        If ``destruction_size`` is not an integer in ``[1, n-1]``,
        ``temperature_factor`` is negative, the stopping criterion kind is
        unrecognized, the stopping criterion value is not positive, the
        ``destruction`` operator name is unrecognized, or ``alpha`` is not in
        the range ``0 < alpha <= 1``.

    Notes
    -----
    Satisfies Requirements 3.5, 5.6, 6.6 (Phase 2 validation) and 6.2, 7.2
    (guided destruction): validation occurs before the loop starts; a
    descriptive error is raised and the run is not executed.
    """
    # --- destruction_size (d) ---
    d = config.destruction_size
    if not isinstance(d, int):
        raise ValueError(
            f"destruction_size must be an integer, got {type(d).__name__} ({d!r})."
        )
    if d < 1 or d > n - 1:
        raise ValueError(
            f"destruction_size must be in [1, {n - 1}] for n={n} jobs, got d={d}."
        )

    # --- temperature_factor (Tp) ---
    tp = config.temperature_factor
    if tp < 0:
        raise ValueError(f"temperature_factor must be >= 0, got Tp={tp}.")

    # --- stopping criterion kind ---
    kind = config.stop.kind
    if kind not in _VALID_STOP_KINDS:
        raise ValueError(
            f"Unrecognized stopping criterion kind: {kind!r}. "
            f"Valid kinds are: {sorted(_VALID_STOP_KINDS)}."
        )

    # --- stopping criterion value ---
    value = config.stop.value
    if value <= 0:
        raise ValueError(f"Stopping criterion value must be positive, got {value}.")

    # --- destruction operator ---
    destruction = config.destruction
    if destruction not in _VALID_DESTRUCTION:
        raise ValueError(
            f"Unrecognized destruction operator: {destruction!r}. "
            f"Valid operators are: {sorted(_VALID_DESTRUCTION)}."
        )

    # --- alpha (RCL size fraction) ---
    alpha = config.alpha
    if not (0 < alpha <= 1):
        raise ValueError(
            f"alpha must be in the range 0 < alpha <= 1, got alpha={alpha}."
        )


def reconstruct(
    partial_sequence: list[int],
    removed_jobs: list[int],
    processing_times: NDArray[np.int64],
) -> tuple[list[int], int]:
    """Reinsert ``removed_jobs`` one at a time (in order) at the best position.

    This implements the construction (reinsertion) phase of the Iterated Greedy
    algorithm, reusing the shared :func:`~pfsp.core.insertion.best_insertion`
    helper (NEH mechanics). Each removed job is inserted at the position that
    minimizes the makespan of the growing sequence; ties are broken by the
    lowest insertion index (inherited from ``best_insertion``).

    Parameters
    ----------
    partial_sequence:
        The remaining jobs after destruction, preserving relative order.
        Not mutated.
    removed_jobs:
        Jobs to reinsert, in the order they should be processed (one at a time).
    processing_times:
        Full processing-time matrix of shape ``(m, n)`` for all ``n`` jobs.

    Returns
    -------
    tuple[list[int], int]
        A pair ``(permutation, makespan_value)`` where ``permutation`` contains
        each job index exactly once and ``makespan_value`` is the makespan of
        that complete permutation.

    Notes
    -----
    Satisfies Requirements 4.1, 4.2, 4.3, 4.4: reinserts in given order using
    ``best_insertion``, returns a valid complete permutation with its makespan.
    """
    sequence = list(partial_sequence)
    ms = 0

    for job in removed_jobs:
        sequence, ms = best_insertion(sequence, job, processing_times)

    return sequence, ms


def compute_temperature(
    processing_times: NDArray[np.int64],
    temperature_factor: float,
) -> float:
    """Compute the constant temperature for the SA acceptance criterion.

    The formula is ``T = Tp * sum(p) / (n * m * 10)`` where ``Tp`` is the
    temperature factor, ``sum(p)`` is the total sum of all processing times,
    ``n`` is the number of jobs, and ``m`` is the number of machines.

    Parameters
    ----------
    processing_times:
        Processing-time matrix of shape ``(m, n)``.
    temperature_factor:
        Non-negative scaling factor ``Tp``.

    Returns
    -------
    float
        The computed temperature value.

    Notes
    -----
    Satisfies Requirement 5.1: T = Tp · Σp / (n · m · 10).
    """
    m, n = processing_times.shape
    total_processing = float(processing_times.sum())
    return temperature_factor * total_processing / (n * m * 10)


def accept(
    candidate_makespan: int,
    current_makespan: int,
    temperature: float,
    rng: np.random.Generator,
) -> bool:
    """Decide whether to accept a candidate solution (SA acceptance criterion).

    The acceptance rule follows the Iterated Greedy of Ruiz & Stützle (2005):

    - If the candidate is at least as good as the current solution
      (``candidate_makespan <= current_makespan``), always accept.
    - If the candidate is worse and ``temperature > 0``, accept with
      probability ``exp(-(candidate_makespan - current_makespan) / temperature)``
      using a draw from ``rng``.
    - If ``temperature == 0``, never accept a worsening solution.

    Parameters
    ----------
    candidate_makespan:
        Makespan of the candidate solution.
    current_makespan:
        Makespan of the current solution.
    temperature:
        Constant temperature (non-negative).
    rng:
        NumPy random generator (seeded) for the probabilistic decision.

    Returns
    -------
    bool
        ``True`` if the candidate is accepted, ``False`` otherwise.

    Notes
    -----
    Satisfies Requirements 5.2, 5.3, 5.4.
    """
    if candidate_makespan <= current_makespan:
        return True

    # Worsening candidate
    if temperature > 0:
        delta = candidate_makespan - current_makespan
        probability = math.exp(-delta / temperature)
        return rng.random() <= probability

    # temperature == 0: reject worsening
    return False


def build_destruct_fn(
    destruction: str,
    processing_times: NDArray[np.int64],
    alpha: float,
) -> DestructFn:
    """Return the destruction operator selected by ``destruction``.

    Binds the chosen operator to the shared destruction call site
    ``destruct_fn(permutation, d, rng)`` used by :func:`iterated_greedy`, so the
    loop code path is identical across operators (Req 6.3).

    Parameters
    ----------
    destruction:
        Operator name, one of ``"random"``, ``"idle-greedy"``, or
        ``"idle-rcl"`` (assumed already validated by :func:`validate_config`).
    processing_times:
        Processing-time matrix of shape ``(m, n)``. Captured by the guided
        operators; never mutated.
    alpha:
        RCL size fraction, used only by the ``"idle-rcl"`` operator.

    Returns
    -------
    DestructFn
        A ``destruct_fn(permutation, d, rng) -> (partial, removed)`` operator.

    Notes
    -----
    For ``"random"`` this returns the Phase 2 :func:`~pfsp.core.destruction.destruct`
    function itself, so the random path consumes the generator exactly as in the
    classic IG and the baseline stays byte-identical (Req 1.6, 6.2).
    """
    if destruction == "idle-rcl":
        return make_idle_rcl_destruct(processing_times, alpha)
    if destruction == "idle-greedy":
        return make_idle_greedy_destruct(processing_times)
    # "random": the classic Phase 2 operator, unchanged call site and rng usage.
    return destruct


#: Algorithm label associated with each destruction operator. The label is what
#: the results framework uses as the ``ResultRecord.algorithm`` value and, via
#: :func:`~pfsp.results.record.raw_csv_path`, as the name of the operator's own
#: raw CSV file (``raw/<label>.csv``). Keeping the classic random operator as
#: plain ``"IG"`` preserves comparability with the Phase 2 baseline (EXP-004) and
#: the NEH file, while the guided operators write to their own files (Req 8.2,
#: 8.4).
_ALGORITHM_LABELS: dict[str, str] = {
    "random": "IG",
    "idle-greedy": "IG-idle-greedy",
    "idle-rcl": "IG-idle-rcl",
}


def algorithm_label(config: IGConfig) -> str:
    """Return the algorithm label for the destruction operator in ``config``.

    The label distinguishes the operator used so that operators can be compared
    within the same summary tables without changing the ResultRecord schema:
    ``"IG"`` for the classic Random_Destruction, ``"IG-idle-greedy"`` for O1, and
    ``"IG-idle-rcl"`` for the guided O2 (Req 8.2). Because the raw CSV path is
    derived from this label, a guided campaign persists to its own
    ``raw/IG-idle-*.csv`` file and never overwrites ``raw/IG.csv`` (EXP-004) nor
    ``raw/NEH.csv`` (Req 8.4).

    Parameters
    ----------
    config:
        IG configuration whose ``destruction`` selects the operator. Assumed
        already validated by :func:`validate_config`.

    Returns
    -------
    str
        The algorithm label associated with ``config.destruction``.

    Raises
    ------
    KeyError
        If ``config.destruction`` is not a recognized operator name (which
        :func:`validate_config` rejects before a run starts).
    """
    return _ALGORITHM_LABELS[config.destruction]


def manifest_parameters(config: IGConfig) -> dict[str, Any]:
    """Return the IG parameters recorded in the Run_Manifest for ``config``.

    Reuses the existing manifest facility (the ``parameters`` mapping of
    :func:`~pfsp.results.manifest.build_manifest`) to record the destruction
    operator and, for a guided operator, the Alpha value, alongside the rest of
    the IG parameters (``d``, ``tp``, stopping criterion, local search). This
    keeps every knob that varies across executions traceable and the runs
    regenerable (Req 8.1).

    Parameters
    ----------
    config:
        The IG configuration whose parameters are captured.

    Returns
    -------
    dict[str, Any]
        A JSON-serializable mapping with ``destruction`` always present and
        ``alpha`` present only for a guided operator (``"idle-greedy"`` or
        ``"idle-rcl"``); the classic ``"random"`` operator omits ``alpha`` since
        it does not use it.
    """
    parameters: dict[str, Any] = {
        "d": config.destruction_size,
        "tp": config.temperature_factor,
        "stop_kind": config.stop.kind,
        "stop_value": config.stop.value,
        "local_search": config.local_search,
        "destruction": config.destruction,
    }
    # Alpha is a parameter only of the guided operators; the random baseline
    # does not consume it, so it is not recorded for "random".
    if config.destruction != "random":
        parameters["alpha"] = config.alpha
    return parameters


_DEFAULT_CONFIG = IGConfig()
"""Module-level default IGConfig instance (avoids mutable default argument)."""


def iterated_greedy(
    instance: Instance,
    seed: int,
    config: IGConfig = _DEFAULT_CONFIG,
) -> tuple[list[int], int]:
    """Run the classic Iterated Greedy and return ``(best_permutation, makespan)``.

    Implements the IG algorithm of Ruiz & Stützle (2005): starting from a NEH
    solution (optionally improved by local search), it iterates a cycle of
    destruction, reconstruction, optional local search, and SA acceptance until
    the stopping criterion is met. Tracks the best solution found (strict
    improvement only).

    Parameters
    ----------
    instance:
        The PFSP instance to solve. Its ``processing_times`` matrix is read
        but never mutated.
    seed:
        Non-negative integer seed for the random generator. All randomness in
        the run (destruction, acceptance, local search scan order) comes from
        a single generator seeded with this value.
    config:
        IG configuration (destruction size, temperature factor, local search
        flag, stopping criterion). Validated before starting; invalid
        parameters raise ``ValueError``.

    Returns
    -------
    tuple[list[int], int]
        A pair ``(best_permutation, best_makespan)`` where ``best_permutation``
        is a valid permutation of ``0..n-1`` and ``best_makespan`` is its
        makespan computed by the Makespan_Calculator.

    Raises
    ------
    ValueError
        If ``config`` contains invalid parameters (see :func:`validate_config`).

    Notes
    -----
    Satisfies Requirements 2.1, 2.2, 2.3, 5.5, 7.1–7.7, 8.3, 9.1.

    Stopping criterion counting:

    - ``iterations``: one iteration = one loop cycle.
    - ``evaluations``: counted per iteration as an approximation.
      Each reconstruction evaluates approximately ``d * (current_length)``
      candidates (d insertions), and each local search pass evaluates many
      more. For a pragmatic and deterministic count, each iteration increments
      the counter by 1, and the default budget is calibrated accordingly.
      **Actual implementation:** counts one evaluation per iteration for the
      ``evaluations`` mode (equivalent to iterations with a scale factor of 1),
      consistent with the design recommendation for this task. Refinement is
      deferred to task 15 (calibration).
    - ``time``: checks ``time.perf_counter()`` each iteration.
    """
    # --- 1. Validate config (fail early) ---
    validate_config(config, instance.n)

    # --- 1b. Bind the selected destruction operator to the shared call site.
    # Building the operator never consumes the generator, so the "random" path
    # stays byte-identical to the Phase 2 baseline (Req 1.6, 6.2, 6.3).
    destruct_fn = build_destruct_fn(
        config.destruction, instance.processing_times, config.alpha
    )

    # --- 2. Create seeded RNG ---
    rng = make_rng(seed)

    # --- 3. Initial solution: NEH + optional local search ---
    pi, _ = neh(instance)

    if config.local_search:
        pi, _ = local_search(pi, instance.processing_times, rng)

    # --- 4. Best solution tracking ---
    best_perm = list(pi)
    best_makespan = makespan(instance.processing_times, best_perm)
    current_perm = list(pi)
    current_makespan = best_makespan

    # --- 5. Compute temperature ---
    temperature = compute_temperature(
        instance.processing_times, config.temperature_factor
    )

    # --- 6. Main loop ---
    stop_kind = config.stop.kind
    stop_value = config.stop.value
    counter = 0
    start_time = time.perf_counter()

    while True:
        # Check stopping criterion
        if stop_kind == "iterations":
            if counter >= stop_value:
                break
        elif stop_kind == "evaluations":
            # Count one evaluation unit per iteration (pragmatic approach;
            # refinement deferred to calibration task 15).
            if counter >= stop_value:
                break
        elif stop_kind == "time":
            elapsed = time.perf_counter() - start_time
            if elapsed >= stop_value:
                break

        # --- Destruction (operator selected by config; same call site) ---
        partial, removed = destruct_fn(current_perm, config.destruction_size, rng)

        # --- Construction (reinsertion) ---
        pi_cand, c_cand = reconstruct(partial, removed, instance.processing_times)

        # --- Optional local search ---
        if config.local_search:
            pi_cand, c_cand = local_search(pi_cand, instance.processing_times, rng)

        # --- Acceptance criterion ---
        if accept(c_cand, current_makespan, temperature, rng):
            current_perm = pi_cand
            current_makespan = c_cand

        # --- Update best (strict improvement only) ---
        if c_cand < best_makespan:
            best_perm = list(pi_cand)
            best_makespan = c_cand

        # Increment counter
        counter += 1

    # --- 7. Return best solution ---
    return best_perm, best_makespan
