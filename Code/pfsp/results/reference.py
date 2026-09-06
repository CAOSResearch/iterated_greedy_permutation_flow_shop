"""Reference_Provider: best-known reference value of a PFSP instance.

The reference value ``ref`` of an instance is the baseline against which the
Relative Percentage Deviation (RPD) of a makespan is later computed by the
results framework. This module resolves ``ref`` from benchmark metadata without
ever fabricating a value (Reqs 10.3, 10.5):

* **Taillard** instances carry their best-known upper bound in the instance file
  itself, so the reference is simply :attr:`Instance.upper_bound` (which may be
  ``None`` if the source did not provide one).
* **VRF** instances do not embed their bounds; the best-known values live in the
  official ``BestSolutionsAndBounds.xlsx``. That workbook is converted **once**
  to a normalized CSV (``data/vrf/best_solutions_and_bounds.csv`` with columns
  ``instance,set,n,m,UB,LB``) so the runtime can read it with the standard
  library :mod:`csv` module and never depends on an Excel-reading library. The
  conversion script is documented in ``Code/data/vrf/README.md``.

The VRF lookup table is parsed lazily and cached in memory between calls. If the
CSV is absent (the ``data/`` tree is not versioned) or an instance is not found,
:func:`get_reference` returns ``None`` rather than raising or guessing.
"""

from __future__ import annotations

import csv
from pathlib import Path

from pfsp.instance import Instance

#: Location of the normalized VRF bounds CSV, relative to the repository root
#: (the ``Code/`` directory that contains both ``pfsp`` and ``data``).
_VRF_BOUNDS_RELATIVE_PATH = Path("data") / "vrf" / "best_solutions_and_bounds.csv"

#: Column of the CSV holding the base instance name (e.g. ``VFR10_5_1``).
_CSV_INSTANCE_COLUMN = "instance"

#: Column of the CSV holding the best-known upper bound used as the reference.
_CSV_UPPER_BOUND_COLUMN = "UB"

#: Suffix the VRF instance files carry in their stem (``VFR10_5_1_Gap``) but the
#: CSV ``instance`` column omits (``VFR10_5_1``); stripped when building the key.
_VRF_GAP_SUFFIX = "_Gap"

#: In-memory cache of the parsed VRF references, mapping base instance name to
#: its best-known upper bound. ``None`` means "not loaded yet"; an empty mapping
#: is a valid loaded state (e.g. when the CSV is missing).
_vrf_reference_cache: dict[str, int] | None = None


def _repository_root() -> Path:
    """Return the repository root (the ``Code/`` directory) for this package.

    Resolved relative to this file so the data path is portable across
    checkouts: ``reference.py`` -> ``results`` -> ``pfsp`` -> repository root.
    """
    return Path(__file__).resolve().parents[2]


def _vrf_bounds_path() -> Path:
    """Return the absolute path of the normalized VRF bounds CSV."""
    return _repository_root() / _VRF_BOUNDS_RELATIVE_PATH


def _load_vrf_references() -> dict[str, int]:
    """Parse and cache the VRF bounds CSV as a ``{instance: UB}`` mapping.

    The CSV is read once with the standard-library :mod:`csv` module and the
    result is memoized in :data:`_vrf_reference_cache`. A missing CSV yields an
    empty mapping (so every lookup resolves to ``None``); rows whose ``UB`` is
    missing or non-integer are skipped rather than fabricated.

    Returns
    -------
    dict[str, int]
        Mapping from base instance name (e.g. ``"VFR10_5_1"``) to its best-known
        upper bound.
    """
    global _vrf_reference_cache
    if _vrf_reference_cache is not None:
        return _vrf_reference_cache

    references: dict[str, int] = {}
    path = _vrf_bounds_path()
    try:
        with open(path, encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                name = (row.get(_CSV_INSTANCE_COLUMN) or "").strip()
                raw_ub = (row.get(_CSV_UPPER_BOUND_COLUMN) or "").strip()
                if not name or not raw_ub:
                    continue
                try:
                    references[name] = int(raw_ub)
                except ValueError:
                    # Never fabricate a value from an unparseable cell; skip it.
                    continue
    except OSError:
        # The data/ tree is not versioned; a missing CSV is a valid state and
        # must degrade to "no reference resolved" (Req 10.5).
        references = {}

    _vrf_reference_cache = references
    return _vrf_reference_cache


def _vrf_lookup_key(instance: Instance) -> str:
    """Derive the CSV lookup key for a VRF ``instance``.

    The key is the instance file stem with the trailing ``_Gap`` suffix removed
    so it matches the base name used in the CSV ``instance`` column. For
    example, a ``source_path`` of ``.../VFR10_5_1_Gap.txt`` yields ``VFR10_5_1``.
    """
    stem = Path(instance.source_path).stem
    if stem.endswith(_VRF_GAP_SUFFIX):
        stem = stem[: -len(_VRF_GAP_SUFFIX)]
    return stem


def get_reference(instance: Instance) -> int | None:
    """Resolve the best-known reference value of ``instance``.

    Parameters
    ----------
    instance:
        The instance whose reference value is requested.

    Returns
    -------
    int | None
        For Taillard instances, the upper bound recorded in the instance
        (:attr:`Instance.upper_bound`), which may itself be ``None``. For VRF
        instances, the best-known upper bound looked up by base instance name in
        the normalized bounds CSV. ``None`` when the value cannot be resolved
        (unknown benchmark, missing CSV, or instance not listed); a reference is
        never fabricated (Req 10.5).
    """
    if instance.source_benchmark == "taillard":
        return instance.upper_bound
    if instance.source_benchmark == "vrf":
        return _load_vrf_references().get(_vrf_lookup_key(instance))
    return None


__all__ = ["get_reference"]
