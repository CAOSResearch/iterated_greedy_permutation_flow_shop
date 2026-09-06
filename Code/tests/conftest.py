"""Session-wide test isolation: redirect _DEFAULT_RESULTS_DIR to a tempdir.

This conftest prevents any test from accidentally writing to the real
``Code/results/`` directory, even if it forgets to pass an explicit
``--results-dir`` or ``results_dir`` parameter. It works by monkeypatching the
module-level ``_DEFAULT_RESULTS_DIR`` in every module that defines one, AND the
default parameter values of functions that capture it at definition time.

Two layers of protection (belt and suspenders):
1. **Monkeypatch (session autouse):** redirects the default to a temporary
   directory for the entire test session.
2. **Post-suite check:** a test that fails if new files appear under the real
   ``Code/results/`` after the suite runs.

Requirement references: Req 12 (isolation), Req 10.2.
"""

from __future__ import annotations

import importlib
import inspect
import tempfile
from pathlib import Path

import pytest

# The real results directory that must NOT be touched by tests.
_REPO_RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# Modules (by import path) that define _DEFAULT_RESULTS_DIR and need patching.
_MODULES_TO_PATCH: tuple[str, ...] = (
    "pfsp.results.record",
    "pfsp.results.aggregate",
    "pfsp.cli",
    "pfsp.cli_ig",  # Future IG CLI; patched conditionally.
)


def _snapshot_results_dir(results_dir: Path) -> set[str]:
    """Return a set of relative file paths currently under *results_dir*."""
    if not results_dir.is_dir():
        return set()
    return {
        str(p.relative_to(results_dir)) for p in results_dir.rglob("*") if p.is_file()
    }


def _patch_function_defaults(mod: object, old_value: Path, new_value: Path) -> None:
    """Patch default argument values that captured the old _DEFAULT_RESULTS_DIR.

    Python evaluates default parameter values at function definition time, so
    monkeypatching the module attribute alone doesn't affect functions that used
    it as a default. This helper walks all functions in the module and replaces
    the old Path in their __defaults__ with the new one.
    """
    for _name, obj in inspect.getmembers(mod, inspect.isfunction):
        defaults = obj.__defaults__
        if defaults is None:
            continue
        new_defaults = tuple(
            new_value if isinstance(d, Path) and d == old_value else d for d in defaults
        )
        if new_defaults != defaults:
            obj.__defaults__ = new_defaults


# Take a snapshot at session start (before any test runs).
_INITIAL_SNAPSHOT: set[str] = _snapshot_results_dir(_REPO_RESULTS_DIR)


@pytest.fixture(autouse=True, scope="session")
def _isolate_results_dir():
    """Monkeypatch _DEFAULT_RESULTS_DIR in all known modules to a tempdir.

    This session-scoped, autouse fixture ensures that no test can write to the
    real ``Code/results/`` directory through the default path, regardless of
    whether it passes an explicit ``results_dir``.

    It patches both the module-level attribute AND the captured default values
    in function signatures.
    """
    with tempfile.TemporaryDirectory(prefix="pfsp_test_results_") as tmp:
        tmp_results = Path(tmp)
        # Keep track of original defaults so we can restore them.
        originals: list[tuple[object, Path]] = []
        for mod_name in _MODULES_TO_PATCH:
            try:
                mod = importlib.import_module(mod_name)
            except (ModuleNotFoundError, ImportError):
                # Module doesn't exist yet (e.g. pfsp.cli_ig); skip gracefully.
                continue
            if hasattr(mod, "_DEFAULT_RESULTS_DIR"):
                old_value = mod._DEFAULT_RESULTS_DIR
                originals.append((mod, old_value))
                mod._DEFAULT_RESULTS_DIR = tmp_results
                _patch_function_defaults(mod, old_value, tmp_results)
        yield tmp_results
        # Restore originals (good citizenship for session teardown).
        for mod, old_value in originals:
            _patch_function_defaults(mod, mod._DEFAULT_RESULTS_DIR, old_value)
            mod._DEFAULT_RESULTS_DIR = old_value


def test_no_new_files_in_real_results_dir() -> None:
    """FAIL if new files appeared under Code/results/ during the test suite.

    This is the second layer of isolation: even if the monkeypatch somehow
    failed, this test catches any leakage to the real results directory.
    """
    current_snapshot = _snapshot_results_dir(_REPO_RESULTS_DIR)
    new_files = current_snapshot - _INITIAL_SNAPSHOT
    assert (
        not new_files
    ), "Tests leaked files into Code/results/! New files:\n" + "\n".join(
        f"  - {f}" for f in sorted(new_files)
    )
