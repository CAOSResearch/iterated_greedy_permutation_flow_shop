"""Test that the results-directory isolation fixture works correctly.

Verifies that when code uses the default results path (without an explicit
``results_dir`` argument), writes land in the temporary directory provided by
the conftest fixture — and NOT in the real ``Code/results/``.

Requirement references: Req 12 (isolation), Req 10.2.
"""

from __future__ import annotations

import pfsp.results.record as record_mod
from tests.conftest import _REPO_RESULTS_DIR


def test_default_results_dir_is_redirected_to_temp() -> None:
    """Without explicit results_dir, the module-level default points to tmp."""
    current_default = record_mod._DEFAULT_RESULTS_DIR
    # The fixture should have redirected it away from the real dir.
    assert current_default != _REPO_RESULTS_DIR, (
        "_DEFAULT_RESULTS_DIR was NOT patched — it still points to the real "
        f"results directory: {current_default}"
    )
    # It should be inside a temporary directory (not under Code/results/).
    assert not str(current_default).startswith(
        str(_REPO_RESULTS_DIR)
    ), "_DEFAULT_RESULTS_DIR still resolves under Code/results/"


def test_write_without_results_dir_goes_to_temp() -> None:
    """A write using the default path lands in the temp dir, not Code/results/."""
    from pfsp.results.record import raw_csv_path

    # raw_csv_path uses _DEFAULT_RESULTS_DIR when no explicit dir is passed.
    csv_path = raw_csv_path("TEST_ISOLATION")

    # The path should be inside the patched temp directory.
    assert not str(csv_path).startswith(
        str(_REPO_RESULTS_DIR)
    ), f"raw_csv_path resolved under Code/results/: {csv_path}"

    # Actually write a small file to prove it lands in temp, not in real results.
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("test\n", encoding="utf-8")
    assert csv_path.exists(), "File was not created"

    # Confirm nothing new appeared in the real results dir.
    real_test_csv = _REPO_RESULTS_DIR / "raw" / "TEST_ISOLATION.csv"
    assert (
        not real_test_csv.exists()
    ), f"File leaked to real results directory: {real_test_csv}"


def test_aggregate_default_also_redirected() -> None:
    """The aggregate module's default is also patched away from real dir."""
    import pfsp.results.aggregate as agg_mod

    current_default = agg_mod._DEFAULT_RESULTS_DIR
    assert current_default != _REPO_RESULTS_DIR
    assert not str(current_default).startswith(str(_REPO_RESULTS_DIR))


def test_cli_default_also_redirected() -> None:
    """The CLI module's default is also patched away from real dir."""
    import pfsp.cli as cli_mod

    current_default = cli_mod._DEFAULT_RESULTS_DIR
    assert current_default != _REPO_RESULTS_DIR
    assert not str(current_default).startswith(str(_REPO_RESULTS_DIR))


def test_cli_ig_patched_conditionally() -> None:
    """If pfsp.cli_ig exists, its default is patched; if not, no error."""
    try:
        import pfsp.cli_ig as cli_ig_mod

        current_default = cli_ig_mod._DEFAULT_RESULTS_DIR
        assert current_default != _REPO_RESULTS_DIR
    except (ModuleNotFoundError, ImportError):
        # Module doesn't exist yet — the fixture should have skipped it
        # without raising. This test passes by verifying no crash occurred.
        pass
