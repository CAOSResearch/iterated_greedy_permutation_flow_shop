"""Tests for the algorithm label and Run_Manifest of the guided destruction.

Covers (spec destruccion-inteligente, task 5):

* Algorithm label (Req 8.2): the ResultRecord/campaign label reflects the
  selected destruction operator (``IG`` / ``IG-idle-greedy`` / ``IG-idle-rcl``).
* Run_Manifest (Req 8.1): the manifest parameters record the ``destruction``
  operator and the ``alpha`` value (for a guided operator), reusing the existing
  ``build_manifest`` facility.
* Schema stability (Req 12.8): the ``ResultRecord`` fields and the raw CSV header
  are unchanged, so guided operators stay comparable with NEH/IG in the same
  summary tables.
* No-overwrite (Req 8.4, 8.5): a guided label writes to its own
  ``raw/IG-idle-*.csv`` and never to ``raw/IG.csv`` (EXP-004) nor ``raw/NEH.csv``;
  a ``fresh`` restart is scoped to the campaign's own file, leaving the classic
  IG and NEH files untouched.

All persistence tests use ``tmp_path``; none writes to ``Code/results/``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from pfsp.core.ig import (
    IGConfig,
    StoppingCriterion,
    algorithm_label,
    iterated_greedy,
    manifest_parameters,
)
from pfsp.experiments.runner import run_campaign
from pfsp.instance import Instance
from pfsp.results.manifest import build_manifest
from pfsp.results.record import CSV_HEADER, ResultRecord, raw_csv_path

# ---------------------------------------------------------------------------
# Paths and skip conditions
# ---------------------------------------------------------------------------

_CODE_DIR = Path(__file__).resolve().parent.parent
_TAILLARD_INSTANCE = _CODE_DIR / "data" / "taillard" / "tai20_5_0.fsp"

_skip_without_taillard = pytest.mark.skipif(
    not _TAILLARD_INSTANCE.exists(),
    reason="Taillard benchmark data not present (data/taillard/tai20_5_0.fsp)",
)


# ---------------------------------------------------------------------------
# Test 1: Algorithm label reflects the operator (Req 8.2)
# ---------------------------------------------------------------------------


class TestAlgorithmLabel:
    """The label distinguishes the destruction operator without new schema."""

    @pytest.mark.parametrize(
        ("operator", "expected"),
        [
            ("random", "IG"),
            ("idle-greedy", "IG-idle-greedy"),
            ("idle-rcl", "IG-idle-rcl"),
        ],
    )
    def test_label_reflects_operator(self, operator: str, expected: str) -> None:
        """Each operator maps to its documented label (Req 8.2)."""
        config = IGConfig(destruction=operator)
        assert algorithm_label(config) == expected

    def test_default_config_labels_as_classic_ig(self) -> None:
        """The default (random) config keeps the plain 'IG' label."""
        assert algorithm_label(IGConfig()) == "IG"


# ---------------------------------------------------------------------------
# Test 2: Manifest parameters record destruction and alpha (Req 8.1)
# ---------------------------------------------------------------------------


class TestManifestParameters:
    """manifest_parameters records the operator and (guided) alpha."""

    def test_idle_rcl_records_destruction_and_alpha(self) -> None:
        """A guided idle-rcl config records both destruction and alpha (Req 8.1)."""
        config = IGConfig(destruction="idle-rcl", alpha=0.25)
        params = manifest_parameters(config)

        assert params["destruction"] == "idle-rcl"
        assert params["alpha"] == 0.25

    def test_idle_greedy_records_destruction_and_alpha(self) -> None:
        """idle-greedy is a guided operator, so alpha is recorded too."""
        config = IGConfig(destruction="idle-greedy", alpha=0.10)
        params = manifest_parameters(config)

        assert params["destruction"] == "idle-greedy"
        assert params["alpha"] == 0.10

    def test_random_records_destruction_without_alpha(self) -> None:
        """The random baseline records destruction but no (unused) alpha."""
        params = manifest_parameters(IGConfig(destruction="random"))

        assert params["destruction"] == "random"
        assert "alpha" not in params

    def test_records_the_rest_of_the_ig_parameters(self) -> None:
        """The other IG knobs are also captured for full traceability."""
        config = IGConfig(
            destruction_size=5,
            temperature_factor=0.4,
            local_search=True,
            stop=StoppingCriterion(kind="iterations", value=123),
            destruction="idle-rcl",
        )
        params = manifest_parameters(config)

        assert params["d"] == 5
        assert params["tp"] == 0.4
        assert params["local_search"] is True
        assert params["stop_kind"] == "iterations"
        assert params["stop_value"] == 123

    def test_parameters_land_in_the_built_manifest(self) -> None:
        """build_manifest carries destruction and alpha into its JSON payload."""
        config = IGConfig(destruction="idle-rcl", alpha=0.20)
        manifest = build_manifest(
            instance_id="taillard/tai20_5_0",
            algorithm=algorithm_label(config),
            outputs=["raw/IG-idle-rcl.csv"],
            seed=1,
            parameters=manifest_parameters(config),
            makespan=1300,
        )
        payload = manifest.to_dict()

        assert payload["algorithm"] == "IG-idle-rcl"
        assert payload["parameters"]["destruction"] == "idle-rcl"
        assert payload["parameters"]["alpha"] == 0.20


# ---------------------------------------------------------------------------
# Test 3: ResultRecord schema is unchanged (Req 12.8)
# ---------------------------------------------------------------------------


class TestResultRecordSchemaUnchanged:
    """Adding guided operators must not change the record schema."""

    def test_csv_header_is_the_fixed_schema(self) -> None:
        """The raw CSV header stays exactly the Phase 1/2 schema (Req 12.8)."""
        assert CSV_HEADER == (
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
        )

    def test_result_record_fields_match_header(self) -> None:
        """The dataclass fields still line up one-to-one with the CSV header."""
        field_names = tuple(f.name for f in dataclasses.fields(ResultRecord))
        assert field_names == CSV_HEADER

    def test_guided_record_serializes_with_the_same_columns(self) -> None:
        """A guided-label record produces exactly len(header) cells."""
        record = ResultRecord(
            instance_id="taillard/tai20_5_0",
            instance_set=None,
            n=20,
            m=5,
            algorithm="IG-idle-rcl",
            seed=1,
            makespan=1300,
            runtime_s=0.5,
            rpd=1.23,
            manifest_id="run-20260716-000000-abcd",
        )
        assert len(record.to_row()) == len(CSV_HEADER)


# ---------------------------------------------------------------------------
# Test 4: Raw CSV path derived from the label (no-overwrite, Req 8.4)
# ---------------------------------------------------------------------------


class TestRawCsvPathIsolation:
    """A guided label resolves to its own raw CSV, isolated from IG/NEH."""

    def test_each_label_has_its_own_file(self, tmp_path: Path) -> None:
        """The three operators resolve to three distinct raw files (Req 8.4)."""
        ig = raw_csv_path("IG", tmp_path)
        greedy = raw_csv_path("IG-idle-greedy", tmp_path)
        rcl = raw_csv_path("IG-idle-rcl", tmp_path)

        assert ig.name == "IG.csv"
        assert greedy.name == "IG-idle-greedy.csv"
        assert rcl.name == "IG-idle-rcl.csv"
        assert len({ig, greedy, rcl}) == 3

    def test_guided_label_never_targets_ig_or_neh(self, tmp_path: Path) -> None:
        """The guided labels never resolve to the IG.csv or NEH.csv files."""
        for operator in ("idle-greedy", "idle-rcl"):
            label = algorithm_label(IGConfig(destruction=operator))
            path = raw_csv_path(label, tmp_path)
            assert path != raw_csv_path("IG", tmp_path)
            assert path != raw_csv_path("NEH", tmp_path)


# ---------------------------------------------------------------------------
# Test 5: End-to-end campaign persists to its own files (Req 8.4, 8.5)
# ---------------------------------------------------------------------------


@_skip_without_taillard
class TestGuidedCampaignNoOverwrite:
    """A guided campaign writes to its own CSV and leaves IG/NEH untouched."""

    def _guided_config(self) -> IGConfig:
        return IGConfig(
            destruction_size=4,
            destruction="idle-rcl",
            alpha=0.10,
            stop=StoppingCriterion(kind="iterations", value=20),
        )

    def _solver(self, config: IGConfig):
        def solve(instance: Instance, seed: int) -> tuple[list[int], int]:
            return iterated_greedy(instance, seed, config)

        return solve

    def test_campaign_writes_own_csv_and_manifest(self, tmp_path: Path) -> None:
        """The guided campaign creates raw/IG-idle-rcl.csv, not IG.csv/NEH.csv."""
        config = self._guided_config()
        label = algorithm_label(config)

        exit_code = run_campaign(
            [_TAILLARD_INSTANCE],
            tmp_path,
            id_prefix="taillard/",
            algorithm=label,
            stochastic_solver=self._solver(config),
            seeds=[1],
            parameters=manifest_parameters(config),
            verbose=False,
        )

        assert exit_code == 0
        assert (tmp_path / "raw" / "IG-idle-rcl.csv").exists()
        assert not (tmp_path / "raw" / "IG.csv").exists()
        assert not (tmp_path / "raw" / "NEH.csv").exists()

        # The manifest of the run records destruction and alpha.
        manifests = list((tmp_path / "manifests").glob("*.json"))
        assert manifests, "expected at least one manifest file"
        text = manifests[0].read_text(encoding="utf-8")
        assert '"destruction": "idle-rcl"' in text
        assert '"alpha": 0.1' in text

    def test_fresh_is_scoped_to_own_file(self, tmp_path: Path) -> None:
        """--fresh on the guided campaign never touches IG.csv or NEH.csv."""
        # Pre-existing classic-IG and NEH files that must survive a guided --fresh.
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True)
        ig_csv = raw_dir / "IG.csv"
        neh_csv = raw_dir / "NEH.csv"
        ig_before = "prior IG results, must not be touched\n"
        neh_before = "prior NEH results, must not be touched\n"
        ig_csv.write_text(ig_before, encoding="utf-8")
        neh_csv.write_text(neh_before, encoding="utf-8")

        config = self._guided_config()
        exit_code = run_campaign(
            [_TAILLARD_INSTANCE],
            tmp_path,
            id_prefix="taillard/",
            algorithm=algorithm_label(config),
            stochastic_solver=self._solver(config),
            seeds=[1],
            parameters=manifest_parameters(config),
            fresh=True,
            verbose=False,
        )

        assert exit_code == 0
        # The unrelated campaigns' raw files are byte-for-byte intact.
        assert ig_csv.read_text(encoding="utf-8") == ig_before
        assert neh_csv.read_text(encoding="utf-8") == neh_before
        # And the guided campaign wrote its own file.
        assert (raw_dir / "IG-idle-rcl.csv").exists()


# ---------------------------------------------------------------------------
# Test 6: The recorded label round-trips through the raw record (Req 8.2)
# ---------------------------------------------------------------------------


def _make_synthetic_instance(n: int = 8, m: int = 3, seed: int = 42) -> Instance:
    """Create a small synthetic PFSP instance for a hermetic label round-trip."""
    rng = np.random.default_rng(seed)
    processing_times = rng.integers(1, 100, size=(m, n), dtype=np.int64)
    return Instance(
        processing_times=processing_times,
        n=n,
        m=m,
        source_benchmark="synthetic",
        source_path="synthetic/manifest_instance.fsp",
    )


@pytest.mark.parametrize(
    ("operator", "expected"),
    [
        ("idle-greedy", "IG-idle-greedy"),
        ("idle-rcl", "IG-idle-rcl"),
    ],
)
def test_label_matches_run_and_record(operator: str, expected: str) -> None:
    """The label used for a real guided run matches the operator (Req 8.2)."""
    instance = _make_synthetic_instance()
    config = IGConfig(
        destruction=operator,
        alpha=0.30,
        stop=StoppingCriterion(kind="iterations", value=30),
    )
    # The run itself is valid, and the label derived for its record is correct.
    perm, ms = iterated_greedy(instance, seed=1, config=config)
    assert sorted(perm) == list(range(instance.n))
    assert algorithm_label(config) == expected
