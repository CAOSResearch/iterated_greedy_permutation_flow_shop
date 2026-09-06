# pfsp

Reproducible Permutation Flow-Shop Scheduling Problem (PFSP) toolkit: the NEH
constructive baseline, the classic Iterated Greedy (IG) and the **guided idle-RCL
destruction operator** proposed by the thesis, together with the instance I/O,
makespan computation, reproducibility utilities, results framework and analysis layer
that support the experimental campaigns.

The package `pfsp` is the **single source of truth** for the algorithmic logic. The
command-line entry points are clients that import the package; they do not reimplement
core logic. The only objective modeled is the makespan (`C_max`); no energy or power
metric is ever computed — the sustainability argument of the thesis is narrative,
derived from reducing makespan and idle times.

> El código va en **inglés** por convención técnica; este README documenta el mapa de
> artefactos y su relación con la metodología de la memoria. Para preparar el entorno,
> ejecutar los CLI y depurar componentes, ver **`EXECUTION_GUIDE.md`**.

## Quick start

```bash
pip install -r requirements.txt        # runtime (NumPy, pinned)
pip install -e .                       # editable install (registers pfsp-neh, pfsp-ig)
pfsp-neh --instance data/taillard/tai20_5_0.fsp
pfsp-ig --instance data/taillard/tai20_5_0.fsp --seed 42
pfsp-ig --instance data/taillard/tai20_5_0.fsp --seed 42 --destruction idle-rcl --alpha 0.30
```

(Paths are relative to `Code/`. From the repository root, prefix them with `Code/`.)

## Artifact map

```
Code/
├── pfsp/                       # the package — single source of truth
│   ├── instance.py             # Instance contract (frozen dataclass) + identifier
│   ├── io/
│   │   ├── taillard.py         # Taillard .fsp reader
│   │   ├── vrf.py              # VRF VFR* reader
│   │   └── loader.py           # load_instance: format-agnostic dispatch
│   ├── core/
│   │   ├── makespan.py         # vectorized C_max (NumPy)
│   │   ├── neh.py              # NEH constructive heuristic (baseline)
│   │   ├── insertion.py        # shared best-insertion mechanic (NEH + IG)
│   │   ├── ig.py               # IGConfig + iterated_greedy loop + acceptance
│   │   ├── destruction.py      # destruction operators: random, idle-greedy, idle-rcl
│   │   └── local_search.py     # insertion-based local search (seeded)
│   ├── results/
│   │   ├── reference.py        # best-known reference (UB Taillard / CSV VRF)
│   │   ├── record.py           # ResultRecord schema + RPD + raw CSV persistence
│   │   ├── aggregate.py        # Aggregator: summary tables derived from raw
│   │   └── manifest.py         # Run manifest + environment capture
│   ├── repro/
│   │   └── seeds.py            # Seed manager (explicit-seed RNG)
│   ├── experiments/
│   │   └── runner.py           # Resumable campaign engine (multi-seed)
│   ├── analysis/
│   │   ├── data.py             # Load RPD series / calibration sweeps from raw CSV
│   │   ├── stats.py            # Friedman test + Nemenyi critical difference
│   │   └── plots.py            # Figures (matplotlib)
│   ├── cli.py                  # pfsp-neh entry point
│   └── cli_ig.py               # pfsp-ig entry point (Iterated Greedy)
├── tests/                      # pytest suite (unit + property-based)
├── data/                       # benchmark instances (versioned; self-contained repo)
│   ├── taillard/               # 120 .fsp files + provenance README
│   └── vrf/                    # Small/ + Large/ .txt + best_solutions_and_bounds.csv
├── results/                    # experimental evidence — laptop environment
├── results-server/             # experimental evidence — laboratory server
├── pyproject.toml              # package + black + ruff + pytest config
├── requirements.txt            # runtime dependency, pinned (numpy)
├── requirements-dev.txt        # dev: pytest, black, ruff, matplotlib
├── EXECUTION_GUIDE.md          # setup, run, debug
└── README.md                   # this file
```

## Algorithms and operators

| Label | Meaning |
|---|---|
| `NEH` | Constructive baseline (deterministic). |
| `IG` | Classic Iterated Greedy with **random** destruction (Phase 2 baseline). |
| `IG-idle-greedy` | Guided operator **O1**: purely greedy on the idle-time score (ablation). |
| `IG-idle-rcl` | Guided operator **O2**: idle-time score + Restricted Candidate List with parameter α — the thesis contribution. |

Each label writes to its **own** raw results file, so no campaign ever overwrites
another's evidence.

## Experimental evidence (`results/`, `results-server/`)

Two environments were used: a conventional laptop (`results/`) and the laboratory
server (`results-server/`), so that runtimes could be compared on a single hardware
platform. Both follow the same layout.

| Artifact | Path | Producer | Content |
|---|---|---|---|
| Raw result record | `raw/<algorithm>.csv` | `pfsp.results.record.append_record` | One **append-only** row per execution. Fixed header: `instance_id, instance_set, n, m, algorithm, seed, makespan, runtime_s, rpd, manifest_id, timestamp`. Every individual run is preserved, not just aggregates: multi-seed campaigns yield one row per instance × seed. |
| Calibration sweeps | `calibration/*.csv` | campaign engine | Computational-budget sweep (where RPD stabilizes) and α sweep for the RCL. These fix the parameters used in the final campaigns. |
| Instrumentation | `instrumentation/*` | campaign engine | Local-search work measured per iteration, plus the analysis report that explains the mechanism behind the observed behaviour. |
| Summary table | `summary/summary.csv` | `pfsp.results.aggregate.write_summary` | **Derived** from `raw/`: one row per `(algorithm, instance_set, n, m)` group with `n_instances, rpd_mean, rpd_std, rpd_min, rpd_max, makespan_mean`. Regenerated from the raw rows; never written by hand. |
| Figures | `figures/*.pdf` | `pfsp.analysis.plots` | Plots derived from the raw records and the sweeps. |

Regenerate the summary from the raw records at any time:

```bash
python -c "from pfsp.results.aggregate import write_summary, default_raw_dir, default_summary_path; print(write_summary(default_raw_dir(), default_summary_path()))"
```

**Comparison metric.** `RPD = 100 * (C_max - ref) / ref` against the best-known
reference: the upper bound for Taillard instances, the best-known value from the VRF
CSV for VRF instances, and `None` if unresolved — a reference is **never fabricated**.

## Components

| Component | Module | Role |
|---|---|---|
| Instance contract | `pfsp/instance.py` | Single normalized `Instance` type (matrix `(m, n)`, `p[i][j]` = time of job `j` on machine `i`), read-only matrix, stable `identifier` (`<benchmark>/<stem>`). |
| Taillard reader | `pfsp/io/taillard.py` | Parse `.fsp` files to an `Instance` (records `n, m, seed, UB, LB` when present). |
| VRF reader | `pfsp/io/vrf.py` | Parse `VFR*_Gap.txt` files to an `Instance` (transposes to `(m, n)`). |
| Loader | `pfsp/io/loader.py` | `load_instance(path)`: detect format by name (fallback by content) and dispatch. |
| Makespan | `pfsp/core/makespan.py` | `makespan(matrix, permutation)`: vectorized completion-time recurrence; never mutates the input. |
| Insertion helper | `pfsp/core/insertion.py` | `best_insertion(...)`: shared mechanic used by NEH and IG reconstruction. |
| NEH | `pfsp/core/neh.py` | `neh(instance) -> (permutation, makespan)`: deterministic constructive baseline. |
| Iterated Greedy | `pfsp/core/ig.py` | `iterated_greedy(instance, seed, config)`: main loop, acceptance criterion, `IGConfig`, `StoppingCriterion`. |
| Destruction | `pfsp/core/destruction.py` | `destruct` (random), `idle_scores`, `build_rcl`, `make_idle_greedy_destruct` (O1) and `make_idle_rcl_destruct` (O2). All seeded. |
| Local search | `pfsp/core/local_search.py` | `local_search(...)`: insertion-based LS until no improvement. |
| Reference provider | `pfsp/results/reference.py` | `get_reference(instance)`: UB for Taillard, best-known CSV value for VRF. |
| Result record / RPD | `pfsp/results/record.py` | `ResultRecord`, `compute_rpd`, append-only persistence to `raw/<algorithm>.csv`. |
| Aggregator | `pfsp/results/aggregate.py` | `summarize` / `write_summary`: summary tables derived from the raw records. |
| Run manifest | `pfsp/results/manifest.py` | `build_manifest` / `write_manifest` + `capture_environment`. |
| Seed manager | `pfsp/repro/seeds.py` | `make_rng(seed)`: explicit-seed NumPy generator. |
| Campaign runner | `pfsp/experiments/runner.py` | `run_campaign`: resumable multi-seed engine with per-instance checkpoint, logging and sanity checks. |
| Analysis layer | `pfsp/analysis/` | RPD series and sweeps from raw CSV, Friedman + Nemenyi contrasts, figures. |
| CLI | `pfsp/cli.py`, `pfsp/cli_ig.py` | `pfsp-neh` and `pfsp-ig`, wiring loading → algorithm → RPD → record + manifest. |

## Reproducibility

- **Pinned dependencies** (`requirements.txt`), **explicit seed management**
  (`pfsp/repro/seeds.py`) and a defined raw-results format were established before any
  experimental logic, not added afterwards.
- **Raw data preserved; summaries derived.** Per-execution records are persisted
  individually; the summary tables are always computed by the aggregator, never
  hand-edited. This is what lets a table in the thesis be regenerated and audited.
- **Provenance.** Each run writes a manifest (instance, algorithm and parameters, seed,
  captured environment, timestamp) and every raw record carries its `manifest_id`, so a
  number traces back to its exact execution context. The manifests of the published
  campaigns (3.700+ JSON files) are not distributed here; new ones are produced under
  `results/manifests/` as soon as you run the CLI.
- **Resumable campaigns.** The engine persists each instance × seed as it finishes, so a
  long campaign can be interrupted and resumed without duplicating rows.

## Verification

```bash
pip install -r requirements-dev.txt
pytest tests           # unit + property-based suite
ruff check .           # linter
black --check .        # formatting
```

## Note on this distribution

This is the distribution of the thesis deliverable: the package, its tests, the
instances and the experimental evidence. 