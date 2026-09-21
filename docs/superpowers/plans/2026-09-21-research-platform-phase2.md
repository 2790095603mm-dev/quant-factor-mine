# Research Platform Phase Two Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add versioned Dataset/Universe management, deterministic Jobs/Cache, Factor Compare, Multi-Factor Lab, and Strategy Compare without changing the numerical contracts of the existing Experiment, Factor Library, Factor Pipeline, or Backtest Engine.

**Architecture:** Add focused local-registry modules around the existing engines. Dataset/Universe identities and exact factor definitions become the reproducibility envelope; synchronous jobs persist lifecycle state and cache artifacts by a canonical request hash; analysis and UI layers call existing computation primitives through these wrappers.

**Tech Stack:** Python 3.13, Pandas, NumPy, Plotly, Streamlit, JSON/CSV/Parquet/pickle local artifacts, pytest, Streamlit AppTest.

## Global Constraints

- Preserve existing Experiment, Factor Library, Factor Pipeline, and Backtest Engine behavior and public entry points.
- Reuse `factor_report`, `synthesize`, `run_backtest`, `standard_metrics`, `ResearchStore`, and factor registration instead of duplicating their logic.
- Do not add AI factor generation, genetic algorithms, symbolic-regression expansion, React, a distributed queue, or cloud storage.
- Existing v1/v2 research records remain readable and are displayed as `legacy_unbound`; their dataset version is never inferred from current data.
- Every user-facing newly saved Experiment includes `dataset_id`, `dataset_version`, `universe_id`, and `universe_version`.
- Job statuses are exactly `PENDING`, `RUNNING`, `SUCCESS`, and `FAILED`.
- Cache identity includes factor versions, dataset version, universe identity/version, date range, pipeline config, backtest config, job type, and cache schema version.
- All file edits preserve unrelated dirty-worktree changes; only plan-owned hunks and new files are changed.

---

### Task 1: Deterministic Job and Cache Core

**Files:**
- Create: `qfm/jobs/__init__.py`
- Create: `qfm/jobs/models.py`
- Create: `qfm/jobs/store.py`
- Create: `qfm/jobs/service.py`
- Test: `tests/test_jobs.py`

**Interfaces:**
- Produces: `JobStatus`, `JobRecord`, `canonicalise(value)`, `build_cache_key(job_type, request)`, `JobStore`, and `JobService.run(job_type, request, compute)`.
- Consumers: Factor Compare, Multi-Factor Lab, Backtest UI, and Job Center.

- [ ] **Step 1: Write failing model and key tests**

```python
def test_cache_key_is_stable_for_reordered_dicts(tmp_path):
    left = build_cache_key("FACTOR_ANALYSIS", {"pipeline_config": {"b": 2, "a": 1}})
    right = build_cache_key("FACTOR_ANALYSIS", {"pipeline_config": {"a": 1, "b": 2}})
    assert left == right

def test_cache_key_changes_with_dataset_or_factor_version():
    base = full_request()
    assert build_cache_key("BACKTEST", base) != build_cache_key(
        "BACKTEST", {**base, "dataset_version": "dataset_changed"}
    )
```

- [ ] **Step 2: Run the model tests and confirm missing imports fail**

Run: `/opt/anaconda3/bin/python -m pytest -q tests/test_jobs.py`

- [ ] **Step 3: Implement immutable job records and canonical hashing**

```python
class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"

def build_cache_key(job_type: str, request: Mapping[str, Any]) -> str:
    payload = {"cache_schema_version": 1, "job_type": job_type, **canonicalise(request)}
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "cache_" + sha256(raw.encode("utf-8")).hexdigest()
```

- [ ] **Step 4: Write failing lifecycle/cache tests**

```python
def test_success_then_cache_hit_records_two_jobs(tmp_path):
    service = JobService(tmp_path)
    calls = []
    first = service.run("FACTOR_COMPUTE", full_request(), lambda: calls.append(1) or {"x": 1})
    second = service.run("FACTOR_COMPUTE", full_request(), lambda: calls.append(2) or {"x": 2})
    assert first.value == second.value == {"x": 1}
    assert calls == [1]
    assert first.job.cache_hit is False and second.job.cache_hit is True

def test_failure_persists_failed_job_without_cache(tmp_path):
    with pytest.raises(RuntimeError):
        JobService(tmp_path).run("BACKTEST", full_request(), lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert JobStore(tmp_path).list_jobs()[0].status == "FAILED"
```

- [ ] **Step 5: Implement atomic manifests and pickle result cache**

`JobService.run` creates PENDING, updates RUNNING, reads a valid existing result as SUCCESS/cache-hit, or calls `compute`, atomically writes the result, and updates SUCCESS. On exceptions it writes FAILED with the error and re-raises. Corrupt pickle files are treated as misses and replaced only after successful recomputation.

- [ ] **Step 6: Run job tests**

Run: `/opt/anaconda3/bin/python -m pytest -q tests/test_jobs.py`
Expected: all tests pass.

---

### Task 2: Versioned Dataset and Universe Catalog

**Files:**
- Create: `qfm/data/catalog.py`
- Create: `qfm/data/catalog_views.py`
- Modify: `qfm/data/universe.py`
- Modify: `qfm/data/__init__.py`
- Test: `tests/test_data_catalog.py`

**Interfaces:**
- Produces: `DatasetVersion`, `UniverseDefinition`, `DataCatalog`, `normalise_symbols`, `register_panel_dataset`, `render_data_catalog`.
- Consumes: `build_data_snapshot` fingerprints and existing `get_index_cons` / cached fundamentals.

- [ ] **Step 1: Write failing catalog contract tests**

```python
def test_dataset_version_is_content_stable(tmp_path, panel):
    catalog = DataCatalog(tmp_path)
    first = catalog.register_panel("cn_equity_daily", panel, snapshot(panel), source="akshare")
    second = catalog.register_panel("cn_equity_daily", panel, snapshot(panel), source="akshare")
    assert first.dataset_version == second.dataset_version
    assert set(first.to_dict()) >= {
        "dataset_id", "dataset_version", "source", "start_date", "end_date",
        "last_update", "symbols", "fields",
    }

def test_custom_universe_normalises_and_versions_symbols(tmp_path):
    universe = DataCatalog(tmp_path).create_custom_universe("自选池", ["1", "600000", "000001"])
    assert universe.symbols == ("000001", "600000")
    assert universe.universe_id.startswith("custom_")
```

- [ ] **Step 2: Confirm tests fail before implementation**

Run: `/opt/anaconda3/bin/python -m pytest -q tests/test_data_catalog.py`

- [ ] **Step 3: Implement dataclasses, fingerprints, and atomic JSON registries**

```python
@dataclass(frozen=True)
class DatasetVersion:
    dataset_id: str
    dataset_version: str
    source: str
    start_date: str
    end_date: str
    last_update: str
    symbols: tuple[str, ...]
    fields: tuple[str, ...]
    market_fingerprint: str = ""
    fundamental_fingerprint: str = ""

@dataclass(frozen=True)
class UniverseDefinition:
    universe_id: str
    universe_version: str
    name: str
    source: str
    symbols: tuple[str, ...]
```

`DataCatalog` stores datasets and universes under its configured root using temporary-file replacement and tolerates malformed individual entries with warnings.

- [ ] **Step 4: Extend universe resolution without breaking old aliases**

Add `cn_hs300`, `cn_zz500`, `cn_zz1000`, and `cn_all_a`. Preserve `index800` and `full`. Custom universe IDs resolve from `DataCatalog`; unknown IDs raise a clear `ValueError`.

- [ ] **Step 5: Add the catalog Streamlit page**

Render built-in and custom universe tables, a custom-symbol text area, CSV/TXT uploader, create action, and Dataset version table. Invalid inputs stay on the page with actionable errors.

- [ ] **Step 6: Run catalog and existing universe tests**

Run: `/opt/anaconda3/bin/python -m pytest -q tests/test_data_catalog.py tests/test_adjustment_factor.py`
Expected: all tests pass.

---

### Task 3: Bind New Experiments to Dataset and Universe Versions

**Files:**
- Modify: `qfm/research/models.py`
- Modify: `qfm/research/snapshot.py`
- Modify: `qfm/research/payload.py`
- Modify: `qfm/research/store.py`
- Modify: `qfm/research/replay.py`
- Modify: `qfm/research/views.py`
- Modify: `qfm/simulation/views.py`
- Test: `tests/test_research_dataset_binding.py`
- Modify tests: `tests/test_research_models.py`, `tests/test_research_payload.py`, `tests/test_research_reopen.py`

**Interfaces:**
- Produces: Research format v3 bindings in `data_snapshot`; `ResearchRun.dataset_binding()`; legacy `legacy_unbound` display.
- Consumes: `DataCatalog.register_panel` and current payload generation.

- [ ] **Step 1: Write failing payload-binding tests**

```python
def test_new_payload_contains_exact_dataset_and_universe_binding(panel, tmp_path):
    payload = build_strategy_run_payload(..., catalog_root=tmp_path, universe_id="cn_hs300")
    snapshot = payload["data_snapshot"]
    assert snapshot["dataset_id"]
    assert snapshot["dataset_version"].startswith("dataset_")
    assert snapshot["universe_id"] == "cn_hs300"
    assert snapshot["universe_version"].startswith("universe_")

def test_v2_run_loads_as_legacy_unbound():
    run = ResearchRun.from_dict(v2_manifest())
    assert run.legacy_unbound is True
```

- [ ] **Step 2: Run focused tests and capture failures**

Run: `/opt/anaconda3/bin/python -m pytest -q tests/test_research_dataset_binding.py`

- [ ] **Step 3: Upgrade the model format compatibly**

Set `FORMAT_VERSION = 3`, keep `(1, 2, 3)` readable, and expose binding helpers without changing old artifact loading. A run is legacy-unbound when any required binding field is absent.

- [ ] **Step 4: Attach catalog bindings in payload construction**

`build_strategy_run_payload` receives optional `catalog_root` and `universe_id`; user-facing callers provide both. It registers the panel Dataset using the already computed data snapshot and adds all four identifiers before save.

- [ ] **Step 5: Guard replay and render binding status**

Bound runs show Dataset/Universe chips and replay normally. Legacy-unbound runs remain viewable/exportable; replay UI is disabled with a message requiring explicit rebinding rather than silently using current data.

- [ ] **Step 6: Run research compatibility tests**

Run: `/opt/anaconda3/bin/python -m pytest -q tests/test_research_dataset_binding.py tests/test_research_models.py tests/test_research_payload.py tests/test_research_reopen.py tests/test_research_store.py tests/test_research_views.py`
Expected: all tests pass.

---

### Task 4: Factor Compare and Correlation Analysis

**Files:**
- Create: `qfm/analysis/__init__.py`
- Create: `qfm/analysis/factor_compare.py`
- Create: `qfm/analysis/factor_views.py`
- Test: `tests/test_factor_compare.py`
- Test: `tests/test_factor_compare_view.py`

**Interfaces:**
- Produces: `FactorCompareResult`, `compare_factors(panel, names, horizon, start, end, pipeline_config)`, `cross_sectional_corr`, `high_correlation_pairs`, `render_factor_compare`.
- Consumes: `compute_factor`, `factor_report`, `run_pipeline`, `compute_ic`, and `JobService`.

- [ ] **Step 1: Write failing metric and correlation tests**

```python
def test_compare_factors_returns_all_required_metrics(wide_panel):
    result = compare_factors(wide_panel, ["mom_20", "rev_20"], horizon=20)
    assert set(result.metrics.columns) >= {
        "factor", "ic", "rank_ic", "icir", "long_short_return",
        "turnover", "coverage", "stability",
    }

def test_spearman_detects_monotonic_duplicate():
    a = frame_with_cross_section([1, 2, 3, 4])
    b = frame_with_cross_section([1, 4, 9, 16])
    result = cross_sectional_corr({"a": a, "b": b}, method="spearman")
    assert result.loc["a", "b"] == pytest.approx(1.0)
```

- [ ] **Step 2: Confirm tests fail**

Run: `/opt/anaconda3/bin/python -m pytest -q tests/test_factor_compare.py`

- [ ] **Step 3: Implement one-pipeline-pass factor comparison**

For each selected factor, compute its raw frame once, call the existing pipeline once, calculate Pearson IC and Rank IC against the same forward-return matrix, reuse `factor_report(..., preprocessed=True)` for layers/turnover, and derive stability as the positive Rank-IC share.

- [ ] **Step 4: Implement both correlation matrices and duplicate pairs**

```python
def high_correlation_pairs(matrix: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    rows = [
        {"factor_a": a, "factor_b": b, "correlation": matrix.loc[a, b]}
        for i, a in enumerate(matrix.index)
        for b in matrix.columns[i + 1:]
        if abs(matrix.loc[a, b]) > threshold
    ]
    return pd.DataFrame(rows).sort_values("correlation", key=lambda s: s.abs(), ascending=False)
```

- [ ] **Step 5: Add Streamlit result rendering through FACTOR_ANALYSIS Job**

The page selects 2–12 factors and renders the metric table, Pearson heatmap, Spearman heatmap, and a prominent red-marked duplicate table. Job requests include exact factor versions and catalog binding.

- [ ] **Step 6: Run calculation and AppTest suites**

Run: `/opt/anaconda3/bin/python -m pytest -q tests/test_factor_compare.py tests/test_factor_compare_view.py`
Expected: all tests pass.

---

### Task 5: Multi-Factor Lab and Persistent Composite Factors

**Files:**
- Modify: `qfm/portfolio/synthesis.py`
- Create: `qfm/multifactor/__init__.py`
- Create: `qfm/multifactor/models.py`
- Create: `qfm/multifactor/registry.py`
- Create: `qfm/multifactor/views.py`
- Modify: `qfm/factors/base.py`
- Modify: `qfm/factors/__init__.py`
- Test: `tests/test_multifactor_lab.py`
- Test: `tests/test_multifactor_registry.py`
- Test: `tests/test_multifactor_view.py`

**Interfaces:**
- Produces: `mode="ic_x_ir"` in `synthesize`; `CompositeDefinition`; `CompositeRegistry.save/load/register_all`; `render_multifactor_lab`.
- Consumes: factor registry/version APIs, `synthesize`, and `JobService`.

- [ ] **Step 1: Write failing IC×IR and no-lookahead tests**

```python
def test_ic_x_ir_weights_are_normalised(panel):
    score, weights = synthesize(panel, ["mom_20", "roe"], mode="ic_x_ir", horizon=20)
    history = score.attrs["weight_history"]
    assert np.allclose(history.sum(axis=1), 1.0)

def test_ic_x_ir_history_before_cutoff_ignores_future_prices(panel):
    original, _ = synthesize(panel, NAMES, mode="ic_x_ir", horizon=20)
    changed = replace_future_prices(panel, cutoff=180)
    mutated, _ = synthesize(changed, NAMES, mode="ic_x_ir", horizon=20)
    pd.testing.assert_frame_equal(original.iloc[:160], mutated.iloc[:160])
```

- [ ] **Step 2: Implement `ic_x_ir` as a walk-forward mode**

For each factor/window use `mean_ic * (mean_ic / std_ic)`, clip negative contributions to zero, normalize, and fall back to equal weights exactly like existing modes.

- [ ] **Step 3: Write failing persistence/reload/cycle tests**

```python
def test_saved_composite_reloads_into_factor_library(panel, tmp_path):
    registry = CompositeRegistry(tmp_path)
    saved = registry.save(definition("value_quality", ["bp", "roe"]))
    clear_runtime_factor("value_quality")
    registry.register_all()
    assert get_factor("value_quality").family == "多因子"
    assert not compute_factor("value_quality", panel).empty

def test_registry_rejects_direct_and_indirect_cycles(tmp_path):
    registry = CompositeRegistry(tmp_path)
    registry.save(definition("a", ["mom_20"]))
    registry.save(definition("b", ["a"]))
    with pytest.raises(ValueError, match="循环"):
        registry.save(definition("a", ["b"]))
```

- [ ] **Step 4: Implement atomic composite registry and runtime factories**

Each definition carries component factor version/hash, weighting parameters, source hash, and version. The runtime closure calls `synthesize` with the stored configuration. `source_key` is the canonical definition JSON so existing factor versioning remains authoritative.

- [ ] **Step 5: Register saved composites during factor package startup**

Load after built-in families are registered. Missing components or version drift do not crash startup; keep the definition, skip unsafe registration, and expose warnings to the Lab page.

- [ ] **Step 6: Add Multi-Factor Lab UI through MULTI_FACTOR Job**

The page selects factors, weight mode, horizon/lookback/rebalance/orthogonalization, previews latest and historical weights, runs a composite score, and requires a name/description before saving into Factor Library.

- [ ] **Step 7: Run Multi-Factor and existing synthesis tests**

Run: `/opt/anaconda3/bin/python -m pytest -q tests/test_multifactor_lab.py tests/test_multifactor_registry.py tests/test_multifactor_view.py tests/test_synthesis.py tests/test_pipeline_no_lookahead.py`
Expected: all tests pass.

---

### Task 6: Strategy Compare

**Files:**
- Create: `qfm/analysis/strategy_compare.py`
- Create: `qfm/analysis/strategy_views.py`
- Modify: `qfm/research/store.py`
- Test: `tests/test_strategy_compare.py`
- Test: `tests/test_strategy_compare_view.py`

**Interfaces:**
- Produces: `StrategyCompareResult`, `compare_strategies(loaded_runs)`, `ResearchStore.list_all_runs()`, and `render_strategy_compare`.
- Consumes: saved `LoadedResearchRun`, `standard_metrics`, and `drawdown`.

- [ ] **Step 1: Write failing cross-project comparison tests**

```python
def test_strategy_compare_has_required_metrics_and_curves(saved_runs):
    result = compare_strategies(saved_runs)
    assert set(result.metrics.columns) >= {
        "strategy", "annual_return", "excess_return", "sharpe",
        "max_drawdown", "calmar", "turnover",
    }
    assert set(result.nav.columns) == set(result.drawdown.columns)
    assert set(result.excess.columns) == set(result.nav.columns)
```

- [ ] **Step 2: Confirm tests fail**

Run: `/opt/anaconda3/bin/python -m pytest -q tests/test_strategy_compare.py`

- [ ] **Step 3: Add all-project read API without changing existing list behavior**

`ResearchStore.list_all_runs()` returns `(project, run)` pairs while retaining `list_runs(project_id)` unchanged.

- [ ] **Step 4: Implement metric and curve assembly**

Normalize each saved strategy and benchmark to 1 at its own first valid date. Excess curve is `strategy_nav / benchmark_nav - 1` after inner alignment. Drawdown uses the existing helper. Metrics use each full saved series and its matching benchmark.

- [ ] **Step 5: Build the Strategy Compare page**

Allow 2–8 completed runs across projects. Show version-binding columns, the metric table, strategy+benchmark cumulative chart, excess chart, and drawdown chart. Different benchmarks retain distinct names; identical benchmark content is deduplicated by fingerprint.

- [ ] **Step 6: Run strategy comparison and research tests**

Run: `/opt/anaconda3/bin/python -m pytest -q tests/test_strategy_compare.py tests/test_strategy_compare_view.py tests/test_research_store.py tests/test_research_views.py`
Expected: all tests pass.

---

### Task 7: Integrate Navigation, Backtest Jobs, and Job Center

**Files:**
- Create: `qfm/jobs/views.py`
- Modify: `app.py`
- Modify: `qfm/simulation/views.py`
- Modify: `README.md`
- Test: `tests/test_phase2_pages_smoke.py`
- Modify: `tests/test_factor_simulation_ui.py`
- Modify: `tests/test_pages_smoke.py`

**Interfaces:**
- Produces: sidebar routes for Factor Compare, Multi-Factor Lab, Strategy Compare, Data/Universe, and Job Center; BACKTEST and FACTOR_ANALYSIS job-wrapped execution.
- Consumes: page renderers and `JobService` from Tasks 1–6.

- [ ] **Step 1: Write failing navigation/page smoke tests**

```python
def test_phase2_navigation_is_present():
    app = AppTest.from_file("app.py", default_timeout=90).run()
    options = app.radio(key="section").options
    assert {"因子对比", "多因子实验室", "策略对比", "数据与股票池", "任务中心"} <= set(options)

def test_job_center_renders_statuses(tmp_path):
    seed_jobs(tmp_path, ["PENDING", "RUNNING", "SUCCESS", "FAILED"])
    app = run_job_view(tmp_path)
    assert not app.exception
    assert all(status in rendered_text(app) for status in ["PENDING", "RUNNING", "SUCCESS", "FAILED"])
```

- [ ] **Step 2: Add page routes with lazy panel loading**

Catalog, Strategy Compare, and Job Center must render without loading 799-stock market data. Factor Compare and Multi-Factor Lab load the selected panel only when those pages are selected.

- [ ] **Step 3: Wrap user-facing analysis and backtest buttons**

Construct the complete cache request before execution. On cache hit render the same result plus a “已读取缓存” caption. On failed jobs keep the existing page usable and surface the error.

- [ ] **Step 4: Add Job Center**

Render status counts, recent jobs, type/status filters, cache-hit marker, duration, compact request JSON, and errors. Do not add job cancellation because jobs are synchronous in this phase.

- [ ] **Step 5: Update README**

Document the five new capabilities, storage locations, cache identity, legacy-unbound behavior, and the explicit non-goals.

- [ ] **Step 6: Run UI and compatibility smoke tests**

Run: `/opt/anaconda3/bin/python -m pytest -q tests/test_phase2_pages_smoke.py tests/test_factor_simulation_ui.py tests/test_pages_smoke.py`
Expected: all tests pass.

---

### Task 8: Full Regression and Live Verification

**Files:**
- Modify: `scripts/verify_phase1.py`
- Create: `scripts/verify_phase2.py`
- Test: `tests/test_verify_phase2_script.py`

**Interfaces:**
- Produces: an offline zero-network Phase 2 verification command.
- Consumes: all public interfaces from Tasks 1–7.

- [ ] **Step 1: Add an offline verification script test**

```python
def test_phase2_verifier_passes(capsys):
    module = load_verify_phase2()
    assert module.main([]) == 0
    output = capsys.readouterr().out
    assert "0 项失败" in output
    for label in ("Job/Cache", "Dataset/Universe", "Factor Compare", "Multi-Factor", "Strategy Compare"):
        assert label in output
```

- [ ] **Step 2: Implement Phase 2 synthetic verification**

Use the existing synthetic panel and a temporary root. Verify one cache hit, one custom universe, one bound experiment, both correlation modes/high-correlation detection, all four multi-factor modes, composite reload, and multi-strategy curves.

- [ ] **Step 3: Run targeted Phase 2 verification**

Run: `/opt/anaconda3/bin/python scripts/verify_phase2.py`
Expected: all checks pass and report `0 项失败`.

- [ ] **Step 4: Run static checks and the complete test suite**

Run: `git diff --check`

Run: `/opt/anaconda3/bin/python -m compileall -q app.py qfm scripts`

Run: `/opt/anaconda3/bin/python -m pytest -q`

Expected: zero failures; pre-existing numerical/deprecation warnings may remain but no new warning class is introduced.

- [ ] **Step 5: Restart the LaunchAgent and verify localhost**

Run: `launchctl kickstart -k gui/$(id -u)/com.zxt.quant-factor-mine`

Run: `curl -fsS http://localhost:8501/_stcore/health`

Expected: `ok`.

- [ ] **Step 6: Perform browser acceptance checks**

Open `http://localhost:8501/` and verify: all old routes open; Factor Compare produces tables/heatmaps; Multi-Factor saves and reappears in Factor Library; Catalog creates a custom universe; Strategy Compare renders saved experiments; Job Center shows SUCCESS/cache hits; no page displays an exception.

- [ ] **Step 7: Inspect only fresh service logs**

Record the log line before restart, inspect only later lines for `Traceback`, `ImportError`, `NameError`, `KeyError`, `AttributeError`, and `TypeError`, and resolve any new issue through the systematic-debugging workflow.

## Self-Review

- Spec coverage: all five user requirements map to Tasks 1–7; Task 8 covers compatibility and live acceptance.
- Placeholder scan: no deferred implementation placeholders remain.
- Type consistency: Dataset/Universe bindings, Job request keys, factor comparison outputs, composite definitions, and strategy comparison outputs use one name throughout the plan.
- Compatibility: old aliases and v1/v2 research formats remain readable; existing engine functions are reused, not replaced.

## Execution Choice

The user explicitly requested execution in this task. Use inline execution because subagent delegation was not requested and current collaboration rules prohibit unsolicited subagents. Execute Tasks 1–8 in order with focused test checkpoints.
