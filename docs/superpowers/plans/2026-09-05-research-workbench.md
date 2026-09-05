# 研究项目工作台 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有策略回测之上提供可保存、可复现、可横向比较的本地研究项目工作台。

**Architecture:** 新增 `qfm.research` 作为独立的文件存储领域层。`app.py` 仅把成功回测的输入、摘要和表格产物交给该层，不改变 `synthesize` 或 `run_backtest` 的计算。研究页从存储层加载 manifest 与 CSV，用同一项目下的运行作对比。

**Tech Stack:** Python 3.13、pandas、Streamlit、Plotly、pytest、JSON 与 CSV 文件系统存储。

## Global Constraints

- 本期仅支持 A 股日频策略回测运行归档；不接入实盘、云同步、数据库或任务队列。
- 所有文件写入位于 `data_cache/research/`；不复制行情原始数据。
- 运行保存必须由用户在界面明确触发，不能自动写盘。
- 运行 manifest 必须包含格式版本、研究配置、数据快照、结果摘要和产物文件名。
- 对既有 `app.py` 的未提交改动只做最小追加，不格式化、不还原，也不将其与本功能无关的内容纳入提交。
- 新增单元测试不得访问网络或读取真实缓存行情。

---

### Task 1: 定义研究项目与运行的数据模型

**Files:**
- Create: `qfm/research/models.py`
- Create: `qfm/research/__init__.py`
- Create: `tests/test_research_models.py`

**Interfaces:**
- Produces: `ResearchProject`, `ResearchRun`, `LoadedResearchRun` dataclasses.
- Produces: `ResearchProject.to_dict()`, `ResearchProject.from_dict(data)`, `ResearchRun.to_dict()`, `ResearchRun.from_dict(data)`.
- Consumes: only standard library `dataclasses`, `datetime`, `typing` and pandas annotations.

- [ ] **Step 1: Write failing round-trip and validation tests**

```python
import pytest

from qfm.research.models import ResearchProject, ResearchRun


def test_project_round_trip_preserves_metadata():
    project = ResearchProject.create(name="价值+质量", description="月频选股")
    restored = ResearchProject.from_dict(project.to_dict())
    assert restored == project


def test_project_rejects_blank_or_oversized_name():
    with pytest.raises(ValueError, match="项目名称"):
        ResearchProject.create(name="   ")
    with pytest.raises(ValueError, match="80"):
        ResearchProject.create(name="x" * 81)


def test_run_round_trip_preserves_config_and_summary():
    run = ResearchRun.create(
        project_id="project_1", name="2024 月频", config={"top_n": 30},
        data_snapshot={"pool": "index800"}, summary={"夏普比率": 1.2},
        artifacts={"nav": "nav.csv"},
    )
    assert ResearchRun.from_dict(run.to_dict()) == run
```

- [ ] **Step 2: Run the model tests to verify they fail**

Run: `/opt/anaconda3/bin/python3 -m pytest tests/test_research_models.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'qfm.research'`.

- [ ] **Step 3: Implement immutable models and explicit schema validation**

```python
def _required_string(data: Mapping[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"缺少或非法字段: {key}")
    return value


@dataclass(frozen=True)
class ResearchProject:
    id: str
    name: str
    description: str
    created_at: str
    updated_at: str

    @classmethod
    def create(cls, name: str, description: str = "") -> "ResearchProject":
        cleaned = name.strip()
        if not cleaned or len(cleaned) > 80:
            raise ValueError("项目名称不能为空且最长 80 个字符")
        now = datetime.now(timezone.utc).isoformat()
        return cls(f"project_{uuid4().hex}", cleaned, description.strip(), now, now)

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ResearchProject":
        return cls(*(_required_string(data, key) for key in
                     ("id", "name", "description", "created_at", "updated_at")))
```

Implement `ResearchRun` with `format_version: int = 1`, IDs, timestamps, `config`, `data_snapshot`, `summary` and `artifacts`. Its `create` method must call `_required_string` for `project_id`, generate `run_<uuid4().hex>`, default a blank run name to `运行 <UTC ISO timestamp>` and reject name lengths over 80. Its `from_dict` must reject any `format_version != 1`, missing dictionaries and empty artifact file names. `LoadedResearchRun` contains a `ResearchRun` plus `nav`, `benchmark_nav`, `weights`, `yearly_performance` and `trades` pandas objects. Export all three types from `qfm/research/__init__.py`.

- [ ] **Step 4: Run the model tests to verify they pass**

Run: `/opt/anaconda3/bin/python3 -m pytest tests/test_research_models.py -q`

Expected: PASS.

- [ ] **Step 5: Commit only new model files and tests**

```bash
git add qfm/research/models.py qfm/research/__init__.py tests/test_research_models.py
git commit -m "feat: add research ledger models"
```

### Task 2: 实现原子化本地研究账本

**Files:**
- Create: `qfm/research/store.py`
- Create: `tests/test_research_store.py`
- Modify: `qfm/research/__init__.py`

**Interfaces:**
- Produces: `ResearchStore(root: Path)`.
- Produces: `create_project(name: str, description: str = "") -> ResearchProject`.
- Produces: `list_projects() -> list[ResearchProject]`, `get_project(project_id: str) -> ResearchProject`.
- Produces: `save_run(project_id: str, name: str, config: dict, data_snapshot: dict, summary: dict, nav: pd.Series, benchmark_nav: pd.Series, weights: pd.DataFrame, yearly_performance: pd.DataFrame, trades: pd.DataFrame) -> ResearchRun`.
- Produces: `list_runs(project_id: str) -> tuple[list[ResearchRun], list[str]]` and `load_run(run_id: str) -> LoadedResearchRun`.
- Consumes: the Task 1 models and pandas DataFrames/Series.

- [ ] **Step 1: Write failing store tests using `tmp_path`**

```python
import pandas as pd
import pytest


@pytest.fixture
def artifacts():
    index = pd.date_range("2026-01-01", periods=2, freq="D", name="date")
    nav = pd.Series([1.0, 1.02], index=index, name="nav")
    benchmark_nav = pd.Series([1.0, 1.01], index=index, name="benchmark_nav")
    weights = pd.DataFrame({"weight": [0.5, 0.5]}, index=pd.Index(["ep_ttm", "roe"], name="factor"))
    yearly = pd.DataFrame({"年份": [2026], "收益": [0.02]})
    trades = pd.DataFrame({"date": index[:1], "stock": ["600000"], "side": ["BUY"]})
    return nav, benchmark_nav, weights, yearly, trades


def test_store_round_trips_project_and_run(tmp_path, artifacts):
    nav, bench_nav, weights, yearly, trades = artifacts
    store = ResearchStore(tmp_path / "research")
    project = store.create_project("质量组合", "成本后月度回测")
    run = store.save_run(project.id, "2026-09-05", {"top_n": 30},
                         {"pool": "index800", "stocks": 799}, {"夏普比率": 1.1},
                         nav, bench_nav, weights, yearly, trades)
    loaded = store.load_run(run.id)
    assert loaded.run.project_id == project.id
    pd.testing.assert_series_equal(loaded.nav, nav, check_freq=False)
    pd.testing.assert_frame_equal(loaded.trades, trades)


def test_store_skips_corrupt_manifest_and_returns_warning(tmp_path):
    store = ResearchStore(tmp_path / "research")
    project = store.create_project("测试")
    (store.runs_dir / "broken").mkdir(parents=True)
    (store.runs_dir / "broken" / "manifest.json").write_text("{broken", encoding="utf-8")
    runs, warnings = store.list_runs(project.id)
    assert runs == []
    assert any("broken" in warning for warning in warnings)


def test_failed_write_never_exposes_final_run_directory(tmp_path, monkeypatch, artifacts):
    nav, bench_nav, weights, yearly, trades = artifacts
    store = ResearchStore(tmp_path / "research")
    project = store.create_project("测试")
    def raise_os_error(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(store, "_write_csv", raise_os_error)
    with pytest.raises(OSError):
        store.save_run(project.id, "失败", {}, {}, {}, nav, bench_nav, weights, yearly, trades)
    assert list(store.runs_dir.glob("run_*")) == []
```

- [ ] **Step 2: Run the store tests to verify they fail**

Run: `/opt/anaconda3/bin/python3 -m pytest tests/test_research_store.py -q`

Expected: FAIL with `ImportError` or `AttributeError` for `ResearchStore`.

- [ ] **Step 3: Implement storage and atomic artifact persistence**

Implement directories lazily with `Path.mkdir(parents=True, exist_ok=True)`. Write project JSON atomically through a `*.tmp` file and `Path.replace`. In `save_run`, write a UUID-named temporary directory under `runs/`, write every CSV with explicit index labels (`date` for Series, `factor` for weights), write `manifest.json` last, then rename the directory to `runs/<run_id>`. On an exception, remove only that known temporary directory and re-raise.

Use `pd.read_csv(nav_path, index_col="date", parse_dates=True)` for NAV artifacts and return a `LoadedResearchRun` with all five persisted tables. Sort valid runs by `created_at` descending. Do not silently swallow corrupt manifests: add their IDs and exception messages to the warning list.

- [ ] **Step 4: Run store and model tests to verify they pass**

Run: `/opt/anaconda3/bin/python3 -m pytest tests/test_research_models.py tests/test_research_store.py -q`

Expected: PASS.

- [ ] **Step 5: Commit only research storage files and their tests**

```bash
git add qfm/research/store.py qfm/research/__init__.py tests/test_research_store.py
git commit -m "feat: persist research projects and runs"
```

### Task 3: 将策略回测结果封装为可保存的运行载荷

**Files:**
- Modify: `app.py`
- Create: `tests/test_research_payload.py`

**Interfaces:**
- Produces: `build_strategy_run_payload(panel, pool: str, names: list[str], mode: str, horizon: int, weight_lookback: int, orthogonalize: bool, ortho_controls: tuple[str, ...], top_n: int, start_date: str, rebalance: str, bench_mode: str, costs: dict[str, float], max_participation: float, initial_capital: float, weights: dict[str, float], backtest: BacktestResult) -> dict[str, object]`.
- Consumes: `BacktestResult`, `perf_stats` and `yearly_perf` from `qfm.portfolio`.
- Produces: a JSON-safe config, data snapshot, summary, NAV, benchmark NAV, factor-weight table, annual-performance table and trade table.

- [ ] **Step 1: Write a failing payload test with a synthetic `BacktestResult`**

```python
import numpy as np
import pandas as pd


def test_strategy_payload_records_execution_assumptions(panel):
    index = panel.close.index[:3]
    columns = panel.close.columns
    nav = pd.Series([1.0, 1.01, 1.02], index=index)
    bench = pd.Series([1.0, 1.005, 1.01], index=index)
    holdings = pd.DataFrame(0.0, index=index, columns=columns)
    cash = pd.Series([1.0, 0.0, 0.0], index=index)
    trades = pd.DataFrame({"date": [index[1]], "stock": [columns[0]], "side": ["BUY"]})
    result = BacktestResult(nav=nav, bench_nav=bench, holdings=holdings,
                            trades=trades, turnover=3.2, cost_pct=0.001,
                            cost_total=0.02, cash_weight=cash, params={"execution": "next_open"})
    payload = build_strategy_run_payload(
        panel, "index800", ["ep_ttm", "roe"], "equal", 20, 252,
        False, (), 30, "2021-01-01", "ME", "equal",
        {"commission": 0.0003, "stamp": 0.0005, "impact": 0.001}, 0.05,
        1_000_000, {"ep_ttm": 0.5, "roe": 0.5}, result,
    )
    assert payload["data_snapshot"]["pool"] == "index800"
    assert payload["config"]["execution"] == "next_open"
    assert payload["summary"]["成交笔数"] == len(trades)
```

- [ ] **Step 2: Run the payload test to verify it fails**

Run: `/opt/anaconda3/bin/python3 -m pytest tests/test_research_payload.py -q`

Expected: FAIL because `build_strategy_run_payload` is not defined.

- [ ] **Step 3: Add the pure payload helper and store latest successful result**

Add the helper near existing non-UI helpers in `app.py`; it must contain no Streamlit calls. After `run_backtest` completes, create the payload and assign it to `st.session_state["latest_strategy_run"]`. Keep the existing result visualisation intact and source it from local variables during the run that created it.

The payload must record `panel.close.index.min()`, `panel.close.index.max()`, `panel.close.shape`, pool, factor names, all weighting/orthogonalisation fields, trade execution assumptions, cost parameters, `perf_stats`, turnover, costs, final NAV and trade count. Convert numpy scalar values to native `float`/`int` before passing them to `ResearchStore`.

- [ ] **Step 4: Run the payload test to verify it passes**

Run: `/opt/anaconda3/bin/python3 -m pytest tests/test_research_payload.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the payload test; defer `app.py` staging if it still contains unrelated dirty changes**

```bash
git add tests/test_research_payload.py
git commit -m "test: cover strategy research payload"
```

Do not run `git add app.py` unless its pre-existing dirty changes have been separately reviewed and intentionally included.

### Task 4: 增加研究项目页和显式保存动作

**Files:**
- Modify: `app.py`
- Create: `qfm/research/ui_state.py`
- Create: `tests/test_research_ui_smoke.py`

**Interfaces:**
- Consumes: `ResearchStore`, `ResearchProject`, `LoadedResearchRun` and `st.session_state["latest_strategy_run"]` from Tasks 1–3.
- Produces: `page_research(store: ResearchStore) -> None` and `render_run_details(loaded: LoadedResearchRun) -> None`.
- Produces: session key `active_project_id: str | None` and page section `研究项目`.

- [ ] **Step 1: Write a failing Streamlit-independent UI state test**

```python
import pandas as pd

from qfm.research.ui_state import can_save_strategy_run, set_active_project

def test_active_project_selection_is_preserved():
    state = {"active_project_id": None}
    set_active_project(state, "project_123")
    assert state["active_project_id"] == "project_123"


def test_save_disabled_without_project_or_result():
    assert can_save_strategy_run(active_project_id=None, payload={}) is False
    payload = {"nav": pd.Series([1.0])}
    assert can_save_strategy_run(active_project_id="project_123", payload=payload) is True
```

- [ ] **Step 2: Run the UI-state test to verify it fails**

Run: `/opt/anaconda3/bin/python3 -m pytest tests/test_research_ui_smoke.py -q`

Expected: FAIL because `set_active_project` and `can_save_strategy_run` are not defined.

- [ ] **Step 3: Implement navigation, project management, run comparison and detail views**

Create `qfm/research/ui_state.py` with the following two helpers and import them in both tests and `app.py`:

```python
from collections.abc import MutableMapping

import pandas as pd


def set_active_project(state: MutableMapping[str, object], project_id: str | None) -> None:
    state["active_project_id"] = project_id


def can_save_strategy_run(active_project_id: str | None, payload: dict[str, object] | None) -> bool:
    return bool(active_project_id and payload and isinstance(payload.get("nav"), pd.Series))
```

Add “研究项目” as the first sidebar option. Instantiate `ResearchStore(Path(__file__).parent / "data_cache" / "research")` once after imports. Render a project creation `st.form` with name and description; on success set `active_project_id`, show a success message and rerun. Use a selectbox to change active project without creating any project implicitly.

For the selected project, render its runs as a dataframe with config and performance columns; render each corrupt-record warning with `st.warning`. Offer a multiselect of saved run IDs, limit the Plotly comparison to 8 selected runs, load their saved `nav.csv`, and draw each curve by run name. A run details selectbox must show manifest JSON, weights, yearly performance, trades, and direct downloads of each saved CSV.

On the strategy page, show the active project badge. Once `latest_strategy_run` exists, show a run-name input and a “保存至当前研究项目” button. Disable it and explain why if no project is active. On click call `store.save_run(active_project_id, run_name, payload["config"], payload["data_snapshot"], payload["summary"], payload["nav"], payload["benchmark_nav"], payload["weights"], payload["yearly_performance"], payload["trades"])`; show the returned run ID and a button that switches `st.session_state["section"]` to `研究项目`. Catch `OSError` and `ValueError`, show the message, and retain the in-memory result for a retry.

- [ ] **Step 4: Run UI-state and all research unit tests**

Run: `/opt/anaconda3/bin/python3 -m pytest tests/test_research_models.py tests/test_research_store.py tests/test_research_payload.py tests/test_research_ui_smoke.py -q`

Expected: PASS.

- [ ] **Step 5: Review the app diff before any commit**

Run: `git diff -- app.py`

Expected: the diff contains only the new research-workbench imports, helpers, sidebar entry, saving controls and research page. If earlier user changes are interleaved, leave `app.py` unstaged and report that state rather than staging unrelated changes.

### Task 5: 回归验证与浏览器验收

**Files:**
- Modify: `README.md`
- Create: `scripts/verify_research_workbench.py`

**Interfaces:**
- Consumes: public `qfm.research` API and the localhost Streamlit UI.
- Produces: a zero-network smoke script that creates a temporary research root, writes a run and reads it back.

- [ ] **Step 1: Write the failing command-line smoke script**

```python
import pandas as pd
from pathlib import Path
from tempfile import TemporaryDirectory


def main() -> None:
    with TemporaryDirectory() as temp_dir:
        store = ResearchStore(Path(temp_dir) / "research")
        project = store.create_project("smoke")
        dates = pd.date_range("2026-01-01", periods=2, freq="D", name="date")
        nav = pd.Series([1.0, 1.01], index=dates)
        bench = pd.Series([1.0, 1.005], index=dates)
        weights = pd.DataFrame({"weight": [1.0]}, index=pd.Index(["ep_ttm"], name="factor"))
        yearly = pd.DataFrame({"年份": [2026], "收益": [0.01]})
        trades = pd.DataFrame({"date": [dates[1]], "stock": ["600000"], "side": ["BUY"]})
        run = store.save_run(project.id, "smoke", {}, {}, {}, nav, bench, weights, yearly, trades)
        assert store.load_run(run.id).run.id == run.id
```

- [ ] **Step 2: Run the smoke script before its dependencies are complete**

Run: `/opt/anaconda3/bin/python3 scripts/verify_research_workbench.py`

Expected: FAIL until Tasks 1–4 are complete.

- [ ] **Step 3: Complete the script and document user workflow**

Add a README section describing: create project → set active project → run strategy backtest → explicitly save → compare runs. State that saved runs retain result artifacts, not a copy of source market data, and are historical simulations rather than investment advice.

- [ ] **Step 4: Run all automated checks**

Run: `/opt/anaconda3/bin/python3 -m pytest tests/ -q && /opt/anaconda3/bin/python3 scripts/verify_research_workbench.py`

Expected: all tests PASS and the smoke script exits 0.

- [ ] **Step 5: Perform browser acceptance using the local server**

First inspect the local web-app helper:

Run: `python /Users/zxt/.agents/skills/webapp-testing/scripts/with_server.py --help`

Then use a native Python Playwright script against `http://localhost:8501` to verify the sidebar includes “研究项目”, a project can be created, and the empty-state guidance appears before a run is saved. Capture one screenshot and inspect browser console errors.

- [ ] **Step 6: Commit only clean, scoped files**

```bash
git add qfm/research tests/test_research_models.py tests/test_research_store.py \
  tests/test_research_payload.py tests/test_research_ui_smoke.py \
  scripts/verify_research_workbench.py README.md
git commit -m "feat: add reproducible research workbench"
```

Only include `app.py` if its entire staged diff has been reviewed and is known to contain no unrelated edits.

## Post-implementation optimization roadmap

1. **数据可信度（最高优先级）**：历史指数成分、退市股票、公告时点审计和数据质量面板，先解决幸存者偏差和点时可得性。
2. **组合约束**：行业/风格暴露上限、单票权重上限、换手预算和风险模型约束；把当前“诊断”升级为“构建时约束”。
3. **研究效率**：将单因子、表达式挖掘和 ML 合成也写入同一研究账本，支持同一数据快照下的完整 lineage 与实验比较。
4. **模型可靠性**：LightGBM 与 XGBoost 基线对比、滚动样本外评估、特征覆盖率报告和特征重要性稳定性。
5. **任务运行**：引入后台任务、进度、可取消任务和运行日志；仅在本地研究账本稳定后再做调度与通知。
6. **纸面交易再到实盘**：先生成每日目标权重和成交差异报告，确认回测/纸面交易语义一致后，才评估券商或 vn.py 接口。
