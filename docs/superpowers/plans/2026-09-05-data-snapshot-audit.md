# 数据快照审计 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为保存的策略回测提供可复现的数据审计快照，记录实际样本、覆盖率、数据来源口径和确定性内容指纹。

**Architecture:** 新增独立的 `qfm.research.snapshot` 纯计算模块，读取 `DataPanel` 并返回仅含 JSON 标准类型的字典；策略载荷调用它，研究项目页以向后兼容方式展示它。原始数据仍由既有缓存管理，研究账本只保存小型审计元数据而不复制行情。

**Tech Stack:** Python 3.13、pandas、hashlib、Streamlit、pytest、JSON/CSV 本地研究账本。

## Global Constraints

- 只在成功回测后构建会话内快照，只有用户点击保存才写入 `data_cache/research/`。
- 不改变任何因子、回测、交易成本或收益计算；不访问网络、不写入行情缓存、不复制原始行情。
- 保持 `ResearchRun` 格式版本 1；新增字段是 `data_snapshot` 的向后兼容扩展，旧 manifest 必须继续可读。
- 指纹必须确定性地包含字段边界、日期、代码和值；同面板相同，纳入字段的值变化会影响对应指纹。
- 所有新增单元测试使用已有合成 `panel` 夹具或临时数据，不能依赖真实缓存或网络。
- 不暂存或提交现有 `app.py` 等用户未提交改动；本期文档和纯新增模块可独立提交。

---

### Task 1: 构建纯数据快照与内容指纹

**Files:**
- Create: `qfm/research/snapshot.py`
- Create: `tests/test_research_snapshot.py`

**Interfaces:**
- Produces: `build_data_snapshot(panel: DataPanel, pool: str) -> dict[str, object]`。
- Consumes: `DataPanel.close` 作为日期与证券样本基准；可选的 OHLCV、成交额、换手率、流通市值、行业与 `fund` 面板。
- Produces: 兼容的顶层 `pool`、`stocks`、`trading_days`、`data_start`、`data_end`，以及 `snapshot_version`、`stock_codes`、`universe_fingerprint`、`sources`、`coverage`、`fingerprints` 和 `quality_warnings`。

- [ ] **Step 1: 写出确定性、隔离变化与缺失数据的失败测试**

```python
from copy import deepcopy

import numpy as np

from qfm.research.snapshot import build_data_snapshot


def test_snapshot_is_deterministic_and_records_actual_sample(panel):
    first = build_data_snapshot(panel, "index800")
    second = build_data_snapshot(panel, "index800")

    assert first == second
    assert first["snapshot_version"] == 2
    assert first["stocks"] == panel.close.shape[1]
    assert first["stock_codes"] == sorted(panel.close.columns.tolist())
    assert first["coverage"]["close"] == 1.0
    assert first["universe_fingerprint"].startswith("sha256:")
    assert first["fingerprints"]["market"].startswith("sha256:")


def test_snapshot_fingerprints_change_only_for_the_relevant_data_family(panel):
    base = build_data_snapshot(panel, "index800")
    market_changed = deepcopy(panel)
    market_changed.close.iloc[0, 0] += 0.01
    market = build_data_snapshot(market_changed, "index800")
    fund_changed = deepcopy(panel)
    fund_changed.fund["roe"].iloc[0, 0] += 0.01
    fundamentals = build_data_snapshot(fund_changed, "index800")

    assert market["fingerprints"]["market"] != base["fingerprints"]["market"]
    assert market["fingerprints"]["fundamentals"] == base["fingerprints"]["fundamentals"]
    assert fundamentals["fingerprints"]["market"] == base["fingerprints"]["market"]
    assert fundamentals["fingerprints"]["fundamentals"] != base["fingerprints"]["fundamentals"]


def test_snapshot_reports_missing_optional_data_without_network(panel):
    sparse = deepcopy(panel)
    sparse.open.iloc[:, :] = np.nan
    sparse.amount.iloc[:, :] = np.nan
    sparse.industry = sparse.industry.iloc[:0, :0]
    sparse.fund = {}

    snapshot = build_data_snapshot(sparse, "index800")

    assert snapshot["coverage"]["open"] == 0.0
    assert snapshot["coverage"]["amount"] == 0.0
    assert snapshot["coverage"]["industry"] == 0.0
    assert snapshot["coverage"]["fundamentals"] == {}
    assert any("开盘价" in warning for warning in snapshot["quality_warnings"])
    assert any("财务" in warning for warning in snapshot["quality_warnings"])
```

- [ ] **Step 2: 运行新测试，确认因模块不存在而失败**

Run: `/opt/anaconda3/bin/python3 -m pytest tests/test_research_snapshot.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'qfm.research.snapshot'`.

- [ ] **Step 3: 实现最小且有边界的快照模块**

```python
MARKET_FIELDS = ("open", "high", "low", "close", "volume", "amount", "turnover", "mv_float")


def build_data_snapshot(panel: DataPanel, pool: str) -> dict[str, object]:
    if panel.close.empty:
        raise ValueError("无法构建数据快照：收盘价面板为空")
    dates = pd.DatetimeIndex(panel.close.index)
    codes = sorted(_normalise_code(code) for code in panel.close.columns)
    coverage = _coverage_summary(panel, dates, codes)
    return {
        "snapshot_version": 2,
        "pool": str(pool),
        "stocks": len(codes),
        "trading_days": len(dates),
        "data_start": dates.min().isoformat(),
        "data_end": dates.max().isoformat(),
        "stock_codes": codes,
        "universe_fingerprint": _fingerprint_universe(dates, codes),
        "sources": {"market": "akshare:sina-qfq", "fundamentals": "akshare:eastmoney-yjbb"},
        "coverage": coverage,
        "fingerprints": {
            "market": _fingerprint_market(panel, dates, codes),
            "fundamentals": _fingerprint_fundamentals(panel, dates, codes),
        },
        "quality_warnings": _quality_warnings(coverage),
    }
```

Implement `_normalise_code` with string conversion plus `zfill(6)` only for numeric codes up to six digits. Implement a hasher initialized with a format label; update it with each field name and `pd.util.hash_pandas_object` row hashes for a frame reindexed to the canonical date/code axes. Hash market fields and fundamental/industry fields in separate hashers. For an absent optional frame, hash its field name plus an explicit `absent` marker and report zero coverage. Return every hash as `sha256:<hex>`. Calculate coverage as non-null cells divided by the close-grid size, nested under `coverage["fundamentals"]` for sorted fund names. Return Chinese warnings for low close/open/amount coverage and absent industry/fundamental fields.

- [ ] **Step 4: 运行快照测试，确认通过**

Run: `/opt/anaconda3/bin/python3 -m pytest tests/test_research_snapshot.py -q`

Expected: PASS with 3 tests.

- [ ] **Step 5: 提交独立快照模块与测试**

```bash
git add qfm/research/snapshot.py tests/test_research_snapshot.py
git commit -m "feat: add research data snapshot audit"
```

### Task 2: 将审计快照接入策略研究载荷

**Files:**
- Modify: `qfm/research/payload.py:36-89`
- Modify: `qfm/research/__init__.py`
- Modify: `tests/test_research_payload.py`

**Interfaces:**
- Consumes: `build_data_snapshot(panel, pool)` from `qfm.research.snapshot`。
- Produces: the existing payload-builder result with the full version-2 snapshot under `payload["data_snapshot"]`; its public function signature is unchanged.
- Preserves: all existing config, summary and artifact output fields and their JSON-safe conversion.

- [ ] **Step 1: 扩展现有载荷测试，断言可持久化审计字段**

```python
def test_strategy_payload_records_execution_assumptions(panel):
    # 保留现有 BacktestResult 构造与 build_strategy_run_payload 调用。
    snapshot = payload["data_snapshot"]
    assert snapshot["pool"] == "index800"
    assert snapshot["stocks"] == panel.close.shape[1]
    assert snapshot["snapshot_version"] == 2
    assert snapshot["stock_codes"][0] == "600000"
    assert snapshot["fingerprints"]["market"].startswith("sha256:")
    assert snapshot["coverage"]["fundamentals"]["roe"] == 1.0
```

- [ ] **Step 2: 运行载荷测试，确认新增字段尚不存在而失败**

Run: `/opt/anaconda3/bin/python3 -m pytest tests/test_research_payload.py -q`

Expected: FAIL with `KeyError: 'snapshot_version'`.

- [ ] **Step 3: 用快照构建器替换载荷中的手写简略字典，并导出接口**

```python
from qfm.research.snapshot import build_data_snapshot


data_snapshot = build_data_snapshot(panel, pool)
return {
    "config": _json_safe(config),
    "data_snapshot": _json_safe(data_snapshot),
    "summary": _json_safe(summary),
    "nav": backtest.nav.rename("nav").copy(),
    "benchmark_nav": backtest.bench_nav.rename("benchmark_nav").copy(),
    "weights": factor_weights,
    "yearly_performance": yearly_perf(backtest.nav),
    "trades": backtest.trades.copy(),
}
```

In `qfm/research/__init__.py`, export `build_data_snapshot`. Do not modify `ResearchRun.FORMAT_VERSION` or the `ResearchStore` read/write format.

- [ ] **Step 4: 运行载荷和快照测试，确认通过**

Run: `/opt/anaconda3/bin/python3 -m pytest tests/test_research_snapshot.py tests/test_research_payload.py -q`

Expected: PASS with 4 tests.

- [ ] **Step 5: 提交纯研究层接入与测试**

```bash
git add qfm/research/payload.py qfm/research/__init__.py tests/test_research_payload.py
git commit -m "feat: record data audit in research payloads"
```

### Task 3: 在研究项目页审核新旧数据快照

**Files:**
- Modify: `qfm/research/views.py:132-137`
- Modify: `README.md:86-96`
- Modify: `scripts/verify_research_workbench.py`

**Interfaces:**
- Consumes: a saved run's `data_snapshot`, including optional `snapshot_version`, `coverage`, `fingerprints` and `quality_warnings`.
- Produces: `_render_data_snapshot(snapshot: dict[str, Any]) -> None`, a Streamlit-only display helper.
- Preserves: existing full JSON detail and all run comparison/download controls.

- [ ] **Step 1: 为页面抽取快照呈现帮助函数**

```python
def _render_data_snapshot(snapshot: dict[str, Any]) -> None:
    if snapshot.get("snapshot_version") != 2:
        st.info("该运行未记录扩展数据审计信息；下次重新运行并保存后可查看覆盖率与数据指纹。")
        st.json(snapshot)
        return
    coverage = snapshot.get("coverage", {})
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("样本证券", f"{snapshot.get('stocks', '—')}")
    col_b.metric("交易日", f"{snapshot.get('trading_days', '—')}")
    col_c.metric("收盘覆盖", f"{float(coverage.get('close', 0.0)):.1%}")
    col_d.metric("成交额覆盖", f"{float(coverage.get('amount', 0.0)):.1%}")
    for warning in snapshot.get("quality_warnings", []):
        st.warning(warning)
    with st.expander("完整数据快照"):
        st.json(snapshot)
```

Place the helper above `page_research` and call it inside the existing `配置与数据快照` expander. Also show short universe and market fingerprints when present. Use `.get` defaults for all optional values so a manually edited/incomplete manifest renders a notice rather than crashing. Do not add a new persistence write path.

- [ ] **Step 2: 更新零网络账本冒烟脚本，验证扩展快照能往返保存**

```python
data_snapshot = {
    "snapshot_version": 2, "pool": "index800", "stocks": 1, "trading_days": 2,
    "coverage": {"close": 1.0, "amount": 1.0, "fundamentals": {}},
    "fingerprints": {"market": "sha256:test", "fundamentals": "sha256:test"},
    "quality_warnings": [],
}
run = store.save_run(
    project.id, "smoke", {"top_n": 30}, data_snapshot, {"夏普比率": 1.0},
    nav, benchmark_nav, weights, yearly, trades,
)
loaded = store.load_run(run.id)
assert loaded.run.data_snapshot["snapshot_version"] == 2
```

- [ ] **Step 3: 更新 README 中研究项目说明**

Replace the existing claim that each save writes only “股票池/日期快照” with wording that it also saves actual stock-code sample, coverage and deterministic data fingerprints. State that hashes detect a changed recomputation input but do not replace raw-data backup or point-in-time index constituent history.

- [ ] **Step 4: 运行纯测试、冒烟和静态校验**

Run:

```bash
/opt/anaconda3/bin/python3 -m py_compile qfm/research/snapshot.py qfm/research/payload.py qfm/research/views.py scripts/verify_research_workbench.py
/opt/anaconda3/bin/python3 -m pytest tests/test_research_snapshot.py tests/test_research_payload.py tests/test_research_models.py tests/test_research_store.py tests/test_research_ui_state.py -q
/opt/anaconda3/bin/python3 scripts/verify_research_workbench.py
git diff --check
```

Expected: all named tests pass, the smoke output is `research workbench smoke: ok`, and `git diff --check` produces no output.

- [ ] **Step 5: 提交不重叠的说明与冒烟脚本；视现有脏改动决定是否暂存 views.py/README**

```bash
git add scripts/verify_research_workbench.py
git commit -m "test: cover audited research snapshot"
```

Before staging `qfm/research/views.py` or `README.md`, inspect `git diff` and stage only if the file contains no unrelated user changes. Never stage `app.py` in this task.

### Task 4: 完整回归与本地 UI 验收

**Files:**
- Verify only: `app.py`, `qfm/research/views.py`, `qfm/research/payload.py`, full test suite.

**Interfaces:**
- Consumes: local Streamlit process at `http://localhost:8501` and its existing Research Project entry.
- Produces: verification evidence only; no external data fetch and no mutation of the user's main research store.

- [ ] **Step 1: 运行全量测试与服务健康检查**

Run:

```bash
/opt/anaconda3/bin/python3 -m pytest tests/ -q
curl -fsS http://localhost:8501/_stcore/health
```

Expected: every test passes and health endpoint prints `ok`.

- [ ] **Step 2: 用临时研究目录启动隔离 Streamlit 实例并检查页面降级路径**

```bash
QFM_RESEARCH_ROOT="$(mktemp -d)" /opt/anaconda3/bin/python /opt/anaconda3/bin/streamlit run app.py --server.headless true --server.port 8502
```

Open `http://localhost:8502` with the available UI automation. Create a project, then verify the research page still shows its empty-state guidance. Stop only that temporary port-8502 process after inspection; do not change the main port-8501 store.

- [ ] **Step 3: Report exact verification evidence and next scope**

Report the full test count, smoke result, health result, and that the current audit detects differences but does not provide point-in-time index constituents. The next work item is portfolio construction constraints: individual-position caps, industry caps and a turnover budget.
