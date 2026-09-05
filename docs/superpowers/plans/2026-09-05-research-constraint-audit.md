# 研究运行约束执行审计归档 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将非空的组合约束执行审计作为可选 CSV 随研究运行保存、加载、展示和导出。

**Architecture:** 回测结果经 `build_strategy_run_payload` 产生审计 DataFrame，保存面板将它作为最后一个可选参数交给 `ResearchStore`。Store 仅为非空数据创建 `constraint_history.csv` 并在 artifact manifest 中索引；加载模型用可选字段兼容历史运行。研究项目页只渲染已加载的表。

**Tech Stack:** Python 3.13、pandas、Streamlit、pytest。

## Global Constraints

- 只保存已经在 `BacktestResult.constraint_history` 中生成的审计行；不得在研究页重新计算目标、实际暴露或超限。
- 仅在非空审计表存在时创建 `constraint_history.csv` 和第六个下载按钮。
- 历史 manifest 缺少该 artifact 时必须可读取，并以 `constraint_history is None` 表示。
- `signal_date` 和 `execution_date` 存取后必须保持 pandas datetime 语义；CSV 不保存 DataFrame 索引。
- 不升级研究 manifest 格式版本，不重写旧研究运行，不改变普通运行的五个原有产物。
- 对现有脏改动仅做局部追加；所有测试零网络。

---

### Task 1: 测试可选审计 CSV 的存取兼容性

**Files:**
- Modify: `tests/test_research_store.py`
- Modify: `tests/test_research_payload.py`
- Modify: `scripts/verify_research_workbench.py`

**Interfaces:**
- Consumes: `ResearchStore.save_run(..., trades, constraint_history=None)` and `BacktestResult.constraint_history`.
- Produces: assertions for manifest artifact indexing, date round trip, absent-artifact compatibility and payload copy semantics.

- [ ] **Step 1: 写出可选产物的失败测试**

Add a dedicated store test with this data and save call:

```python
constraint_history = pd.DataFrame({
    "signal_date": [pd.Timestamp("2026-01-01")],
    "execution_date": [pd.Timestamp("2026-01-02")],
    "target_cash": [0.0],
    "actual_cash": [0.25],
    "target_tracking_error": [0.25],
    "stock_cap_exceeded": [False],
    "industry_cap_exceeded": [False],
})
run = store.save_run(..., trades, constraint_history)
loaded = store.load_run(run.id)
assert run.artifacts["constraint_history"] == "constraint_history.csv"
pd.testing.assert_frame_equal(loaded.constraint_history, constraint_history)
```

In the existing no-optional-artifact round-trip test, add:

```python
assert loaded.constraint_history is None
assert "constraint_history" not in run.artifacts
```

Set `constraint_history` on the `BacktestResult` fixture in `test_research_payload.py` and assert `payload["constraint_history"].equals(constraint_history)` and is not the same object. Add a one-row audit DataFrame to the smoke script and assert its loaded `actual_cash` is `0.25`.

- [ ] **Step 2: 运行测试，确认接口尚不存在**

Run: `/opt/anaconda3/bin/python3 -m pytest tests/test_research_store.py tests/test_research_payload.py -q`

Expected: FAIL with `TypeError` for the extra `save_run` argument or an assertion that the payload lacks `constraint_history`.

- [ ] **Step 3: 提交独立测试**

```bash
git add tests/test_research_store.py tests/test_research_payload.py scripts/verify_research_workbench.py
git commit -m "test: cover archived constraint execution audit"
```

### Task 2: 实现可选产物的保存、读取与载荷透传

**Files:**
- Modify: `qfm/research/models.py:163-171`
- Modify: `qfm/research/store.py:78-181`
- Modify: `qfm/research/payload.py:63-71`
- Modify: `qfm/research/views.py:79-91`

**Interfaces:**
- Produces: `LoadedResearchRun.constraint_history: pd.DataFrame | None`, `ResearchStore.save_run(..., constraint_history: pd.DataFrame | None = None)`, and payload key `constraint_history`.
- Preserves: all original positional `save_run` callers and manifests with only the original five artifact keys.

- [ ] **Step 1: 扩展模型和写入接口**

Append a default field to `LoadedResearchRun`:

```python
constraint_history: pd.DataFrame | None = None
```

Append the optional argument to `ResearchStore.save_run`. Build the existing five-item artifact mapping first, then add the sixth mapping only for a nonempty DataFrame:

```python
if constraint_history is not None and not constraint_history.empty:
    artifacts["constraint_history"] = "constraint_history.csv"
```

After writing `trades.csv`, write the optional table without an index:

```python
if "constraint_history" in artifacts:
    self._write_csv(constraint_history, temporary / artifacts["constraint_history"])
```

- [ ] **Step 2: 实现兼容读取**

Add `_read_constraint_history(path)` that calls `pd.read_csv(path)`, then for each existing key in `("signal_date", "execution_date")` assigns `pd.to_datetime(frame[key])`. In `load_run`, only call it when the artifact key exists:

```python
constraint_history=(
    self._read_constraint_history(artifact("constraint_history"))
    if "constraint_history" in run.artifacts else None
)
```

Pass that value to the new `LoadedResearchRun` field. Keep all current exception wrapping.

- [ ] **Step 3: 将回测审计从载荷传到 store**

Before the payload return mapping, create a copied optional table:

```python
constraint_history = (
    backtest.constraint_history.copy()
    if backtest.constraint_history is not None and not backtest.constraint_history.empty else None
)
```

Include it as `"constraint_history": constraint_history`. In `render_strategy_save_panel`, pass `payload.get("constraint_history")` as the final positional argument to `save_run`.

- [ ] **Step 4: 运行定向测试和冒烟**

Run:

```bash
/opt/anaconda3/bin/python3 -m pytest tests/test_research_store.py tests/test_research_payload.py -q
/opt/anaconda3/bin/python3 scripts/verify_research_workbench.py
```

Expected: all selected tests pass and smoke prints `research workbench smoke: ok`.

### Task 3: 呈现与导出归档审计表

**Files:**
- Modify: `qfm/research/views.py:278-298`
- Modify: `README.md:64-72`

**Interfaces:**
- Consumes: `LoadedResearchRun.constraint_history` from Task 2.
- Produces: optional audit expander and sixth CSV download, while ordinary runs retain the original layout.

- [ ] **Step 1: 增加条件详情表**

After the trades expander, add:

```python
if loaded.constraint_history is not None:
    with st.expander(f"组合约束执行审计（{len(loaded.constraint_history)} 次）"):
        st.dataframe(loaded.constraint_history, hide_index=True, use_container_width=True)
```

Do not nest this expander inside “配置与数据快照”, because Streamlit does not permit nested expanders.

- [ ] **Step 2: 增加条件导出按钮**

Create five columns when the audit is absent and six columns when it is present:

```python
export_columns = st.columns(6 if loaded.constraint_history is not None else 5)
```

Keep the original five buttons in indexes 0–4. When present, write index 5:

```python
export_columns[5].download_button(
    "约束审计 CSV",
    loaded.constraint_history.to_csv(index=False).encode("utf-8-sig"),
    "research_constraint_history.csv", "text/csv", use_container_width=True,
)
```

Update README’s archive list to include “约束执行审计（如本次策略有约束记录）”.

- [ ] **Step 3: 全量验证与隔离页面验收**

Run:

```bash
/opt/anaconda3/bin/python3 -m py_compile app.py qfm/research/models.py qfm/research/store.py qfm/research/payload.py qfm/research/views.py
/opt/anaconda3/bin/python3 -m pytest tests/ -q
/opt/anaconda3/bin/python3 scripts/verify_research_workbench.py
curl -fsS http://localhost:8501/_stcore/health
git diff --check
```

Create one temporary project/run with an optional audit table under a temporary `QFM_RESEARCH_ROOT`, start only a port-8502 Streamlit process, and open “研究项目”. Verify its accessibility tree contains “组合约束执行审计（1 次）” and “约束审计 CSV”. Stop only the temporary PID.

- [ ] **Step 4: 提交新的文档**

```bash
git add docs/superpowers/specs/2026-09-05-research-constraint-audit-design.md docs/superpowers/plans/2026-09-05-research-constraint-audit.md
git commit -m "docs: define archived constraint audit"
```

