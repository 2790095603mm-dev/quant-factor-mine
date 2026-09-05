# 约束执行差异审计 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为每条受约束调仓记录追加执行日收盘的实际仓位、目标偏离与实际超限提示。

**Architecture:** 回测循环在开盘阶段暂存目标和预算诊断，在同一交易日收盘盯市后基于 `shares`、`cash`、收盘价和行业面板计算实际审计字段，再写入 `constraint_history`。策略页只消费该表，不改变目标构建、换手预算或成交顺序。

**Tech Stack:** Python 3.13、pandas、Streamlit、pytest。

## Global Constraints

- 新字段只描述执行日收盘实际暴露；不得把实际超限解释为目标构建器违反约束。
- 未知或缺失行业必须统一归类为 `未知行业`，不能在实际行业暴露中被忽略。
- `target_tracking_error` 为实际股票权重与预算后执行目标的绝对差之和，使用二者的证券并集。
- `stock_cap_exceeded` 与 `industry_cap_exceeded` 仅在实际最大暴露大于相应上限加 `1e-8` 时为真。
- 不修改订单状态机、成本、容量、交易限制、研究文件格式或默认回测结果。
- 对现有脏改动的 `backtest.py` 和 `app.py` 仅做局部追加；所有新测试零网络。

---

### Task 1: 在收盘盯市时合并实际执行审计

**Files:**
- Modify: `qfm/portfolio/backtest.py:118-239`
- Modify: `tests/test_backtest_constraints.py`

**Interfaces:**
- Produces: existing `BacktestResult.constraint_history` with `actual_invested`, `actual_cash`, `actual_positions`, `actual_max_stock_weight`, `actual_max_industry_weight`, `target_tracking_error`, `stock_cap_exceeded` and `industry_cap_exceeded`.
- Preserves: target diagnostics and turnover diagnostics already present in every audit row.

- [ ] **Step 1: 写出成交受阻时目标与实际分离的失败测试**

```python
def test_constraint_history_records_actual_execution_gap(panel):
    blocked = deepcopy(panel)
    dates = blocked.close.index
    blocked.open.loc[dates[1], :] = float("nan")
    score = blocked.close.rank(axis=1, method="first")

    result = run_backtest(
        blocked, score, top_n=10, rebalance="B",
        start=str(dates[0].date()), end=str(dates[3].date()),
        constraints=PortfolioConstraints(),
    )

    first = result.constraint_history.iloc[0]
    assert first["target_cash"] == pytest.approx(0.0)
    assert first["actual_cash"] == pytest.approx(1.0)
    assert first["actual_positions"] == 0
    assert first["target_tracking_error"] == pytest.approx(1.0)
    assert first["actual_max_stock_weight"] == pytest.approx(0.0)
    assert first["actual_max_industry_weight"] == pytest.approx(0.0)
```

Extend the existing constrained-target test with:

```python
history = result.constraint_history
assert {"actual_invested", "actual_cash", "target_tracking_error"}.issubset(history.columns)
assert history["actual_cash"].between(0.0, 1.0).all()
assert history["actual_positions"].ge(0).all()
```

- [ ] **Step 2: 运行约束回测测试，确认新字段缺失**

Run: `/opt/anaconda3/bin/python3 -m pytest tests/test_backtest_constraints.py -q`

Expected: FAIL with `KeyError: 'actual_cash'`.

- [ ] **Step 3: 将审计行从开盘阶段延后到收盘阶段**

At the start of each date loop define `execution_audit: dict[str, object] | None = None` and `execution_target: pd.Series | None = None`. Replace the current immediate `constraint_rows.append` after `apply_turnover_budget` with assignment to `execution_audit` and save the budget-adjusted target in `execution_target`.

After the existing close-of-day `values`, `value` and `cash_weight` calculation, when `execution_audit is not None`:

```python
actual_weights = values / value if value > 0 else pd.Series(dtype=float)
actual_all = actual_weights.reindex(actual_weights.index.union(execution_target.index), fill_value=0.0)
target_all = execution_target.reindex(actual_all.index, fill_value=0.0)
labels = _actual_industry_labels(panel, dt, actual_weights.index)
actual_industry = actual_weights.groupby(labels).sum()
actual_max_stock = float(actual_weights.max()) if len(actual_weights) else 0.0
actual_max_industry = float(actual_industry.max()) if len(actual_industry) else 0.0
constraint_rows.append({
    **execution_audit,
    "actual_invested": float(actual_weights.sum()),
    "actual_cash": float(cash_weight.loc[dt]),
    "actual_positions": int(len(actual_weights)),
    "actual_max_stock_weight": actual_max_stock,
    "actual_max_industry_weight": actual_max_industry,
    "target_tracking_error": float((actual_all - target_all).abs().sum()),
    "stock_cap_exceeded": actual_max_stock > constraints.max_stock_weight + 1e-8,
    "industry_cap_exceeded": actual_max_industry > constraints.max_industry_weight + 1e-8,
})
```

Implement `_actual_industry_labels(panel, dt, index)` in `backtest.py`: when a same-date industry series exists, reindex it; otherwise make a string series of `未知行业`; fill missing values with that same label. Update the existing actual industry-exposure block to use this helper rather than dropping missing labels.

- [ ] **Step 4: 运行约束与相关回归测试**

Run: `/opt/anaconda3/bin/python3 -m pytest tests/test_backtest_constraints.py tests/test_portfolio_constraints.py tests/test_mining_trials.py -q`

Expected: PASS with no network access.

- [ ] **Step 5: 提交独立测试**

```bash
git add tests/test_backtest_constraints.py
git commit -m "test: cover actual constraint execution audit"
```

Do not stage the dirty `qfm/portfolio/backtest.py` without separately reviewing all pre-existing changes.

### Task 2: 将实际执行差异呈现在策略页

**Files:**
- Modify: `app.py:798-826`
- Verify: isolated Streamlit strategy page.

**Interfaces:**
- Consumes: the expanded `bt.constraint_history` table.
- Produces: six latest-execution metrics and a full formatted audit table.

- [ ] **Step 1: 替换目标专属摘要为目标/实际并排摘要**

Replace the four current metrics with:

```python
cc1.metric("目标现金", f"{latest_constraint['target_cash']:.1%}")
cc2.metric("实际现金", f"{latest_constraint['actual_cash']:.1%}")
cc3.metric("目标偏离", f"{latest_constraint['target_tracking_error']:.1%}")
cc4.metric("实际最大单股", f"{latest_constraint['actual_max_stock_weight']:.1%}")
cc5.metric("实际最大行业", f"{latest_constraint['actual_max_industry_weight']:.1%}")
cc6.metric(
    "实际超限",
    "是" if latest_constraint["stock_cap_exceeded"] or latest_constraint["industry_cap_exceeded"] else "否",
)
```

Keep the existing planned gross turnover and budget-binding values in the detailed table. Update the caption to state that “实际超限” is an execution-day closing snapshot and may arise from execution barriers or price movement.

- [ ] **Step 2: 扩展详细表格字段与格式**

Append `actual_invested`, `actual_cash`, `actual_positions`, `actual_max_stock_weight`, `actual_max_industry_weight`, `target_tracking_error`, `stock_cap_exceeded` and `industry_cap_exceeded` to `constraint_columns`. Apply `{:.1%}` formatting to all actual weight and tracking-error fields, preserve dates and booleans unformatted.

- [ ] **Step 3: 运行静态和完整验证**

Run:

```bash
/opt/anaconda3/bin/python3 -m py_compile app.py qfm/portfolio/backtest.py
/opt/anaconda3/bin/python3 -m pytest tests/ -q
/opt/anaconda3/bin/python3 scripts/verify_research_workbench.py
curl -fsS http://localhost:8501/_stcore/health
git diff --check
```

Expected: every test passes, smoke prints `research workbench smoke: ok`, health prints `ok`, and `git diff --check` produces no output.

- [ ] **Step 4: 隔离页面验收**

Start a temporary port-8502 Streamlit instance with an isolated research root. Open “策略回测”, verify the constraints panel remains visible, and confirm the source-level constraint history section can render the new actual-field columns after a completed run. Stop only the temporary service.
