# 组合风险约束 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在日频多头回测中加入单股、行业与单次调仓成交额约束，并把目标组合和实际执行的约束状态归档、展示。

**Architecture:** 新建 `qfm.portfolio.constraints` 作为纯目标构建和换手缩放层，保持 `backtest.py` 只负责 T+1 成交、成本和盯市。回测结果新增可选约束历史，策略页和研究载荷从该统一结果读取信息，默认参数不改变旧的 TOP-N 等权行为。

**Tech Stack:** Python 3.13、pandas、dataclasses、Streamlit、pytest、本地 JSON/CSV 研究账本。

## Global Constraints

- 单股与行业上限均在 `(0, 1]`，单次调仓成交额预算为 `None` 或 `(0, 2]`；无效配置立即抛出 `ValueError`。
- `max_stock_weight=1.0`、`max_industry_weight=1.0`、预算 `None` 必须得到旧的 TOP-N 等权目标。
- 目标权重不满足满仓条件时保留现金，不得突破单股或行业上限；行业缺失统一归为 `未知行业`。
- 成交额预算限制开盘前的计划买卖名义金额，不保证涨跌停、停牌或容量受阻时的实际持仓立刻回到目标上限。
- 不添加优化器、网络调用、数据库、最小交易单位、空头或点时指数成分数据。
- 不暂存或提交既有 `app.py`、`qfm/portfolio/backtest.py` 等用户未提交改动之外的内容；对这些文件仅做局部追加。
- 新测试只用合成数据，完整回归、研究账本冒烟和隔离 Streamlit 验收必须通过。

---

### Task 1: 实现纯约束目标构建与换手缩放

**Files:**
- Create: `qfm/portfolio/constraints.py`
- Create: `tests/test_portfolio_constraints.py`
- Modify: `qfm/portfolio/__init__.py`

**Interfaces:**
- Produces: `PortfolioConstraints(max_stock_weight: float = 1.0, max_industry_weight: float = 1.0, max_rebalance_turnover: float | None = None)`.
- Produces: `constrained_target_weights(score, volume, industry, top_n, constraints) -> tuple[pd.Series, dict[str, object]]`.
- Produces: `apply_turnover_budget(current, target, max_rebalance_turnover) -> tuple[pd.Series, dict[str, object]]`.
- Consumes: score and volume as same-index `pd.Series`; industry can be `None` or a series indexed by stock.

- [ ] **Step 1: 写出默认兼容、硬上限、未知行业和换手预算的失败测试**

```python
import pandas as pd
import pytest

from qfm.portfolio.constraints import (
    PortfolioConstraints,
    apply_turnover_budget,
    constrained_target_weights,
)


def _inputs():
    index = pd.Index(["A", "B", "C", "D", "E", "F"], name="stock")
    return (
        pd.Series([6.0, 5.0, 4.0, 3.0, 2.0, 1.0], index=index),
        pd.Series(1.0, index=index),
        pd.Series(["银行", "银行", "科技", "科技", None, "医药"], index=index),
    )


def test_default_constraints_keep_top_n_equal_weight():
    score, volume, industry = _inputs()
    target, diag = constrained_target_weights(
        score, volume, industry, 3, PortfolioConstraints(),
    )

    pd.testing.assert_series_equal(
        target, pd.Series([1 / 3] * 3, index=pd.Index(["A", "B", "C"]), dtype=float),
    )
    assert diag["target_cash"] == pytest.approx(0.0)


def test_stock_and_industry_caps_leave_cash_instead_of_breaking_limits():
    score, volume, industry = _inputs()
    target, diag = constrained_target_weights(
        score, volume, industry, 4,
        PortfolioConstraints(max_stock_weight=0.20, max_industry_weight=0.25),
    )
    exposure = target.groupby(industry.reindex(target.index).fillna("未知行业")).sum()

    assert target.max() <= 0.20
    assert exposure.max() <= 0.25
    assert target.sum() < 1.0
    assert diag["target_cash"] == pytest.approx(1 - target.sum())


def test_unknown_industry_obeys_industry_cap():
    score, volume, industry = _inputs()
    target, _ = constrained_target_weights(
        score, volume, industry, 6,
        PortfolioConstraints(max_industry_weight=0.15),
    )

    assert target.get("E", 0.0) <= 0.15


def test_turnover_budget_scales_target_without_budget_changing_target():
    index = pd.Index(["A", "B"])
    current = pd.Series([0.5, 0.0], index=index)
    target = pd.Series([0.0, 0.5], index=index)

    unchanged, off = apply_turnover_budget(current, target, None)
    scaled, on = apply_turnover_budget(current, target, 0.4)

    pd.testing.assert_series_equal(unchanged, target)
    assert off["budget_binding"] is False
    assert on["gross_turnover"] == pytest.approx(1.0)
    assert on["applied_gross_turnover"] == pytest.approx(0.4)
    assert scaled.tolist() == pytest.approx([0.3, 0.2])
```

- [ ] **Step 2: 运行测试，确认模块缺失**

Run: `/opt/anaconda3/bin/python3 -m pytest tests/test_portfolio_constraints.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'qfm.portfolio.constraints'`.

- [ ] **Step 3: 实现不可变配置、贪心分配与预算缩放**

```python
@dataclass(frozen=True)
class PortfolioConstraints:
    max_stock_weight: float = 1.0
    max_industry_weight: float = 1.0
    max_rebalance_turnover: float | None = None

    def __post_init__(self) -> None:
        _validate_unit_interval("max_stock_weight", self.max_stock_weight)
        _validate_unit_interval("max_industry_weight", self.max_industry_weight)
        if self.max_rebalance_turnover is not None:
            _validate_turnover_budget(self.max_rebalance_turnover)

    def to_dict(self) -> dict[str, float | None]:
        return {
            "max_stock_weight": self.max_stock_weight,
            "max_industry_weight": self.max_industry_weight,
            "max_rebalance_turnover": self.max_rebalance_turnover,
        }
```

In `constrained_target_weights`, reject non-positive `top_n`, retain candidates with non-null score and positive volume, sort descending, assign `min(1 / top_n, remaining_stock_cap, remaining_industry_cap, remaining_cash)`, and stop after `top_n` positive allocations. Build diagnostics with `target_positions`, `target_invested`, `target_cash`, `target_max_stock_weight` and `target_max_industry_weight`. In `apply_turnover_budget`, reindex both series to their union with zeros, compute `gross_turnover = sum(abs(target - current))`, and when the budget binds return `current + budget / gross_turnover * (target - current)` plus `gross_turnover`, `applied_gross_turnover`, `budget`, `budget_binding` and `turnover_scale`. Export all three public symbols from `qfm.portfolio`.

- [ ] **Step 4: 运行纯约束测试**

Run: `/opt/anaconda3/bin/python3 -m pytest tests/test_portfolio_constraints.py -q`

Expected: PASS with 4 tests.

- [ ] **Step 5: 提交纯模块与测试**

```bash
git add qfm/portfolio/constraints.py qfm/portfolio/__init__.py tests/test_portfolio_constraints.py
git commit -m "feat: add portfolio constraint builder"
```

### Task 2: 将约束目标接入 T+1 回测并保存审计历史

**Files:**
- Modify: `qfm/portfolio/backtest.py:13-232`
- Create: `tests/test_backtest_constraints.py`
- Modify: `qfm/research/payload.py:64-82`
- Modify: `tests/test_research_payload.py`

**Interfaces:**
- Consumes: `PortfolioConstraints`, `constrained_target_weights` and `apply_turnover_budget`.
- Produces: the existing `run_backtest` result with a new optional final keyword parameter `constraints: PortfolioConstraints | None = None`.
- Produces: optional `BacktestResult.constraint_history: pd.DataFrame | None`.
- Preserves: prior `run_backtest` callers when `constraints` is omitted.

- [ ] **Step 1: 写出回测记录约束配置和预算审计的失败测试**

```python
from copy import deepcopy

import pandas as pd

from qfm.portfolio import PortfolioConstraints, run_backtest


def test_backtest_records_constrained_targets_and_turnover_budget(panel):
    constrained_panel = deepcopy(panel)
    constrained_panel.industry.loc[:, :] = "银行"
    score = constrained_panel.close.rank(axis=1, method="first")
    result = run_backtest(
        constrained_panel, score, top_n=10, rebalance="ME", start="2024-02-01",
        constraints=PortfolioConstraints(
            max_stock_weight=0.08,
            max_industry_weight=0.20,
            max_rebalance_turnover=0.30,
        ),
    )

    assert result.params["constraints"]["max_stock_weight"] == 0.08
    assert result.constraint_history is not None
    assert not result.constraint_history.empty
    assert result.constraint_history["target_max_stock_weight"].max() <= 0.08
    assert result.constraint_history["target_max_industry_weight"].max() <= 0.20
    assert result.constraint_history["applied_gross_turnover"].max() <= 0.30
    assert result.constraint_history["target_cash"].max() > 0.0
```

Add to the existing payload test:

```python
result.params["constraints"] = {"max_stock_weight": 0.08}
payload = build_strategy_run_payload(
    panel=panel, pool="index800", names=["ep_ttm", "roe"], mode="equal",
    horizon=20, weight_lookback=252, orthogonalize=False, ortho_controls=(),
    top_n=30, start_date="2021-01-01", rebalance="ME", bench_mode="equal",
    costs={"commission": 0.0003, "stamp": 0.0005, "impact": 0.001},
    max_participation=0.05, initial_capital=1_000_000,
    weights={"ep_ttm": 0.5, "roe": 0.5}, backtest=result,
)
assert payload["config"]["portfolio_constraints"]["max_stock_weight"] == 0.08
```

- [ ] **Step 2: 运行两组测试，确认缺少新参数和配置字段而失败**

Run: `/opt/anaconda3/bin/python3 -m pytest tests/test_backtest_constraints.py tests/test_research_payload.py -q`

Expected: FAIL with `TypeError` for unknown `constraints` or `KeyError: 'portfolio_constraints'`.

- [ ] **Step 3: 在回测循环的目标与执行边界加入约束**

Extend `BacktestResult` with `constraint_history: pd.DataFrame | None = None` after the existing optional fields. Add `constraints: PortfolioConstraints | None = None` at the end of `run_backtest` parameters, normalize it with `constraints = constraints or PortfolioConstraints()`, and replace the signal-day `_target_weights` call with `constrained_target_weights(score.loc[dt], panel.volume.loc[dt], industry_at_signal, top_n, constraints)`.

Store the pending signal as `(signal_date, raw_target, target_diag)`. On the next opening, calculate the current stock weights from `current / pretrade_value`, call `apply_turnover_budget(current_weights, raw_target, constraints.max_rebalance_turnover)`, then calculate `desired = execution_target * pretrade_value` before existing sells and buys. Append one row containing `signal_date`, `execution_date`, target diagnostics and turnover diagnostics before clearing pending. Return a date-sorted DataFrame or `None` when no target was generated. Add `"constraints": constraints.to_dict()` to `params`.

In the payload builder, add `"portfolio_constraints": dict(backtest.params.get("constraints", {}))` to its config dict; this uses an empty dict for old synthetic results.

- [ ] **Step 4: 运行约束、载荷和现有回测相关测试**

Run: `/opt/anaconda3/bin/python3 -m pytest tests/test_portfolio_constraints.py tests/test_backtest_constraints.py tests/test_research_payload.py tests/test_mining_trials.py -q`

Expected: PASS with no network access.

- [ ] **Step 5: 暂存前检查与提交**

Run: `git diff -- qfm/portfolio/backtest.py qfm/research/payload.py`. Stage only the dedicated new test; do not stage dirty `backtest.py` or `app.py` without confirming their unrelated changes remain intact.

```bash
git add tests/test_backtest_constraints.py
git commit -m "test: cover constrained backtest targets"
```

### Task 3: 增加策略页控件、执行记录与最终验收

**Files:**
- Modify: `app.py:651-789`
- Modify: `README.md:65-75`
- Verify: `qfm/portfolio/backtest.py`, `qfm/research/payload.py`, local Streamlit UI.

**Interfaces:**
- Consumes: `PortfolioConstraints` and `BacktestResult.constraint_history`.
- Produces: strategy page controls for single-stock cap, industry cap and optional gross turnover budget, all defaulting to no additional constraint.
- Produces: saved run config containing the normalized portfolio constraint dictionary.

- [ ] **Step 1: 添加默认关闭的风险约束控件与回测参数**

```python
with st.expander("组合风险约束（目标组合）"):
    max_stock_weight = st.slider("单股目标权重上限", 0.01, 1.00, 1.00, 0.01)
    max_industry_weight = st.slider("单行业目标权重上限", 0.01, 1.00, 1.00, 0.01)
    use_turnover_budget = st.checkbox("限制单次调仓成交额", value=False)
    max_rebalance_turnover = (
        st.slider("单次调仓成交额上限（组合净值）", 0.05, 2.00, 0.50, 0.05)
        if use_turnover_budget else None
    )
constraints = PortfolioConstraints(
    max_stock_weight=max_stock_weight,
    max_industry_weight=max_industry_weight,
    max_rebalance_turnover=max_rebalance_turnover,
)
bt = run_backtest(
    panel, score, top_n=top_n, start=start_d, rebalance=rebalance,
    bench_mode=bench_mode, initial_capital=float(initial_capital),
    max_participation=max_participation, cost=costs, constraints=constraints,
)
```

Use a local import beside existing portfolio imports. Rename the existing visible `持仓数量` label to `最多持仓数量`. Explain that restrictive caps can intentionally retain cash, and that execution barriers can delay actual exposure reductions.

- [ ] **Step 2: 显示执行记录**

After the existing cost metrics, when `bt.constraint_history` is nonempty, render an expander named `组合约束执行记录`, show the latest target position count, target cash, applied gross turnover and whether the turnover budget bound, then render the full history with percent formatting for weights and turnover. Do not claim actual holdings always satisfy a target cap; preserve the existing industry exposure chart as actual close-of-day exposure.

- [ ] **Step 3: 更新 README 的策略能力说明**

Add one short paragraph under the research-workbench section stating that strategy backtests can record target stock/industry caps and an optional per-rebalance gross-trade budget; restrictive settings may retain cash, and execution constraints can defer convergence to the target.

- [ ] **Step 4: 运行完整验证**

Run:

```bash
/opt/anaconda3/bin/python3 -m py_compile app.py qfm/portfolio/constraints.py qfm/portfolio/backtest.py qfm/research/payload.py
/opt/anaconda3/bin/python3 -m pytest tests/ -q
/opt/anaconda3/bin/python3 scripts/verify_research_workbench.py
curl -fsS http://localhost:8501/_stcore/health
git diff --check
```

Expected: all tests pass, smoke prints `research workbench smoke: ok`, health prints `ok`, and `git diff --check` produces no output.

- [ ] **Step 5: 隔离 UI 验收**

Start the existing app with a temporary `QFM_RESEARCH_ROOT` on port 8502. In browser automation open the strategy page and verify all three constraint controls are visible, with stock/industry defaults at 100% and the turnover slider hidden until its checkbox is selected. Stop only the temporary 8502 service.
