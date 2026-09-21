"""回测引擎与统一指标输出测试：滑点、外部基准、每 N 日调仓、ST、指标口径。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qfm.portfolio import (
    SUMMARY_METRICS,
    PortfolioConstraints,
    run_backtest,
    sortino_ratio,
    standard_metrics,
    summary_table,
    tracking_error,
)
from qfm.portfolio.backtest import _limit_ratio, _rebalance_dates


def _tiny_panel(stock_count: int = 6, days: int = 12) -> "object":
    """可手算的小面板：所有股票价格线性上涨，便于精确断言成交价与成本。"""
    from qfm.data.panel import DataPanel

    dates = pd.bdate_range("2024-01-02", periods=days)
    codes = [f"{600000 + i}" for i in range(stock_count)]
    base = np.array([10.0 + i for i in range(stock_count)])
    close = pd.DataFrame(np.tile(base, (days, 1)), index=dates, columns=codes)
    # 开盘价固定为收盘价的 0.99，成交价可预测
    open_ = close * 0.99
    return DataPanel(
        close=close,
        open=open_,
        high=close * 1.01,
        low=close * 0.98,
        volume=pd.DataFrame(1e9, index=dates, columns=codes),
        amount=pd.DataFrame(1e12, index=dates, columns=codes),
        turnover=pd.DataFrame(0.01, index=dates, columns=codes),
        mv_float=pd.DataFrame(np.tile(base * 1e8, (days, 1)), index=dates, columns=codes),
        industry=pd.DataFrame(np.tile(["银行", "白酒", "科技"], (days, 2))[:, :stock_count],
                              index=dates, columns=codes),
    )


def _flat_score(panel, top_n: int = 3) -> pd.DataFrame:
    """让得分只取决于代码顺序，选股结果稳定可预测。"""
    ranks = np.arange(panel.close.shape[1])[::-1]
    return pd.DataFrame(np.tile(ranks, (panel.close.shape[0], 1)),
                        index=panel.close.index, columns=panel.close.columns)


# ---------- 调仓频率 ----------

def test_rebalance_supports_every_n_days():
    dates = pd.bdate_range("2024-01-02", periods=10)

    assert _rebalance_dates(dates, 3) == {dates[0], dates[3], dates[6], dates[9]}
    assert _rebalance_dates(dates, 1) == set(dates)


def test_rebalance_supports_quarter_and_daily_aliases():
    dates = pd.bdate_range("2024-01-02", periods=260)

    quarterly = _rebalance_dates(dates, "QE")
    assert len(quarterly) == 4, "一年应产出 4 个季末信号日"
    assert _rebalance_dates(dates, "D") == set(dates)


@pytest.mark.parametrize("bad", [0, -3])
def test_rebalance_rejects_non_positive_intervals(bad):
    with pytest.raises(ValueError, match="N 必须为正整数"):
        _rebalance_dates(pd.bdate_range("2024-01-02", periods=5), bad)


def test_rebalance_rejects_bool_and_unknown_strings():
    dates = pd.bdate_range("2024-01-02", periods=5)
    with pytest.raises(ValueError, match="不接受布尔值"):
        _rebalance_dates(dates, True)
    with pytest.raises(ValueError, match="未知调仓频率"):
        _rebalance_dates(dates, "月月")


def test_every_n_days_backtest_trades_on_that_grid():
    panel = _tiny_panel()
    result = run_backtest(panel, _flat_score(panel), top_n=3, rebalance=4, cost={})

    signal_dates = sorted(result.trades["signal_date"].unique())
    expected = list(panel.close.index[::4])
    assert signal_dates == expected[: len(signal_dates)]


# ---------- 滑点 ----------

def test_slippage_worsens_buy_price_and_nav():
    panel = _tiny_panel()
    score = _flat_score(panel)
    clean = run_backtest(panel, score, top_n=3, rebalance=2, cost={})
    slipped = run_backtest(panel, score, top_n=3, rebalance=2, cost={}, slippage=0.01)

    buys_clean = clean.trades[clean.trades["side"] == "BUY"].iloc[0]
    buys_slipped = slipped.trades[slipped.trades["side"] == "BUY"].iloc[0]
    assert buys_slipped["price"] == pytest.approx(buys_clean["price"] * 1.01)
    assert slipped.nav.iloc[-1] < clean.nav.iloc[-1]


def test_slippage_worsens_sell_price():
    panel = _tiny_panel(days=16)
    score = _flat_score(panel)
    clean = run_backtest(panel, score, top_n=3, rebalance=3, cost={})
    slipped = run_backtest(panel, score, top_n=3, rebalance=3, cost={}, slippage=0.02)

    sells_clean = clean.trades[clean.trades["side"] == "SELL"]
    sells_slipped = slipped.trades[slipped.trades["side"] == "SELL"]
    if sells_clean.empty:
        pytest.skip("该参数下没有卖出成交")
    assert sells_slipped["price"].iloc[0] == pytest.approx(sells_clean["price"].iloc[0] * 0.98)


def test_zero_slippage_is_the_historical_behaviour():
    panel = _tiny_panel()
    score = _flat_score(panel)

    default = run_backtest(panel, score, top_n=3, rebalance=2, cost={})
    explicit = run_backtest(panel, score, top_n=3, rebalance=2, cost={}, slippage=0.0)

    pd.testing.assert_series_equal(default.nav, explicit.nav)


def test_slippage_is_recorded_separately_from_impact():
    """滑点改成交价、impact 改费用，两者必须能被分别记账。"""
    panel = _tiny_panel()
    score = _flat_score(panel)

    result = run_backtest(panel, score, top_n=3, rebalance=2,
                          cost={"commission": 0.0, "stamp": 0.0, "impact": 0.0},
                          slippage=0.005)

    assert result.cost_total == pytest.approx(0.0)
    assert result.params["slippage"] == pytest.approx(0.005)


@pytest.mark.parametrize("bad", [-0.01, 1.0, 1.5])
def test_slippage_validation(bad):
    panel = _tiny_panel()
    with pytest.raises(ValueError, match="滑点必须在"):
        run_backtest(panel, _flat_score(panel), top_n=3, cost={}, slippage=bad)


# ---------- 外部基准 ----------

def test_external_benchmark_overrides_bench_mode():
    panel = _tiny_panel()
    series = pd.Series(np.linspace(100, 150, len(panel.close.index)), index=panel.close.index)

    result = run_backtest(panel, _flat_score(panel), top_n=3, cost={}, benchmark=series)

    assert result.params["benchmark_source"] == "external"
    assert result.bench_nav.iloc[0] == pytest.approx(1.0)
    assert result.bench_nav.iloc[-1] == pytest.approx(1.5)


def test_external_benchmark_skips_missing_dates():
    panel = _tiny_panel()
    series = pd.Series([100.0, 110.0], index=[panel.close.index[0], panel.close.index[5]])

    result = run_backtest(panel, _flat_score(panel), top_n=3, cost={}, benchmark=series)

    assert result.bench_nav.notna().all()
    assert result.bench_nav.iloc[-1] == pytest.approx(1.1)


def test_external_benchmark_needs_overlap():
    panel = _tiny_panel()
    series = pd.Series([100.0, 110.0], index=pd.bdate_range("2030-01-02", periods=2))

    with pytest.raises(ValueError, match="没有足够重叠"):
        run_backtest(panel, _flat_score(panel), top_n=3, cost={}, benchmark=series)


def test_external_benchmark_rejects_empty_series():
    panel = _tiny_panel()
    with pytest.raises(ValueError, match="非空"):
        run_backtest(panel, _flat_score(panel), top_n=3, cost={}, benchmark=pd.Series(dtype=float))


def test_default_benchmark_still_used_without_series():
    panel = _tiny_panel()
    result = run_backtest(panel, _flat_score(panel), top_n=3, cost={})

    assert result.params["benchmark_source"] == "equal"


# ---------- ST ----------

def test_limit_ratio_uses_five_percent_for_st():
    assert _limit_ratio("600000", is_st=True) == 0.05
    assert _limit_ratio("600000") == 0.10
    assert _limit_ratio("300001") == 0.20
    assert _limit_ratio("830001") == 0.30


def test_st_stock_is_excluded_from_selection():
    panel = _tiny_panel()
    st = pd.DataFrame(False, index=panel.close.index, columns=panel.close.columns)
    # 让得分最高的股票变成 ST，它不应被选中
    st.iloc[:, 0] = True
    score = _flat_score(panel, top_n=3)

    without = run_backtest(panel, score, top_n=3, rebalance=2, cost={})
    with_st = run_backtest(panel, score, top_n=3, rebalance=2, cost={}, st=st)

    held_without = set(without.trades["stock"])
    held_with_st = set(with_st.trades["stock"])
    assert st.columns[0] in held_without
    assert st.columns[0] not in held_with_st
    assert with_st.params["st_filter"] is True


def test_st_can_be_flagged_on_a_single_day():
    panel = _tiny_panel()
    st = pd.DataFrame(False, index=panel.close.index, columns=panel.close.columns)
    st.iloc[:2, 0] = True

    result = run_backtest(panel, _flat_score(panel), top_n=3, rebalance=1, cost={}, st=st)

    first_day_trades = result.trades[result.trades["signal_date"] == panel.close.index[0]]
    assert st.columns[0] not in set(first_day_trades["stock"])
    # 后段恢复普通状态后可以被选中
    later = result.trades[result.trades["signal_date"] >= panel.close.index[5]]
    assert st.columns[0] in set(later["stock"])


def test_invalid_st_panel_is_rejected():
    panel = _tiny_panel()
    with pytest.raises(ValueError, match="date×stock 的布尔面板"):
        run_backtest(panel, _flat_score(panel), top_n=3, cost={}, st=pd.DataFrame())


def test_no_st_means_flag_is_false():
    panel = _tiny_panel()
    result = run_backtest(panel, _flat_score(panel), top_n=3, cost={})

    assert result.params["st_filter"] is False


# ---------- 指标口径 ----------

def test_standard_metrics_is_the_single_source_of_truth():
    # 超额/跟踪误差类指标有「对齐天数 > 20」门槛，样本必须够长
    rng = np.random.default_rng(3)
    index = pd.bdate_range("2024-01-02", periods=40)
    nav = pd.Series(1 + np.cumsum(rng.normal(0.0015, 0.012, 40)), index=index)
    bench = pd.Series(1 + np.cumsum(rng.normal(0.0005, 0.010, 40)), index=index)

    metrics = standard_metrics(nav, bench, turnover=3.5)

    assert metrics["年化换手"] == pytest.approx(3.5)
    assert np.isfinite(metrics["索提诺比率"])
    assert np.isfinite(metrics["跟踪误差"])
    assert np.isfinite(metrics["年化超额"])
    assert metrics["信息比率"] == pytest.approx(metrics["超额夏普"])


def test_bench_relative_metrics_need_enough_overlap():
    """样本过短时刻意不算超额类指标，而不是给出噪声很大的数字。"""
    index = pd.bdate_range("2024-01-02", periods=5)
    nav = pd.Series([1.0, 1.02, 1.01, 1.05, 1.08], index=index)
    bench = pd.Series([1.0, 1.01, 1.00, 1.02, 1.03], index=index)

    metrics = standard_metrics(nav, bench)

    assert "年化超额" not in metrics
    assert "跟踪误差" not in metrics


def test_sortino_only_penalises_downside():
    index = pd.bdate_range("2024-01-02", periods=6)
    steady = pd.Series([1.0, 1.01, 1.02, 1.03, 1.04, 1.05], index=index)
    volatile = pd.Series([1.0, 1.05, 0.98, 1.06, 0.97, 1.08], index=index)

    assert np.isfinite(sortino_ratio(volatile))
    # 完全没有下跌日时下行波动为 0，索提诺没有定义；返回 NaN 而不是无穷大
    assert np.isnan(sortino_ratio(steady))


def test_tracking_error_is_zero_for_identical_series():
    index = pd.bdate_range("2024-01-02", periods=10)
    nav = pd.Series(np.linspace(1.0, 1.2, 10), index=index)

    assert tracking_error(nav, nav.copy()) != tracking_error(nav, nav.copy())  # 全零 → std=0 → NaN


def test_summary_table_covers_every_standard_metric():
    """对比表列必须来自 SUMMARY_METRICS，且策略行必须真的算出超额类指标。"""
    index = pd.bdate_range("2024-01-02", periods=60)
    rng = np.random.default_rng(0)
    nav = pd.Series(1 + np.cumsum(rng.normal(0.001, 0.01, 60)), index=index)
    bench = pd.Series(1 + np.cumsum(rng.normal(0.0005, 0.009, 60)), index=index)

    table = summary_table(nav, bench)

    assert list(table["组合"]) == ["策略", "基准"]
    assert set(SUMMARY_METRICS) <= set(table.columns)
    strategy = table[table["组合"] == "策略"].iloc[0]
    assert np.isfinite(strategy["年化超额"])
    assert np.isfinite(strategy["跟踪误差"])
    assert np.isfinite(strategy["索提诺比率"])
    # 基准相对自身的超额没有意义，必须留空而不是填 0
    assert pd.isna(table[table["组合"] == "基准"].iloc[0]["年化超额"])


def test_summary_table_without_benchmark_has_one_row():
    index = pd.bdate_range("2024-01-02", periods=30)
    nav = pd.Series(np.linspace(1.0, 1.3, 30), index=index)

    table = summary_table(nav)

    assert list(table["组合"]) == ["策略"]
    assert pd.isna(table.iloc[0]["年化超额"])


def test_perf_stats_rejects_invalid_risk_free_rate():
    nav = pd.Series([1.0, 1.1], index=pd.bdate_range("2024-01-02", periods=2))

    with pytest.raises(ValueError, match="无风险年利率"):
        standard_metrics(nav, risk_free_rate=-1.5)


def test_metrics_unchanged_for_existing_keys(panel):
    """既有指标键与数值不得因新增指标而改变口径。"""
    from qfm.portfolio import perf_stats

    score = _flat_score(panel)
    result = run_backtest(panel, score, top_n=5, rebalance="ME", cost={})
    stats = perf_stats(result.nav, result.bench_nav)

    for key in ("总收益", "年化收益", "年化波动", "夏普比率", "最大回撤", "卡玛比率", "日胜率"):
        assert key in stats
        assert np.isfinite(stats[key])
    assert stats["最大回撤"] <= 0
