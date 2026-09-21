"""单因子模拟的时间、方向、成本及指标口径。无需网络。"""
import numpy as np
import pandas as pd
import pytest

from qfm.portfolio.performance import perf_stats, yearly_perf
from qfm.simulation.engine import SimulationSettings, prepare_signal, run_factor_simulation, research_fitness


def test_sharpe_uses_mean_daily_excess_returns():
    returns = pd.Series([0.02, -0.01, 0.03, -0.015])
    nav = pd.Series(np.r_[1.0, (1 + returns).cumprod()])
    expected = (returns.mean() - (1.03 ** (1 / 252) - 1)) / returns.std() * np.sqrt(252)
    assert perf_stats(nav, risk_free_rate=0.03)["夏普比率"] == pytest.approx(expected)


def test_yearly_includes_first_trading_day_and_drawdown_from_prior_year():
    nav = pd.Series([1.0, 0.9, 0.99], index=pd.to_datetime(["2023-12-29", "2024-01-02", "2024-01-03"]))
    row = yearly_perf(nav).set_index("年份").loc[2024]
    assert row["收益"] == pytest.approx(-0.01)
    assert row["最大回撤"] == pytest.approx(-0.10)


def test_signal_delay_direction_and_future_invariance(panel):
    raw = panel.close.pct_change(5, fill_method=None)
    settings = SimulationSettings(start="2024-02-01", end="2024-09-30", delay=3, decay=3)
    positive = prepare_signal(panel, raw, "positive", settings)
    negative = prepare_signal(panel, raw, "negative", settings)
    pd.testing.assert_frame_equal(negative, -positive)
    now = prepare_signal(panel, raw, "positive", SimulationSettings(**{**settings.to_dict(), "delay": 1}))
    pd.testing.assert_frame_equal(positive.loc[settings.start:], now.shift(2).loc[settings.start:])
    changed = raw.copy()
    changed.loc[changed.index > "2024-06-03"] *= -10
    rerun = prepare_signal(panel, changed, "positive", settings)
    pd.testing.assert_frame_equal(positive.loc[:"2024-06-03"], rerun.loc[:"2024-06-03"])


def test_neutralization_removes_industry_and_size(panel):
    settings = SimulationSettings(start="2024-03-01", end="2024-03-08", neutralization="industry_size")
    signal = prepare_signal(panel, panel.close, "positive", settings)
    for date, row in signal.iterrows():
        assert row.groupby(panel.industry.loc[date]).mean().abs().max() < 1e-8
        assert abs(row.corr(np.log(panel.mv_float.loc[date]))) < 1e-8


def test_long_only_simulation_outputs_matching_run_metrics(panel):
    config = SimulationSettings(start="2024-02-01", end="2024-05-31", top_n=10, rebalance="W-FRI")
    result = run_factor_simulation(panel, "mom_5", config)
    assert not result.backtest.trades.empty
    assert (result.backtest.trades.date > result.backtest.trades.signal_date).all()
    assert result.backtest.nav.index.max() <= pd.Timestamp(config.end)
    assert result.backtest.nav.iloc[0] == 1
    assert result.backtest.cost_total > 0
    assert np.isfinite(result.stats["夏普比率"])
    assert result.stats["最大回撤"] <= 0
    daily = result.backtest.daily_turnover.iloc[1:]
    assert result.stats["日均换手"] == pytest.approx(daily.mean())
    assert result.stats["Fitness"] == pytest.approx(research_fitness(result.backtest.nav, daily, 0.0))
    assert result.report["horizon"] == config.horizon
    assert result.report["ic_series"].index.equals(result.backtest.nav.index)
    expected_signal = prepare_signal(panel, panel.close.pct_change(5), "positive", config).reindex(result.backtest.nav.index)
    pd.testing.assert_frame_equal(result.report["cleaned"], expected_signal)
    assert result.gross.cost_total == 0


def test_zero_cost_runs_agree(panel):
    config = SimulationSettings(start="2024-03-01", end="2024-04-30", top_n=10,
                                commission=0, stamp=0, impact=0, borrow_rate=0)
    result = run_factor_simulation(panel, "vol_20", config)
    pd.testing.assert_series_equal(result.backtest.nav, result.gross.nav)


def test_long_short_targets_both_legs_and_applies_fees(panel):
    config = SimulationSettings(start="2024-02-01", end="2024-03-29", top_n=10,
                                mode="long_short", rebalance="B")
    result = run_factor_simulation(panel, "mom_5", config)
    held = result.backtest.holdings.iloc[1:]
    assert held.gt(0).any(axis=1).all()
    assert held.lt(0).any(axis=1).all()
    assert result.backtest.cost_total > 0
    assert result.backtest.params["model"] == "theoretical_long_short"
    assert (result.backtest.trades.date > result.backtest.trades.signal_date).all()
    assert np.isfinite(result.backtest.nav).all()


def test_date_validation_and_insufficient_cross_section(panel):
    with pytest.raises(ValueError, match="开始日期"):
        run_factor_simulation(panel, "mom_5", SimulationSettings(start="2024-05-01", end="2024-02-01"))
    with pytest.raises(ValueError, match="Delay"):
        SimulationSettings(delay=0).validate()
    with pytest.raises(ValueError, match="股票"):
        run_factor_simulation(panel, "mom_5", SimulationSettings(mode="long_short", top_n=40))


def test_fitness_returns_missing_for_flat_nav():
    nav = pd.Series([1.0] * 30)
    assert np.isnan(research_fitness(nav, pd.Series([0.0] * 29), 0))


def test_long_short_hand_calculated_self_financing_path():
    from qfm.data.panel import DataPanel
    from qfm.simulation.long_short import run_long_short
    dates = pd.bdate_range("2024-01-05", periods=3)
    close = pd.DataFrame([[100, 100], [102, 98], [104, 96]], index=dates, columns=["600000", "600001"])
    panel = DataPanel(close=close, open=close.shift(1).fillna(100), volume=close * 0 + 1000)
    score = close * 0
    score.iloc[:, 0] = 1
    score.iloc[:, 1] = -1
    settings = SimulationSettings(top_n=1, max_weight=0.5, rebalance="W-FRI", borrow_rate=0)
    result = run_long_short(panel, score, settings, zero_cost=True)
    assert result.nav.to_list() == pytest.approx([1.0, 1.02, 1.04])
    assert result.daily_turnover.to_list() == pytest.approx([0, 1, 0])
    assert len(result.trades) == 2
    assert result.trades.date.min() == dates[1]


def test_fitness_hand_calculated_daily_basis():
    ret = pd.Series([0.01, -0.008, 0.005] * 10)
    nav = pd.Series(np.r_[1, (1 + ret).cumprod()])
    expected = ret.mean() / ret.std() * np.sqrt(252) * np.sqrt(abs(ret.mean() * 252) / 0.2)
    assert research_fitness(nav, pd.Series([0.2] * 30)) == pytest.approx(expected)
