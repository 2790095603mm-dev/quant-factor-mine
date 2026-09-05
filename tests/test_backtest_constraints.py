"""约束目标接入 T+1 回测后的审计记录测试。"""

from __future__ import annotations

from copy import deepcopy

import pytest

from qfm.portfolio import PortfolioConstraints, run_backtest


def test_backtest_records_constrained_targets_and_turnover_budget(panel):
    constrained_panel = deepcopy(panel)
    constrained_panel.industry.loc[:, :] = "银行"
    score = constrained_panel.close.rank(axis=1, method="first")

    result = run_backtest(
        constrained_panel,
        score,
        top_n=10,
        rebalance="ME",
        start="2024-02-01",
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
    assert {"actual_invested", "actual_cash", "target_tracking_error"}.issubset(result.constraint_history.columns)
    assert result.constraint_history["actual_cash"].between(0.0, 1.0).all()
    assert result.constraint_history["actual_positions"].ge(0).all()


def test_constraint_history_records_actual_execution_gap(panel):
    blocked = deepcopy(panel)
    dates = blocked.close.index
    blocked.open.loc[dates[1], :] = float("nan")
    score = blocked.close.rank(axis=1, method="first")

    result = run_backtest(
        blocked,
        score,
        top_n=10,
        rebalance="B",
        start=str(dates[0].date()),
        end=str(dates[3].date()),
        constraints=PortfolioConstraints(),
    )

    first = result.constraint_history.iloc[0]
    assert first["target_cash"] == pytest.approx(0.0)
    assert first["actual_cash"] == pytest.approx(1.0)
    assert first["actual_positions"] == 0
    assert first["target_tracking_error"] == pytest.approx(1.0)
    assert first["actual_max_stock_weight"] == pytest.approx(0.0)
    assert first["actual_max_industry_weight"] == pytest.approx(0.0)
