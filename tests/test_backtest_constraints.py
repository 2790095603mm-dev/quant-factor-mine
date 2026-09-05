"""约束目标接入 T+1 回测后的审计记录测试。"""

from __future__ import annotations

from copy import deepcopy

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
