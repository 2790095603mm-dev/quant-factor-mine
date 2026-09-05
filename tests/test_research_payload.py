"""把策略回测结果转换为可持久化研究运行的测试。"""

from __future__ import annotations

import pandas as pd

from qfm.portfolio import BacktestResult
from qfm.research.payload import build_strategy_run_payload


def test_strategy_payload_records_execution_assumptions(panel):
    index = panel.close.index[:3]
    columns = panel.close.columns
    nav = pd.Series([1.0, 1.01, 1.02], index=index)
    benchmark = pd.Series([1.0, 1.005, 1.01], index=index)
    holdings = pd.DataFrame(0.0, index=index, columns=columns)
    cash = pd.Series([1.0, 0.0, 0.0], index=index)
    trades = pd.DataFrame({"date": [index[1]], "stock": [columns[0]], "side": ["BUY"]})
    result = BacktestResult(
        nav=nav,
        bench_nav=benchmark,
        holdings=holdings,
        trades=trades,
        turnover=3.2,
        cost_pct=0.001,
        cost_total=0.02,
        cash_weight=cash,
        params={"execution": "next_open"},
    )

    payload = build_strategy_run_payload(
        panel=panel,
        pool="index800",
        names=["ep_ttm", "roe"],
        mode="equal",
        horizon=20,
        weight_lookback=252,
        orthogonalize=False,
        ortho_controls=(),
        top_n=30,
        start_date="2021-01-01",
        rebalance="ME",
        bench_mode="equal",
        costs={"commission": 0.0003, "stamp": 0.0005, "impact": 0.001},
        max_participation=0.05,
        initial_capital=1_000_000,
        weights={"ep_ttm": 0.5, "roe": 0.5},
        backtest=result,
    )

    assert payload["data_snapshot"]["pool"] == "index800"
    assert payload["data_snapshot"]["stocks"] == panel.close.shape[1]
    assert payload["data_snapshot"]["snapshot_version"] == 2
    assert payload["data_snapshot"]["stock_codes"][0] == "600000"
    assert payload["data_snapshot"]["fingerprints"]["market"].startswith("sha256:")
    assert payload["data_snapshot"]["coverage"]["fundamentals"]["roe"] == 1.0
    assert payload["config"]["execution"] == "next_open"
    assert payload["config"]["orthogonalize"] is False
    assert payload["summary"]["成交笔数"] == len(trades)
    assert payload["weights"].index.name == "factor"
    assert payload["nav"].equals(nav.rename("nav"))
