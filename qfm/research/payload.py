"""将策略计算结果整理为可持久化、JSON 安全的研究运行载荷。"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import pandas as pd

from qfm.portfolio import BacktestResult, perf_stats, yearly_perf
from qfm.research.snapshot import build_data_snapshot


def _json_safe(value: Any) -> Any:
    """将 pandas/numpy 标量递归转换为 JSON 标准类型。"""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if pd.isna(value):
        return None
    raise TypeError(f"无法序列化研究运行字段: {type(value).__name__}")


def build_strategy_run_payload(
    *,
    panel: Any,
    pool: str,
    names: list[str],
    mode: str,
    horizon: int,
    weight_lookback: int,
    orthogonalize: bool,
    ortho_controls: tuple[str, ...],
    top_n: int,
    start_date: str,
    rebalance: str,
    bench_mode: str,
    costs: dict[str, float],
    max_participation: float,
    initial_capital: float,
    weights: dict[str, float],
    backtest: BacktestResult,
) -> dict[str, Any]:
    """构造一次已完成策略回测的可保存载荷，不触发任何重新计算。"""
    stats = perf_stats(backtest.nav, backtest.bench_nav)
    summary = {
        **stats,
        "净值终值": float(backtest.nav.iloc[-1]),
        "基准净值终值": float(backtest.bench_nav.iloc[-1]),
        "年化换手": float(backtest.turnover),
        "累计交易成本": float(backtest.cost_total),
        "成交成本率": float(backtest.cost_pct),
        "成交笔数": int(len(backtest.trades)),
    }
    config = {
        "factors": list(names),
        "weight_mode": mode,
        "signal_horizon": horizon,
        "weight_lookback": weight_lookback,
        "orthogonalize": orthogonalize,
        "ortho_controls": list(ortho_controls),
        "top_n": top_n,
        "start_date": start_date,
        "rebalance": rebalance,
        "benchmark": bench_mode,
        "costs": dict(costs),
        "max_participation": max_participation,
        "initial_capital": initial_capital,
        "execution": backtest.params.get("execution", "next_open"),
        "portfolio_constraints": dict(backtest.params.get("constraints", {})),
    }
    data_snapshot = build_data_snapshot(panel, pool)
    factor_weights = pd.DataFrame.from_dict(weights, orient="index", columns=["weight"])
    factor_weights.index.name = "factor"
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
