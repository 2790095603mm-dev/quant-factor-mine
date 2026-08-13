"""组合回测：月度调仓日历 + T+1 成交 + 涨跌停过滤 + 交易成本建模"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# A 股成本参数（可配置）：佣金双边、印花税卖出单边、冲击成本
DEFAULT_COST = {"commission": 0.0003, "stamp": 0.0005, "impact": 0.0010}
LIMIT_UP = 0.098  # 近似涨停阈值（10% 板，略留余量防四舍五入）


@dataclass
class BacktestResult:
    nav: pd.Series                    # 策略净值（日频）
    bench_nav: pd.Series              # 基准净值（全池等权）
    holdings: pd.DataFrame            # 每日持仓（date × stock 的权重，0=空仓）
    trades: pd.DataFrame              # 交易记录（date, stock, side, amount, price）
    turnover: float                   # 年化单边换手
    cost_pct: float                   # 总成本占成交额比例
    params: dict = field(default_factory=dict)


def run_backtest(panel, score: pd.DataFrame, top_n: int = 50, rebalance: str = "ME",
                 cost: dict = None, start: str = None, end: str = None,
                 bench_mode: str = "equal") -> BacktestResult:
    """组合回测主入口

    score:    综合得分（date × stock），由因子合成产出
    top_n:    持仓数量
    rebalance: 调仓频率（"ME"=月末）
    cost:     成本参数覆盖
    bench_mode: 基准构造：equal=全池等权 | mv=流通市值加权
    成交模型：T 日收盘决策 → T+1 起新仓生效（当日先吃旧仓收益）；涨停股不可买入
    """
    cost = {**DEFAULT_COST, **(cost or {})}
    close = panel.close
    close_prev = close.shift(1)
    ret = close / close_prev - 1  # 全量日收益矩阵（预计算，避免循环内 shift）

    dates = close.index
    if start:
        dates = dates[dates >= pd.Timestamp(start)]
    if end:
        dates = dates[dates <= pd.Timestamp(end)]

    # 调仓日：每月最后交易日
    rebal_dates = set(pd.Series(dates).groupby(dates.to_period("M")).max().tolist())
    # 基准权重（市值加权模式预取）
    mv = panel.mv_float if bench_mode == "mv" else None

    nav = pd.Series(1.0, index=dates)
    bench_nav = pd.Series(1.0, index=dates)
    holdings = pd.DataFrame(0.0, index=dates, columns=close.columns)
    trades = []
    total_cost = 0.0
    total_turn = 0.0

    prev_nav = 1.0
    prev_bench = 1.0
    weights = None  # 昨日收盘后的目标持仓（今日实际持仓）

    for dt in dates:
        # ---- 基准 ----
        bmask = ret.loc[dt].notna()
        if bmask.sum() > 0:
            if mv is not None:
                w = mv.loc[dt, bmask]
                w = w / w.sum()
                prev_bench *= 1 + float((w * ret.loc[dt, bmask]).sum())
            else:
                prev_bench *= 1 + ret.loc[dt, bmask].mean()
        bench_nav.loc[dt] = prev_bench

        # ---- 当日持仓收益：吃"昨日持仓"的今日收益（T+1 模型）----
        if weights is not None:
            port_ret = (weights * ret.loc[dt].fillna(0.0)).sum()
            prev_nav *= 1 + port_ret
        nav.loc[dt] = prev_nav
        holdings.loc[dt] = weights if weights is not None else 0.0

        # ---- 调仓日：T 日收盘决策，次日生效 ----
        if dt in rebal_dates:
            s = score.loc[dt]
            mask = s.notna() & panel.volume.loc[dt].gt(0) & close_prev.loc[dt].notna()
            s = s[mask]
            new_w = pd.Series(0.0, index=close.columns)
            if len(s) >= top_n:
                # 剔除当日涨停（近似买不进）
                s = s[ret.loc[dt, s.index].fillna(0) < LIMIT_UP]
                if len(s) >= top_n:
                    picks = s.nlargest(top_n)
                    new_w[picks.index] = 1.0 / top_n

            if weights is not None:
                change = (new_w - weights).abs().sum()
                total_turn += change
                sell = (weights - new_w).clip(lower=0).sum()
                total_cost += change * cost["commission"] + sell * cost["stamp"] + change * cost["impact"]
            else:
                total_cost += new_w.sum() * (cost["commission"] + cost["impact"])
            weights = new_w
            holdings.loc[dt] = new_w

    years = max(len(dates) / 252, 1e-9)
    return BacktestResult(
        nav=nav, bench_nav=bench_nav, holdings=holdings,
        trades=pd.DataFrame(trades), turnover=total_turn / years,
        cost_pct=total_cost / max(total_turn, 1e-9),
        params={"top_n": top_n, "rebalance": rebalance, "cost": cost, "bench_mode": bench_mode},
    )
