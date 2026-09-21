"""独立多空研究模型：次日开盘、50/50 目标、固定股数盯市及费用。

不模拟融券券源、涨跌停排队与参与率限制，不作为 A 股可执行组合。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from qfm.portfolio.backtest import BacktestResult, _rebalance_dates


def run_long_short(panel, score, settings, zero_cost=False) -> BacktestResult:
    dates, columns = panel.close.index, panel.close.columns
    signal_dates = _rebalance_dates(dates, settings.rebalance)
    prices = panel.close.ffill()
    shares = pd.Series(0.0, index=columns)
    cash = settings.initial_capital
    nav = pd.Series(1.0, index=dates)
    holdings = pd.DataFrame(0.0, index=dates, columns=columns)
    cash_weight = pd.Series(1.0, index=dates)
    turnover = pd.Series(0.0, index=dates)
    records, pending = [], None
    total_cost, total_notional = 0.0, 0.0
    borrow_rate = 0.0 if zero_cost else settings.borrow_rate
    commission = 0.0 if zero_cost else settings.commission + settings.impact
    stamp = 0.0 if zero_cost else settings.stamp

    for pos, date in enumerate(dates):
        # 上一收盘已有的空头计一天借券费；本日新开空头从下一期计费。
        if pos:
            short_value = float((-shares.clip(upper=0) * prices.iloc[pos - 1]).sum())
            fee = short_value * borrow_rate / 252
            cash -= fee
            total_cost += fee
        if pending is not None:
            signal_date, target = pending
            open_px = panel.open.loc[date]
            mark = open_px.where(open_px.gt(0), prices.iloc[max(0, pos - 1)])
            equity = cash + float((shares * mark).sum())
            if not np.isfinite(equity) or equity <= 0:
                raise ValueError("多空研究组合净资产已耗尽，请缩短区间或降低仓位")
            tradable = open_px.gt(0) & panel.volume.loc[date].gt(0)
            desired_shares = (target * equity / open_px).where(tradable, shares)
            delta = desired_shares - shares
            for stock in delta.index[delta.abs().gt(1e-12) & tradable]:
                qty, px = float(delta[stock]), float(open_px[stock])
                notional = abs(qty * px)
                fee = notional * (commission + (stamp if qty < 0 else 0.0))
                cash -= qty * px + fee
                shares[stock] += qty
                total_notional += notional
                total_cost += fee
                turnover.loc[date] += notional / equity
                records.append({"date": date, "signal_date": signal_date, "stock": stock,
                                "side": "BUY" if qty > 0 else "SELL", "notional": notional,
                                "price": px, "quantity": abs(qty), "cost": fee})
            pending = None
        values = shares * prices.loc[date]
        equity = cash + float(values.sum())
        if not np.isfinite(equity) or equity <= 0:
            raise ValueError("多空研究组合净资产已耗尽，请缩短区间或降低仓位")
        nav.loc[date] = equity / settings.initial_capital
        holdings.loc[date] = values.fillna(0) / equity
        cash_weight.loc[date] = cash / equity
        if date in signal_dates:
            ranked = score.loc[date].where(panel.volume.loc[date].gt(0)).dropna().sort_values(kind="stable")
            target = pd.Series(0.0, index=columns)
            if len(ranked) >= 2 * settings.top_n:
                weight = min(0.5 / settings.top_n, settings.max_weight)
                target.loc[ranked.index[:settings.top_n]] = -weight
                target.loc[ranked.index[-settings.top_n:]] = weight
            pending = date, target

    returns = panel.close.pct_change(fill_method=None).mean(axis=1).fillna(0)
    returns.iloc[0] = 0
    bench = (1 + returns).cumprod()
    years = max((len(dates) - 1) / 252, 1 / 252)
    trades = pd.DataFrame(records, columns=["date", "signal_date", "stock", "side", "notional", "price", "quantity", "cost"])
    return BacktestResult(
        nav=nav, bench_nav=bench, holdings=holdings, trades=trades,
        turnover=total_notional / 2 / settings.initial_capital / years,
        cost_pct=float(trades.cost.sum()) / total_notional if total_notional else 0,
        cost_total=total_cost / settings.initial_capital, cash_weight=cash_weight,
        daily_turnover=turnover,
        params={"execution": "next_open", "model": "theoretical_long_short", "constraints": {"max_stock_weight": settings.max_weight}},
    )
