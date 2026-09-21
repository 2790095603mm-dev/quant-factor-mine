"""组合回测：T 日收盘信号、T+1 开盘成交、成本和可交易性约束。"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from qfm.portfolio.constraints import (
    PortfolioConstraints,
    apply_turnover_budget,
    constrained_target_weights,
)

# A 股成本参数（可由页面覆盖）：佣金双边、印花税卖出单边、冲击成本双边。
DEFAULT_COST = {"commission": 0.0003, "stamp": 0.0005, "impact": 0.0010}
UNKNOWN_INDUSTRY = "未知行业"
# ST 股票涨跌幅限制为 5%
ST_LIMIT_RATIO = 0.05


@dataclass
class BacktestResult:
    nav: pd.Series                    # 策略净值（日频，已扣成本）
    bench_nav: pd.Series              # 基准净值（未计交易成本）
    holdings: pd.DataFrame            # 每日收盘持仓权重（不含现金）
    trades: pd.DataFrame              # 实际成交记录（T+1 开盘）
    turnover: float                   # 年化单边换手
    cost_pct: float                   # 成本 / 实际成交额
    cost_total: float                 # 累计成本 / 初始资金
    cash_weight: pd.Series            # 每日收盘现金权重
    industry_exposure: pd.DataFrame | None = None
    constraint_history: pd.DataFrame | None = None
    params: dict = field(default_factory=dict)
    daily_turnover: pd.Series | None = None  # 当日买卖成交额之和 / 开盘交易前净资产


def _rebalance_dates(dates: pd.DatetimeIndex, rebalance: str | int) -> set[pd.Timestamp]:
    """返回信号生成日；成交统一在下一个交易日开盘。

    rebalance 支持：
    - 日历口径字符串："B"（每日）、"W-FRI"/"W"（每周五）、"ME"（月末）、"QE"（季末）
    - 整数 N：每 N 个交易日一次（研究里常用的固定间隔调仓）
    """
    if isinstance(rebalance, bool):
        raise ValueError("调仓频率不接受布尔值")
    if isinstance(rebalance, int):
        if rebalance < 1:
            raise ValueError("每 N 个交易日调仓时 N 必须为正整数")
        return set(dates[::rebalance])
    aliases = {"ME": "M", "W": "W-FRI", "W-FRI": "W-FRI", "B": "B", "D": "B", "QE": "Q"}
    if rebalance not in aliases:
        raise ValueError(f"未知调仓频率: {rebalance}")
    period = aliases[rebalance]
    if period == "B":
        return set(dates)
    periods = dates.to_period(period)
    return set(pd.Series(dates, index=dates).groupby(periods).max().tolist())


def _limit_ratio(code: str, is_st: bool = False) -> float:
    """按代码给出近似涨跌停幅度。

    is_st=True 时按 5%（ST 股票涨跌幅限制更严）。当前数据源不提供 ST 状态，
    调用方需要通过 `st` 参数显式传入标记面板，否则无从判断。
    """
    if is_st:
        return ST_LIMIT_RATIO
    code = str(code)
    if code.startswith("8"):
        return 0.30
    if code.startswith(("30", "68")):
        return 0.20
    return 0.10


def _target_weights(score: pd.Series, volume: pd.Series, top_n: int) -> pd.Series:
    """基于 T 日可见分数生成等权目标；不足 top_n 时保留空仓。"""
    valid = score.notna() & volume.gt(0)
    ranked = score[valid].nlargest(top_n)
    if len(ranked) < top_n:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / top_n, index=ranked.index, dtype=float)


def _benchmark_step(panel, dt: pd.Timestamp, prev_close: pd.Series,
                    bench_mode: str) -> float:
    """基准以收盘到收盘收益计算；无仓位时返回 0。"""
    close = panel.close.loc[dt]
    ret = close / prev_close - 1
    mask = ret.notna()
    if not mask.any():
        return 0.0
    if bench_mode == "mv":
        mv = panel.mv_float.loc[dt, mask].clip(lower=0).dropna()
        if mv.sum() > 0:
            return float((ret.loc[mv.index] * (mv / mv.sum())).sum())
    return float(ret.loc[mask].mean())


def _actual_industry_labels(panel, dt: pd.Timestamp, index: pd.Index) -> pd.Series:
    """按执行日收盘口径获取行业；缺失字段统一归入未知行业。"""
    industry = getattr(panel, "industry", None)
    if isinstance(industry, pd.DataFrame) and dt in industry.index:
        labels = industry.loc[dt].reindex(index).astype(object)
    else:
        labels = pd.Series(UNKNOWN_INDUSTRY, index=index, dtype=object)
    return labels.where(labels.notna(), UNKNOWN_INDUSTRY).astype(str)


def _normalise_st(st: pd.DataFrame | None, dates: pd.DatetimeIndex,
                  columns: pd.Index) -> pd.DataFrame | None:
    """把 ST 标记面板对齐到回测网格；缺失视为非 ST。"""
    if st is None:
        return None
    if not isinstance(st, pd.DataFrame) or st.empty:
        raise ValueError("st 必须是 date×stock 的布尔面板")
    return (
        st.reindex(index=dates, columns=columns)
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )


def _normalise_benchmark(benchmark: pd.Series | None,
                         dates: pd.DatetimeIndex) -> pd.Series | None:
    """把外部基准序列对齐到回测日期并归一化到起点 1.0。"""
    if benchmark is None:
        return None
    if not isinstance(benchmark, pd.Series) or benchmark.empty:
        raise ValueError("benchmark 必须是非空的价格/净值序列")
    series = pd.Series(benchmark).astype(float).sort_index()
    series = series[~series.index.duplicated(keep="last")].reindex(dates).ffill()
    valid = series.dropna()
    if len(valid) < 2:
        raise ValueError("benchmark 与回测区间没有足够重叠，无法作为基准")
    base = float(valid.iloc[0])
    if base <= 0:
        raise ValueError("benchmark 起始值必须为正数")
    normalised = (series / base).ffill()
    # 区间起点之前没有历史时用 1.0 补齐，保证与 nav 同轴
    return normalised.fillna(1.0)


def run_backtest(panel, score: pd.DataFrame, top_n: int = 50, rebalance: str | int = "ME",
                 cost: dict | None = None, start: str | None = None, end: str | None = None,
                 bench_mode: str = "equal", initial_capital: float = 1_000_000,
                 max_participation: float = 0.05,
                 store_holdings: bool = True,
                 constraints: PortfolioConstraints | None = None,
                 slippage: float = 0.0,
                 benchmark: pd.Series | None = None,
                 st: pd.DataFrame | None = None) -> BacktestResult:
    """运行一个成本后、T+1 开盘成交的多头组合回测。

    T 日收盘形成选股信号，T+1 以开盘价成交。买入涨停、卖出跌停、停牌和
    超过当日成交额参与率上限的订单会部分成交或不成交，剩余资金保留为现金。

    slippage : 单边滑点（比例，如 0.0005 = 5bp）。买入成交价 = 开盘 × (1+slippage)，
               卖出 = 开盘 × (1-slippage)。与 impact（按成交额计提的冲击成本）相互独立：
               slippage 改的是**成交价**，impact 改的是**费用**。
    benchmark: 外部基准净值序列（如指数收盘价）。提供时覆盖 bench_mode 的等权/市值加权基准。
    st       : 布尔面板（date×stock），True 表示当日为 ST。ST 股票按 5% 涨跌幅限制，
               且**不进入选股目标**（已持有的仍可卖出）。当前数据源不提供 ST 状态，
               不传该参数时按普通股票处理 —— 这是数据缺口，不是已实现的功能。
    """
    if top_n <= 0:
        raise ValueError("top_n 必须为正数")
    if initial_capital <= 0:
        raise ValueError("initial_capital 必须为正数")
    if not 0 < max_participation <= 1:
        raise ValueError("max_participation 必须在 (0, 1] 内")
    if bench_mode not in {"equal", "mv"}:
        raise ValueError(f"未知基准模式: {bench_mode}")
    if not 0 <= slippage < 1:
        raise ValueError("滑点必须在 [0, 1) 内")

    constraints = constraints or PortfolioConstraints()
    cost = {**DEFAULT_COST, **(cost or {})}
    close = panel.close
    dates = close.index
    if start:
        dates = dates[dates >= pd.Timestamp(start)]
    if end:
        dates = dates[dates <= pd.Timestamp(end)]
    if len(dates) < 2:
        raise ValueError("回测区间至少需要两个交易日")

    score = score.reindex(index=close.index, columns=close.columns)
    rebal_dates = _rebalance_dates(dates, rebalance)
    columns = close.columns
    st_flags = _normalise_st(st, dates, columns)
    external_bench = _normalise_benchmark(benchmark, dates)
    shares = pd.Series(0.0, index=columns)
    cash = float(initial_capital)
    nav = pd.Series(index=dates, dtype=float)
    bench_nav = pd.Series(index=dates, dtype=float)
    cash_weight = pd.Series(index=dates, dtype=float)
    daily_turnover = pd.Series(0.0, index=dates)
    holdings = (pd.DataFrame(0.0, index=dates, columns=columns)
                if store_holdings else pd.DataFrame(index=dates, columns=columns, dtype=float))
    trades: list[dict] = []
    total_cost = 0.0
    total_notional = 0.0
    pending: tuple[pd.Timestamp, pd.Series, dict[str, object]] | None = None
    prev_bench = 1.0
    industry_rows: list[pd.Series] = []
    constraint_rows: list[dict[str, object]] = []
    close_prev = close.shift(1)

    for pos, dt in enumerate(dates):
        # ---- T+1 开盘执行前一调仓日的选股信号 ----
        did_rebalance = False
        execution_audit: dict[str, object] | None = None
        execution_target: pd.Series | None = None
        if pending is not None:
            notional_before = total_notional
            signal_date, raw_target, target_diag = pending
            open_px = panel.open.loc[dt].reindex(columns)
            prev_px = close_prev.loc[dt].reindex(columns)
            amount = panel.amount.loc[dt].reindex(columns).fillna(0.0)
            volume = panel.volume.loc[dt].reindex(columns).fillna(0.0)
            is_st = (
                st_flags.loc[dt].reindex(columns).fillna(False)
                if st_flags is not None else pd.Series(False, index=columns)
            )
            tradeable = open_px.notna() & open_px.gt(0) & prev_px.notna() & volume.gt(0)
            move = open_px / prev_px - 1
            limits = pd.Series({_code: _limit_ratio(_code, bool(is_st.get(_code, False)))
                                for _code in columns})
            # 滑点体现在成交价上；ST 股票不允许新建仓位（已持有的仍可卖出离场）
            buy_px = open_px * (1 + slippage)
            sell_px = open_px * (1 - slippage)
            can_buy = tradeable & ~is_st & move.lt(limits - 0.0005)
            can_sell = tradeable & move.gt(-limits + 0.0005)

            active = shares[shares.gt(1e-12)].index
            active_values = (shares.loc[active] * open_px.loc[active]).dropna()
            pretrade_value = cash + float(active_values.sum())
            current = active_values.reindex(columns, fill_value=0.0)
            current_weights = current / pretrade_value if pretrade_value > 0 else pd.Series(0.0, index=columns)
            target, turnover_diag = apply_turnover_budget(
                current_weights,
                raw_target,
                constraints.max_rebalance_turnover,
            )
            desired = target * pretrade_value
            capacity = amount * max_participation
            execution_target = target
            execution_audit = {
                "signal_date": signal_date,
                "execution_date": dt,
                **target_diag,
                **turnover_diag,
            }

            # 先卖出，为新持仓腾出现金；跌停/停牌的股票保留原仓位。
            desired_all = desired.reindex(columns, fill_value=0.0)
            sell_names = current[current.gt(desired_all + 1e-8)].index
            for code in sell_names:
                if not bool(can_sell.get(code, False)):
                    continue
                wanted = float(current[code] - desired_all[code])
                notional = min(wanted, float(capacity.get(code, 0.0)))
                if notional <= 1e-8:
                    continue
                price = float(sell_px[code])
                quantity = min(float(shares[code]), notional / price)
                notional = quantity * price
                fee = notional * (cost["commission"] + cost["stamp"] + cost["impact"])
                shares[code] -= quantity
                cash += notional - fee
                total_cost += fee
                total_notional += notional
                trades.append({"date": dt, "signal_date": signal_date, "stock": code,
                               "side": "SELL", "notional": notional, "price": price,
                               "quantity": quantity, "cost": fee})

            # 再买入。若前序卖单受限，按可用现金部分成交，绝不借款。
            active = shares[shares.gt(1e-12)].index
            current = (shares.loc[active] * open_px.loc[active]).reindex(columns, fill_value=0.0)
            buy_gap = (desired_all - current).clip(lower=0.0)
            buy_rate = cost["commission"] + cost["impact"]
            for code, wanted in buy_gap.sort_values(ascending=False).items():
                if wanted <= 1e-8 or not bool(can_buy.get(code, False)):
                    continue
                notional = min(float(wanted), float(capacity.get(code, 0.0)), cash / (1 + buy_rate))
                if notional <= 1e-8:
                    continue
                price = float(buy_px[code])
                quantity = notional / price
                fee = notional * buy_rate
                shares[code] += quantity
                cash -= notional + fee
                total_cost += fee
                total_notional += notional
                trades.append({"date": dt, "signal_date": signal_date, "stock": code,
                               "side": "BUY", "notional": notional, "price": price,
                               "quantity": quantity, "cost": fee})
            pending = None
            daily_turnover.loc[dt] = (total_notional - notional_before) / pretrade_value if pretrade_value > 0 else 0.0
            did_rebalance = True

        # ---- 收盘盯市：已成交仓位获得开盘到收盘收益，现金保留。 ----
        active = shares[shares.gt(1e-12)].index
        values = (shares.loc[active] * close.loc[dt, active]).dropna()
        value = cash + float(values.sum())
        nav.loc[dt] = value / initial_capital
        cash_weight.loc[dt] = cash / value if value > 0 else 1.0
        if store_holdings and value > 0 and len(values):
            holdings.loc[dt, values.index] = values / value

        if did_rebalance and len(values) and value > 0:
            industries = _actual_industry_labels(panel, dt, values.index)
            weights = (values / value).groupby(industries).sum()
            industry_rows.append(weights.rename(dt))

        if execution_audit is not None and execution_target is not None:
            actual_weights = values / value if value > 0 else pd.Series(dtype=float)
            audit_index = actual_weights.index.union(execution_target.index)
            actual_all = actual_weights.reindex(audit_index, fill_value=0.0)
            target_all = execution_target.reindex(audit_index, fill_value=0.0)
            actual_industry = actual_weights.groupby(
                _actual_industry_labels(panel, dt, actual_weights.index)
            ).sum()
            actual_max_stock = float(actual_weights.max()) if len(actual_weights) else 0.0
            actual_max_industry = float(actual_industry.max()) if len(actual_industry) else 0.0
            constraint_rows.append({
                **execution_audit,
                "actual_invested": float(actual_weights.sum()),
                "actual_cash": float(cash_weight.loc[dt]),
                "actual_positions": int(actual_weights.gt(1e-12).sum()),
                "actual_max_stock_weight": actual_max_stock,
                "actual_max_industry_weight": actual_max_industry,
                "target_tracking_error": float((actual_all - target_all).abs().sum()),
                "stock_cap_exceeded": actual_max_stock > constraints.max_stock_weight + 1e-8,
                "industry_cap_exceeded": actual_max_industry > constraints.max_industry_weight + 1e-8,
            })

        # 基准：外部序列优先；否则第二个交易日起按收盘到收盘收益累计。
        if external_bench is not None:
            bench_nav.loc[dt] = float(external_bench.loc[dt])
        else:
            if pos > 0:
                prev_bench *= 1 + _benchmark_step(panel, dt, close_prev.loc[dt], bench_mode)
            bench_nav.loc[dt] = prev_bench

        # ---- T 日收盘形成信号，下一交易日开盘成交 ----
        if dt in rebal_dates:
            industry_at_signal = None
            if isinstance(getattr(panel, "industry", None), pd.DataFrame) and dt in panel.industry.index:
                industry_at_signal = panel.industry.loc[dt]
            signal_score = score.loc[dt]
            if st_flags is not None and dt in st_flags.index:
                # ST 不进入选股目标；已持有的 ST 仍可按 5% 跌停规则卖出离场
                signal_score = signal_score.where(
                    ~st_flags.loc[dt].reindex(columns).fillna(False)
                )
            target, target_diag = constrained_target_weights(
                signal_score,
                panel.volume.loc[dt],
                industry_at_signal,
                top_n,
                constraints,
            )
            if len(target):
                pending = (dt, target, target_diag)

    years = max((len(dates) - 1) / 252, 1e-9)
    industry_exposure = pd.DataFrame(industry_rows).fillna(0.0) if industry_rows else None
    constraint_history = pd.DataFrame(constraint_rows) if constraint_rows else None
    trades_df = pd.DataFrame(trades, columns=["date", "signal_date", "stock", "side", "notional",
                                               "price", "quantity", "cost"])
    return BacktestResult(
        nav=nav,
        bench_nav=bench_nav,
        holdings=holdings,
        trades=trades_df,
        turnover=(total_notional / 2) / initial_capital / years,
        cost_pct=total_cost / total_notional if total_notional > 0 else 0.0,
        cost_total=total_cost / initial_capital,
        cash_weight=cash_weight,
        industry_exposure=industry_exposure,
        constraint_history=constraint_history,
        daily_turnover=daily_turnover,
        params={"top_n": top_n, "rebalance": rebalance, "cost": cost,
                "bench_mode": bench_mode, "initial_capital": initial_capital,
                "max_participation": max_participation, "execution": "next_open",
                "slippage": slippage,
                "benchmark_source": "external" if external_bench is not None else bench_mode,
                "st_filter": st_flags is not None,
                "constraints": constraints.to_dict()},
    )
