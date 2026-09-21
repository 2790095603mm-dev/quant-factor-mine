"""同一信号的一次性诊断、交易模拟和绩效汇总。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import numpy as np
import pandas as pd

from qfm.data.panel import DataPanel
from qfm.factors import compute_factor, get_factor
from qfm.pipeline.pipeline import PipelineConfig, SignalResult, run_pipeline
from qfm.pipeline.tests import factor_report
from qfm.portfolio import BacktestResult, PortfolioConstraints, run_backtest
from qfm.portfolio.performance import perf_stats, yearly_perf


@dataclass(frozen=True)
class SimulationSettings:
    start: str = "2024-01-01"
    end: str = "2026-12-31"
    horizon: int = 20
    mode: str = "long_only"
    top_n: int = 30
    rebalance: str = "ME"
    delay: int = 1
    decay: int = 0
    neutralization: str = "none"
    max_weight: float = 0.10
    commission: float = 0.0003
    stamp: float = 0.0005
    impact: float = 0.001
    borrow_rate: float = 0.03
    risk_free_rate: float = 0.0
    initial_capital: float = 1_000_000
    max_participation: float = 0.05

    def to_dict(self):
        return asdict(self)

    def validate(self):
        if pd.Timestamp(self.start) >= pd.Timestamp(self.end):
            raise ValueError("开始日期必须早于结束日期")
        if self.delay < 1 or self.delay > 60:
            raise ValueError("Delay 必须在 1 到 60 之间；1 表示下一交易日开盘执行")
        if not 0 <= self.decay <= 120:
            raise ValueError("Decay 必须在 0 到 120 之间")
        if self.top_n < 1 or self.horizon < 1:
            raise ValueError("持仓数量和前瞻天数必须为正数")
        if self.mode not in {"long_only", "long_short"}:
            raise ValueError("未知组合模式")
        if self.neutralization not in {"none", "industry", "size", "industry_size"}:
            raise ValueError("未知中性化选项")
        if self.rebalance not in {"B", "W-FRI", "ME"}:
            raise ValueError("未知调仓频率")
        for name in ("commission", "stamp", "impact", "borrow_rate", "risk_free_rate"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError("费用和无风险利率必须为非负有限值")
        if not 0 < self.max_weight <= 1 or not 0 < self.max_participation <= 1 or self.initial_capital <= 0:
            raise ValueError("资金、权重上限和参与率必须为正数，比例不超过 100%")


@dataclass
class FactorSimulation:
    name: str
    settings: SimulationSettings
    report: dict
    backtest: BacktestResult
    gross: BacktestResult
    stats: dict
    yearly: pd.DataFrame
    coverage: float
    warnings: list[str]
    pipeline: SignalResult | None = None   # 统一管线留痕（各阶段覆盖率/中性化诊断）


def slice_panel(panel, dates):
    data = {}
    for field in fields(DataPanel):
        value = getattr(panel, field.name)
        if isinstance(value, pd.DataFrame):
            data[field.name] = value.reindex(index=dates, columns=panel.close.columns)
        elif field.name == "fund":
            data[field.name] = {k: v.reindex(index=dates, columns=panel.close.columns) for k, v in value.items()}
        else:
            data[field.name] = value
    return DataPanel(**data)


def build_signal(panel, raw, direction, settings) -> SignalResult:
    """信号处理入口：委托统一管线 qfm.pipeline.run_pipeline，返回完整管线留痕。

    历史实现把「清洗 → 平滑 → 逐日截面 OLS 中性化」内联在本函数里，与检验页使用的
    clean_factor 不是同一套代码，导致看到的 IC 与实际持仓来自不同信号对象。现在
    全部走统一管线，本函数只负责确定预热窗口。

    处理顺序：缺失 → 去极值 → 中性化 → 标准化 → 方向 → Decay 平滑 → Lag。
    引擎本身已按 T 日信号、T+1 开盘执行，因此 Lag 仅再 shift(delay-1)。
    """
    settings.validate()
    dates = panel.close.index
    start_pos = max(0, dates.searchsorted(pd.Timestamp(settings.start)) - settings.decay - settings.delay - 1)
    history = dates[start_pos:][dates[start_pos:] <= pd.Timestamp(settings.end)]
    # 只在本窗口上计算（含预热段），避免在全历史（8000+ 日）上做逐日 OLS
    window = slice_panel(panel, history)
    return run_pipeline(
        raw, panel=window, config=PipelineConfig.from_settings(settings),
        direction=direction, close=window.close,
    )


def prepare_signal(panel, raw, direction, settings):
    """历史接口：只返回管线输出信号，等价于 build_signal(...).signal。"""
    return build_signal(panel, raw, direction, settings).signal


def research_fitness(nav, daily_turnover, risk_free_rate=0.0):
    """本地 Fitness：日频 Sharpe × sqrt(|算术年化收益| / max(日均成交比例, .125))。"""
    returns = nav.pct_change(fill_method=None).dropna()
    if len(returns) < 20 or daily_turnover.empty:
        return float("nan")
    sharpe = perf_stats(nav, risk_free_rate=risk_free_rate).get("夏普比率", np.nan)
    turn = daily_turnover.mean()
    if not np.isfinite(sharpe) or not np.isfinite(turn):
        return float("nan")
    return float(sharpe * np.sqrt(abs(returns.mean() * 252) / max(turn, 0.125)))


def run_factor_simulation(panel, name, settings, progress=None, raw=None):
    settings.validate()
    factor = get_factor(name)
    if factor is None:
        raise ValueError(f"未知因子：{name}")
    dates = panel.close.index[(panel.close.index >= pd.Timestamp(settings.start)) & (panel.close.index <= pd.Timestamp(settings.end))]
    if len(dates) < max(21, settings.horizon + 2):
        raise ValueError("当前区间有效交易日不足，请扩大日期范围")
    required = settings.top_n * (2 if settings.mode == "long_short" else 1)
    if required > len(panel.close.columns):
        raise ValueError(f"股票数量不足：当前 {len(panel.close.columns)} 只，此配置至少需要 {required} 只")
    if progress:
        progress("计算因子与信号设置…")
    raw = compute_factor(name, panel) if raw is None else raw
    pipeline = build_signal(panel, raw, factor.direction, settings)
    signal = pipeline.signal.reindex(dates)
    if not signal.notna().any().any():
        raise ValueError("该区间没有有效因子值，请检查财务覆盖、Decay 窗口或中性化设置")
    sample = slice_panel(panel, dates)
    report = factor_report(signal, sample.close, horizon=settings.horizon, direction="positive",
                           preprocessed=True, pipeline=pipeline)
    costs = {key: getattr(settings, key) for key in ("commission", "stamp", "impact")}
    if progress:
        progress("模拟成本后组合与无费用对照…")
    if settings.mode == "long_only":
        kwargs = dict(top_n=settings.top_n, rebalance=settings.rebalance,
                      initial_capital=settings.initial_capital, max_participation=settings.max_participation,
                      constraints=PortfolioConstraints(max_stock_weight=settings.max_weight))
        backtest = run_backtest(sample, signal, cost=costs, **kwargs)
        gross = run_backtest(sample, signal, cost={key: 0.0 for key in costs}, **kwargs)
    else:
        from qfm.simulation.long_short import run_long_short
        backtest = run_long_short(sample, signal, settings)
        gross = run_long_short(sample, signal, settings, zero_cost=True)
    stats = perf_stats(backtest.nav, backtest.bench_nav, risk_free_rate=settings.risk_free_rate)
    turns = backtest.daily_turnover.iloc[1:]
    stats["日均换手"] = float(turns.mean())
    stats["Fitness"] = research_fitness(backtest.nav, turns, settings.risk_free_rate)
    stats["累计交易成本"] = backtest.cost_total
    coverage = float(signal.notna().to_numpy().mean())
    warnings = []
    if backtest.trades.empty:
        warnings.append("未产生成交：请检查有效股票数量、开盘价和调仓日期。净值为现金收益，不能据此判断因子有效。")
    if coverage < 0.8:
        warnings.append(f"信号覆盖率为 {coverage:.1%}；部分日期或股票缺少有效因子值。")
    if report["layer"].empty:
        warnings.append("五分层统计需要每日至少 100 只有效股票；当前样本不足，模拟绩效仍按实际成交计算。")
    if settings.mode == "long_short":
        warnings.append("多空研究模拟：目标多头 50% / 空头 50%，不含融券券源、涨跌停排队及成交参与率限制；每日计入设定的借券费。")
    yearly = yearly_perf(backtest.nav, risk_free_rate=settings.risk_free_rate)
    yearly["IC"] = yearly["年份"].map(report["ic_by_year"])
    return FactorSimulation(name, settings, report, backtest, gross, stats, yearly, coverage,
                            warnings, pipeline=pipeline)
