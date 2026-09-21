"""单因子 Tear Sheet：一次调用给出全部检验维度。

同时提供**两种口径**，页面上分区块展示，避免把预测统计误读成可交易业绩：

- 预测统计口径：各层「未来 h 日收益均值」、分层单调性、多空价差标量 —— 重叠窗口、不复利，
  回答"因子有没有预测力"；
- 可交易净值口径：按固定间隔（默认每 horizon 个交易日）非重叠调仓、层内等权、逐期复利的
  分层净值与多空净值 —— 回答"照着这个因子做组合会怎样"。

复用的既有实现：IC/ICIR/滚动/换手全部来自 qfm.pipeline.tests.factor_report，
分层等频分桶沿用 layer_test 的口径，截面分布与暴露为本模块新增。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DIST_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
MIN_STOCKS_PER_GROUP = 5
DEFAULT_EXPOSURE_SAMPLE = 5  # 行业/市值暴露每 N 个交易日采样一次（均值口径不受影响）


@dataclass
class TearSheet:
    """单因子全景诊断结果。"""

    name: str
    horizon: int
    direction: str
    meta: dict = field(default_factory=dict)
    # 预测能力
    ic: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    ic_summary: dict = field(default_factory=dict)
    ic_rolling: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    ic_rolling_ir: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    ic_cumulative: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    ic_by_year: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    # 分层（预测统计口径）
    layer_stat: pd.DataFrame = field(default_factory=pd.DataFrame)
    long_short_spread: float = float("nan")
    monotonicity: dict = field(default_factory=dict)
    # 分层（可交易净值口径）
    layer_nav: pd.DataFrame = field(default_factory=pd.DataFrame)
    long_short_nav: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    layer_period_returns: pd.DataFrame = field(default_factory=pd.DataFrame)
    # 换手与覆盖
    factor_turnover: float = float("nan")
    turnover_series: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    coverage: dict = field(default_factory=dict)
    # 分布
    distribution: pd.DataFrame = field(default_factory=pd.DataFrame)
    latest_snapshot: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    histogram: dict = field(default_factory=dict)
    # 暴露
    industry_exposure: pd.DataFrame = field(default_factory=pd.DataFrame)
    size_exposure: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    size_exposure_mean: float = float("nan")
    warnings: list[str] = field(default_factory=list)

    def kpis(self) -> dict:
        """页头 KPI（一次取齐，避免各处各自拼装口径）。"""
        s = self.ic_summary
        return {
            "平均 IC": s.get("ic_mean", float("nan")),
            "IC 均值绝对值": abs(s.get("ic_mean", float("nan"))) if pd.notna(s.get("ic_mean")) else float("nan"),
            "IC_IR": s.get("ic_ir", float("nan")),
            "IC t 值": s.get("ic_t", float("nan")),
            "IC 正占比": s.get("pos_ratio", float("nan")),
            "有效天数": s.get("n_days", 0),
            "多空价差（统计）": self.long_short_spread,
            "多空累计（可交易）": (self.long_short_nav.iloc[-1] - 1) if len(self.long_short_nav) else float("nan"),
            "组合换手": self.factor_turnover,
            "信号覆盖率": self.coverage.get("signal", float("nan")),
            "市值暴露": self.size_exposure_mean,
        }


def _quantile_layer_returns(signal: pd.DataFrame, close: pd.DataFrame, n_layers: int,
                            rebalance: int) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """非重叠分层的等权持有收益。

    在第 t 个调仓日按信号分 n 层，层内等权买入并持有到下一个调仓日收盘。
    返回 (各层逐期收益, 多空逐期收益, 各层净值)。
    """
    dates = signal.index
    grid = list(dates[::rebalance])
    if len(grid) < 3:
        empty = pd.DataFrame()
        return empty, pd.Series(dtype=float), empty

    rows: dict[pd.Timestamp, dict[int, float]] = {}
    for i in range(len(grid) - 1):
        start, end = grid[i], grid[i + 1]
        valid = signal.loc[start].notna() & close.loc[start].notna() & close.loc[end].notna()
        scores = signal.loc[start][valid]
        if len(scores) < n_layers * MIN_STOCKS_PER_GROUP:
            continue
        ranks = scores.rank(method="first")
        # 等频分层：名次 → 层号 1..n（1 为得分最低层）
        layer_of = np.ceil(ranks * n_layers / len(scores)).clip(upper=n_layers).astype(int)
        period_return = close.loc[end] / close.loc[start] - 1
        rows[end] = {
            int(layer): float(period_return[scores.index[layer_of == layer]].mean())
            for layer in range(1, n_layers + 1)
        }

    if not rows:
        empty = pd.DataFrame()
        return empty, pd.Series(dtype=float), empty

    period = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    period.index.name = "date"
    nav = (1 + period).cumprod()
    long_short = period.iloc[:, -1] - period.iloc[:, 0]
    return period, long_short, nav


def _distribution(signal: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict]:
    """截面分布时序 + 最新截面快照与直方图。"""
    counts = signal.notna().sum(axis=1)
    usable = signal.loc[counts >= MIN_STOCKS_PER_GROUP * 2]
    if usable.empty:
        return pd.DataFrame(), pd.Series(dtype=float), {}
    dist = pd.DataFrame({
        "样本数": counts.loc[usable.index],
        "均值": usable.mean(axis=1),
        "标准差": usable.std(axis=1),
        "偏度": usable.skew(axis=1),
        "峰度": usable.kurt(axis=1),
    })
    for q in DIST_QUANTILES:
        dist[f"P{int(q * 100)}"] = usable.quantile(q, axis=1)

    latest = usable.iloc[-1].dropna()
    histogram: dict = {}
    if len(latest) >= 10:
        values = latest.to_numpy(dtype=float)
        counts_hist, edges = np.histogram(values, bins=30)
        histogram = {
            "date": usable.index[-1],
            "bin_edges": edges.tolist(),
            "counts": counts_hist.tolist(),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return dist, latest, histogram


def _industry_exposure(signal: pd.DataFrame, industry: pd.DataFrame,
                       sample: int) -> pd.DataFrame:
    """各行业内的因子均值暴露（按交易日平均）。

    对 800 只 × 8000 日直接 stack 会产生千万行长表，这里按 sample 抽样交易日，
    对"平均暴露"这一口径没有实质影响。
    """
    if not isinstance(industry, pd.DataFrame) or industry.empty:
        return pd.DataFrame()
    dates = signal.index[::sample]
    rows = []
    for date in dates:
        if date not in industry.index:
            continue
        labels = industry.loc[date]
        values = signal.loc[date]
        valid = values.notna() & labels.notna()
        if valid.sum() < MIN_STOCKS_PER_GROUP:
            continue
        frame = pd.DataFrame({"industry": labels[valid], "signal": values[valid]})
        grouped = frame.groupby("industry")["signal"].agg(["mean", "count"])
        grouped["date"] = date
        rows.append(grouped.reset_index())
    if not rows:
        return pd.DataFrame()
    long = pd.concat(rows, ignore_index=True)
    summary = long.groupby("industry").agg(
        平均暴露=("mean", "mean"),
        暴露标准差=("mean", "std"),
        平均成分数=("count", "mean"),
        样本日数=("mean", "size"),
    )
    return summary.sort_values("平均暴露", ascending=False)


def _size_exposure(signal: pd.DataFrame, mv_float: pd.DataFrame, sample: int) -> pd.Series:
    """因子与 log(流通市值) 的逐日截面相关（市值暴露）。"""
    if not isinstance(mv_float, pd.DataFrame) or mv_float.empty:
        return pd.Series(dtype=float)
    log_mv = np.log(mv_float.where(mv_float > 0))
    out = {}
    for date in signal.index[::sample]:
        if date not in log_mv.index:
            continue
        x, y = signal.loc[date], log_mv.loc[date]
        valid = x.notna() & y.notna()
        if valid.sum() < MIN_STOCKS_PER_GROUP * 2 or y[valid].std() == 0:
            continue
        out[date] = float(x[valid].corr(y[valid]))
    return pd.Series(out, dtype=float).sort_index()


def build_tear_sheet(panel, signal: pd.DataFrame, horizon: int = 20, name: str = "factor",
                     direction: str = "positive", report: dict | None = None,
                     n_layers: int = 5, rebalance: int | None = None, pipeline=None,
                     exposure_sample: int = DEFAULT_EXPOSURE_SAMPLE,
                     meta: dict | None = None) -> TearSheet:
    """构建单因子 Tear Sheet。

    signal   : 统一管线输出的交易信号（date×stock，值越大越好）
    report   : 可传入 factor_report 的结果以复用已算好的 IC/分层，避免重复计算
    pipeline : 可传入 SignalResult，用于展示各阶段覆盖率与中性化诊断
    """
    if not isinstance(signal, pd.DataFrame) or signal.empty:
        raise ValueError("信号为空，无法构建 Tear Sheet")
    from qfm.pipeline.tests import factor_report, turnover_ratio

    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("前瞻天数必须为正整数")
    if rebalance is None:
        step = horizon
    else:
        step = int(rebalance)
        if step < 1:
            raise ValueError("调仓间隔必须为正整数")

    close = panel.close
    signal = signal.reindex(index=close.index, columns=close.columns)

    if report is None:
        report = factor_report(signal, close, horizon=horizon, preprocessed=True, pipeline=pipeline)

    sheet = TearSheet(name=str(name), horizon=horizon, direction=str(direction), meta=dict(meta or {}))
    sheet.ic = report["ic_series"]
    sheet.ic_summary = dict(report["ic_summary"])
    sheet.ic_rolling = report["ic_rolling"]
    sheet.ic_by_year = report["ic_by_year"]
    # 累计 IC 与滚动 ICIR：滚动 IC 已经复用报告里的 120 日均线，ICIR 由 IC/滚动标准差得到
    sheet.ic_cumulative = sheet.ic.fillna(0.0).cumsum()
    rolling_std = sheet.ic.rolling(120, min_periods=30).std()
    sheet.ic_rolling_ir = (sheet.ic_rolling / rolling_std.replace(0, np.nan)).rename("ICIR120")

    sheet.layer_stat = report["layer"]
    sheet.monotonicity = dict(report["monotonicity"])
    sheet.long_short_spread = float(report["monotonicity"].get("spread", float("nan")))
    sheet.factor_turnover = report["turnover"]
    sheet.coverage = dict(report.get("coverage") or {})

    period, long_short, nav = _quantile_layer_returns(signal, close, n_layers, step)
    sheet.layer_period_returns = period
    sheet.layer_nav = nav
    # 多空净值同样按复利口径，与分层净值一致（层均值为个股收益均值，1+r 恒为正）
    sheet.long_short_nav = (
        (1 + long_short).cumprod().rename("多空净值") if len(long_short) else pd.Series(dtype=float)
    )

    # 换手时序：与 factor_turnover 同口径（前 10% 名单的 Jaccard 距离），逐期展开
    grid = list(signal.index[::step])
    prev = None
    turns = {}
    for date in grid:
        row = signal.loc[date].dropna()
        if len(row) < 20:
            continue
        current = set(row.nlargest(max(1, int(len(row) * 0.1))).index)
        if prev:
            turns[date] = 1 - len(prev & current) / len(prev | current)
        prev = current
    sheet.turnover_series = pd.Series(turns, dtype=float).sort_index()
    if not np.isfinite(sheet.factor_turnover) and len(sheet.turnover_series):
        sheet.factor_turnover = float(sheet.turnover_series.mean())

    sheet.distribution, sheet.latest_snapshot, sheet.histogram = _distribution(signal)
    sheet.industry_exposure = _industry_exposure(signal, getattr(panel, "industry", None), exposure_sample)
    sheet.size_exposure = _size_exposure(signal, getattr(panel, "mv_float", None), exposure_sample)
    if len(sheet.size_exposure):
        sheet.size_exposure_mean = float(sheet.size_exposure.mean())

    if sheet.layer_stat.empty:
        sheet.warnings.append(
            f"分层统计需要每日至少 {20 * n_layers} 只有效股票；当前样本不足，分层与多空结果不可用。"
        )
    if sheet.layer_nav.empty:
        sheet.warnings.append(
            f"可交易分层净值需要每次调仓至少 {n_layers * MIN_STOCKS_PER_GROUP} 只有效股票；当前样本不足。"
        )
    if sheet.coverage and sheet.coverage.get("signal", 1.0) < 0.8:
        sheet.warnings.append(f"信号覆盖率仅 {sheet.coverage['signal']:.1%}，部分日期或股票缺少有效值。")
    if np.isfinite(sheet.size_exposure_mean) and abs(sheet.size_exposure_mean) > 0.3:
        sheet.warnings.append(
            f"因子与流通市值相关性达 {sheet.size_exposure_mean:+.2f}，规模暴露显著，建议中性化后再评估。"
        )
    return sheet
