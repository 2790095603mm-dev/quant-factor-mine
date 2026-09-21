"""绩效分析：净值统计 / 回撤 / 分年绩效 / 统一指标输出

`standard_metrics` 是全项目**唯一**的指标构造器：策略页、Tear Sheet、实验归档
都从它取数，避免同一份净值在不同页面显示成不同口径。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# 并排对比表的固定列：新增指标必须加到这里，避免出现"某个页面漏掉年化超额"的情况
SUMMARY_METRICS = (
    "总收益", "年化收益", "年化波动", "夏普比率", "索提诺比率",
    "最大回撤", "卡玛比率", "日胜率", "年化超额", "超额夏普", "信息比率", "跟踪误差",
)


def drawdown(nav: pd.Series) -> pd.Series:
    """回撤序列"""
    return nav / nav.cummax() - 1


def _daily_risk_free(risk_free_rate: float) -> float:
    if not np.isfinite(risk_free_rate) or risk_free_rate <= -1:
        raise ValueError("无风险年利率必须大于 -100%")
    return (1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1


def sortino_ratio(nav: pd.Series, risk_free_rate: float = 0.0) -> float:
    """索提诺比率：超额收益均值 / 下行波动（只惩罚负超额）。"""
    ret = nav.pct_change(fill_method=None).dropna()
    if len(ret) < 2:
        return float("nan")
    excess = ret - _daily_risk_free(risk_free_rate)
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("nan")
    deviation = downside.std()
    if not np.isfinite(deviation) or deviation == 0:
        return float("nan")
    return float(excess.mean() / deviation * np.sqrt(TRADING_DAYS))


def tracking_error(nav: pd.Series, bench: pd.Series) -> float:
    """年化跟踪误差：策略与基准日收益差的标准差 × √252（收益空间，非净值空间）。"""
    left = nav.pct_change(fill_method=None).dropna()
    right = bench.pct_change().dropna()
    aligned = pd.concat([left.rename("n"), right.rename("b")], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return float("nan")
    diff = aligned["n"] - aligned["b"]
    if not np.isfinite(diff.std()) or diff.std() == 0:
        return float("nan")
    return float(diff.std() * np.sqrt(TRADING_DAYS))


def perf_stats(nav: pd.Series, bench: pd.Series | None = None, risk_free_rate: float = 0.0) -> dict:
    """核心绩效指标"""
    daily_rf = _daily_risk_free(risk_free_rate)
    ret = nav.pct_change(fill_method=None).dropna()
    n = len(ret)
    if n < 2:
        return {}
    years = n / TRADING_DAYS
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    ann_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else np.nan
    ann_vol = ret.std() * np.sqrt(TRADING_DAYS)
    sharpe = (ret.mean() - daily_rf) / ret.std() * np.sqrt(TRADING_DAYS) if ann_vol > 0 else np.nan
    dd = drawdown(nav)
    mdd = dd.min()
    calmar = ann_ret / abs(mdd) if mdd < 0 else np.nan
    win = (ret > 0).mean()

    out = {
        "总收益": total_ret, "年化收益": ann_ret, "年化波动": ann_vol,
        "夏普比率": sharpe, "最大回撤": mdd, "卡玛比率": calmar,
        "日胜率": win, "交易天数": n,
        "索提诺比率": sortino_ratio(nav, risk_free_rate),
    }
    if bench is not None:
        b_ret = bench.pct_change().dropna()
        aligned = pd.concat([ret, b_ret], axis=1, join="inner").dropna()
        if len(aligned) > 20:
            excess = aligned.iloc[:, 0] - aligned.iloc[:, 1]
            out["年化超额"] = (1 + excess.mean()) ** TRADING_DAYS - 1
            out["超额夏普"] = excess.mean() / excess.std() * np.sqrt(TRADING_DAYS) if excess.std() > 0 else np.nan
            out["信息比率"] = out["超额夏普"]
            out["相关性"] = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
            out["基准年化"] = (1 + (bench.iloc[-1] / bench.iloc[0] - 1)) ** (1 / (len(b_ret) / TRADING_DAYS)) - 1
            out["跟踪误差"] = tracking_error(nav, bench)
    return out


def standard_metrics(nav: pd.Series, bench: pd.Series | None = None,
                     turnover: float | None = None, risk_free_rate: float = 0.0) -> dict:
    """统一指标输出：策略页 / Tear Sheet / 实验归档共用同一份口径。

    turnover 为年化单边换手（回测对象给出），不在净值里，需要显式传入。
    """
    out = perf_stats(nav, bench, risk_free_rate)
    if turnover is not None:
        out["年化换手"] = float(turnover)
    return out


def yearly_perf(nav: pd.Series, risk_free_rate: float = 0.0) -> pd.DataFrame:
    """分年绩效：收益/回撤/胜率"""
    ret = nav.pct_change(fill_method=None)
    rows = []
    for year, g in ret.groupby(ret.index.year):
        returns = g.dropna()
        # 年初第一天相对上年末的收益必须保留，回撤也从年初基准 1 开始。
        y_nav = pd.Series(np.r_[1.0, (1 + returns).cumprod().to_numpy()])
        rows.append({
            "年份": year,
            "收益": y_nav.iloc[-1] - 1 if len(returns) else np.nan,
            "最大回撤": drawdown(y_nav).min() if len(y_nav) > 1 else np.nan,
            "日胜率": (returns > 0).mean(),
            "夏普": (returns.mean() - ((1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1)) / returns.std() * np.sqrt(TRADING_DAYS) if len(returns) > 1 and returns.std() > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def summary_table(nav: pd.Series, bench: pd.Series | None = None) -> pd.DataFrame:
    """策略 vs 基准 对比表（列固定为 SUMMARY_METRICS，不会遗漏任何标准指标）。

    「策略」行以基准为参照计算超额/信息比率/跟踪误差；「基准」行的这些列恒为 NaN
    —— 基准相对自身的超额没有意义，留空比填 0 更不容易误读。
    """
    rows = []
    pairs = [("策略", nav, bench)]
    if bench is not None:
        pairs.append(("基准", bench, None))
    for label, series, reference in pairs:
        s = perf_stats(series, bench=reference)
        rows.append({"组合": label, **{key: s.get(key, np.nan) for key in SUMMARY_METRICS}})
    return pd.DataFrame(rows)
