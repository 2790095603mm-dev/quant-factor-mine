"""绩效分析：净值统计 / 回撤 / 分年绩效"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def drawdown(nav: pd.Series) -> pd.Series:
    """回撤序列"""
    return nav / nav.cummax() - 1


def perf_stats(nav: pd.Series, bench: pd.Series | None = None) -> dict:
    """核心绩效指标"""
    ret = nav.pct_change().dropna()
    n = len(ret)
    if n < 2:
        return {}
    years = n / TRADING_DAYS
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    ann_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else np.nan
    ann_vol = ret.std() * np.sqrt(TRADING_DAYS)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    dd = drawdown(nav)
    mdd = dd.min()
    calmar = ann_ret / abs(mdd) if mdd < 0 else np.nan
    win = (ret > 0).mean()

    out = {
        "总收益": total_ret, "年化收益": ann_ret, "年化波动": ann_vol,
        "夏普比率": sharpe, "最大回撤": mdd, "卡玛比率": calmar,
        "日胜率": win, "交易天数": n,
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
    return out


def yearly_perf(nav: pd.Series) -> pd.DataFrame:
    """分年绩效：收益/回撤/胜率"""
    ret = nav.pct_change()
    rows = []
    for year, g in ret.groupby(ret.index.year):
        y_nav = nav[nav.index.year == year]
        rows.append({
            "年份": year,
            "收益": y_nav.iloc[-1] / y_nav.iloc[0] - 1 if len(y_nav) > 1 else np.nan,
            "最大回撤": drawdown(y_nav).min() if len(y_nav) > 1 else np.nan,
            "日胜率": (g.dropna() > 0).mean(),
        })
    return pd.DataFrame(rows)


def summary_table(nav: pd.Series, bench: pd.Series | None = None) -> pd.DataFrame:
    """策略 vs 基准 对比表"""
    rows = []
    for label, series in [("策略", nav)] + ([("基准", bench)] if bench is not None else []):
        s = perf_stats(series)
        rows.append({"组合": label, **{k: s.get(k, np.nan) for k in
                     ["总收益", "年化收益", "年化波动", "夏普比率", "最大回撤", "卡玛比率", "日胜率"]}})
    return pd.DataFrame(rows)
