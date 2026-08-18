"""「实战」家族因子（行业格局 / 情绪类，2026-08）：9 个因子

- 行业级（算出行业值后广播给行业内每只股票）：lead_cap / vol_div / volret_cov / lead_ret_pre
- 个股级：range_bias / gap_sent / res_mom / sent_beta / rel_turn

公式按公开口径独立实现；全部基于 T 日已知数据（结构性无未来函数）。
注：lead_cap / lead_ret_pre 的"龙头"取行业内流通市值前 3（mv_float 近似总市值）；
行业成分过少（<3 只）时 top3 退化为全行业，数值失真，建议在成分 ≥5 的行业上解读。
sent_beta 的"市场情绪指数"公式未定义，采用文档化代理：20 日平滑市场广度（涨家数-跌家数）。
市场级指标（ADL 市场广度累计 / Disp 全市场振幅分歧）不注册为个股因子——横截面 IC 无法计算，
如需市场择时请走策略层功能。
"""

from __future__ import annotations

import inspect
import sys

import numpy as np
import pandas as pd

from qfm.factors.base import register_factor

# 模块源码字符串：供无未来函数静态扫描（页面/测试复用）
FACTOR_SOURCE = inspect.getsource(sys.modules[__name__])


def _stack(panel, df, val: str):
    """date×stock 面板 → 长表 (date, stock, val, ind)（dropna=False 保持全形状）"""
    s = df.stack(future_stack=True).rename(val).reset_index()
    s.columns = ["date", "stock", val]
    s["ind"] = panel.industry.stack(future_stack=True).values
    return s


def _broadcast(panel, per_ind: pd.Series) -> pd.DataFrame:
    """(date, industry) 级序列 → 按行业归属广播回 date×stock"""
    long = panel.industry.stack(future_stack=True).rename("ind").reset_index()
    long.columns = ["date", "stock", "ind"]
    idx = pd.MultiIndex.from_arrays([long["date"], long["ind"]])
    long["v"] = per_ind.reindex(idx).values
    return long.pivot_table(index="date", columns="stock", values="v", aggfunc="first") \
               .reindex(index=panel.close.index, columns=panel.close.columns)


@register_factor("lead_cap", "实战",
                 "龙头市值集中度 = 行业前3市值龙头总市值 / 行业总市值（行业资源向头部集聚程度；"
                 "高=寡头垄断/龙头溢价，低=格局分散/中小盘空间）", "positive")
def lead_cap(d):
    s = _stack(d, d.mv_float, "cap").dropna(subset=["cap", "ind"])
    total = s.groupby(["date", "ind"])["cap"].sum()
    top3 = (s.sort_values(["date", "ind", "cap"], ascending=[True, True, False])
             .groupby(["date", "ind"]).head(3).groupby(["date", "ind"])["cap"].sum())
    return _broadcast(d, top3 / total)


@register_factor("vol_div", "实战",
                 "行业成交分化 = 行业成分股当日成交额 Std/Mean（资金分布异质性；"
                 "高=流动性向局部集中/聚焦龙头，低=交易同质化/补涨扩散）", "positive")
def vol_div(d):
    s = _stack(d, d.amount, "amt").dropna(subset=["amt", "ind"])
    g = s.groupby(["date", "ind"])["amt"]
    return _broadcast(d, g.std() / g.mean())


@register_factor("volret_cov", "实战",
                 "行业量价协方差 = Cov(成分股成交额, 日收益)（横截面，量价联动抱团；"
                 "正=高成交标的溢价/结构性行情，低=普涨扩散）", "positive")
def volret_cov(d):
    s = _stack(d, d.amount, "amt")
    s["ret"] = d.close.pct_change(fill_method=None).stack(future_stack=True).values
    s = s.dropna(subset=["amt", "ret", "ind"])
    s["ar"] = s["amt"] * s["ret"]
    g = s.groupby(["date", "ind"])
    return _broadcast(d, g["ar"].mean() - g["amt"].mean() * g["ret"].mean())


@register_factor("lead_ret_pre", "实战",
                 "龙头收益溢价 = 行业前3市值龙头均收益 - 行业等权收益（抱团偏好强度；"
                 "持续为正=龙头溢价稳固，转负=行情扩散至中小盘）", "positive")
def lead_ret_pre(d):
    s = _stack(d, d.mv_float, "cap")
    s["ret"] = d.close.pct_change(fill_method=None).stack(future_stack=True).values
    s = s.dropna(subset=["cap", "ret", "ind"])
    top3_ret = (s.sort_values(["date", "ind", "cap"], ascending=[True, True, False])
                 .groupby(["date", "ind"]).head(3).groupby(["date", "ind"])["ret"].mean())
    ind_ret = s.groupby(["date", "ind"])["ret"].mean()
    return _broadcast(d, top3_ret - ind_ret)


@register_factor("range_bias", "实战",
                 "振幅乖离 = 当日振幅 - 过去20日振幅均值（波动情绪偏离历史中枢；"
                 "正乖离扩大=短线博弈升温，负乖离=情绪钝化）", "positive")
def range_bias(d):
    rng = (d.high - d.low) / d.close.shift(1)
    return rng - rng.rolling(20).mean()


@register_factor("gap_sent", "实战",
                 "隔夜跳空 = (今开 - 昨收) / 昨收（盘外信息驱动的隔夜一致预期情绪；"
                 "正跳空=乐观预期前置，负跳空=悲观扩散）", "positive")
def gap_sent(d):
    return (d.open - d.close.shift(1)) / d.close.shift(1)


@register_factor("res_mom", "实战",
                 "滚动残差动量 = 60日动量 - 其20日滚动均值（剥离动量长期中枢，捕捉短期情绪边际偏离；"
                 "实证：偏离越高短期收益越负（A股反转），方向为负）", "negative")
def res_mom(d):
    mom60 = d.close.pct_change(60, fill_method=None)
    return mom60 - mom60.rolling(20).mean()


@register_factor("sent_beta", "实战",
                 "情绪Beta = 个股收益对市场情绪指数的60日滚动回归系数（情绪弹性；"
                 "市场情绪代理=20日平滑市场广度(涨家数-跌家数)，公式中 MarketSent 未定义，此为文档化代理）", "positive")
def sent_beta(d):
    ret = d.close.pct_change(fill_method=None)
    breadth = (ret > 0).sum(axis=1) - (ret < 0).sum(axis=1)   # 市场广度
    sent = breadth.rolling(20).mean()                          # 平滑为市场情绪代理
    var_s = sent.rolling(60).var()
    return ret.rolling(60).cov(sent).div(var_s, axis=0)


@register_factor("rel_turn", "实战",
                 "相对换手率 = 当日换手 / 过去20日换手均值（相对近期中枢的倍数；"
                 ">1.8 放量情绪爆发，<0.5 持续缩量）", "positive")
def rel_turn(d):
    return d.turnover / d.turnover.rolling(20).mean()
