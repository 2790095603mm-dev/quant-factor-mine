"""「每天一个因子」系列因子（博主公开整理，2026-08）：9 个新增因子

来源：博主公开渠道整理的 19 个因子中，与现库重复 / 缺数据源的 10 个已跳过
（20日动量=现有 mom_20；PIV=现有 bp；PITTM=现有 ep_ttm；20日波动=现有 vol_20；
低波动=vol_20 negative 方向；规模=现有 ln_mv_float；长期动量=现有 mom_250；
Amihud=现有 amihud_20；多空组合=分层检验"多空价差"指标；大单净流入=缺 Level2 数据源）。

公式按博主公开口径独立实现；全部基于 T 日已知数据（结构性无未来函数）。

方向标签说明（2026-08-18 实证校准）：bb_break_20 / turnover_heat / rav_4 /
sentiment_20 / vol_ratio_20 在真实缓存数据上 IC 显著为负（A 股反转效应），
方向由博主原说法 positive 调整为 negative；alpha144_191 / rev_5 / amplitude_3 与实证一致。
"""

from __future__ import annotations

import inspect
import sys

import numpy as np

from qfm.factors.base import register_factor

# 模块源码字符串：供无未来函数静态扫描（页面/测试复用）
FACTOR_SOURCE = inspect.getsource(sys.modules[__name__])


def _rsi(close, n: int):
    """RSI：简单均值口径（n 日平均涨幅 / 平均跌幅）"""
    diff = close.diff()
    gain = diff.clip(lower=0)
    loss = (-diff).clip(lower=0)
    ag = gain.rolling(n).mean()
    al = loss.rolling(n).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


@register_factor("bb_break_20", "动量反转",
                 "布林上轨突破 = close > MA20+2σ 取 1 否则 0（个股突破信号；"
                 "全市场宽度版 = 对全部个股求均值，作市场广度指标；"
                 "实证：A 股突破后易回调，方向为负）", "negative")
def bb_break_20(d):
    mid = d.close.rolling(20).mean()
    upper = mid + 2 * d.close.rolling(20).std()
    return (d.close > upper).astype(float)


@register_factor("amplitude_3", "波动",
                 "振幅 = (high-low)/前收（连续 3 日 <1% 为窒息状态，低振幅异象应取反）", "negative")
def amplitude_3(d):
    return (d.high - d.low) / d.close.shift(1)


@register_factor("turnover_heat", "流动性",
                 "换手升温倍数 = 近20日均换手 / 近250日均换手（资金关注度突变；"
                 "实证：高换手升温后短期收益为负，方向为负）", "negative")
def turnover_heat(d):
    return d.turnover.rolling(20).mean() / d.turnover.rolling(250).mean()


@register_factor("rav_4", "动量反转",
                 "RAV = (RSI4 - RSI4[4日前]) / RSI4[4日前]（短期强弱极端变化；"
                 "实证：强弱极端后收益为负，方向为负）", "negative")
def rav_4(d):
    rsi = _rsi(d.close, 4)
    den = rsi.shift(4).replace(0, np.nan)   # RSI=0（连续下跌）时除零保护
    return (rsi - den) / den


@register_factor("gm_yoy", "成长",
                 "毛利率同比变化 ≈ 最新毛利率 / 一年前(250交易日)毛利率 - 1（公告日对齐后的近似同比）", "positive")
def gm_yoy(d):
    gm = d.fund["gross_margin"].reindex(index=d.close.index, columns=d.close.columns)
    return gm / gm.shift(250) - 1


@register_factor("sentiment_20", "流动性",
                 "情绪 = (换手/20日前换手) × (收盘/20日前收盘)（换手升温 × 短期涨幅；"
                 "实证：情绪过热后收益为负，方向为负）", "negative")
def sentiment_20(d):
    return (d.turnover / d.turnover.shift(19)) * (d.close / d.close.shift(19))


@register_factor("alpha144_191", "流动性",
                 "Alpha144（国泰君安191）= 过去20日下跌日 |日收益|/成交量 累计（流动性冲击，超跌溢价）", "positive")
def alpha144_191(d):
    ret = d.close.pct_change(fill_method=None)
    down = (ret < 0).astype(float)
    impact = ret.abs() / d.volume
    return (impact * down).rolling(20).sum()


@register_factor("rev_5", "动量反转",
                 "5日反转 = 5日前收盘/今收 - 1（短期超跌反弹；口径：分母为今收，"
                 "与 rev_20（-20日动量）略有差异）", "positive")
def rev_5(d):
    return d.close.shift(5) / d.close - 1


@register_factor("vol_ratio_20", "流动性",
                 "量能比 = 今量/20日前量（成交量突变；实证：量能放大后收益为负，方向为负）", "negative")
def vol_ratio_20(d):
    return d.volume / d.volume.shift(19)
