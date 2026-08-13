"""波动族：波动率、下行波动、偏度、最大涨幅（低波动异象）"""

import numpy as np
import pandas as pd

from qfm.factors.base import register_factor

TRADING_DAYS = 252


@register_factor("vol_10", "波动", "10 日年化波动率（日收益标准差），低波动异象", "negative")
def vol_10(d) -> pd.DataFrame:
    return d.close.pct_change(fill_method=None).rolling(10).std() * np.sqrt(TRADING_DAYS)


@register_factor("vol_20", "波动", "20 日年化波动率，低波动异象", "negative")
def vol_20(d) -> pd.DataFrame:
    return d.close.pct_change(fill_method=None).rolling(20).std() * np.sqrt(TRADING_DAYS)


@register_factor("vol_60", "波动", "60 日年化波动率，低波动异象", "negative")
def vol_60(d) -> pd.DataFrame:
    return d.close.pct_change(fill_method=None).rolling(60).std() * np.sqrt(TRADING_DAYS)


@register_factor("down_vol_20", "波动", "20 日下行波动（仅负收益的标准差），下行风险", "negative")
def down_vol_20(d) -> pd.DataFrame:
    ret = d.close.pct_change(fill_method=None)
    neg = ret.where(ret < 0)
    return neg.rolling(20, min_periods=10).std() * np.sqrt(TRADING_DAYS)


@register_factor("vol_120", "波动", "120 日年化波动率，低波动异象", "negative")
def vol_120(d) -> pd.DataFrame:
    return d.close.pct_change(fill_method=None).rolling(120).std() * np.sqrt(TRADING_DAYS)


@register_factor("max_ret_20", "波动", "20 日最大单日涨幅（彩票偏好：高者常高估），应取反", "negative")
def max_ret_20(d) -> pd.DataFrame:
    return d.close.pct_change(fill_method=None).rolling(20).max()


@register_factor("skew_60", "波动", "60 日收益偏度（彩票偏好因子）", "negative")
def skew_60(d) -> pd.DataFrame:
    return d.close.pct_change(fill_method=None).rolling(60).skew()
