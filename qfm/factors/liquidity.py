"""流动性族：换手率、Amihud 非流动性、成交额"""

import numpy as np
import pandas as pd

from qfm.factors.base import register_factor


@register_factor("turnover_5", "流动性", "5 日均换手率（高换手 = 散户过度交易，应取反）", "negative")
def turnover_5(d) -> pd.DataFrame:
    return d.turnover.rolling(5).mean()


@register_factor("turnover_20", "流动性", "20 日均换手率，高换手应取反", "negative")
def turnover_20(d) -> pd.DataFrame:
    return d.turnover.rolling(20).mean()


@register_factor("amihud_20", "流动性", "Amihud 非流动性 = |日收益|/成交额 的 20 日均值（×1e8），越高流动性越差", "negative")
def amihud_20(d) -> pd.DataFrame:
    ret = d.close.pct_change(fill_method=None).abs()
    illiq = ret / d.amount.replace(0, np.nan) * 1e8
    return illiq.rolling(20).mean()


@register_factor("turnover_60", "流动性", "60 日均换手率，高换手应取反", "negative")
def turnover_60(d) -> pd.DataFrame:
    return d.turnover.rolling(60).mean()


@register_factor("amount_5", "流动性", "5 日均成交额（对数），交投活跃度", "negative")
def amount_5(d) -> pd.DataFrame:
    return np.log(d.amount.rolling(5).mean())


@register_factor("amount_20", "流动性", "20 日均成交额（对数），交投活跃度", "negative")
def amount_20(d) -> pd.DataFrame:
    return np.log(d.amount.rolling(20).mean())
