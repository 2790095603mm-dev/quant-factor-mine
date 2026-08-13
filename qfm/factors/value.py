"""价值族：从财务数据推导的估值因子（价格 = 收盘价）"""

import numpy as np
import pandas as pd

from qfm.factors.base import register_factor


def _safe_div(num: pd.DataFrame, den: pd.DataFrame) -> pd.DataFrame:
    """防除零/防 inf"""
    out = num / den.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


@register_factor("ep_ttm", "价值", "盈利收益率 TTM = EPS_TTM / 价格（1/PE_TTM），越高越便宜", "positive")
def ep_ttm(d) -> pd.DataFrame:
    eps = d.fund.get("eps_ttm")
    if eps is None:
        return pd.DataFrame(index=d.close.index, columns=d.close.columns, dtype=float)
    return _safe_div(eps, d.close)


@register_factor("bp", "价值", "账面市值比 = 每股净资产 / 价格（1/PB），越高越便宜", "positive")
def bp(d) -> pd.DataFrame:
    bvps = d.fund.get("bvps")
    if bvps is None:
        return pd.DataFrame(index=d.close.index, columns=d.close.columns, dtype=float)
    return _safe_div(bvps, d.close)


@register_factor("ep", "价值", "盈利收益率（最新报告期累计 EPS / 价格，简化口径）", "positive")
def ep(d) -> pd.DataFrame:
    eps = d.fund.get("eps")
    if eps is None:
        return pd.DataFrame(index=d.close.index, columns=d.close.columns, dtype=float)
    return _safe_div(eps, d.close)
