"""质量族：盈利质量因子（财务数据直出）"""

import numpy as np
import pandas as pd

from qfm.factors.base import register_factor


@register_factor("roe", "质量", "净资产收益率（最新公告），盈利能力强", "positive")
def roe(d) -> pd.DataFrame:
    return d.fund.get("roe", pd.DataFrame(index=d.close.index, columns=d.close.columns, dtype=float))


@register_factor("gross_margin", "质量", "销售毛利率（最新公告），护城河与议价能力", "positive")
def gross_margin(d) -> pd.DataFrame:
    return d.fund.get("gross_margin", pd.DataFrame(index=d.close.index, columns=d.close.columns, dtype=float))


@register_factor("ocf_eps", "质量", "盈余质量 = 每股经营现金流 / 每股收益，利润含金量", "positive")
def ocf_eps(d) -> pd.DataFrame:
    ocf = d.fund.get("ocfps")
    eps = d.fund.get("eps")
    if ocf is None or eps is None:
        return pd.DataFrame(index=d.close.index, columns=d.close.columns, dtype=float)
    out = ocf / eps.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)
