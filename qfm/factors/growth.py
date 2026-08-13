"""成长族：业绩增速因子（财务数据直出）"""

import pandas as pd

from qfm.factors.base import register_factor


@register_factor("rev_yoy", "成长", "营业总收入同比增速（最新公告）", "positive")
def rev_yoy(d) -> pd.DataFrame:
    return d.fund.get("rev_yoy", pd.DataFrame(index=d.close.index, columns=d.close.columns, dtype=float))


@register_factor("profit_yoy", "成长", "净利润同比增速（最新公告）", "positive")
def profit_yoy(d) -> pd.DataFrame:
    return d.fund.get("profit_yoy", pd.DataFrame(index=d.close.index, columns=d.close.columns, dtype=float))


@register_factor("rev_qoq", "成长", "营业总收入季度环比增速（最新公告）", "positive")
def rev_qoq(d) -> pd.DataFrame:
    return d.fund.get("rev_qoq", pd.DataFrame(index=d.close.index, columns=d.close.columns, dtype=float))


@register_factor("profit_qoq", "成长", "净利润季度环比增速（最新公告）", "positive")
def profit_qoq(d) -> pd.DataFrame:
    return d.fund.get("profit_qoq", pd.DataFrame(index=d.close.index, columns=d.close.columns, dtype=float))
