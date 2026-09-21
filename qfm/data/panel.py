"""数据面板：因子计算所需的所有透视表，财务数据按公告日对齐（防未来函数）"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# loader 重命名后的英文列 → fund 键名（口径均为累计值，除 bvps 外）
FUND_FIELDS = {
    "roe": "roe",
    "gross_margin": "gross_margin",
    "bvps": "bvps",
    "eps": "eps",
    "rev_yoy": "rev_yoy",
    "profit_yoy": "profit_yoy",
    "rev_qoq": "rev_qoq",
    "profit_qoq": "profit_qoq",
    "ocfps": "ocfps",
}


@dataclass
class DataPanel:
    """因子函数的统一输入：透视表（index=日期, columns=股票）+ 财务对齐表

    fund 中的每个 DataFrame：index=交易日, columns=股票, 值=该日"已公告"的最新财务值

    复权口径：close/open/high/low 为**前复权**价，factor 为精确复权因子
    （= 前复权价 / 真实价，日内 OHLC 共享同一因子）。因此
    - `close_raw`（真实收盘价）= close / factor
    - `mv_float`（流通市值）= close_raw × 流通股本
    估值类因子（ep/bp）必须用 close_raw，否则分子（真实每股指标）与分母（前复权价）
    量纲不一致；收益率类因子用前复权价反而正确（复权不改变收益率）。
    """
    close: pd.DataFrame = field(default_factory=pd.DataFrame)
    open: pd.DataFrame = field(default_factory=pd.DataFrame)
    high: pd.DataFrame = field(default_factory=pd.DataFrame)
    low: pd.DataFrame = field(default_factory=pd.DataFrame)
    volume: pd.DataFrame = field(default_factory=pd.DataFrame)
    amount: pd.DataFrame = field(default_factory=pd.DataFrame)
    turnover: pd.DataFrame = field(default_factory=pd.DataFrame)
    factor: pd.DataFrame = field(default_factory=pd.DataFrame)      # 精确复权因子（1.0 = 无需复权）
    close_raw: pd.DataFrame = field(default_factory=pd.DataFrame)   # 真实收盘价 = close / factor
    mv_float: pd.DataFrame = field(default_factory=pd.DataFrame)    # 流通市值 = close_raw × 流通股本
    industry: pd.DataFrame = field(default_factory=pd.DataFrame)  # 行业（date×stock 字符串，按公告日对齐）
    fund: dict = field(default_factory=dict)  # {字段: date×stock DataFrame}
    fund_names: list = field(default_factory=list)


def raw_price(panel: DataPanel, field: str) -> pd.DataFrame:
    """真实（不复权）价格：`panel.<field> / panel.factor`，日内 OHLC 共享同一因子。

    factor 缺失（合成面板）时视为 1.0，即价格本身已经是真实价。
    """
    price = getattr(panel, field)
    if not isinstance(price, pd.DataFrame) or price.empty:
        return pd.DataFrame()
    factor = panel.factor
    if not isinstance(factor, pd.DataFrame) or factor.empty:
        return price
    return price / factor.replace(0, np.nan)


def valuation_price(panel: DataPanel) -> pd.DataFrame:
    """估值口径的真实收盘价。

    优先用 close_raw；手工构造的合成面板（无 factor / close_raw）退回 close，
    因为此时 factor 恒为 1，二者等价。
    """
    if isinstance(panel.close_raw, pd.DataFrame) and not panel.close_raw.empty:
        return panel.close_raw
    return panel.close


def build_panel(bars: pd.DataFrame, indicators: pd.DataFrame) -> DataPanel:
    """由长表构建 DataPanel

    bars:       [date, stock, open, high, low, close, volume, amount, outstanding_share, turnover, factor]
                factor 列可缺省（旧版缓存或合成数据），缺省时视为 1.0（价格本身即真实价）
    indicators: [ann_date, report_date, stock, roe, gross_margin, bvps, eps, rev_yoy, profit_yoy, ocfps]
    """
    bars = bars.sort_values(["stock", "date"])
    dates = sorted(bars["date"].unique())

    def pivot(col: str) -> pd.DataFrame:
        return bars.pivot_table(index="date", columns="stock", values=col, aggfunc="last").reindex(dates)

    close = pivot("close")
    if "factor" in bars.columns:
        factor = pivot("factor")
        # 因子必须为正；异常值置 NaN，避免污染市值与真实价
        factor = factor.where(factor > 0)
    else:
        factor = pd.DataFrame(1.0, index=close.index, columns=close.columns)

    p = DataPanel(
        close=close,
        open=pivot("open"),
        high=pivot("high"),
        low=pivot("low"),
        volume=pivot("volume"),
        amount=pivot("amount"),
        turnover=pivot("turnover"),
        factor=factor,
        # 真实收盘价：估值因子与市值的口径基准
        close_raw=close / factor,
        # 流通市值必须用真实价，前复权价会随分红送股历史漂移（逐股幅度不同 → 横截面污染）
        mv_float=(close / factor) * pivot("outstanding_share"),
    )

    # 财务字段按公告日对齐（merge_asof backward）：保证只用"当天已公告"的数据
    # ind 必须在条件块外定义：无财务表时下方行业/EPS_TTM 分支仍需判空
    ind = indicators.copy() if indicators is not None else pd.DataFrame()
    if len(ind):
        # 全市场交易日 × 股票 长表
        dates_long = bars[["date", "stock"]].drop_duplicates().sort_values(["stock", "date"])
        for src_col, key in FUND_FIELDS.items():
            if src_col not in ind.columns:
                continue
            right = ind[["ann_date", "stock", src_col]].rename(
                columns={"ann_date": "date", src_col: "value"}
            ).dropna(subset=["value"]).sort_values("date")
            if right.empty:
                continue
            aligned = pd.merge_asof(
                dates_long.sort_values("date"),
                right,
                on="date",
                by="stock",
                direction="backward",
            )
            p.fund[key] = aligned.pivot_table(
                index="date", columns="stock", values="value", aggfunc="last"
            ).reindex(dates)
            p.fund_names.append(key)

    # 行业：分类字段，按公告日对齐为字符串面板（date×stock）
    if "industry" in ind.columns:
        right = ind[["ann_date", "stock", "industry"]].rename(
            columns={"ann_date": "date", "industry": "value"}
        ).dropna(subset=["value"]).sort_values("date")
        aligned = pd.merge_asof(
            dates_long.sort_values("date"),
            right,
            on="date",
            by="stock",
            direction="backward",
        )
        p.industry = aligned.pivot_table(
            index="date", columns="stock", values="value", aggfunc="last"
        ).reindex(dates)

    # EPS_TTM 拼接：累计口径 → TTM 口径
    # ttm(非年报期) = 本期累计 + 去年年报 - 去年同期累计；年报期 ttm = 年报本身
    if "eps" in ind.columns:
        eps_raw = ind.pivot_table(index="report_date", columns="stock", values="eps", aggfunc="last")
        eps_ttm = eps_raw.copy()
        for r in eps_raw.index:
            if r.month != 12:
                prev_same = r - pd.DateOffset(years=1)      # 去年同期累计
                prev_annual = pd.Timestamp(r.year - 1, 12, 31)  # 去年年报
                if prev_same in eps_raw.index and prev_annual in eps_raw.index:
                    eps_ttm.loc[r] = eps_raw.loc[r] + eps_raw.loc[prev_annual] - eps_raw.loc[prev_same]
        ttm_long = eps_ttm.stack().rename("value").reset_index()
        ttm_long = ttm_long.merge(
            ind[["report_date", "stock", "ann_date"]].drop_duplicates(["report_date", "stock"]),
            on=["report_date", "stock"],
        )
        aligned = pd.merge_asof(
            dates_long.sort_values("date"),
            ttm_long.rename(columns={"ann_date": "date"}).sort_values("date"),
            on="date",
            by="stock",
            direction="backward",
        )
        p.fund["eps_ttm"] = aligned.pivot_table(
            index="date", columns="stock", values="value", aggfunc="last"
        ).reindex(dates)
        p.fund_names.append("eps_ttm")
    return p
