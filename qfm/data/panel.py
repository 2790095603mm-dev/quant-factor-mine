"""数据面板：因子计算所需的所有透视表，财务数据按公告日对齐（防未来函数）"""

from __future__ import annotations

from dataclasses import dataclass, field

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
    """
    close: pd.DataFrame = field(default_factory=pd.DataFrame)
    volume: pd.DataFrame = field(default_factory=pd.DataFrame)
    amount: pd.DataFrame = field(default_factory=pd.DataFrame)
    turnover: pd.DataFrame = field(default_factory=pd.DataFrame)
    mv_float: pd.DataFrame = field(default_factory=pd.DataFrame)  # 流通市值 = close × 流通股本
    fund: dict = field(default_factory=dict)  # {字段: date×stock DataFrame}
    fund_names: list = field(default_factory=list)


def build_panel(bars: pd.DataFrame, indicators: pd.DataFrame) -> DataPanel:
    """由长表构建 DataPanel

    bars:       [date, stock, open, high, low, close, volume, amount, outstanding_share, turnover]
    indicators: [ann_date, report_date, stock, roe, gross_margin, bvps, eps, rev_yoy, profit_yoy, ocfps]
    """
    bars = bars.sort_values(["stock", "date"])
    dates = sorted(bars["date"].unique())

    def pivot(col: str) -> pd.DataFrame:
        return bars.pivot_table(index="date", columns="stock", values=col, aggfunc="last").reindex(dates)

    p = DataPanel(
        close=pivot("close"),
        volume=pivot("volume"),
        amount=pivot("amount"),
        turnover=pivot("turnover"),
        mv_float=pivot("close") * pivot("outstanding_share"),
    )

    # 财务字段按公告日对齐（merge_asof backward）：保证只用"当天已公告"的数据
    if indicators is not None and len(indicators):
        ind = indicators.copy()
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
