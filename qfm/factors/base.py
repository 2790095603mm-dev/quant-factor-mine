"""因子注册表：@register_factor 一行注册一个因子"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from qfm.data.panel import DataPanel

FAMILIES = ["价值", "质量", "成长", "动量反转", "波动", "流动性", "规模", "实战"]


@dataclass
class Factor:
    name: str
    func: Callable[[DataPanel], pd.DataFrame]
    family: str
    description: str
    direction: str = "positive"  # positive: 值越大预期收益越高；negative: 值越小越高（用于报告展示）


FACTORS: dict[str, Factor] = {}

# 因子中文名映射（单一数据源）：页面显示 "英文（中文）"
ZH_NAMES: dict[str, str] = {
    # 价值
    "ep_ttm": "盈利收益率TTM", "bp": "账面市值比", "ep": "盈利收益率",
    # 质量
    "roe": "净资产收益率", "gross_margin": "销售毛利率", "ocf_eps": "盈余质量",
    # 成长
    "rev_yoy": "营收同比增速", "profit_yoy": "净利同比增速",
    "rev_qoq": "营收环比增速", "profit_qoq": "净利环比增速",
    # 动量反转
    "mom_5": "5日动量", "mom_10": "10日动量", "mom_20": "20日动量", "mom_60": "60日动量",
    "mom_120": "120日动量", "mom_250": "250日动量", "rev_20": "20日反转", "rev_5": "5日反转",
    # 波动
    "vol_10": "10日波动率", "vol_20": "20日波动率", "vol_60": "60日波动率",
    "vol_120": "120日波动率", "down_vol_20": "20日下行波动",
    "max_ret_20": "20日最大涨幅", "skew_60": "60日收益偏度",
    # 流动性
    "turnover_5": "5日均换手率", "turnover_20": "20日均换手率", "turnover_60": "60日均换手率",
    "amount_5": "5日均成交额", "amount_20": "20日均成交额", "amihud_20": "Amihud非流动性",
    # 规模
    "ln_mv_float": "流通市值对数",
    # 博主「每天一个因子」
    "bb_break_20": "布林上轨突破", "amplitude_3": "振幅", "turnover_heat": "换手升温倍数",
    "rav_4": "相对强弱极端", "gm_yoy": "毛利率同比", "sentiment_20": "情绪因子",
    "alpha144_191": "流动性冲击Alpha144", "vol_ratio_20": "量能比",
    # 「实战」家族
    "lead_cap": "龙头市值集中度", "vol_div": "行业成交分化", "volret_cov": "行业量价协方差",
    "lead_ret_pre": "龙头收益溢价", "range_bias": "振幅乖离", "gap_sent": "隔夜跳空",
    "res_mom": "滚动残差动量", "sent_beta": "情绪Beta", "rel_turn": "相对换手率",
    # 机器学习
    "ml_synth": "ML合成因子",
}


def factor_label(name: str) -> str:
    """页面显示名：英文（中文）；无中文名时退回原名"""
    zh = ZH_NAMES.get(name)
    return f"{name}（{zh}）" if zh else name


def register_factor(name: str, family: str, description: str = "", direction: str = "positive"):
    """注册因子：装饰器用法

    @register_factor("mom_20", "动量反转", "过去20日动量", "positive")
    def mom_20(d: DataPanel) -> pd.DataFrame:
        return d.close.pct_change(20)
    """

    def deco(func: Callable[[DataPanel], pd.DataFrame]):
        FACTORS[name] = Factor(
            name=name, func=func, family=family, description=description, direction=direction
        )
        return func

    return deco


def get_factor(name: str) -> Factor | None:
    return FACTORS.get(name)


def list_factors(family: str | None = None) -> list[Factor]:
    fs = list(FACTORS.values())
    if family:
        fs = [f for f in fs if f.family == family]
    return sorted(fs, key=lambda f: (FAMILIES.index(f.family) if f.family in FAMILIES else 99, f.name))


def compute_factor(name: str, panel: DataPanel) -> pd.DataFrame:
    f = get_factor(name)
    if f is None:
        raise KeyError(f"未知因子: {name}")
    return f.func(panel)
