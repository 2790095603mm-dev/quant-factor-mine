"""因子注册表：@register_factor 一行注册一个因子"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from qfm.data.panel import DataPanel

FAMILIES = ["价值", "质量", "成长", "动量反转", "波动", "流动性", "规模"]


@dataclass
class Factor:
    name: str
    func: Callable[[DataPanel], pd.DataFrame]
    family: str
    description: str
    direction: str = "positive"  # positive: 值越大预期收益越高；negative: 值越小越高（用于报告展示）


FACTORS: dict[str, Factor] = {}


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
