"""因子库：注册所有家族"""

from qfm.factors.base import (
    FACTORS,
    ZH_NAMES,
    Factor,
    compute_factor,
    factor_label,
    get_factor,
    list_factors,
    register_factor,
)

# 导入各家族以触发注册
from qfm.factors import blogger, growth, liquidity, momentum, practical, quality, size, value, volatility  # noqa: F401

__all__ = [
    "FACTORS",
    "ZH_NAMES",
    "Factor",
    "compute_factor",
    "factor_label",
    "get_factor",
    "list_factors",
    "register_factor",
]
