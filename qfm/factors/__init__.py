"""因子库：注册所有家族"""

from qfm.factors.base import (
    FACTORS,
    FACTOR_HISTORY,
    FAMILIES,
    FAMILY_TAGS,
    ZH_NAMES,
    Factor,
    all_tags,
    compute_factor,
    factor_definitions,
    factor_label,
    get_factor,
    list_factors,
    list_families,
    list_factor_versions,
    load_registry,
    register_factor,
    save_registry,
)

# 导入各家族以触发注册
from qfm.factors import blogger, growth, liquidity, momentum, practical, quality, size, value, volatility  # noqa: F401

# 内置因子完成注册后，再恢复用户保存的综合因子。单条损坏/漂移不会阻断应用启动。
try:
    from qfm.multifactor.registry import CompositeRegistry

    COMPOSITE_WARNINGS = CompositeRegistry().register_all()
except Exception as exc:  # noqa: BLE001 - 启动时保留内置因子可用，并把错误交给 Lab 展示
    COMPOSITE_WARNINGS = [f"综合因子注册表加载失败: {exc}"]

__all__ = [
    "FACTORS",
    "FACTOR_HISTORY",
    "FAMILIES",
    "FAMILY_TAGS",
    "ZH_NAMES",
    "Factor",
    "all_tags",
    "compute_factor",
    "factor_definitions",
    "factor_label",
    "get_factor",
    "list_factors",
    "list_families",
    "list_factor_versions",
    "load_registry",
    "register_factor",
    "save_registry",
    "COMPOSITE_WARNINGS",
]
