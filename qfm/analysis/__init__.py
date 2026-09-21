"""跨因子、跨策略的统一比较分析。"""

from qfm.analysis.factor_compare import (
    FactorCompareResult,
    compare_factors,
    cross_sectional_corr,
    high_correlation_pairs,
)
from qfm.analysis.strategy_compare import StrategyCompareResult, compare_strategies

__all__ = [
    "FactorCompareResult",
    "compare_factors",
    "cross_sectional_corr",
    "high_correlation_pairs",
    "StrategyCompareResult",
    "compare_strategies",
]
