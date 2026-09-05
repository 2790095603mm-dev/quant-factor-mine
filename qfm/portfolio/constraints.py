"""多头目标组合的确定性风险约束与调仓预算。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import pandas as pd


EPSILON = 1e-12
UNKNOWN_INDUSTRY = "未知行业"


def _validate_unit_interval(name: str, value: float) -> None:
    number = float(value)
    if not math.isfinite(number) or not 0 < number <= 1:
        raise ValueError(f"{name} 必须在 (0, 1] 内")


def _validate_turnover_budget(value: float) -> None:
    number = float(value)
    if not math.isfinite(number) or not 0 < number <= 2:
        raise ValueError("max_rebalance_turnover 必须在 (0, 2] 内")


@dataclass(frozen=True)
class PortfolioConstraints:
    """目标组合硬上限与可选的单次调仓成交额预算。"""

    max_stock_weight: float = 1.0
    max_industry_weight: float = 1.0
    max_rebalance_turnover: float | None = None

    def __post_init__(self) -> None:
        _validate_unit_interval("max_stock_weight", self.max_stock_weight)
        _validate_unit_interval("max_industry_weight", self.max_industry_weight)
        if self.max_rebalance_turnover is not None:
            _validate_turnover_budget(self.max_rebalance_turnover)

    def to_dict(self) -> dict[str, float | None]:
        return {
            "max_stock_weight": float(self.max_stock_weight),
            "max_industry_weight": float(self.max_industry_weight),
            "max_rebalance_turnover": (
                None if self.max_rebalance_turnover is None else float(self.max_rebalance_turnover)
            ),
        }


def _industries(index: pd.Index, industry: pd.Series | None) -> pd.Series:
    if industry is None:
        return pd.Series(UNKNOWN_INDUSTRY, index=index, dtype=object)
    values = industry.reindex(index).astype(object)
    return values.where(values.notna(), UNKNOWN_INDUSTRY).astype(str)


def constrained_target_weights(
    score: pd.Series,
    volume: pd.Series,
    industry: pd.Series | None,
    top_n: int,
    constraints: PortfolioConstraints,
) -> tuple[pd.Series, dict[str, object]]:
    """按分数降序贪心分配目标仓位，并严格遵守单股及行业上限。"""
    if top_n <= 0:
        raise ValueError("top_n 必须为正数")
    if score.index.has_duplicates:
        raise ValueError("score 证券代码不能重复")

    volume = volume.reindex(score.index).fillna(0.0)
    valid = score.notna() & volume.gt(0)
    ranked = score.loc[valid].sort_values(ascending=False, kind="mergesort")
    labels = _industries(score.index, industry)
    base_weight = 1.0 / top_n
    remaining_cash = 1.0
    industry_weights: dict[str, float] = {}
    allocations: dict[object, float] = {}

    for code in ranked.index:
        if len(allocations) >= top_n or remaining_cash <= EPSILON:
            break
        industry_name = labels.loc[code]
        industry_remaining = constraints.max_industry_weight - industry_weights.get(industry_name, 0.0)
        allocation = min(
            base_weight,
            constraints.max_stock_weight,
            max(0.0, industry_remaining),
            remaining_cash,
        )
        if allocation <= EPSILON:
            continue
        allocations[code] = float(allocation)
        industry_weights[industry_name] = industry_weights.get(industry_name, 0.0) + float(allocation)
        remaining_cash -= float(allocation)

    target = pd.Series(allocations, dtype=float)
    target.index.name = score.index.name
    invested = float(target.sum())
    return target, {
        "target_positions": int(len(target)),
        "target_invested": invested,
        "target_cash": max(0.0, 1.0 - invested),
        "target_max_stock_weight": float(target.max()) if len(target) else 0.0,
        "target_max_industry_weight": max(industry_weights.values(), default=0.0),
    }


def apply_turnover_budget(
    current: pd.Series,
    target: pd.Series,
    max_rebalance_turnover: float | None,
) -> tuple[pd.Series, dict[str, Any]]:
    """将目标向当前持仓线性缩放，使计划买卖额不超过给定预算。"""
    if max_rebalance_turnover is not None:
        _validate_turnover_budget(max_rebalance_turnover)
    index = current.index.union(target.index)
    current_all = current.reindex(index, fill_value=0.0).fillna(0.0).astype(float)
    target_all = target.reindex(index, fill_value=0.0).fillna(0.0).astype(float)
    gross_turnover = float((target_all - current_all).abs().sum())
    if max_rebalance_turnover is None or gross_turnover <= float(max_rebalance_turnover) + EPSILON:
        result = target_all
        scale = 1.0
        binding = False
    else:
        scale = float(max_rebalance_turnover) / gross_turnover
        result = current_all + scale * (target_all - current_all)
        binding = True
    result.name = target.name
    return result, {
        "gross_turnover": gross_turnover,
        "applied_gross_turnover": gross_turnover * scale,
        "turnover_budget": max_rebalance_turnover,
        "budget_binding": binding,
        "turnover_scale": scale,
    }
