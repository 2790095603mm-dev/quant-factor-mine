"""组合目标约束和换手预算的纯函数测试。"""

from __future__ import annotations

import pandas as pd
import pytest

from qfm.portfolio.constraints import (
    PortfolioConstraints,
    apply_turnover_budget,
    constrained_target_weights,
)


def _inputs() -> tuple[pd.Series, pd.Series, pd.Series]:
    index = pd.Index(["A", "B", "C", "D", "E", "F"], name="stock")
    return (
        pd.Series([6.0, 5.0, 4.0, 3.0, 2.0, 1.0], index=index),
        pd.Series(1.0, index=index),
        pd.Series(["银行", "银行", "科技", "科技", None, "医药"], index=index),
    )


def test_default_constraints_keep_top_n_equal_weight():
    score, volume, industry = _inputs()

    target, diag = constrained_target_weights(
        score, volume, industry, 3, PortfolioConstraints(),
    )

    expected = pd.Series(
        [1 / 3] * 3,
        index=pd.Index(["A", "B", "C"], name="stock"),
        dtype=float,
    )
    pd.testing.assert_series_equal(target, expected)
    assert diag["target_cash"] == pytest.approx(0.0)


def test_stock_and_industry_caps_leave_cash_instead_of_breaking_limits():
    score, volume, industry = _inputs()

    target, diag = constrained_target_weights(
        score,
        volume,
        industry,
        4,
        PortfolioConstraints(max_stock_weight=0.20, max_industry_weight=0.25),
    )
    exposure = target.groupby(industry.reindex(target.index).fillna("未知行业")).sum()

    assert target.max() <= 0.20
    assert exposure.max() <= 0.25
    assert target.sum() < 1.0
    assert diag["target_cash"] == pytest.approx(1 - target.sum())


def test_unknown_industry_obeys_industry_cap():
    score, volume, industry = _inputs()

    target, _ = constrained_target_weights(
        score,
        volume,
        industry,
        6,
        PortfolioConstraints(max_industry_weight=0.15),
    )

    assert target.get("E", 0.0) <= 0.15


def test_turnover_budget_scales_target_without_budget_changing_target():
    index = pd.Index(["A", "B"])
    current = pd.Series([0.5, 0.0], index=index)
    target = pd.Series([0.0, 0.5], index=index)

    unchanged, off = apply_turnover_budget(current, target, None)
    scaled, on = apply_turnover_budget(current, target, 0.4)

    pd.testing.assert_series_equal(unchanged, target)
    assert off["budget_binding"] is False
    assert on["gross_turnover"] == pytest.approx(1.0)
    assert on["applied_gross_turnover"] == pytest.approx(0.4)
    assert scaled.tolist() == pytest.approx([0.3, 0.2])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_stock_weight": 0.0},
        {"max_industry_weight": 1.1},
        {"max_rebalance_turnover": 0.0},
        {"max_rebalance_turnover": 2.1},
    ],
)
def test_constraints_reject_invalid_limits(kwargs):
    with pytest.raises(ValueError):
        PortfolioConstraints(**kwargs)
