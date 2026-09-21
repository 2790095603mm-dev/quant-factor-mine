"""多因子统一指标与截面相关性测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qfm.analysis.factor_compare import (
    compare_factors,
    cross_sectional_corr,
    high_correlation_pairs,
)


def test_compare_factors_returns_all_required_metrics(panel):
    result = compare_factors(panel, ["mom_20", "rev_20"], horizon=20)

    assert set(result.metrics.columns) >= {
        "factor", "ic", "rank_ic", "icir", "long_short_return",
        "turnover", "coverage", "stability",
    }
    assert set(result.pearson_corr.index) == {"mom_20", "rev_20"}
    assert set(result.spearman_corr.columns) == {"mom_20", "rev_20"}
    assert result.signals["mom_20"].shape == panel.close.shape


def test_spearman_detects_monotonic_duplicate():
    dates = pd.date_range("2025-01-01", periods=4)
    stocks = [f"{index:06d}" for index in range(30)]
    values = np.arange(1, 31, dtype=float)
    a = pd.DataFrame(np.tile(values, (4, 1)), index=dates, columns=stocks)
    b = a.pow(2)

    matrix = cross_sectional_corr({"a": a, "b": b}, method="spearman")

    assert matrix.loc["a", "b"] == pytest.approx(1.0)
    pairs = high_correlation_pairs(matrix)
    assert pairs.iloc[0]["factor_a"] == "a"
    assert pairs.iloc[0]["factor_b"] == "b"
    assert pairs.iloc[0]["correlation"] == pytest.approx(1.0)


def test_pearson_and_spearman_are_distinct_for_nonlinear_relation():
    dates = pd.date_range("2025-01-01", periods=3)
    stocks = [f"{index:06d}" for index in range(30)]
    values = np.arange(1, 31, dtype=float)
    signals = {
        "linear": pd.DataFrame(np.tile(values, (3, 1)), index=dates, columns=stocks),
        "square": pd.DataFrame(np.tile(values ** 2, (3, 1)), index=dates, columns=stocks),
    }

    pearson = cross_sectional_corr(signals, method="pearson")
    spearman = cross_sectional_corr(signals, method="spearman")

    assert spearman.loc["linear", "square"] == pytest.approx(1.0)
    assert pearson.loc["linear", "square"] < 1.0


def test_duplicate_threshold_uses_absolute_correlation():
    matrix = pd.DataFrame(
        [[1.0, -0.91, 0.4], [-0.91, 1.0, 0.2], [0.4, 0.2, 1.0]],
        index=["a", "b", "c"], columns=["a", "b", "c"],
    )

    pairs = high_correlation_pairs(matrix, threshold=0.7)

    assert pairs[["factor_a", "factor_b"]].values.tolist() == [["a", "b"]]
    assert pairs.iloc[0]["correlation"] == pytest.approx(-0.91)


def test_compare_requires_at_least_two_unique_factors(panel):
    with pytest.raises(ValueError, match="至少选择 2 个"):
        compare_factors(panel, ["mom_20", "mom_20"])
