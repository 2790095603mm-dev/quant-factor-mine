"""ML 合成因子测试：样本外 IC / 无未来函数 / 注册可用 / 小数据跑通（零网络）"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qfm.data.panel import DataPanel
from qfm.factors import FACTORS, get_factor, register_factor
from qfm.ml_synthesizer import synthesize_ml_factor
from qfm.pipeline.tests import compute_ic, forward_returns


def _predictive_panel() -> tuple[DataPanel, pd.DataFrame]:
    """构造可预测面板：ret[t] = 0.02·x[t-1] + eps，特征 x[t] 可预测未来收益（无泄露）"""
    rng = np.random.default_rng(5)
    n, T = 40, 300
    dates = pd.bdate_range("2024-01-02", periods=T)
    cols = [f"{600000 + i}" for i in range(n)]
    x = pd.DataFrame(rng.normal(0, 1, (T, n)), index=dates, columns=cols)
    ret = 0.02 * x.shift(1).fillna(0.0).values + rng.normal(0, 0.02, (T, n))
    close = pd.DataFrame(100.0 * np.exp(np.cumsum(ret, axis=0)), index=dates, columns=cols)
    panel = DataPanel(
        close=close,
        volume=pd.DataFrame(rng.integers(1e6, 5e7, (T, n)).astype(float), index=dates, columns=cols),
        amount=pd.DataFrame(rng.uniform(1e8, 5e9, (T, n)), index=dates, columns=cols),
        turnover=pd.DataFrame(rng.uniform(0.005, 0.05, (T, n)), index=dates, columns=cols),
        mv_float=pd.DataFrame(np.tile(rng.uniform(2e9, 5e11, n), (T, 1)), index=dates, columns=cols),
        industry=pd.DataFrame(np.tile(np.array([["银行", "白酒", "科技", "医药"][i % 4]
                                                for i in range(n)], dtype=object), (T, 1)),
                               index=dates, columns=cols),
    )
    return panel, x


def test_ml_synth_oos_ic_positive():
    panel, x = _predictive_panel()
    register_factor(name="x_sig", family="测试", description="", direction="positive")(lambda d: x)
    try:
        cutoff = "2024-12-31"          # 面板 300 交易日止于 ~2025-02，样本外取 2025 年初
        pred, meta = synthesize_ml_factor(panel, names=["x_sig"], horizon=5,
                                          train_cutoff=cutoff, n_folds=2)
        # 无未来函数：预测面板只含 cutoff 之后
        assert (pred.index >= pd.Timestamp(cutoff)).all()
        # 样本外 IC 显著为正（评估前需重索引到全日期轴，compute_ic 按位置计算）
        pred_full = pred.reindex(panel.close.index)
        ic = compute_ic(pred_full, forward_returns(panel.close, 5)).dropna()
        assert ic.mean() > 0.05
        assert meta["n_folds"] >= 1
        assert "importance_top" in meta and len(meta["importance_top"]) >= 1
    finally:
        FACTORS.pop("x_sig", None)


def test_ml_synth_no_lookahead_index():
    panel, x = _predictive_panel()
    register_factor(name="x_sig", family="测试", description="", direction="positive")(lambda d: x)
    try:
        pred, _ = synthesize_ml_factor(panel, names=["x_sig"], horizon=5,
                                       train_cutoff="2024-10-01", n_folds=2)
        assert (pred.index >= pd.Timestamp("2024-10-01")).all()
        assert pred.notna().sum().sum() > 0
    finally:
        FACTORS.pop("x_sig", None)


def test_ml_synth_register_factor_usable():
    panel, x = _predictive_panel()
    register_factor(name="x_sig", family="测试", description="", direction="positive")(lambda d: x)
    try:
        pred, _ = synthesize_ml_factor(panel, names=["x_sig"], horizon=5,
                                       train_cutoff="2024-12-31", n_folds=2)
        # 页面同款注册方式
        register_factor(name="ml_synth", family="机器学习", description="test",
                        direction="positive")(lambda d: pred)
        f = get_factor("ml_synth")
        assert f is not None
        out = f.func(panel)
        assert out.shape == pred.shape
        assert abs(float(out.iloc[0, 0] - pred.iloc[0, 0])) < 1e-12
    finally:
        FACTORS.pop("x_sig", None)
        FACTORS.pop("ml_synth", None)


def test_ml_synth_small_panel_runs(panel):
    pred, meta = synthesize_ml_factor(panel, names=["mom_20", "turnover_20"], horizon=5,
                                      train_cutoff="2024-06-30", n_folds=2)
    assert pred.shape[1] >= 1
    assert (pred.index >= pd.Timestamp("2024-06-30")).all()
    assert meta["n_folds"] >= 1


def test_ml_synth_invalid_cutoff_raises(panel):
    with pytest.raises(ValueError, match="训练截止日"):
        synthesize_ml_factor(panel, names=["mom_20"], horizon=5, train_cutoff="not-a-date", n_folds=2)


def test_factor_report_partial_index_alignment(panel):
    """回归：部分日期因子的报告 IC 必须等于全量报告在重叠日期的 IC（同一清洗路径）
    防止按位置错位导致的静默失真（ML 样本外预测等部分日期因子）"""
    from qfm.factors import compute_factor
    from qfm.pipeline.tests import factor_report

    full = compute_factor("mom_20", panel)
    partial = full.iloc[100:].copy()
    rep_full = factor_report(full, panel.close, horizon=5, direction="positive")
    rep_partial = factor_report(partial, panel.close, horizon=5, direction="positive")
    ic_full = rep_full["ic_series"]
    ic_partial = rep_partial["ic_series"].dropna()
    assert ic_partial.index.isin(partial.index).all()
    assert np.allclose(ic_partial.values, ic_full.loc[ic_partial.index].values, atol=1e-12)


def test_factor_report_subset_columns_no_crash(panel):
    """回归：因子股票列少于 close 列时（如个股缺数据）不崩溃，IC 在共同股票上计算"""
    from qfm.factors import compute_factor
    from qfm.pipeline.tests import factor_report

    full = compute_factor("mom_20", panel)
    subset = full.iloc[:, :40]
    rep = factor_report(subset, panel.close, horizon=5, direction="positive")
    s = rep["ic_summary"]
    assert pd.notna(s["ic_mean"])
