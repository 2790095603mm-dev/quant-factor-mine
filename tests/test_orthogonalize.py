"""正交化模块测试：行业暴露清除 / IC 保留率 / 诊断字段 / 最小样本跳过"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qfm.orthogonalize import (
    controls_from_panel,
    orthogonalize,
    orthogonalize_factor,
    winsorize_zscore,
)


def _industry_signal(panel, seed: int = 0) -> pd.DataFrame:
    """行业强暴露信号：银行/科技 取 2，其余 0，叠加噪声"""
    rng = np.random.default_rng(seed)
    ind = panel.industry.iloc[0]
    eff = np.array([2.0 if s in ("银行", "科技") else 0.0 for s in ind])
    noise = pd.DataFrame(rng.normal(0, 1, panel.close.shape),
                         index=panel.close.index, columns=panel.close.columns)
    return pd.DataFrame(np.tile(eff, (panel.close.shape[0], 1)),
                        index=panel.close.index, columns=panel.close.columns) + noise


def _industry_controls(panel) -> dict:
    return {k: v for k, v in controls_from_panel(panel, controls=("industry",)).items()
            if k.startswith("industry_")}


def test_winsorize_zscore_standardized(panel):
    x = pd.Series(np.r_[np.random.default_rng(1).normal(0, 1, 100), 50.0, -50.0])
    z = winsorize_zscore(x)
    assert abs(z.mean()) < 1e-9
    assert abs(z.std() - 1.0) < 1e-9
    assert z.abs().max() < 5.0          # 极端值被截尾


def test_controls_from_panel_shapes(panel):
    ctrls = controls_from_panel(panel)
    assert "size" in ctrls and "beta" in ctrls and "vol" in ctrls
    ind_ctrls = {k: v for k, v in ctrls.items() if k.startswith("industry_")}
    assert len(ind_ctrls) == 4 - 1                      # 4 行业 one-hot 后 drop_first
    for v in ind_ctrls.values():
        assert v.shape == panel.close.shape
    assert ctrls["size"].shape == panel.close.shape


def test_orthogonalize_removes_industry_exposure(panel):
    signal = _industry_signal(panel)
    ctrls = _industry_controls(panel)
    resid, diag = orthogonalize(signal, ctrls)
    assert diag["exposure_before"] > 0.5            # 行业可解释大部分方差
    assert diag["exposure_after"] < 0.05            # 残差对行业暴露清零
    assert diag["days_skipped"] == 0
    assert diag["coverage_after"] > 0.9


def test_orthogonalize_min_samples_skips(panel):
    signal = _industry_signal(panel, seed=2)
    ctrls = _industry_controls(panel)
    _, diag = orthogonalize(signal, ctrls, min_samples=10_000)
    assert diag["days_skipped"] == panel.close.shape[0]
    assert diag["coverage_after"] == 0.0


def test_orthogonalize_factor_diagnostics(panel):
    signal = _industry_signal(panel, seed=3)
    resid, diag = orthogonalize_factor(panel, signal, controls=("industry", "size"), horizon=5)
    assert np.isfinite(diag["ic_before"]) and np.isfinite(diag["ic_after"])
    # 残差因子 IC 保留率可为负（噪声下符号翻转属正常），仅约束量级
    assert abs(diag["ic_retention"]) <= 1.5
    assert "turnover_before" in diag and "turnover_after" in diag
    assert resid.shape == signal.shape
