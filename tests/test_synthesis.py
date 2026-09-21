"""synthesize 正交化升级测试：size 正交后截面相关≈0 / 默认行为兼容"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qfm.portfolio import synthesize


def test_synthesize_size_orthogonalized(panel):
    score, w = synthesize(panel, ["mom_20", "turnover_20"], mode="equal",
                          orthogonalize=True, ortho_controls=("size",))
    log_mv = np.log(panel.mv_float)
    corrs = []
    for dt in score.index:
        y, x = score.loc[dt], log_mv.loc[dt]
        mask = y.notna() & x.notna()
        if mask.sum() > 30 and y[mask].std() > 0:
            corrs.append(y[mask].corr(x[mask]))
    assert abs(float(np.nanmean(corrs))) < 0.1
    # 权重轨迹仍供策略页展示，但不能干扰正交化内部的 Pandas 运算。
    assert isinstance(score.attrs["weight_history"], pd.DataFrame)
    assert score.attrs["weight_history"].index.equals(panel.close.index)


def test_synthesize_default_backward_compatible(panel):
    score_a, _ = synthesize(panel, ["mom_20", "turnover_20"], mode="equal",
                            orthogonalize=True, ortho_controls=None)
    score_b, _ = synthesize(panel, ["mom_20", "turnover_20"], mode="equal",
                            orthogonalize=True, ortho_controls=("size",))
    assert score_a.shape == score_b.shape
    assert score_a.notna().sum().sum() > 0


def test_synthesize_no_ortho_unchanged(panel):
    score, w = synthesize(panel, ["mom_20", "turnover_20"], mode="equal", orthogonalize=False)
    assert set(w) == {"mom_20", "turnover_20"}
    assert score.notna().sum().sum() > 0


def test_ic_x_ir_weights_are_normalised(panel):
    score, weights = synthesize(
        panel, ["mom_20", "roe"], mode="ic_x_ir", horizon=20,
        weight_lookback=126, weight_rebalance="ME",
    )

    history = score.attrs["weight_history"]
    assert np.allclose(history.sum(axis=1), 1.0)
    assert sum(weights.values()) == pytest.approx(1.0)
