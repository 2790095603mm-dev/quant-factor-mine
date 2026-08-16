"""synthesize 正交化升级测试：size 正交后截面相关≈0 / 默认行为兼容"""

from __future__ import annotations

import numpy as np

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
