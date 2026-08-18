"""博主「每天一个因子」10 个新增因子测试：可计算 / 形状 / 定点校验 / 无未来函数"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qfm.factors import compute_factor, get_factor, list_factors
from qfm.pipeline.lookahead import scan_source

NEW_FACTORS = ["bb_break_20", "amplitude_3", "turnover_heat", "rav_4", "gm_yoy",
               "sentiment_20", "alpha144_191", "rev_5", "vol_ratio_20"]


def test_all_new_factors_registered():
    names = {f.name for f in list_factors()}
    for n in NEW_FACTORS:
        assert n in names, f"{n} 未注册"


def test_all_new_factors_computable(panel):
    for n in NEW_FACTORS:
        fdf = compute_factor(n, panel)
        assert fdf.shape == panel.close.shape, f"{n} 形状错误"
        assert fdf.notna().sum().sum() > 0, f"{n} 全 NaN"
        assert np.isfinite(fdf.dropna()).all().all(), f"{n} 含 Inf"


def test_amplitude_formula(panel):
    fdf = compute_factor("amplitude_3", panel)
    expect = (panel.high - panel.low) / panel.close.shift(1)
    assert np.allclose(fdf.values, expect.values, equal_nan=True)


def test_bb_break_binary(panel):
    fdf = compute_factor("bb_break_20", panel)
    vals = fdf.dropna().values.ravel()
    assert set(np.unique(vals)) <= {0.0, 1.0}


def test_rev5_formula(panel):
    """rev_5 忠于博主公式（分母=今收），与 -mom_5 高度相关但不精确相等"""
    rev5 = compute_factor("rev_5", panel)
    expect = panel.close.shift(5) / panel.close - 1
    assert np.allclose(rev5.values, expect.values, equal_nan=True)
    mom5 = compute_factor("mom_5", panel)
    both = (rev5.notna() & mom5.notna()).values
    corr = float(np.corrcoef(rev5.values[both], -mom5.values[both])[0, 1])
    assert corr > 0.99


def test_gm_yoy_needs_fund(panel):
    fdf = compute_factor("gm_yoy", panel)
    assert fdf.notna().sum().sum() > 0          # conftest fund 已含 gross_margin
    assert fdf.shape == panel.close.shape


def test_no_lookahead_scan():
    from qfm.factors.blogger import FACTOR_SOURCE  # 模块源码字符串，供静态扫描
    chk = scan_source(FACTOR_SOURCE)
    assert not chk["leaks"], f"泄漏模式: {chk['leaks']}"
