"""「实战」家族 6 因子测试：可计算 / 行业内同值 / 公式定点 / 无未来函数"""

from __future__ import annotations

import numpy as np

from qfm.factors import compute_factor, list_factors
from qfm.pipeline.lookahead import scan_source

NEW_FACTORS = ["lead_cap", "vol_div", "volret_cov", "lead_ret_pre", "range_bias", "gap_sent",
               "res_mom", "sent_beta", "rel_turn"]


def test_all_practical_factors_registered():
    names = {f.name for f in list_factors()}
    for n in NEW_FACTORS:
        assert n in names, f"{n} 未注册"
        f = list_factors()[0]  # noqa: B018
    from qfm.factors.base import get_factor
    for n in NEW_FACTORS:
        assert get_factor(n).family == "实战", f"{n} 家族应为实战"


def test_all_practical_factors_computable(panel):
    for n in NEW_FACTORS:
        fdf = compute_factor(n, panel)
        assert fdf.shape == panel.close.shape, f"{n} 形状错误"
        assert fdf.notna().sum().sum() > 0, f"{n} 全 NaN"
        assert np.isfinite(fdf.dropna()).all().all(), f"{n} 含 Inf"


def test_industry_factor_constant_within_industry(panel):
    """行业级因子：同一行业、同一交易日，所有股票取值必须相同"""
    for n in ["lead_cap", "vol_div", "volret_cov", "lead_ret_pre"]:
        fdf = compute_factor(n, panel)
        dt = panel.close.index[100]
        row, ind_row = fdf.loc[dt], panel.industry.loc[dt]
        checked = 0
        for ind_name in ind_row.dropna().unique():
            vals = row[ind_row == ind_name].dropna()
            if len(vals) > 1:
                assert vals.nunique() == 1, f"{n} 行业 {ind_name} 内取值不一致"
                checked += 1
        assert checked > 0


def test_gap_sent_formula(panel):
    fdf = compute_factor("gap_sent", panel)
    expect = (panel.open - panel.close.shift(1)) / panel.close.shift(1)
    assert np.allclose(fdf.values, expect.values, equal_nan=True)


def test_rel_turn_formula(panel):
    fdf = compute_factor("rel_turn", panel)
    expect = panel.turnover / panel.turnover.rolling(20).mean()
    assert np.allclose(fdf.values, expect.values, equal_nan=True)


def test_res_mom_formula(panel):
    fdf = compute_factor("res_mom", panel)
    mom60 = panel.close.pct_change(60)
    expect = mom60 - mom60.rolling(20).mean()
    assert np.allclose(fdf.values, expect.values, equal_nan=True)


def test_sent_beta_variation_across_stocks(panel):
    """情绪 Beta 必须是个股异质（同日截面不止一个取值），且回归系数有限"""
    fdf = compute_factor("sent_beta", panel)
    dt = panel.close.index[150]
    row = fdf.loc[dt].dropna()
    assert len(row) > 10
    assert row.nunique() > 1                    # 个股间 beta 有差异
    assert np.isfinite(row).all()


def test_range_bias_formula(panel):
    fdf = compute_factor("range_bias", panel)
    rng = (panel.high - panel.low) / panel.close.shift(1)
    expect = rng - rng.rolling(20).mean()
    assert np.allclose(fdf.values, expect.values, equal_nan=True)


def test_no_lookahead_scan():
    from qfm.factors.practical import FACTOR_SOURCE
    chk = scan_source(FACTOR_SOURCE)
    assert not chk["leaks"], f"泄漏模式: {chk['leaks']}"
