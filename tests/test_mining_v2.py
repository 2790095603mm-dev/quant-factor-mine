"""挖掘引擎 v2 测试：流式候选 / 方向取反 / 变换列 / 候选总数"""

from __future__ import annotations

import numpy as np

from qfm.mining import generate_candidates_v2, run_mining


def test_v2_candidate_count_and_names(panel):
    cands = list(generate_candidates_v2(panel, names=["mom_20"], transforms=("raw", "rank")))
    assert len(cands) == 2 + 48                     # 2 因子变换 + 基础窗口集 48
    names = [n for n, _ in cands[:2]]
    assert names == ["mom_20__raw", "mom_20__rank"]
    for _, fdf in cands:
        assert fdf.shape == panel.close.shape


def test_v2_negative_factor_flipped(panel):
    """负向因子（vol_20）入池取反：raw 候选值 ≈ -vol_20 原值"""
    from qfm.factors import compute_factor

    cands = dict(generate_candidates_v2(panel, names=["vol_20"], transforms=("raw",)))
    raw = compute_factor("vol_20", panel)
    assert np.allclose(cands["vol_20__raw"].values, -raw.values, equal_nan=True)


def test_v2_run_mining_leaderboard(panel, tmp_path):
    df, meta = run_mining(panel, horizon=5, max_candidates=10, save_trials=True,
                          trials_dir=str(tmp_path))
    assert len(df) == 10
    assert "变换" in df.columns
    assert set(df["变换"]) <= {"raw", "rank", "zscore", "detrend20"}
    assert meta["n_trials"] == 10


def test_v2_total_count(panel):
    from qfm.mining.engine import _candidate_total

    total = _candidate_total(panel, names=["mom_20"], transforms=("raw",))
    assert total == 1 + 48                     # 1 因子 × 1 变换 + 基础窗口集 48
