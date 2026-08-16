"""挖掘引擎试验矩阵测试：落盘格式 / manifest / latest_trials / 排行榜新列"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from qfm.mining import run_mining
from qfm.mining.engine import latest_trials, save_trials_matrix


def test_run_mining_returns_leaderboard_and_meta(panel, tmp_path):
    df, meta = run_mining(panel, horizon=5, max_candidates=6,
                          save_trials=True, trials_dir=str(tmp_path))
    assert "无未来函数" in df.columns
    assert set(df["无未来函数"]) == {"pass（结构安全）"}
    assert meta is not None and meta["n_trials"] == 6 and meta["horizon"] == 5
    assert (Path(meta["path"]) / "trials.parquet").exists()
    assert (Path(meta["path"]) / "manifest.json").exists()


def test_trials_matrix_shape(panel, tmp_path):
    _, meta = run_mining(panel, horizon=5, max_candidates=6,
                         save_trials=True, trials_dir=str(tmp_path))
    mat = pd.read_parquet(Path(meta["path"]) / "trials.parquet")
    assert mat.shape[1] == 6                    # 每候选一列
    assert mat.shape[0] >= 6                    # 至少 6 个调仓期
    assert mat.notna().sum().sum() > 0


def test_save_and_latest_trials_roundtrip(panel, tmp_path):
    mat = pd.DataFrame({"a": [0.01, -0.02, 0.03], "b": [0.02, 0.01, -0.01]})
    path = save_trials_matrix(mat, {"horizon": 20, "top_n": 30}, trials_dir=str(tmp_path))
    df, manifest = latest_trials(trials_dir=str(tmp_path))
    assert list(df.columns) == ["a", "b"]
    assert manifest["n_trials"] == 2 and manifest["T_periods"] == 3
    assert "created_at" in manifest


def test_run_mining_no_trials(panel, tmp_path):
    df, meta = run_mining(panel, horizon=5, max_candidates=4,
                          save_trials=False, trials_dir=str(tmp_path))
    assert meta is None
    assert len(df) == 4
