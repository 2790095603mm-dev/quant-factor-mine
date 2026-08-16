"""过拟合检验测试：DSR 削减 / PBO / Haircut 单调性 / MinTRL / 综合判定"""

from __future__ import annotations

import numpy as np

from qfm.overfit import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    haircut_sharpe,
    minimum_track_record_length,
    overfit_report,
    pbo_cscv,
    sharpe_ratio,
)


def _gauss_trials(T: int, N: int, mean: float = 0.0, std: float = 0.01, seed: int = 7) -> np.ndarray:
    return np.random.default_rng(seed).normal(mean, std, (T, N))


def test_sharpe_basic():
    r = np.array([0.01, -0.01, 0.02, 0.0, 0.015, -0.005])
    assert sharpe_ratio(r) > 0
    assert np.isnan(sharpe_ratio(np.array([1.0])))          # 单样本无法估计


def test_expected_max_sharpe_positive_and_increasing():
    assert expected_max_sharpe(1, 1.0) == 0.0
    assert expected_max_sharpe(100, 1.0) > expected_max_sharpe(10, 1.0) > 0


def test_deflated_sharpe_noise_below_threshold():
    T, N = 500, 100
    mat = _gauss_trials(T, N, std=0.01)
    best_col = mat[:, int(np.argmax([sharpe_ratio(mat[:, j]) for j in range(N)]))]
    dsr = deflated_sharpe_ratio(best_col, N, trials_matrix=mat)
    assert 0.0 < dsr < 0.95          # 纯噪声选出最优 → DSR 显著低于 0.95


def test_deflated_sharpe_real_edge_above_threshold():
    T, N = 500, 100
    mat = _gauss_trials(T, N, std=0.01)
    edge = np.random.default_rng(3).normal(0.005, 0.01, T)   # SR≈0.5/期
    dsr = deflated_sharpe_ratio(edge, N, trials_matrix=mat)
    assert dsr > 0.95


def test_pbo_high_on_noise():
    mat = _gauss_trials(500, 40, std=0.01)
    pbo = pbo_cscv(mat)
    assert np.isfinite(pbo) and pbo > 0.3        # 纯噪声 PBO 应接近 0.5


def test_pbo_nan_when_few_trials():
    assert np.isnan(pbo_cscv(np.zeros((100, 5))))
    assert np.isnan(pbo_cscv(np.zeros((5, 50))))


def test_haircut_monotonic_in_trials():
    r = np.random.default_rng(0).normal(0.003, 0.01, 500)   # SR≈0.3/期，n=10 打折后仍为正
    h10 = haircut_sharpe(r, n_trials=10)["sr_adjusted"]
    h1000 = haircut_sharpe(r, n_trials=1000)["sr_adjusted"]
    assert h10 > h1000 >= 0


def test_mintrl():
    assert np.isinf(minimum_track_record_length(-0.1))      # SR≤基准 → ∞
    assert minimum_track_record_length(0.0) == float("inf")
    assert minimum_track_record_length(0.01) > 1e4          # 极低 SR 需要极长样本


def test_overfit_report_fail_on_noise():
    T, N = 500, 100
    mat = _gauss_trials(T, N, std=0.01)
    best = mat[:, int(np.argmax([sharpe_ratio(mat[:, j]) for j in range(N)]))]
    rep = overfit_report(best, N, trials_matrix=mat)
    assert rep.verdict == "FAIL" and rep.passed is False
    assert rep.dsr < 0.95 and np.isfinite(rep.pbo)
    assert rep.reasons


def test_overfit_report_pass_on_edge():
    T, N = 500, 100
    rng = np.random.default_rng(3)
    mat = rng.normal(0.0, 0.01, (T, N))
    edge = rng.normal(0.005, 0.01, T)          # 真实 edge：SR≈0.5/期
    mat[:, 0] = edge                           # edge 必须是试验矩阵中的一列（IS 最优 → OOS 仍最优）
    rep = overfit_report(edge, N, trials_matrix=mat)
    assert rep.verdict == "PASS" and rep.passed is True
    assert rep.dsr > 0.95
    assert np.isfinite(rep.pbo) and rep.pbo < 0.2


def test_overfit_report_degraded_without_matrix():
    r = np.random.default_rng(9).normal(0.005, 0.01, 500)
    rep = overfit_report(r, n_trials=100, trials_matrix=None)
    assert rep.dsr_degraded is True and np.isnan(rep.pbo)
    assert rep.verdict in ("PASS", "FAIL")
