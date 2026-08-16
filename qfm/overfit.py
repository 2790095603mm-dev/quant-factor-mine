"""回测过拟合统计检验：DSR / PSR / MinTRL / PBO / Haircut

公式独立实现（公开学术文献，非 GPL 代码拷贝）：
- Bailey & López de Prado (2012, 2014)：PSR / DSR / MinTRL
- Bailey et al. (2017)：PBO via CSCV
- Harvey & Liu (2015)：多重检验 Haircut（Bonferroni / Holm / BHY）
所有 Sharpe 均为逐期（非年化）口径。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from scipy.stats import norm

EULER_MASCHERONI = 0.5772156649015328606


def sharpe_ratio(returns, benchmark: float = 0.0) -> float:
    """逐期 Sharpe（ddof=1）；序列过短或零波动返回 nan"""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if r.size < 2:
        return float("nan")
    sd = r.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float((r.mean() - benchmark) / sd)


def _moments(returns) -> tuple[float, float]:
    """样本偏度 + 非超额峰度（正态==3）"""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    n = r.size
    if n < 4:
        return 0.0, 3.0
    m = r.mean()
    s = r.std(ddof=0)
    if s == 0:
        return 0.0, 3.0
    skew = float(np.mean(((r - m) / s) ** 3))
    kurt = float(np.mean(((r - m) / s) ** 4))
    return skew, kurt


def probabilistic_sharpe_ratio(observed_sr: float, benchmark_sr: float, n_obs: int,
                               skew: float = 0.0, kurtosis: float = 3.0) -> float:
    """PSR(SR*) = P[真实 SR > SR*]（López de Prado 2012，偏度/峰度修正）"""
    if n_obs < 2:
        return float("nan")
    se = math.sqrt((1 - skew * observed_sr + (kurtosis - 1) / 4 * observed_sr ** 2) / (n_obs - 1))
    if se == 0:
        return float("nan")
    return float(norm.cdf((observed_sr - benchmark_sr) / se))


def expected_max_sharpe(n_trials: int, sr_std: float) -> float:
    """E[max SR_N] ≈ σ·[(1-γ)Φ⁻¹(1-1/N) + γΦ⁻¹(1-1/(N·e))]（Bailey & LdP 2014）"""
    if n_trials <= 1:
        return 0.0
    z1 = norm.ppf(1 - 1.0 / n_trials)
    z2 = norm.ppf(1 - 1.0 / (n_trials * math.e))
    return sr_std * ((1 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2)


def deflated_sharpe_ratio(returns, n_trials: int, trials_matrix=None) -> float:
    """DSR = P[真实 SR > E[max SR_N]]（扣除多重检验选择偏差）

    trials_matrix 提供时用试验 SR 的经验标准差估计 σ；否则退化为
    单次估计（σ = 所选策略 SR 的标准误），偏宽松。
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    sr = sharpe_ratio(r)
    if not np.isfinite(sr) or n_trials < 1:
        return float("nan")
    n_obs = r.size
    skew, kurt = _moments(r)
    sr_se = math.sqrt((1 - skew * sr + (kurt - 1) / 4 * sr ** 2) / (n_obs - 1))
    if trials_matrix is not None:
        m = np.asarray(trials_matrix, dtype=float)
        if m.ndim == 2 and m.shape[1] >= 2 and np.isfinite(m).sum() > 0:
            vals = np.array([sharpe_ratio(m[:, j]) for j in range(m.shape[1])])
            vals = vals[np.isfinite(vals)]
            sr_std = float(vals.std(ddof=1)) if vals.size >= 2 else sr_se
        else:
            sr_std = sr_se
    else:
        sr_std = sr_se
    sr_star = expected_max_sharpe(int(n_trials), sr_std)
    return probabilistic_sharpe_ratio(sr, sr_star, n_obs, skew, kurt)


def minimum_track_record_length(observed_sr: float, benchmark_sr: float = 0.0,
                                prob: float = 0.95, skew: float = 0.0,
                                kurtosis: float = 3.0) -> float:
    """MinTRL：使 PSR ≥ prob 所需的最少期数；SR≤基准 时返回 ∞"""
    if observed_sr <= benchmark_sr:
        return float("inf")
    diff = observed_sr - benchmark_sr
    z = norm.ppf(prob)
    se_coef = math.sqrt(1 - skew * observed_sr + (kurtosis - 1) / 4 * observed_sr ** 2)
    return float(1 + (z * se_coef / diff) ** 2)


def pbo_cscv(trials_matrix, n_blocks: int = 16) -> float:
    """Probability of Backtest Overfitting via CSCV（Bailey et al. 2017）

    输入 T×N 收益矩阵；将 T 均分为偶数块，取所有 C(S, S/2) 个组合为
    IS，其余为 OOS；统计 IS 最优策略在 OOS 落入下半区的比例。
    N<10 或 T<8 返回 nan。
    """
    X = np.asarray(trials_matrix, dtype=float)
    if X.ndim != 2:
        return float("nan")
    T, N = X.shape
    if N < 10 or T < 8:
        return float("nan")
    good = np.isfinite(X).sum(axis=0) >= max(4, T // 4)
    X = X[:, good]
    if X.shape[1] < 10:
        return float("nan")
    S = min(n_blocks, T // 4)
    S = S if S % 2 == 0 else S - 1
    if S < 4:
        return float("nan")
    edges = np.linspace(0, T, S + 1, dtype=int)
    blocks = [np.arange(edges[i], edges[i + 1]) for i in range(S)]

    def _sr(idx) -> np.ndarray:
        sub = X[idx]
        mu = np.nanmean(sub, axis=0)
        sd = np.nanstd(sub, axis=0, ddof=1)
        out = np.full(N, np.nan)
        pos = sd > 0
        out[pos] = mu[pos] / sd[pos]
        return out

    pbo_cnt = 0.0
    total = 0.0
    for sel in combinations(range(S), S // 2):
        is_idx = np.concatenate([blocks[i] for i in sel])
        oos_idx = np.concatenate([blocks[i] for i in range(S) if i not in sel])
        is_sr = _sr(is_idx)
        oos_sr = _sr(oos_idx)
        if not np.isfinite(is_sr).any() or not np.isfinite(oos_sr).any():
            continue
        j_star = int(np.nanargmax(is_sr))
        # 过拟合信号：OOS 中跑赢 IS 最优者的试验占比 > 50%（下半区）
        rank_frac = float((oos_sr > oos_sr[j_star]).mean())
        pbo_cnt += 1.0 if rank_frac > 0.5 else 0.0
        total += 1.0
    return float(pbo_cnt / total) if total > 0 else float("nan")


def haircut_sharpe(returns, n_trials: int, method: str = "holm") -> dict:
    """多重检验下 Sharpe 打折（Harvey & Liu 2015）。

    method: bonferroni | holm | bhy；p 值用 norm.sf 避免大 t 下溢。
    返回 {sr_observed, sr_adjusted, p, p_adj, method}
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    sr = sharpe_ratio(r)
    n_obs = r.size
    if not np.isfinite(sr) or n_obs < 2 or n_trials < 1:
        return {"sr_observed": float("nan"), "sr_adjusted": float("nan"),
                "p": float("nan"), "p_adj": float("nan"), "method": method}
    tstat = sr * math.sqrt(n_obs)
    p = 2.0 * norm.sf(abs(tstat))
    if method == "bhy":
        h_n = sum(1.0 / k for k in range(1, int(n_trials) + 1))
        p_adj = min(1.0, p * n_trials * h_n)
    else:  # bonferroni / holm（对最优者，holm 退化为 bonferroni）
        p_adj = min(1.0, p * n_trials)
    # 用 isf（上尾分位数）而非 ppf(1-x)：避免 p_adj 极小时 1-x 舍入为 1.0 → inf
    sr_adj = norm.isf(p_adj / 2) / math.sqrt(n_obs)
    return {"sr_observed": sr, "sr_adjusted": sr_adj, "p": p, "p_adj": p_adj, "method": method}


@dataclass
class OverfitReport:
    verdict: str                       # PASS | FAIL | INSUFFICIENT
    passed: bool
    dsr: float
    dsr_degraded: bool
    pbo: float
    n_trials: int
    n_obs: int
    sr_observed: float
    sr_adjusted: float
    sr_annual: float
    sr_annual_adjusted: float
    haircut_method: str
    minimum_track_record_length: float
    reasons: list = field(default_factory=list)
    matrix_used: bool = False


def overfit_report(selected_returns, n_trials: int, trials_matrix=None,
                   periods_per_year: int = 252, haircut_method: str = "holm") -> OverfitReport:
    """汇总四项统计 → OverfitReport。

    判定：DSR≥0.95 且（PBO 缺失或 <0.5）且 Haircut 后 SR>0 → PASS。
    收益序列过短（<6）或 Sharpe 无法估计 → INSUFFICIENT。
    """
    r = np.asarray(selected_returns, dtype=float)
    r = r[~np.isnan(r)]
    n_obs = int(r.size)
    if n_obs < 6:
        return OverfitReport(verdict="INSUFFICIENT", passed=False, dsr=float("nan"),
                             dsr_degraded=True, pbo=float("nan"), n_trials=int(n_trials),
                             n_obs=n_obs, sr_observed=float("nan"), sr_adjusted=float("nan"),
                             sr_annual=float("nan"), sr_annual_adjusted=float("nan"),
                             haircut_method=haircut_method,
                             minimum_track_record_length=float("inf"),
                             reasons=["收益序列过短（<6 期），无法做统计检验"])
    dsr = deflated_sharpe_ratio(r, n_trials, trials_matrix)
    matrix_used = trials_matrix is not None
    if not np.isfinite(dsr):
        return OverfitReport(verdict="INSUFFICIENT", passed=False, dsr=float("nan"),
                             dsr_degraded=not matrix_used, pbo=float("nan"),
                             n_trials=int(n_trials), n_obs=n_obs,
                             sr_observed=float("nan"), sr_adjusted=float("nan"),
                             sr_annual=float("nan"), sr_annual_adjusted=float("nan"),
                             haircut_method=haircut_method,
                             minimum_track_record_length=float("inf"),
                             reasons=["Sharpe 无法估计（序列无波动）"])
    pbo = pbo_cscv(trials_matrix) if matrix_used else float("nan")
    hc = haircut_sharpe(r, n_trials, haircut_method)
    mintrl = minimum_track_record_length(hc["sr_observed"])
    ppy = max(periods_per_year, 1)
    checks = [("DSR≥0.95", np.isfinite(dsr) and dsr >= 0.95)]
    if np.isfinite(pbo):
        checks.append(("PBO<0.5", pbo < 0.5))
    checks.append(("Haircut 后 SR>0", np.isfinite(hc["sr_adjusted"]) and hc["sr_adjusted"] > 0))
    reasons = [f"{k} 不满足" for k, v in checks if not v]
    passed = not reasons
    return OverfitReport(
        verdict="PASS" if passed else "FAIL",
        passed=passed,
        dsr=dsr,
        dsr_degraded=not matrix_used,
        pbo=pbo,
        n_trials=int(n_trials),
        n_obs=n_obs,
        sr_observed=hc["sr_observed"],
        sr_adjusted=hc["sr_adjusted"],
        sr_annual=hc["sr_observed"] * math.sqrt(ppy),
        sr_annual_adjusted=hc["sr_adjusted"] * math.sqrt(ppy),
        haircut_method=haircut_method,
        minimum_track_record_length=mintrl,
        reasons=reasons,
        matrix_used=matrix_used,
    )
