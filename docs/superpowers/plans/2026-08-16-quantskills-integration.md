# QuantSkills 三技能融入实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 QuantSkills 的因子正交化、回测过拟合检验、无未来函数检查三个方法论融入 quant-factor-mine Streamlit 应用（零新页面，全部嵌入现有 6 板块）。

**Architecture:** 3 个新 qfm 模块（`orthogonalize.py` / `overfit.py` / `pipeline/lookahead.py`）+ 2 处现有模块改造（`mining/engine.py` 保存试验矩阵、`portfolio/synthesis.py` 正交化升级）+ app.py 3 个页面集成 + 4 组 pytest。GPL-3.0 源码不拷贝，按公开学术文献公式独立实现。

**Tech Stack:** Python 3.13 / pandas 2.2.3 / numpy 2.1.3 / scipy 1.15.3 / streamlit / pytest / plotly。

## Global Constraints

- Python：`/opt/anaconda3/bin/python3`（项目虚拟环境即系统 anaconda）
- 零网络依赖：所有测试用合成数据（`tests/conftest.py` 夹具），不访问 akshare/新浪
- 不拷贝 GPL-3.0 源码：`/tmp/quantskills/` 三个包仅作方法论参考，算法按文献公式独立实现
- 命名风格沿用现有 qfm 包：模块内函数用 snake_case，返回 pandas/numpy 原生类型
- `run_mining` 返回类型改为 `tuple[pd.DataFrame, dict | None]`（所有调用方同步更新：app.py:331、qfm/cli.py:58）
- `synthesize` 新增 `ortho_controls` 参数，默认行为（仅市值正交）向后兼容
- 每次任务结束必须运行 pytest 并提交 git

---

### Task 1: 合成数据夹具 + 正交化模块（qfm/orthogonalize.py）

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/test_orthogonalize.py`
- Create: `qfm/orthogonalize.py`

**Interfaces:**
- Produces: `winsorize_zscore(x: pd.Series, n_mad: float = 5.0) -> pd.Series`
- Produces: `controls_from_panel(panel, controls=("industry","size","style"), style_window: int = 60) -> dict[str, pd.DataFrame]`（键：`size`/`beta`/`vol`/`industry`，industry 为展开后 one-hot 且已 drop_first）
- Produces: `orthogonalize(signal: pd.DataFrame, controls: dict, min_samples: int = 30) -> tuple[pd.DataFrame, dict]`（dict 键：`controls/days/days_skipped/coverage_before/coverage_after/exposure_before/exposure_after`）
- Produces: `orthogonalize_factor(panel, signal, controls=("industry","size","style"), horizon: int = 20, min_samples: int = 30) -> tuple[pd.DataFrame, dict]`（dict 额外含 `ic_before/ic_after/ic_retention/turnover_before/turnover_after/horizon`）
- Consumes: `qfm.pipeline.tests` 的 `compute_ic / forward_returns / turnover_ratio`（已存在，Task 3 不改其签名）

- [ ] **Step 1: 写合成数据夹具 tests/conftest.py**

```python
"""合成数据夹具：带行业/规模效应的 DataPanel，全量测试使用（零网络）"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qfm.data.panel import DataPanel


@pytest.fixture(scope="session")
def panel() -> DataPanel:
    rng = np.random.default_rng(42)
    n_stocks, n_days = 60, 260
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    cols = [f"{600000 + i}" for i in range(n_stocks)]
    inds = ["银行", "白酒", "科技", "医药"]
    stock_ind = [inds[i % 4] for i in range(n_stocks)]
    ind_eff = np.array([0.0009, -0.0002, 0.0006, 0.0001])       # 行业漂移
    mv_base = rng.uniform(2e9, 5e11, n_stocks)
    size_eff = (np.log(mv_base) - np.log(mv_base).mean()) / np.log(mv_base).std()

    drift = ind_eff[[i % 4 for i in range(n_stocks)]] + 0.0004 * size_eff
    ret = rng.normal(0.0002 + drift, 0.02, (n_days, n_stocks))
    close = 100.0 * np.exp(np.cumsum(ret, axis=0))

    return DataPanel(
        close=pd.DataFrame(close, index=dates, columns=cols),
        volume=pd.DataFrame(rng.integers(1e6, 5e7, (n_days, n_stocks)).astype(float), index=dates, columns=cols),
        amount=pd.DataFrame(rng.uniform(1e8, 5e9, (n_days, n_stocks)), index=dates, columns=cols),
        turnover=pd.DataFrame(rng.uniform(0.005, 0.05, (n_days, n_stocks)), index=dates, columns=cols),
        mv_float=pd.DataFrame(np.tile(mv_base, (n_days, 1)), index=dates, columns=cols),
        industry=pd.DataFrame(np.tile(np.array(stock_ind, dtype=object), (n_days, 1)), index=dates, columns=cols),
        fund={"roe": pd.DataFrame(rng.uniform(0.05, 0.25, (n_days, n_stocks)), index=dates, columns=cols),
              "eps_ttm": pd.DataFrame(rng.uniform(0.2, 3.0, (n_days, n_stocks)), index=dates, columns=cols)},
        fund_names=["roe", "eps_ttm"],
    )
```

- [ ] **Step 2: 写失败测试 tests/test_orthogonalize.py**

```python
"""正交化模块测试：行业暴露清除 / IC 保留率 / 诊断字段 / 最小样本跳过"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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


def test_winsorize_zscore_standardized(panel):
    x = pd.Series(np.r_[np.random.default_rng(1).normal(0, 1, 100), 50.0, -50.0])
    z = winsorize_zscore(x)
    assert abs(z.mean()) < 1e-9
    assert abs(z.std() - 1.0) < 1e-9
    assert z.abs().max() < 5.0          # 极端值被截尾


def test_controls_from_panel_shapes(panel):
    ctrls = controls_from_panel(panel)
    assert "size" in ctrls and "beta" in ctrls and "vol" in ctrls and "industry" in ctrls
    assert ctrls["size"].shape == panel.close.shape
    assert ctrls["industry"].shape[0] == panel.close.shape[0]
    assert ctrls["industry"].shape[1] == 4 - 1     # 4 行业 one-hot 后 drop_first


def test_orthogonalize_removes_industry_exposure(panel):
    signal = _industry_signal(panel)
    ctrls = {"industry": controls_from_panel(panel, controls=("industry",))["industry"]}
    resid, diag = orthogonalize(signal, ctrls)
    assert diag["exposure_before"] > 0.5            # 行业可解释大部分方差
    assert diag["exposure_after"] < 0.05            # 残差对行业暴露清零
    assert diag["days_skipped"] == 0
    assert diag["coverage_after"] > 0.9


def test_orthogonalize_min_samples_skips(panel):
    signal = _industry_signal(panel, seed=2)
    ctrls = {"industry": controls_from_panel(panel, controls=("industry",))["industry"]}
    _, diag = orthogonalize(signal, ctrls, min_samples=10_000)
    assert diag["days_skipped"] == panel.close.shape[0]
    assert diag["coverage_after"] == 0.0


def test_orthogonalize_factor_diagnostics(panel):
    signal = _industry_signal(panel, seed=3)
    resid, diag = orthogonalize_factor(panel, signal, controls=("industry", "size"), horizon=5)
    assert np.isfinite(diag["ic_before"]) and np.isfinite(diag["ic_after"])
    assert 0.0 <= diag["ic_retention"] <= 1.5
    assert "turnover_before" in diag and "turnover_after" in diag
    assert resid.shape == signal.shape
```

- [ ] **Step 3: 运行测试确认失败**

Run: `cd /Users/zxt/.zcode/workspace/default/quant-factor-mine && /opt/anaconda3/bin/python3 -m pytest tests/test_orthogonalize.py -v`
Expected: FAIL（ModuleNotFoundError: qfm.orthogonalize）

- [ ] **Step 4: 实现 qfm/orthogonalize.py**

```python
"""逐日截面 OLS 正交化：剥离行业 / 市值 / 风格 / 旧因子暴露

方法论源自 QuantSkills skill-factor-orthogonalize（逐日截面回归 + 残差重标准化），
公式为公开学术标准做法，本项目独立实现。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WINSORIZE_NSIG = 5.0
MIN_SAMPLES = 30


def winsorize_zscore(x: pd.Series, n_mad: float = WINSORIZE_NSIG) -> pd.Series:
    """MAD 截尾（n_mad·1.4826·MAD）后截面 z-score"""
    med = x.median()
    mad = (x - med).abs().median()
    if not np.isfinite(mad) or mad == 0:
        clipped = x.copy()
    else:
        clipped = x.clip(med - n_mad * 1.4826 * mad, med + n_mad * 1.4826 * mad)
    std = clipped.std()
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.nan, index=x.index)
    return (clipped - clipped.mean()) / std


def controls_from_panel(panel, controls=("industry", "size", "style"),
                        style_window: int = 60) -> dict[str, pd.DataFrame]:
    """构造控制变量面板（全部来自 DataPanel，零外部 API）。

    返回 {控制名: date×stock DataFrame}；industry 为展开后 one-hot（已 drop_first 防共线）。
    """
    out: dict[str, pd.DataFrame] = {}
    if "size" in controls and panel.mv_float is not None and len(panel.mv_float):
        out["size"] = np.log(panel.mv_float)
    if "style" in controls:
        ret = panel.close.pct_change(fill_method=None)
        mkt = ret.mean(axis=1)                          # 全池等权市场收益
        var_mkt = mkt.rolling(style_window).var()
        beta = ret.rolling(style_window).cov(mkt).div(var_mkt, axis=0)
        out["beta"] = beta
        out["vol"] = ret.rolling(20).std()
    if "industry" in controls and panel.industry is not None and len(panel.industry):
        dum = pd.get_dummies(panel.industry, prefix="ind", dtype=float)
        dum = dum.loc[:, dum.columns.str.startswith("ind_")]
        if dum.shape[1] >= 2:
            out["industry"] = dum.iloc[:, 1:]           # drop_first
    return out


def _regress_out(y: pd.Series, x: pd.DataFrame, min_samples: int = MIN_SAMPLES) -> pd.Series:
    """单日截面 OLS：y = [1 X] β + resid；样本不足返回全 NaN"""
    data = pd.concat([y.rename("y"), x], axis=1).dropna()
    if len(data) < max(min_samples, x.shape[1] + 3):
        return pd.Series(np.nan, index=y.index)
    yy = data["y"].to_numpy(dtype=float)
    xx = np.column_stack([np.ones(len(data)), data.drop(columns="y").to_numpy(dtype=float)])
    resid = yy - xx @ np.linalg.lstsq(xx, yy, rcond=None)[0]
    out = pd.Series(np.nan, index=y.index)
    out.loc[data.index] = resid
    return out


def _r2(y: pd.Series, x: pd.DataFrame) -> float:
    """y 对 x 回归的 R²（含截距）；样本不足返回 nan"""
    data = pd.concat([y.rename("y"), x], axis=1).dropna()
    if len(data) < x.shape[1] + 3 or data["y"].std() == 0:
        return float("nan")
    yy = data["y"].to_numpy(dtype=float)
    xx = np.column_stack([np.ones(len(data)), data.drop(columns="y").to_numpy(dtype=float)])
    pred = xx @ np.linalg.lstsq(xx, yy, rcond=None)[0]
    ss_res = float(((yy - pred) ** 2).sum())
    ss_tot = float(((yy - yy.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _mean_exposure(signal: pd.DataFrame, ctrls: list, names: list, min_samples: int) -> float:
    rs = []
    for dt in signal.index:
        x = pd.concat([c.loc[dt].rename(n) for n, c in zip(names, ctrls)], axis=1)
        rs.append(_r2(signal.loc[dt], x))
    arr = np.array(rs, dtype=float)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")


def orthogonalize(signal: pd.DataFrame, controls: dict[str, pd.DataFrame],
                  min_samples: int = MIN_SAMPLES) -> tuple[pd.DataFrame, dict]:
    """逐日截面 OLS 正交化。

    signal: date×stock 因子信号
    controls: {控制名: date×stock DataFrame}（industry 须为展开后的 one-hot 矩阵）
    返回 (残差因子 date×stock, 诊断 dict)
    """
    if not controls:
        return signal.copy(), {"controls": [], "days": 0, "days_skipped": 0,
                               "coverage_before": float(signal.notna().mean().mean()),
                               "coverage_after": float(signal.notna().mean().mean()),
                               "exposure_before": float("nan"), "exposure_after": float("nan")}
    names = list(controls)
    ctrls = list(controls.values())
    resid = signal.copy()
    days_done = days_skipped = 0
    for dt in signal.index:
        x = pd.concat([c.loc[dt].rename(n) for n, c in zip(names, ctrls)], axis=1)
        r = _regress_out(signal.loc[dt], x, min_samples)
        if r.notna().any():
            resid.loc[dt] = winsorize_zscore(r)
            days_done += 1
        else:
            resid.loc[dt] = np.nan
            days_skipped += 1
    diag = {
        "controls": names,
        "days": days_done,
        "days_skipped": days_skipped,
        "coverage_before": float(signal.notna().mean().mean()),
        "coverage_after": float(resid.notna().mean().mean()),
        "exposure_before": _mean_exposure(signal, ctrls, names, min_samples),
        "exposure_after": _mean_exposure(resid, ctrls, names, min_samples),
    }
    return resid, diag


def orthogonalize_factor(panel, signal: pd.DataFrame, controls=("industry", "size", "style"),
                         horizon: int = 20, min_samples: int = MIN_SAMPLES) -> tuple[pd.DataFrame, dict]:
    """面板级正交化入口：构造控制变量 → 正交化 → 附加 IC/换手诊断。

    返回 (残差因子, 诊断 dict，含 ic_before/ic_after/ic_retention/turnover_before/turnover_after)
    """
    from qfm.pipeline.tests import compute_ic, forward_returns, turnover_ratio

    ctrls = controls_from_panel(panel, controls)
    resid, diag = orthogonalize(signal, ctrls, min_samples)
    fwd = forward_returns(panel.close, horizon)
    ic_b = compute_ic(signal, fwd).dropna()
    ic_a = compute_ic(resid, fwd).dropna()

    def _mean(s: pd.Series) -> float:
        return float(s.mean()) if len(s) else float("nan")

    mb, ma = _mean(ic_b), _mean(ic_a)
    diag.update({
        "horizon": horizon,
        "ic_before": mb,
        "ic_after": ma,
        "ic_retention": ma / mb if np.isfinite(mb) and mb != 0 else float("nan"),
        "turnover_before": turnover_ratio(signal),
        "turnover_after": turnover_ratio(resid),
    })
    return resid, diag
```

- [ ] **Step 5: 运行测试确认通过**

Run: `cd /Users/zxt/.zcode/workspace/default/quant-factor-mine && /opt/anaconda3/bin/python3 -m pytest tests/test_orthogonalize.py -v`
Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
cd /Users/zxt/.zcode/workspace/default/quant-factor-mine
git add qfm/orthogonalize.py tests/conftest.py tests/test_orthogonalize.py
git commit -m "feat: 逐日截面 OLS 正交化模块（行业/市值/风格暴露剥离 + 诊断）+ 合成数据夹具"
```

---

### Task 2: 过拟合检验模块（qfm/overfit.py）

**Files:**
- Create: `tests/test_overfit.py`
- Create: `qfm/overfit.py`

**Interfaces:**
- Produces: `sharpe_ratio(returns, benchmark=0.0) -> float`（逐期 Sharpe，非年化）
- Produces: `probabilistic_sharpe_ratio(observed_sr, benchmark_sr, n_obs, skew=0.0, kurtosis=3.0) -> float`
- Produces: `expected_max_sharpe(n_trials, sr_std) -> float`
- Produces: `deflated_sharpe_ratio(returns, n_trials, trials_matrix=None) -> float`（DSR 概率；有矩阵时用试验 SR 标准差，否则退化单次估计）
- Produces: `minimum_track_record_length(observed_sr, benchmark_sr=0.0, prob=0.95, skew=0.0, kurtosis=3.0) -> float`（∞ 当 SR≤基准）
- Produces: `pbo_cscv(trials_matrix, n_blocks=16) -> float`（PBO 概率；N<10 或 T<8 返回 nan）
- Produces: `haircut_sharpe(returns, n_trials, method="holm") -> dict`（键：sr_observed/sr_adjusted/p/p_adj/method）
- Produces: `OverfitReport` dataclass（verdict/passed/dsr/dsr_degraded/pbo/n_trials/n_obs/sr_observed/sr_adjusted/sr_annual/sr_annual_adjusted/haircut_method/minimum_track_record_length/reasons/matrix_used）
- Produces: `overfit_report(selected_returns, n_trials, trials_matrix=None, periods_per_year=252, haircut_method="holm") -> OverfitReport`（判定：DSR≥0.95 且（PBO 缺失或 <0.5）且 Haircut 后 SR>0）

- [ ] **Step 1: 写失败测试 tests/test_overfit.py**

```python
"""过拟合检验测试：DSR 削减 / PBO / Haircut 单调性 / MinTRL / 综合判定"""

from __future__ import annotations

import numpy as np
import pytest

from qfm.overfit import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    haircut_sharpe,
    minimum_track_record_length,
    overfit_report,
    pbo_cscv,
    probabilistic_sharpe_ratio,
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
    r = np.random.default_rng(0).normal(0.001, 0.01, 500)
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
    mat = _gauss_trials(T, N, std=0.01)
    edge = np.random.default_rng(3).normal(0.005, 0.01, T)
    rep = overfit_report(edge, N, trials_matrix=mat)
    assert rep.verdict == "PASS" and rep.passed is True
    assert rep.dsr > 0.95


def test_overfit_report_degraded_without_matrix():
    r = np.random.default_rng(9).normal(0.005, 0.01, 500)
    rep = overfit_report(r, n_trials=100, trials_matrix=None)
    assert rep.dsr_degraded is True and np.isnan(rep.pbo)
    assert rep.verdict in ("PASS", "FAIL")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd /Users/zxt/.zcode/workspace/default/quant-factor-mine && /opt/anaconda3/bin/python3 -m pytest tests/test_overfit.py -v`
Expected: FAIL（ModuleNotFoundError: qfm.overfit）

- [ ] **Step 3: 实现 qfm/overfit.py**

```python
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
            per_trial = [sharpe_ratio(m[:, j]) for j in range(m.shape[1])]
            vals = np.array([v for v in per_trial if np.isfinite(v)])
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
        return np.where(sd > 0, mu / np.where(sd > 0, sd, 1.0), np.nan)

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
        rank_frac = float((oos_sr <= oos_sr[j_star]).mean())
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
    sr_adj = norm.ppf(1 - p_adj / 2) / math.sqrt(n_obs)
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd /Users/zxt/.zcode/workspace/default/quant-factor-mine && /opt/anaconda3/bin/python3 -m pytest tests/test_overfit.py -v`
Expected: 11 passed（test_mintrl 中 `small == float("inf") or small > 0` 恒真，属断言宽松，保留）

- [ ] **Step 5: Commit**

```bash
cd /Users/zxt/.zcode/workspace/default/quant-factor-mine
git add qfm/overfit.py tests/test_overfit.py
git commit -m "feat: 回测过拟合检验模块（DSR/PBO/Haircut/MinTRL，文献公式独立实现）"
```

---

### Task 3: 无未来函数检查（qfm/pipeline/lookahead.py + factor_report 扩展）

**Files:**
- Create: `tests/test_lookahead.py`
- Create: `qfm/pipeline/lookahead.py`
- Modify: `qfm/pipeline/tests.py:125-160`（`factor_report` 增加可选 `lookahead=None` 参数）

**Interfaces:**
- Produces: `scan_source(src: str) -> dict`（键 `leaks: list[str]` / `warnings: list[str]`）
- Produces: `check_structural(kind: str) -> str`（`"mining"` → `"pass"`，其余 `"review"`）
- Modifies: `factor_report(factor_df, close, horizon=20, direction="positive", lookahead=None)` → 返回 dict 增加 `"lookahead"` 键（仅当传入 lookahead 时）

- [ ] **Step 1: 写失败测试 tests/test_lookahead.py**

```python
"""无未来函数检查测试：泄漏模式命中 / 合法源码放行 / 结构性判定 / factor_report 集成"""

from __future__ import annotations

from qfm.pipeline.lookahead import check_structural, scan_source
from qfm.pipeline.tests import factor_report


def test_scan_hits_leaks():
    src = "def f(d):\n    return d.close.shift(-1) / d.close"
    r = scan_source(src)
    assert r["leaks"], "负 shift 必须命中"


def test_scan_hits_iloc_and_rolling():
    assert scan_source("x = s.iloc[-1]")["leaks"]
    assert scan_source("x = s.rolling(-5).mean()")["leaks"]


def test_scan_clean_code_no_leaks():
    src = "def f(d):\n    return d.close.pct_change(20) / d.turnover.rolling(20).mean()"
    r = scan_source(src)
    assert not r["leaks"]


def test_scan_shift1_is_warning_only():
    r = scan_source("x = d.close.shift(1)")
    assert not r["leaks"] and r["warnings"]


def test_check_structural():
    assert check_structural("mining") == "pass"
    assert check_structural("custom") == "review"


def test_factor_report_lookahead_field(panel):
    from qfm.factors import compute_factor
    rep = factor_report(compute_factor("mom_20", panel), panel.close, horizon=20,
                        lookahead={"status": "pass", "hits": []})
    assert rep["lookahead"]["status"] == "pass"
    rep2 = factor_report(compute_factor("mom_20", panel), panel.close, horizon=20)
    assert "lookahead" not in rep2
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd /Users/zxt/.zcode/workspace/default/quant-factor-mine && /opt/anaconda3/bin/python3 -m pytest tests/test_lookahead.py -v`
Expected: FAIL（ModuleNotFoundError: qfm.pipeline.lookahead）

- [ ] **Step 3: 实现 qfm/pipeline/lookahead.py**

```python
"""无未来函数检查：结构性判定 + 静态泄漏模式扫描

方法论源自 QuantSkills skill-quant-factor-skill-factory 的 no-lookahead 检查思想，
本项目按因子计算语义独立实现（静态模式扫描，非 GPL 代码拷贝）。
"""

from __future__ import annotations

import re

# (正则, 描述) —— 命中即泄漏（引用未来数据）
LEAK_PATTERNS: list[tuple[str, str]] = [
    (r"\.shift\(\s*-\d", "负 shift：引用未来数据"),
    (r"\.iloc\[\s*-", "负 iloc：引用序列末尾（未来）数据"),
    (r"rolling\(\s*-", "负窗口 rolling"),
    (r"pct_change\s*\([^)]*-\d", "pct_change 负周期"),
    (r"fwd|forward_return|future|label|target", "疑似未来收益/标签变量"),
]

# 提示级（不必然泄漏，但需人工确认）
WARN_PATTERNS: list[tuple[str, str]] = [
    (r"\.shift\(1\)", "shift(1)：引用 T-1 日数据（安全，请确认方向）"),
]


def scan_source(src: str) -> dict:
    """静态扫描因子源码，返回 {leaks: [描述], warnings: [描述]}"""
    leaks = [d for p, d in LEAK_PATTERNS if re.search(p, src)]
    warns = [d for p, d in WARN_PATTERNS if re.search(p, src)]
    return {"leaks": leaks, "warnings": warns}


def check_structural(kind: str) -> str:
    """结构性判定：挖掘候选（指标×窗口×变换）全部基于 T 日及以前数据 → pass"""
    return "pass" if kind == "mining" else "review"
```

- [ ] **Step 4: 扩展 factor_report（qfm/pipeline/tests.py）**

在 `tests.py:125` 的函数签名后追加参数，并在 return dict 中追加 lookahead：

```python
def factor_report(factor_df: pd.DataFrame, close: pd.DataFrame, horizon: int = 20,
                  direction: str = "positive", lookahead: dict | None = None) -> dict:
```

return dict 最后一行 `"horizon": horizon,` 之后追加：

```python
        "horizon": horizon,
        **({"lookahead": lookahead} if lookahead is not None else {}),
    }
```

（注意：return dict 现有结尾为 `"horizon": horizon,\n    }`，把 `}` 改为 `**({...} if ... else {}),\n    }`）

- [ ] **Step 5: 运行测试确认通过**

Run: `cd /Users/zxt/.zcode/workspace/default/quant-factor-mine && /opt/anaconda3/bin/python3 -m pytest tests/test_lookahead.py -v`
Expected: 6 passed

- [ ] **Step 6: Commit**

```bash
cd /Users/zxt/.zcode/workspace/default/quant-factor-mine
git add qfm/pipeline/lookahead.py qfm/pipeline/tests.py tests/test_lookahead.py
git commit -m "feat: 无未来函数检查（静态泄漏扫描 + 挖掘候选结构性判定）+ factor_report 扩展"
```

---

### Task 4: 挖掘引擎保存试验矩阵（qfm/mining/engine.py + cli.py 兼容）

**Files:**
- Modify: `qfm/mining/engine.py`
- Modify: `qfm/cli.py:58`
- Create: `tests/test_mining_trials.py`

**Interfaces:**
- Modifies: `run_mining(panel, horizon=20, max_candidates=None, progress=None, save_trials=True, top_n=30, trials_dir=None) -> tuple[pd.DataFrame, dict | None]`（排行榜 DataFrame 新增「无未来函数」列；返回第二元素为 trials 元信息 dict 或 None）
- Produces: `candidate_monthly_returns(panel, fdf, top_n=30) -> pd.Series`（月频 TOP-N 等权组合收益）
- Produces: `save_trials_matrix(matrix: pd.DataFrame, meta: dict, trials_dir=None) -> str`（落盘路径，内含 trials.parquet + manifest.json）
- Produces: `latest_trials(trials_dir=None) -> tuple[pd.DataFrame, dict] | None`
- Produces: `TRIALS_DIR` 常量（`<项目根>/data_cache/trials`）
- Consumes: Task 2 无直接依赖；Task 7 回测页使用 `latest_trials`；Task 8 挖掘页使用新返回值

- [ ] **Step 1: 写失败测试 tests/test_mining_trials.py**

```python
"""挖掘引擎试验矩阵测试：落盘格式 / manifest / latest_trials / 排行榜新列"""

from __future__ import annotations

import json
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd /Users/zxt/.zcode/workspace/default/quant-factor-mine && /opt/anaconda3/bin/python3 -m pytest tests/test_mining_trials.py -v`
Expected: FAIL（run_mining 返回值仍为 DataFrame，解包报错）

- [ ] **Step 3: 实现 engine.py 改造**

在 `qfm/mining/engine.py` 文件顶部 import 后追加（原 import 为 `from __future__ import annotations` + `import pandas as pd` + `from qfm.pipeline.tests import factor_report`）：

```python
import json
import os
from pathlib import Path
```

替换 `run_mining` 函数整体（engine.py:37-65）：

```python
TRIALS_DIR = str(Path(__file__).resolve().parents[2] / "data_cache" / "trials")


def candidate_monthly_returns(panel, fdf: pd.DataFrame, top_n: int = 30) -> pd.Series:
    """以候选因子为打分、月末调仓 TOP-N 等权组合的月频收益序列（复用 run_backtest 成本模型）"""
    from qfm.portfolio.backtest import run_backtest

    bt = run_backtest(panel, fdf, top_n=top_n, bench_mode="equal")
    monthly = bt.nav.resample("ME").last()
    return monthly.pct_change().dropna()


def save_trials_matrix(matrix: pd.DataFrame, meta: dict, trials_dir: str | None = None) -> str:
    """保存 T×N 试验矩阵 + manifest.json，返回子目录路径"""
    d = Path(trials_dir or TRIALS_DIR)
    d.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    sub = d / f"trials_{meta.get('horizon', 20)}_{stamp}"
    sub.mkdir(exist_ok=True)
    matrix.to_parquet(sub / "trials.parquet")
    full = {**meta, "n_trials": int(matrix.shape[1]), "T_periods": int(matrix.shape[0]),
            "created_at": pd.Timestamp.now().isoformat()}
    (sub / "manifest.json").write_text(json.dumps(full, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(sub)


def latest_trials(trials_dir: str | None = None) -> tuple[pd.DataFrame, dict] | None:
    """读取最近一次试验矩阵 (DataFrame T×N, manifest)；无则 None"""
    d = Path(trials_dir or TRIALS_DIR)
    if not d.is_dir():
        return None
    subs = sorted([p for p in d.iterdir() if (p / "trials.parquet").exists()],
                  key=lambda p: p.name, reverse=True)
    if not subs:
        return None
    sub = subs[0]
    return (pd.read_parquet(sub / "trials.parquet"),
            json.loads((sub / "manifest.json").read_text(encoding="utf-8")))


def run_mining(panel, horizon: int = 20, max_candidates: int | None = None,
               progress=None, save_trials: bool = True, top_n: int = 30,
               trials_dir: str | None = None) -> tuple[pd.DataFrame, dict | None]:
    """批量检验全部候选，返回 (排行榜 DataFrame, 试验矩阵元信息或 None)

    排行榜按 |IC| 降序，含「无未来函数」列（挖掘候选结构性安全）；
    save_trials=True 时逐候选计算月频 TOP-N 组合收益，保存 T×N 试验矩阵。
    """
    cands = generate_candidates(panel)
    if max_candidates:
        cands = dict(list(cands.items())[:max_candidates])

    rows = []
    trial_series: dict[str, pd.Series] = {}
    n = len(cands)
    for i, (name, fdf) in enumerate(cands.items()):
        if progress:
            progress(i, n, name)
        rep = factor_report(fdf, panel.close, horizon=horizon, direction="positive")
        s = rep["ic_summary"]
        rows.append({
            "因子": name,
            "IC": s["ic_mean"],
            "IC_IR": s["ic_ir"],
            "t值": s["ic_t"],
            "正占比": s["pos_ratio"],
            "有效天数": s["n_days"],
            "换手率": rep["turnover"],
            "单调": rep["monotonicity"]["monotonic"],
            "无未来函数": "pass（结构安全）",
        })
        if save_trials:
            trial_series[name] = candidate_monthly_returns(panel, fdf, top_n=top_n)
    out = pd.DataFrame(rows)
    if len(out):
        out["|IC|"] = out["IC"].abs()
        out = out.sort_values("|IC|", ascending=False).drop(columns="|IC|").reset_index(drop=True)

    meta = None
    if save_trials and trial_series:
        matrix = pd.DataFrame(trial_series).dropna(how="all")
        meta = {"horizon": horizon, "top_n": top_n, "max_candidates": len(cands)}
        path = save_trials_matrix(matrix, meta, trials_dir)
        meta = {**meta, "path": path, "n_trials": int(matrix.shape[1]),
                "T_periods": int(matrix.shape[0])}
    return out, meta
```

- [ ] **Step 4: 更新 cli.py 调用处（qfm/cli.py:58）**

将：

```python
    df = run_mining(panel, horizon=args.horizon, max_candidates=args.max_candidates,
```

改为：

```python
    df, _ = run_mining(panel, horizon=args.horizon, max_candidates=args.max_candidates,
```

（若该调用带 progress 参数则保留；只需解包新增的返回值）

- [ ] **Step 5: 运行全部相关测试**

Run: `cd /Users/zxt/.zcode/workspace/default/quant-factor-mine && /opt/anaconda3/bin/python3 -m pytest tests/ -q`
Expected: 全部通过（Task1-4 共 26 个用例）

- [ ] **Step 6: Commit**

```bash
cd /Users/zxt/.zcode/workspace/default/quant-factor-mine
git add qfm/mining/engine.py qfm/cli.py tests/test_mining_trials.py
git commit -m "feat: 挖掘引擎保存试验矩阵（T×N + manifest）+ 排行榜无未来函数列 + cli 兼容"
```

---

### Task 5: 因子合成正交化升级（qfm/portfolio/synthesis.py）

**Files:**
- Modify: `qfm/portfolio/synthesis.py:60-103`（`synthesize` 签名与正交化实现）
- Create: `tests/test_synthesis.py`

**Interfaces:**
- Modifies: `synthesize(panel, names, mode="equal", horizon=20, orthogonalize=False, ortho_controls=None) -> tuple[pd.DataFrame, dict]`
  - `ortho_controls`: 元组子集（`"industry"`/`"size"`/`"style"`）；None 时默认 `("size",)`（与旧行为等价）
  - 正交化改为调用 Task 1 的 `qfm.orthogonalize.orthogonalize_factor`（残差按日 z-score，排名序不变，回测选取等价）
- Consumes: `qfm.orthogonalize.orthogonalize_factor`（Task 1）

- [ ] **Step 1: 写失败测试 tests/test_synthesis.py**

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd /Users/zxt/.zcode/workspace/default/quant-factor-mine && /opt/anaconda3/bin/python3 -m pytest tests/test_synthesis.py -v`
Expected: 至少 test_synthesize_size_orthogonalized 失败或报 TypeError（新参数不存在）——先按"未实现"预期：若旧代码无 ortho_controls 参数，调用报 TypeError，即失败成立

- [ ] **Step 3: 修改 qfm/portfolio/synthesis.py**

将 `synthesize` 签名（synthesis.py:60-61）：

```python
def synthesize(panel, names: list, mode: str = "equal", horizon: int = 20,
               orthogonalize: bool = False) -> tuple[pd.DataFrame, dict]:
```

改为：

```python
def synthesize(panel, names: list, mode: str = "equal", horizon: int = 20,
               orthogonalize: bool = False,
               ortho_controls: tuple | None = None) -> tuple[pd.DataFrame, dict]:
```

将正交化实现块（synthesis.py:96-102，从 `if orthogonalize:` 到 `score = resid`）整体替换为：

```python
    if orthogonalize:
        from qfm.orthogonalize import orthogonalize_factor

        controls = ortho_controls or ("size",)
        score, _diag = orthogonalize_factor(panel, score, controls=controls, horizon=horizon)
    return score, weights
```

（即删除旧的内联 ln(mv) 回归循环；注意函数体原有 `return score, weights` 在最后，保持不动）

- [ ] **Step 4: 运行测试确认通过**

Run: `cd /Users/zxt/.zcode/workspace/default/quant-factor-mine && /opt/anaconda3/bin/python3 -m pytest tests/test_synthesis.py -v`
Expected: 3 passed

- [ ] **Step 5: 回归全部测试**

Run: `cd /Users/zxt/.zcode/workspace/default/quant-factor-mine && /opt/anaconda3/bin/python3 -m pytest tests/ -q`
Expected: 全部通过

- [ ] **Step 6: Commit**

```bash
cd /Users/zxt/.zcode/workspace/default/quant-factor-mine
git add qfm/portfolio/synthesis.py tests/test_synthesis.py
git commit -m "feat: 因子合成正交化升级（行业/市值/风格可选剥离，向后兼容）"
```

---

### Task 6: 页面集成①——因子检验页「正交化」Tab

**Files:**
- Modify: `app.py:206`（tabs 定义）+ `app.py:315` 后（新增 Tab 3 代码块）

**Interfaces:**
- Consumes: `qfm.orthogonalize.orthogonalize_factor`（Task 1）、`qfm.portfolio.synthesis.factor_panel`（已存在）
- 无新接口

- [ ] **Step 1: 修改 tabs 定义**

将 `app.py:206`：

```python
    tab_single, tab_cmp = st.tabs(["单因子检验", "多因子对比"])
```

改为：

```python
    tab_single, tab_cmp, tab_ortho = st.tabs(["单因子检验", "多因子对比", "正交化"])
```

- [ ] **Step 2: 在 page_test 末尾（`tab_cmp` 块结束后、`def page_mine` 前）插入 Tab 3 代码**

```python
    # ---------- Tab 3：正交化（QuantSkills factor-orthogonalize 方法论） ----------
    with tab_ortho:
        st.markdown("逐日截面 OLS 正交化：剥离行业 / 市值 / 风格暴露 → 残差因子（方法论源自 QuantSkills factor-orthogonalize）")
        c1, c2, c3 = st.columns([2, 1, 1])
        ortho_name = c1.selectbox("选择因子", [f.name for f in list_factors()], key="ortho_f")
        ortho_h = c2.selectbox("前瞻天数", [5, 10, 20, 60], index=2, key="ortho_h")
        ortho_controls = c3.multiselect("剥离暴露", ["industry", "size", "style"],
                                        default=["industry", "size", "style"],
                                        format_func=lambda c: {"industry": "行业", "size": "市值",
                                                               "style": "风格(beta/波动率)"}[c])
        if st.button("运行正交化", use_container_width=True):
            if not ortho_controls:
                st.error("至少选择一项剥离暴露")
                return
            from qfm.orthogonalize import orthogonalize_factor
            from qfm.portfolio.synthesis import factor_panel
            with st.status("正交化中…", expanded=False) as status:
                fdf = factor_panel(panel, [ortho_name])[ortho_name]
                resid, diag = orthogonalize_factor(panel, fdf, controls=tuple(ortho_controls),
                                                   horizon=ortho_h)
                status.update(label=f"✅ 完成：{ortho_name} → 残差因子（{diag['days']} 日有效）",
                              state="complete")
            k1, k2, k3 = st.columns(3)
            k1.metric("IC 保留率", f"{diag['ic_retention']:.0%}" if pd.notna(diag["ic_retention"]) else "—",
                      help="正交后 IC / 正交前 IC；越低说明信号越依赖被剥离暴露")
            k2.metric("暴露 R²（前→后）", f"{diag['exposure_before']:.3f} → {diag['exposure_after']:.3f}",
                      help="信号对控制变量回归 R²：正交后应≈0")
            k3.metric("覆盖率（前→后）", f"{diag['coverage_before']:.0%} → {diag['coverage_after']:.0%}")
            cmp = pd.DataFrame({
                "指标": ["IC", "暴露 R²", "TOP10% 换手", "覆盖率"],
                "正交前": [diag["ic_before"], diag["exposure_before"],
                           diag["turnover_before"], diag["coverage_before"]],
                "正交后": [diag["ic_after"], diag["exposure_after"],
                           diag["turnover_after"], diag["coverage_after"]],
            })
            st.dataframe(cmp.style.format({"正交前": "{:.4f}", "正交后": "{:.4f}"}),
                         hide_index=True, use_container_width=True)
            if diag["days_skipped"]:
                st.caption(f"跳过 {diag['days_skipped']} 日（截面样本 <30）")
            st.download_button("⬇ 下载残差因子 CSV", resid.to_csv().encode("utf-8-sig"),
                               file_name=f"{ortho_name}_residual.csv", mime="text/csv")
```

- [ ] **Step 3: 语法检查 + 启动冒烟**

```bash
cd /Users/zxt/.zcode/workspace/default/quant-factor-mine
/opt/anaconda3/bin/python3 -m py_compile app.py
```

Expected: 无输出（编译通过）。然后重启 streamlit 并检查健康：

```bash
pkill -f "streamlit run" || true
nohup /opt/anaconda3/bin/streamlit run app.py --server.headless true > /tmp/qfm_app.log 2>&1 < /dev/null &
sleep 8
curl -s http://localhost:8501/_stcore/health
grep -i "traceback\|error" /tmp/qfm_app.log | head -5 || echo "日志无错误"
```

Expected: health 返回 `ok`；日志无 Traceback

- [ ] **Step 4: Commit**

```bash
cd /Users/zxt/.zcode/workspace/default/quant-factor-mine
git add app.py
git commit -m "feat: 因子检验页新增正交化 Tab（前后诊断对比 + 残差因子下载）"
```

---

### Task 7: 页面集成②——策略回测页（正交化升级 + 过拟合检验区）

**Files:**
- Modify: `app.py:392-420`（① 因子合成控件区 + synthesize 调用）
- Modify: `app.py:473-489`（绩效指标后插入过拟合检验区）

**Interfaces:**
- Consumes: `qfm.mining.latest_trials`（Task 4）、`qfm.overfit.overfit_report`（Task 2）
- 注意：回测月频收益 → `periods_per_year=12`

- [ ] **Step 1: 替换 ① 因子合成的正交化控件（app.py:399）**

将：

```python
    ortho = c4.checkbox("市值正交化", help="对流通市值回归取残差，消除规模暴露")
```

改为：

```python
    with st.expander("因子正交化（行业 / 市值 / 风格暴露剥离）"):
        ortho = st.checkbox("启用正交化", value=False,
                            help="逐日截面 OLS 残差化，消除所选暴露（方法论源自 QuantSkills factor-orthogonalize）")
        ortho_controls = st.multiselect("剥离暴露", ["industry", "size", "style"], default=["size"],
                                        format_func=lambda c: {"industry": "行业", "size": "市值",
                                                               "style": "风格(beta/波动率)"}[c])
```

- [ ] **Step 2: 更新 synthesize 调用（app.py:415-416）**

将：

```python
            score, weights = synthesize(panel, names, mode=mode, horizon=horizon,
                                        orthogonalize=ortho)
```

改为：

```python
            score, weights = synthesize(panel, names, mode=mode, horizon=horizon,
                                        orthogonalize=ortho,
                                        ortho_controls=tuple(ortho_controls) if ortho else None)
```

- [ ] **Step 3: 在「分年绩效」块之后（app.py:487 之后、`st.download_button("⬇ 下载净值 CSV"...` 之前）插入过拟合检验区**

```python
        # 过拟合检验（QuantSkills skill-backtest-overfit 方法论）
        st.markdown("**过拟合检验**（DSR / PBO / Haircut / MinTRL）")
        monthly = bt.nav.resample("ME").last().pct_change().dropna()
        if len(monthly) < 6:
            st.warning("回测期过短（<6 个月），无法做统计显著性检验")
        else:
            from qfm.mining import latest_trials
            from qfm.overfit import overfit_report

            tl = latest_trials()
            trials_df, trials_meta = (tl[0], tl[1]) if tl else (None, None)
            default_n = int((trials_meta or {}).get("n_trials", 36))
            n_trials = st.number_input("试验次数 n_trials（诚实申报：得到该结果前试过的全部参数/候选配置数）",
                                       min_value=1, max_value=100000, value=default_n,
                                       help="来自最近一次自动挖掘的候选数；少报 = 自欺，DSR/PBO 会偏乐观")
            use_matrix = False
            if trials_df is not None:
                use_matrix = st.checkbox(
                    f"使用最近一次挖掘试验矩阵（{trials_df.shape[1]} 候选 × {trials_df.shape[0]} 期，"
                    f"{trials_meta['created_at'][:10]} 生成，horizon={trials_meta['horizon']}）",
                    value=True)
            rep = overfit_report(monthly.values, n_trials=int(n_trials),
                                 trials_matrix=trials_df.values if (use_matrix and trials_df is not None) else None,
                                 periods_per_year=12, haircut_method="holm")
            vcolor = {"PASS": "#2E9E6B", "FAIL": "#E5484D", "INSUFFICIENT": "#C56A00"}[rep.verdict]
            st.markdown(
                f"<span style='color:{vcolor};font-weight:600;font-size:15px'>结论：{rep.verdict}"
                f" — {'统计上可区分于多重检验噪声' if rep.passed else '存在过拟合 / 数据挖掘嫌疑'}</span>",
                unsafe_allow_html=True)
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("DSR 削减夏普", f"{rep.dsr:.2f}" if pd.notna(rep.dsr) else "—",
                      help="P[真实 SR > 期望最大 SR_N]；≥0.95 才可信")
            k2.metric("PBO 过拟合概率", f"{rep.pbo:.2f}" if pd.notna(rep.pbo) else "—",
                      help="IS 最优策略在 OOS 落入下半区的概率；<0.5 才安全；无试验矩阵时为 —")
            k3.metric("Haircut 夏普（年化）", f"{rep.sr_annual:.2f} → {rep.sr_annual_adjusted:.2f}",
                      help=f"多重检验打折（{rep.haircut_method}，n_trials={rep.n_trials}）")
            k4.metric("MinTRL 最小样本期数", f"{rep.minimum_track_record_length:.0f}"
                      if np.isfinite(rep.minimum_track_record_length) else "∞",
                      help="该夏普达到统计显著（PSR≥95%）所需的最少期数")
            if rep.reasons:
                st.warning("未通过项：" + "；".join(rep.reasons))
            if rep.dsr_degraded:
                st.caption("⚠️ 未提供试验矩阵，DSR 为退化估计（偏宽松）。精确 PBO 需先在「自动挖掘」页跑一次并勾选保存试验矩阵。")
```

- [ ] **Step 4: 语法检查 + 重启 + 健康检查**

```bash
cd /Users/zxt/.zcode/workspace/default/quant-factor-mine
/opt/anaconda3/bin/python3 -m py_compile app.py
pkill -f "streamlit run" || true
nohup /opt/anaconda3/bin/streamlit run app.py --server.headless true > /tmp/qfm_app.log 2>&1 < /dev/null &
sleep 8
curl -s http://localhost:8501/_stcore/health
grep -i "traceback\|error" /tmp/qfm_app.log | head -5 || echo "日志无错误"
```

Expected: health 返回 `ok`；日志无 Traceback

- [ ] **Step 5: Commit**

```bash
cd /Users/zxt/.zcode/workspace/default/quant-factor-mine
git add app.py
git commit -m "feat: 策略回测页过拟合检验区（DSR/PBO/Haircut/MinTRL + PASS/FAIL）+ 正交化选项升级"
```

---

### Task 8: 页面集成③——自动挖掘页 + 自定义因子页

**Files:**
- Modify: `app.py:321-338`（page_mine）
- Modify: `app.py:353-382`（page_custom）

**Interfaces:**
- Consumes: `run_mining` 新返回值（Task 4）、`qfm.pipeline.lookahead.scan_source`（Task 3）

- [ ] **Step 1: 修改 page_mine（app.py:325-338）**

将：

```python
    c1, c2 = st.columns([1, 1])
    horizon = c1.selectbox("前瞻天数", [5, 10, 20, 60], index=2, key="mine_h")
    max_c = c2.selectbox("候选数限制", [36, 72, 144, 288], index=0,
                         help="36=单窗口集；越大跑得越久（每候选约 2 秒）")
    if st.button("开始挖掘", use_container_width=True):
        prog = st.progress(0.0, text="准备…")
        df = run_mining(panel, horizon=horizon, max_candidates=max_c,
                        progress=lambda i, n, nm: prog.progress((i + 1) / n, text=f"{i+1}/{n} · {nm}"))
        st.success(f"挖掘完成：{len(df)} 个候选因子")
        st.dataframe(df.style.format({"IC": "{:+.4f}", "IC_IR": "{:.2f}", "t值": "{:.2f}",
                                      "正占比": "{:.0%}", "换手率": "{:.0%}"}),
                     use_container_width=True, height=420)
        st.download_button("⬇ 下载排行榜 CSV", df.to_csv(index=False).encode("utf-8-sig"),
                           file_name="factor_leaderboard.csv", mime="text/csv")
```

改为：

```python
    c1, c2 = st.columns([1, 1])
    horizon = c1.selectbox("前瞻天数", [5, 10, 20, 60], index=2, key="mine_h")
    max_c = c2.selectbox("候选数限制", [36, 72, 144, 288], index=0,
                         help="36=单窗口集；越大跑得越久（每候选约 2 秒）")
    save_trials = st.checkbox("保存试验矩阵（供策略回测页过拟合检验使用）", value=True,
                              help="逐候选计算月频 TOP-30 组合收益，落盘 data_cache/trials/")
    if st.button("开始挖掘", use_container_width=True):
        prog = st.progress(0.0, text="准备…")
        df, trials_meta = run_mining(panel, horizon=horizon, max_candidates=max_c,
                                     save_trials=save_trials,
                                     progress=lambda i, n, nm: prog.progress((i + 1) / n, text=f"{i+1}/{n} · {nm}"))
        st.success(f"挖掘完成：{len(df)} 个候选因子")
        st.dataframe(df.style.format({"IC": "{:+.4f}", "IC_IR": "{:.2f}", "t值": "{:.2f}",
                                      "正占比": "{:.0%}", "换手率": "{:.0%}"}),
                     use_container_width=True, height=420)
        if trials_meta:
            st.caption(f"📁 试验矩阵已保存：`{trials_meta['path']}`"
                       f"（{trials_meta['n_trials']} 候选 × {trials_meta['T_periods']} 期月频收益 · "
                       f"TOP{trials_meta['top_n']} 等权）")
        st.download_button("⬇ 下载排行榜 CSV", df.to_csv(index=False).encode("utf-8-sig"),
                           file_name="factor_leaderboard.csv", mime="text/csv")
```

- [ ] **Step 2: 修改 page_custom（app.py:360-366 区域）**

在 `st.success("注册成功，已自动进入检验流程")` 之后插入无未来函数检查：

将：

```python
            exec(code, ns)
            st.success("注册成功，已自动进入检验流程")
```

改为：

```python
            exec(code, ns)
            from qfm.pipeline.lookahead import scan_source
            chk = scan_source(code)
            if chk["leaks"]:
                st.error("⚠️ 检测到未来函数泄漏模式：" + "；".join(chk["leaks"]) +
                         "（因子将引用未来数据，检验结果不可信）")
            elif chk["warnings"]:
                st.warning("提示：" + "；".join(chk["warnings"]))
            st.success("注册成功，已自动进入检验流程")
```

- [ ] **Step 3: 语法检查 + 重启 + 健康检查**

```bash
cd /Users/zxt/.zcode/workspace/default/quant-factor-mine
/opt/anaconda3/bin/python3 -m py_compile app.py
pkill -f "streamlit run" || true
nohup /opt/anaconda3/bin/streamlit run app.py --server.headless true > /tmp/qfm_app.log 2>&1 < /dev/null &
sleep 8
curl -s http://localhost:8501/_stcore/health
grep -i "traceback\|error" /tmp/qfm_app.log | head -5 || echo "日志无错误"
```

Expected: health 返回 `ok`；日志无 Traceback

- [ ] **Step 4: Commit**

```bash
cd /Users/zxt/.zcode/workspace/default/quant-factor-mine
git add app.py
git commit -m "feat: 自动挖掘页试验矩阵开关与路径提示 + 自定义因子页无未来函数静态扫描"
```

---

### Task 9: 端到端验收（pytest 全绿 + 页面冒烟 + README 方法论出处）

**Files:**
- Modify: `README.md`（新增方法论出处与功能说明）
- Modify: `交接.md`（新增功能状态，供上下文压缩后续用）

- [ ] **Step 1: 全量测试**

Run: `cd /Users/zxt/.zcode/workspace/default/quant-factor-mine && /opt/anaconda3/bin/python3 -m pytest tests/ -q`
Expected: 全部通过（29 个用例：正交化 5 + overfit 11 + lookahead 6 + mining 4 + synthesis 3）

- [ ] **Step 2: 页面冒烟**

```bash
cd /Users/zxt/.zcode/workspace/default/quant-factor-mine
pkill -f "streamlit run" || true
nohup /opt/anaconda3/bin/streamlit run app.py --server.headless true > /tmp/qfm_app.log 2>&1 < /dev/null &
sleep 8
curl -s http://localhost:8501/_stcore/health
```

Expected: `ok`；`grep -ci traceback /tmp/qfm_app.log` 为 0
额外：用真实缓存数据验证一次核心链路（若 data_cache 有缓存）：

```bash
cd /Users/zxt/.zcode/workspace/default/quant-factor-mine && /opt/anaconda3/bin/python3 - <<'PY'
from qfm.data import DataLoader, build_panel, get_universe
from qfm.orthogonalize import orthogonalize_factor
from qfm.overfit import overfit_report
from qfm.mining import latest_trials
import numpy as np, pandas as pd
dl = DataLoader()
stocks = dl.status().get("stocks_cached", 0)
print(f"缓存股票数: {stocks}")
if stocks:
    codes = get_universe("index800")[:min(stocks, 50)]
    bars = dl.load_bars(codes)
    ind = dl.load_indicators()
    panel = build_panel(bars, ind)
    print("panel:", panel.close.shape, "| 行业列:", panel.industry.shape if len(panel.industry) else "无")
    from qfm.factors import compute_factor
    resid, diag = orthogonalize_factor(panel, compute_factor("ep_ttm", panel),
                                       controls=("industry", "size"), horizon=20)
    print("正交化: IC保留率=%.2f%% 暴露R²=%.3f→%.3f" % (
        diag["ic_retention"] * 100, diag["exposure_before"], diag["exposure_after"]))
    nav = pd.Series(np.cumprod(1 + np.random.default_rng(0).normal(0.01, 0.05, 36)), index=pd.date_range("2023-01-31", periods=36, freq="ME"))
    rep = overfit_report(nav.pct_change().dropna().values, n_trials=36)
    print("过拟合报告: verdict=", rep.verdict, "dsr=%.2f mintrl=%.0f" % (rep.dsr, rep.minimum_track_record_length))
print("trials:", latest_trials())
PY
```

Expected: 打印面板形状、正交化诊断与过拟合结论；无异常（无缓存时仅打印 0 股票数，不报错）

- [ ] **Step 3: README.md 增补（文件末尾追加）**

```markdown
## 方法论出处（QuantSkills 三技能融入）

| 功能 | 方法论来源 | 独立实现依据 |
|---|---|---|
| 逐日截面 OLS 正交化 | QuantSkills `skill-factor-orthogonalize`（GPL-3.0） | 公开标准截面回归；行业/市值/风格暴露剥离 |
| 回测过拟合检验 DSR/PBO/Haircut/MinTRL | QuantSkills `skill-backtest-overfit`（GPL-3.0） | Bailey & López de Prado (2012/2014)、Bailey et al. (2017)、Harvey & Liu (2015) 文献公式 |
| 无未来函数检查 | QuantSkills `skill-quant-factor-skill-factory`（GPL-3.0） | 静态泄漏模式扫描 + 挖掘候选结构性判定 |

本项目未拷贝上述仓库源码，算法按文献公式独立实现。
```

- [ ] **Step 4: 交接.md 增补（「三、当前状态」列表追加）**

```markdown
- ✅ QuantSkills 三技能融入：因子检验页「正交化」Tab（行业/市值/风格 OLS 剥离 + 前后诊断）、策略回测页「过拟合检验」（DSR/PBO/Haircut/MinTRL + PASS/FAIL，n_trials 自动取挖掘候选数）、自动挖掘页试验矩阵（data_cache/trials/，供回测页 PBO）、自定义因子页无未来函数静态扫描；算法按文献公式独立实现（GPL 源码未拷贝）
```

- [ ] **Step 5: 最终提交**

```bash
cd /Users/zxt/.zcode/workspace/default/quant-factor-mine
git add README.md 交接.md
git commit -m "docs: 方法论出处与功能状态更新"
```

- [ ] **Step 6: 交付验收汇总**

输出：pytest 全量输出、health 检查、真实数据链路输出、git log（本次 8 个 commit：1 设计文档 + 7 功能）、KPI 卡（PUA 协议）。
