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
        # 行业 one-hot：每个行业一张 date×stock 的 0/1 面板（控制变量统一为 date×stock 形态）
        # 注意：get_dummies 对"逐列常量"的 DataFrame 会生成重复列名，故走长表 pivot
        long = panel.industry.stack().rename("ind").reset_index()
        long.columns = ["date", "stock", "ind"]
        long["one"] = 1.0
        dum = long.pivot_table(index=["date", "stock"], columns="ind", values="one",
                               aggfunc="first", fill_value=0.0)
        wide = dum.unstack("stock")          # index=date, columns=MultiIndex(ind, stock)
        ind_names = sorted(wide.columns.get_level_values(0).unique())
        for ind_name in ind_names[1:]:       # drop_first 防共线
            out[f"industry_{ind_name}"] = (wide[ind_name]
                                           .reindex(index=panel.close.index,
                                                    columns=panel.close.columns)
                                           .astype(float))
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


def residualize_frame(signal: pd.DataFrame, controls: dict[str, pd.DataFrame],
                      min_samples: int = MIN_SAMPLES) -> tuple[pd.DataFrame, dict]:
    """逐日截面 OLS 正交化，**返回原始残差**（不做截尾/标准化）。

    这是唯一的截面剥离实现：`orthogonalize`（残差再标准化，供正交化页）与
    `qfm.pipeline` 的中性化阶段都调用它，避免出现多套互相不一致的 OLS。

    controls: {控制名: date×stock DataFrame}（industry 须为展开后的 one-hot 矩阵）
    返回 (残差 date×stock, 诊断 dict)。样本不足的交易日残差为 NaN。
    """
    if not controls:
        flat = float(signal.notna().mean().mean())
        return signal.copy(), {"controls": [], "days": 0, "days_skipped": 0,
                               "coverage_before": flat, "coverage_after": flat,
                               "exposure_before": float("nan"), "exposure_after": float("nan")}
    names = list(controls)
    ctrls = list(controls.values())
    resid = signal.copy()
    days_done = days_skipped = 0
    for dt in signal.index:
        x = pd.concat([c.loc[dt].rename(n) for n, c in zip(names, ctrls)], axis=1)
        r = _regress_out(signal.loc[dt], x, min_samples)
        if r.notna().any():
            resid.loc[dt] = r
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


def orthogonalize(signal: pd.DataFrame, controls: dict[str, pd.DataFrame],
                  min_samples: int = MIN_SAMPLES) -> tuple[pd.DataFrame, dict]:
    """逐日截面 OLS 正交化 + 残差重标准化（5MAD 截尾 → z-score）。

    行为与历史版本一致；残差的原始回归由 `residualize_frame` 承担。
    返回 (残差因子 date×stock, 诊断 dict)
    """
    resid, diag = residualize_frame(signal, controls, min_samples)
    if not controls:
        return resid, diag
    out = resid.copy()
    for dt in signal.index:
        if resid.loc[dt].notna().any():
            out.loc[dt] = winsorize_zscore(resid.loc[dt])
        else:
            out.loc[dt] = np.nan
    # 重标准化不改变非空模式，也不改变线性回归 R²；此处仅按最终残差复核一遍
    diag["coverage_after"] = float(out.notna().mean().mean())
    diag["exposure_after"] = _mean_exposure(out, list(controls.values()), list(controls), min_samples)
    return out, diag


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
