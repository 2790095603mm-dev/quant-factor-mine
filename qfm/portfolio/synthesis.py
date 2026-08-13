"""因子合成：多因子加权打分 + 相关性矩阵 + 市值正交化"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qfm.factors import compute_factor
from qfm.pipeline.clean import clean_factor


def factor_panel(panel, names: list) -> pd.DataFrame:
    """计算并清洗多个因子，返回 {因子名: date×stock DataFrame}"""
    out = {}
    for name in names:
        fdf = compute_factor(name, panel)
        out[name] = clean_factor(fdf)
    return out


def factor_corr(factors: dict) -> pd.DataFrame:
    """因子间截面相关性矩阵（按每期截面相关系数均值）"""
    common = None
    for fdf in factors.values():
        if common is None:
            common = fdf.notna()
        else:
            common &= fdf.notna()
    corrs = {}
    for a in factors:
        row = {}
        for b in factors:
            if a == b:
                row[b] = 1.0
                continue
            rs = []
            for dt in factors[a].index:
                mask = common.loc[dt]
                va, vb = factors[a].loc[dt, mask], factors[b].loc[dt, mask]
                if len(va) > 20:
                    rs.append(va.corr(vb))
            row[b] = float(np.mean(rs)) if rs else np.nan
        corrs[a] = row
    return pd.DataFrame(corrs)


def weight_by_ic(panel, factors: dict, horizon: int = 20) -> dict:
    """IC 加权：权重 ∝ 全期平均 |IC|（可扩展到滚动 IC）"""
    from qfm.pipeline.tests import compute_ic, forward_returns

    fwd = forward_returns(panel.close, horizon)
    w = {}
    for name, fdf in factors.items():
        ic = compute_ic(fdf, fwd)
        w[name] = abs(ic.mean()) if len(ic) else 0.0
    tot = sum(w.values())
    return {k: (v / tot if tot > 0 else 1 / len(w)) for k, v in w.items()}


def synthesize(panel, names: list, mode: str = "equal", horizon: int = 20,
               orthogonalize: bool = False) -> tuple[pd.DataFrame, dict]:
    """因子合成主入口

    mode: equal=等权 | ic=IC加权 | icir=IC_IR加权
    orthogonalize: 对流通市值(对数)回归取残差，消除规模暴露
    返回 (综合得分 date×stock, 权重 dict)
    """
    factors = factor_panel(panel, names)
    if mode == "equal":
        weights = {n: 1 / len(names) for n in names}
    elif mode == "ic":
        weights = weight_by_ic(panel, factors, horizon)
    elif mode == "icir":
        from qfm.pipeline.tests import compute_ic, forward_returns

        fwd = forward_returns(panel.close, horizon)
        w = {}
        for n, fdf in factors.items():
            ic = compute_ic(fdf, fwd)
            icir = ic.mean() / ic.std(ddof=1) if len(ic) > 1 and ic.std() > 0 else 0.0
            w[n] = abs(icir)
        tot = sum(w.values())
        weights = {k: (v / tot if tot > 0 else 1 / len(w)) for k, v in w.items()}
    else:
        raise ValueError(f"未知权重模式: {mode}")

    score = sum(weights[n] * factors[n] for n in names)
    if orthogonalize:
        mv = np.log(panel.mv_float)
        # 逐期对 ln(市值) 回归，取残差（消除规模因子暴露）
        resid = score.copy()
        for dt in score.index:
            y = score.loc[dt]
            x = mv.loc[dt]
            mask = y.notna() & x.notna()
            if mask.sum() < 20:
                continue
            xx = np.column_stack([np.ones(mask.sum()), x[mask].values])
            beta, *_ = np.linalg.lstsq(xx, y[mask].values, rcond=None)
            resid.loc[dt, mask] = y[mask] - xx @ beta
        score = resid
    return score, weights
