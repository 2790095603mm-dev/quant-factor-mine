"""因子合成：多因子加权打分 + 相关性矩阵 + 市值正交化"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qfm.factors import compute_factor, get_factor
from qfm.pipeline.clean import clean_factor


def factor_panel(panel, names: list, align_direction: bool = False,
                 factor_versions: dict[str, int] | None = None) -> dict[str, pd.DataFrame]:
    """计算并清洗多个因子，返回 {因子名: date×stock DataFrame}。

    ``align_direction=True`` 时，所有因子都会被规范为“数值越高越好”。
    这应当用于合成和选股；原始因子研究可保留自然方向。
    """
    out = {}
    for name in names:
        version = (factor_versions or {}).get(name)
        factor = get_factor(name, version=version)
        if factor is None:
            suffix = f" v{version}" if version is not None else ""
            raise KeyError(f"未知因子: {name}{suffix}")
        fdf = factor.func(panel) if version is not None else compute_factor(name, panel)
        cleaned = clean_factor(fdf)
        out[name] = -cleaned if align_direction and factor and factor.direction == "negative" else cleaned
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


def _normalise_weights(raw: dict[str, float], names: list[str]) -> dict[str, float]:
    """仅保留正向历史贡献；全为噪声时退回等权。"""
    cleaned = {name: max(float(raw.get(name, 0.0)), 0.0) for name in names}
    total = sum(cleaned.values())
    return ({name: value / total for name, value in cleaned.items()}
            if total > 0 else {name: 1 / len(names) for name in names})


def walk_forward_weights(panel, factors: dict[str, pd.DataFrame], horizon: int = 20,
                         mode: str = "ic", lookback: int = 252,
                         min_obs: int = 60, rebalance: str = "ME") -> pd.DataFrame:
    """仅用当日之前已经实现的收益，生成逐日有效的 IC / IC_IR 权重。

    每个信号日都剔除最近 ``horizon`` 天，使当日尚未实现的前瞻收益不会进入
    权重估计；权重在该日收盘形成，并与同日选股信号一起于下一交易日执行。
    """
    if mode not in {"ic", "icir", "ic_x_ir"}:
        raise ValueError(f"未知权重模式: {mode}")
    from qfm.pipeline.tests import compute_ic, forward_returns
    from qfm.portfolio.backtest import _rebalance_dates

    names = list(factors)
    dates = panel.close.index
    fwd = forward_returns(panel.close, horizon)
    ic_history = {name: compute_ic(fdf, fwd) for name, fdf in factors.items()}
    rebal_dates = _rebalance_dates(dates, rebalance)
    current = {name: 1 / len(names) for name in names}
    rows = []
    for i, dt in enumerate(dates):
        if dt in rebal_dates:
            # t+h 之后才能观察 t 的收益，故最多使用 i-horizon 的 IC。
            end = i - horizon
            if end >= min_obs:
                raw = {}
                for name, ic in ic_history.items():
                    hist = ic.iloc[max(0, end - lookback + 1): end + 1].dropna()
                    if len(hist) < min_obs:
                        raw[name] = 0.0
                    elif mode == "ic":
                        raw[name] = hist.mean()
                    elif mode == "icir":
                        sd = hist.std(ddof=1)
                        raw[name] = hist.mean() / sd if sd > 0 else 0.0
                    else:
                        sd = hist.std(ddof=1)
                        mean_ic = hist.mean()
                        raw[name] = mean_ic * (mean_ic / sd) if sd > 0 else 0.0
                current = _normalise_weights(raw, names)
        rows.append(current.copy())
    return pd.DataFrame(rows, index=dates, columns=names)


def synthesize(panel, names: list, mode: str = "equal", horizon: int = 20,
               orthogonalize: bool = False,
               ortho_controls: tuple | None = None,
               weight_lookback: int = 252,
               weight_rebalance: str = "ME",
               factor_versions: dict[str, int] | None = None) -> tuple[pd.DataFrame, dict]:
    """因子合成主入口

    mode: equal=等权 | ic=IC加权 | icir=IC_IR加权 | ic_x_ir=IC×IR加权
    orthogonalize: 逐日截面 OLS 正交化剥离暴露（行业/市值/风格/已有因子）
    ortho_controls: ("industry","size","style") 子集；None 时默认 ("size",)（仅市值，向后兼容）
    返回 (综合得分 date×stock, 权重 dict)
    """
    # 负向因子在这里统一取反；之后每个分数都表示“越高越应被持有”。
    factors = factor_panel(
        panel, names, align_direction=True, factor_versions=factor_versions
    )
    if mode == "equal":
        weights = {n: 1 / len(names) for n in names}
        weight_history = pd.DataFrame([weights] * len(panel.close.index), index=panel.close.index)
    elif mode == "ic":
        weight_history = walk_forward_weights(panel, factors, horizon=horizon, mode="ic",
                                              lookback=weight_lookback,
                                              rebalance=weight_rebalance)
        weights = weight_history.iloc[-1].to_dict()
    elif mode in {"icir", "ic_x_ir"}:
        weight_history = walk_forward_weights(panel, factors, horizon=horizon, mode=mode,
                                              lookback=weight_lookback,
                                              rebalance=weight_rebalance)
        weights = weight_history.iloc[-1].to_dict()
    else:
        raise ValueError(f"未知权重模式: {mode}")

    score = sum(factors[n].mul(weight_history[n], axis=0) for n in names)
    # DataFrame.attrs 会随切片/拼接向下传播。正交化过程会对单期信号做
    # nlargest 等运算；若此时 attrs 内嵌 DataFrame，Pandas 比较 attrs 时会
    # 得到一个布尔 DataFrame 而非布尔值。将展示用元数据延后附加到最终分数。
    score_metadata = {
        "weight_history": weight_history,
        "weight_mode": mode,
        "weight_lookback": weight_lookback,
    }
    if orthogonalize:
        from qfm.orthogonalize import orthogonalize_factor

        controls = ortho_controls or ("size",)
        score, _diag = orthogonalize_factor(panel, score, controls=controls, horizon=horizon)
    score.attrs.update(score_metadata)
    return score, weights
