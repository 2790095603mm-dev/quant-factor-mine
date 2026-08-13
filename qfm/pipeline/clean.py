"""因子预处理：截面去极值 + 标准化（逐日截面处理，因子矩阵 = date × stock）"""

from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize(df: pd.DataFrame, method: str = "mad", n: float = 3.0) -> pd.DataFrame:
    """逐日截面去极值

    method="mad": 中位数 ± n × 1.4826 × MAD（稳健，抗异常值）
    method="quantile": n 视为百分位裁剪比例（如 0.01 → 裁 1%/99%）
    """
    if method == "mad":
        med = df.median(axis=1)
        mad = (df.sub(med, axis=0)).abs().median(axis=1)
        lo = med - n * 1.4826 * mad
        hi = med + n * 1.4826 * mad
        return df.clip(lo, hi, axis=0)
    if method == "quantile":
        lo = df.quantile(n, axis=1)
        hi = df.quantile(1 - n, axis=1)
        return df.clip(lo, hi, axis=0)
    raise ValueError(f"未知去极值方法: {method}")


def zscore(df: pd.DataFrame) -> pd.DataFrame:
    """逐日截面 z-score 标准化（μ=0, σ=1）"""
    mu = df.mean(axis=1)
    sd = df.std(axis=1)
    return df.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)


def clean_factor(factor_df: pd.DataFrame, method: str = "mad", n: float = 3.0) -> pd.DataFrame:
    """完整预处理：去极值 → z-score"""
    return zscore(winsorize(factor_df, method=method, n=n))
