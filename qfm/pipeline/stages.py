"""统一因子管线的各阶段实现（纯函数，单阶段可测）。

阶段顺序固定为：

    缺失处理 → 去极值 → 方向 → 中性化 → 标准化 → Decay 平滑 → Lag → Signal

每个函数只做一件事，输入输出都是 date×stock 矩阵（index=交易日, columns=股票），
不做任何隐式重索引——轴对齐由 pipeline.run_pipeline 统一负责。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 缺失处理方式
MISSING_METHODS = ("none", "row_drop", "cs_median")
# 去极值方式
WINSORIZE_METHODS = ("none", "mad", "quantile")
# 标准化方式
STANDARDIZE_METHODS = ("none", "zscore", "rank")


def coverage_of(frame: pd.DataFrame) -> float:
    """非空占比（整体）。"""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return float("nan")
    return float(frame.notna().to_numpy().mean())


def apply_missing(frame: pd.DataFrame, method: str = "none", min_stocks: int = 20) -> pd.DataFrame:
    """缺失处理。

    none      ：不处理（下游 IC/分层自带截面样本数门槛）
    row_drop  ：当日有效股票数不足 min_stocks 时整行置 NaN
    cs_median ：用当日截面中位数填充缺失（停牌股票会得到"中性"值，谨慎使用）
    """
    if method not in MISSING_METHODS:
        raise ValueError(f"未知缺失处理方式: {method}")
    if method == "none" or not isinstance(frame, pd.DataFrame) or frame.empty:
        return frame.copy()
    if method == "row_drop":
        counts = frame.notna().sum(axis=1)
        out = frame.copy()
        out.loc[counts < min_stocks, :] = np.nan
        return out
    return frame.apply(lambda row: row.fillna(row.median()), axis=1)


def apply_winsorize(frame: pd.DataFrame, method: str = "mad", n: float = 3.0) -> pd.DataFrame:
    """逐日截面去极值（复用 pipeline.clean.winsorize）。"""
    if method not in WINSORIZE_METHODS:
        raise ValueError(f"未知去极值方式: {method}")
    if method == "none" or not isinstance(frame, pd.DataFrame) or frame.empty:
        return frame.copy()
    from qfm.pipeline.clean import winsorize

    return winsorize(frame, method=method, n=n)


def apply_direction(frame: pd.DataFrame, direction: str = "positive") -> tuple[pd.DataFrame, int]:
    """统一方向：输出恒为"值越大预期收益越高"。返回 (方向统一后的因子, 符号)。"""
    if direction not in ("positive", "negative"):
        raise ValueError(f"未知因子方向: {direction}")
    sign = -1 if direction == "negative" else 1
    return (frame * sign if sign < 0 else frame.copy()), sign


def apply_neutralize(
    frame: pd.DataFrame,
    panel,
    controls: tuple[str, ...] = (),
    min_samples: int = 30,
) -> tuple[pd.DataFrame, dict]:
    """逐日截面 OLS 中性化，返回**原始残差**（标准化交给 apply_standardize）。

    复用 qfm.orthogonalize 的截面回归与控制变量构造，避免出现第二套 OLS 实现。
    """
    if not controls or not isinstance(frame, pd.DataFrame) or frame.empty:
        flat = coverage_of(frame)
        return frame.copy(), {
            "controls": [], "days": 0, "days_skipped": 0,
            "coverage_before": flat, "coverage_after": flat,
            "exposure_before": float("nan"), "exposure_after": float("nan"),
        }
    from qfm.orthogonalize import controls_from_panel, residualize_frame

    ctrl = controls_from_panel(panel, controls)
    if not ctrl:
        flat = coverage_of(frame)
        return frame.copy(), {
            "controls": [], "days": 0, "days_skipped": 0,
            "coverage_before": flat, "coverage_after": flat,
            "exposure_before": float("nan"), "exposure_after": float("nan"),
        }
    return residualize_frame(frame, ctrl, min_samples)


def apply_standardize(frame: pd.DataFrame, method: str = "zscore") -> pd.DataFrame:
    """逐日截面标准化。

    zscore：均值 0 / 标准差 1（复用 pipeline.clean.zscore）
    rank  ：截面百分位排名平移到 [-0.5, 0.5]（厚尾数据比 z-score 稳健）

    ⚠️ 与中性化的交互：中性化输出的是对控制变量**线性**正交的残差。z-score 是线性变换，
    保持这种正交性（残差与行业/市值的线性相关仍为 0）；rank 是**非线性单调**变换，会破坏
    线性正交性 —— 排名后的残差与市值仍单调相关，但 Pearson 相关不再为 0（实测可达 0.5+）。
    因此同时启用中性化时应保持 zscore；确需 rank 时，把 rank 放在中性化之前。
    """
    if method not in STANDARDIZE_METHODS:
        raise ValueError(f"未知标准化方式: {method}")
    if method == "none" or not isinstance(frame, pd.DataFrame) or frame.empty:
        return frame.copy()
    if method == "rank":
        return frame.rank(axis=1, pct=True) - 0.5
    from qfm.pipeline.clean import zscore

    return zscore(frame)


def apply_decay(frame: pd.DataFrame, decay: int = 0) -> pd.DataFrame:
    """线性衰减平滑：最近一天权重最高，N 日内权重 N..1。

    逐历史偏移求和，避免累积和遇缺失值时权重漂移；只有 N 天都非空才出值。
    """
    if decay <= 1 or not isinstance(frame, pd.DataFrame) or frame.empty:
        return frame.copy()
    total = frame * 0.0
    count = frame.notna().astype(int) * 0
    for lag in range(decay):
        values = frame.shift(lag)
        total += values.fillna(0) * (decay - lag)
        count += values.notna().astype(int)
    return (total / (decay * (decay + 1) / 2)).where(count == decay)


def apply_lag(frame: pd.DataFrame, lag: int = 0) -> pd.DataFrame:
    """额外延迟 lag 个交易日。

    注意：回测引擎本身已按 T 日收盘信号、T+1 开盘成交执行，因此 lag=0 即为常规设置；
    lag>0 表示再多等若干交易日。lag=1 与 lag=0 等价（引擎已提供一天延迟）。
    """
    if lag <= 1 or not isinstance(frame, pd.DataFrame) or frame.empty:
        return frame.copy()
    return frame.shift(lag - 1)


def finalize_signal(frame: pd.DataFrame, min_stocks: int = 0) -> pd.DataFrame:
    """Signal 阶段：可选地剔除截面样本过薄的交易日。

    min_stocks=0（默认）不掩码，与历史行为一致（IC/分层各自有 MIN_STOCKS 门槛）；
    设为正数时，有效股票数不足的交易日整行置 NaN，使"预测"与"可交易"口径一致。
    """
    if min_stocks <= 0 or not isinstance(frame, pd.DataFrame) or frame.empty:
        return frame.copy()
    counts = frame.notna().sum(axis=1)
    out = frame.copy()
    out.loc[counts < min_stocks, :] = np.nan
    return out
