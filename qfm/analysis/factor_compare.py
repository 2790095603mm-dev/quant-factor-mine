"""Factor Compare：统一指标、Pearson/Spearman 截面相关性与重复因子识别。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from qfm.factors import compute_factor, get_factor
from qfm.pipeline.pipeline import PipelineConfig, run_pipeline
from qfm.pipeline.tests import compute_ic, factor_report, forward_returns, ic_summary


@dataclass(frozen=True)
class FactorCompareResult:
    metrics: pd.DataFrame
    pearson_corr: pd.DataFrame
    spearman_corr: pd.DataFrame
    high_correlations: pd.DataFrame
    signals: dict[str, pd.DataFrame]
    ic_series: dict[str, pd.Series]
    rank_ic_series: dict[str, pd.Series]


def _pipeline_config(value: PipelineConfig | Mapping[str, Any] | None) -> PipelineConfig:
    if value is None:
        return PipelineConfig()
    if isinstance(value, PipelineConfig):
        return value
    settings = dict(value)
    if isinstance(settings.get("neutralize"), list):
        settings["neutralize"] = tuple(settings["neutralize"])
    return PipelineConfig(**settings)


def _mean_or_nan(series: pd.Series) -> float:
    cleaned = series.replace([np.inf, -np.inf], np.nan).dropna()
    return float(cleaned.mean()) if not cleaned.empty else float("nan")


def cross_sectional_corr(
    signals: Mapping[str, pd.DataFrame], method: str = "pearson"
) -> pd.DataFrame:
    """计算因子两两逐日截面相关系数，再对有效日期取平均。"""
    if method not in {"pearson", "spearman"}:
        raise ValueError(f"未知相关系数方法: {method}")
    names = list(signals)
    if not names:
        return pd.DataFrame(dtype=float)
    if any(not isinstance(signals[name], pd.DataFrame) for name in names):
        raise TypeError("signals 的值必须是 DataFrame")

    common_index = signals[names[0]].index
    common_columns = signals[names[0]].columns
    for name in names[1:]:
        common_index = common_index.intersection(signals[name].index)
        common_columns = common_columns.intersection(signals[name].columns)
    aligned = {
        name: signals[name].reindex(index=common_index, columns=common_columns)
        for name in names
    }
    if method == "spearman":
        aligned = {name: frame.rank(axis=1) for name, frame in aligned.items()}

    matrix = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)
    for left_index, left_name in enumerate(names):
        matrix.loc[left_name, left_name] = 1.0
        left = aligned[left_name]
        for right_name in names[left_index + 1:]:
            right = aligned[right_name]
            daily = left.corrwith(right, axis=1, method="pearson")
            valid_counts = (left.notna() & right.notna()).sum(axis=1)
            value = _mean_or_nan(daily.where(valid_counts >= 3))
            matrix.loc[left_name, right_name] = value
            matrix.loc[right_name, left_name] = value
    return matrix


def high_correlation_pairs(matrix: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    """列出非对角线中绝对相关系数严格大于阈值的因子对。"""
    if not 0 <= threshold <= 1:
        raise ValueError("相关性阈值必须在 0 到 1 之间")
    rows: list[dict[str, Any]] = []
    names = list(matrix.index)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            if right not in matrix.columns:
                continue
            value = matrix.loc[left, right]
            if pd.notna(value) and abs(float(value)) > threshold:
                rows.append({
                    "factor_a": left,
                    "factor_b": right,
                    "correlation": float(value),
                    "absolute_correlation": abs(float(value)),
                })
    columns = ["factor_a", "factor_b", "correlation", "absolute_correlation"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(
        "absolute_correlation", ascending=False, ignore_index=True
    )


def compare_factors(
    panel,
    names: list[str] | tuple[str, ...],
    *,
    horizon: int = 20,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    pipeline_config: PipelineConfig | Mapping[str, Any] | None = None,
    correlation_threshold: float = 0.7,
    precomputed_signals: Mapping[str, pd.DataFrame] | None = None,
) -> FactorCompareResult:
    """在同一数据、区间和处理管线下比较多个 Factor Library 因子。"""
    selected = list(dict.fromkeys(str(name) for name in names))
    if len(selected) < 2:
        raise ValueError("至少选择 2 个不同因子进行比较")
    if len(selected) > 12:
        raise ValueError("一次最多比较 12 个因子")
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("前瞻天数必须为正整数")

    config = _pipeline_config(pipeline_config)
    close = panel.close.loc[start:end]
    if close.empty:
        raise ValueError("所选日期区间没有行情数据")
    future_returns = forward_returns(close, horizon)
    signals: dict[str, pd.DataFrame] = {}
    ic_series: dict[str, pd.Series] = {}
    rank_ic_series: dict[str, pd.Series] = {}
    rows: list[dict[str, Any]] = []

    for name in selected:
        definition = get_factor(name)
        if definition is None:
            raise KeyError(f"未知因子: {name}")
        pipeline = None
        if precomputed_signals is not None:
            if name not in precomputed_signals:
                raise KeyError(f"预计算信号缺少因子: {name}")
            signal = precomputed_signals[name].reindex(
                index=close.index, columns=close.columns
            )
        else:
            raw = compute_factor(name, panel)
            pipeline = run_pipeline(raw, panel=panel, config=config, direction=definition.direction)
            signal = pipeline.signal.loc[start:end]
        signals[name] = signal

        pearson = compute_ic(signal, future_returns, method="pearson")
        rank = compute_ic(signal, future_returns, method="spearman")
        report = factor_report(
            signal,
            close,
            horizon=horizon,
            direction="positive",
            preprocessed=True,
            pipeline=pipeline,
        )
        rank_summary = ic_summary(rank)
        ic_series[name] = pearson
        rank_ic_series[name] = rank
        rows.append({
            "factor": name,
            "ic": _mean_or_nan(pearson),
            "rank_ic": _mean_or_nan(rank),
            "icir": float(rank_summary["ic_ir"]),
            "long_short_return": float(report["monotonicity"].get("top_minus_bottom", np.nan)),
            "turnover": float(report["turnover"]),
            "coverage": float(signal.notna().to_numpy().mean()),
            "stability": float((rank.dropna() > 0).mean()) if rank.notna().any() else np.nan,
        })

    pearson_corr = cross_sectional_corr(signals, "pearson")
    spearman_corr = cross_sectional_corr(signals, "spearman")
    duplicates = high_correlation_pairs(spearman_corr, correlation_threshold)
    return FactorCompareResult(
        metrics=pd.DataFrame(rows),
        pearson_corr=pearson_corr,
        spearman_corr=spearman_corr,
        high_correlations=duplicates,
        signals=signals,
        ic_series=ic_series,
        rank_ic_series=rank_ic_series,
    )
