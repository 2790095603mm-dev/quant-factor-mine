"""跨研究项目的历史 Backtest Experiment 对比。"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Iterable

import numpy as np
import pandas as pd

from qfm.portfolio.performance import drawdown as drawdown_series
from qfm.portfolio.performance import standard_metrics
from qfm.research.models import STATUS_COMPLETED, LoadedResearchRun


@dataclass(frozen=True)
class StrategyCompareResult:
    metrics: pd.DataFrame
    nav: pd.DataFrame
    benchmark: pd.DataFrame
    excess: pd.DataFrame
    drawdown: pd.DataFrame
    labels: dict[str, str]


def _normalise(series: pd.Series) -> pd.Series:
    cleaned = series.replace([np.inf, -np.inf], np.nan).dropna().sort_index().astype(float)
    if cleaned.empty or cleaned.iloc[0] == 0:
        return pd.Series(dtype=float)
    return cleaned / cleaned.iloc[0]


def _fingerprint(series: pd.Series) -> str:
    hashed = pd.util.hash_pandas_object(series, index=True).to_numpy().tobytes()
    return sha256(hashed).hexdigest()


def _label(loaded: LoadedResearchRun) -> str:
    return f"{loaded.run.name} · {loaded.run.id[-8:]}"


def compare_strategies(
    loaded_runs: Iterable[LoadedResearchRun],
) -> StrategyCompareResult:
    """统一比较 2–8 个已保存且成功的策略运行。"""
    selected = list(loaded_runs)
    if len(selected) < 2:
        raise ValueError("至少选择 2 个已完成策略")
    if len(selected) > 8:
        raise ValueError("一次最多比较 8 个策略")
    failed = [item.run.name for item in selected if item.run.status != STATUS_COMPLETED]
    if failed:
        raise ValueError(f"只能比较已完成策略: {failed}")

    metric_rows: list[dict[str, Any]] = []
    nav_columns: dict[str, pd.Series] = {}
    excess_columns: dict[str, pd.Series] = {}
    drawdown_columns: dict[str, pd.Series] = {}
    benchmark_columns: dict[str, pd.Series] = {}
    benchmark_fingerprints: dict[str, str] = {}
    labels: dict[str, str] = {}

    for loaded in selected:
        label = _label(loaded)
        labels[loaded.run.id] = label
        nav = _normalise(loaded.nav)
        benchmark = _normalise(loaded.benchmark_nav)
        if nav.empty:
            raise ValueError(f"策略净值为空或非法: {loaded.run.name}")
        nav_columns[label] = nav
        drawdown_columns[label] = drawdown_series(nav)

        aligned = pd.concat(
            [nav.rename("strategy"), benchmark.rename("benchmark")], axis=1, join="inner"
        ).dropna()
        excess_columns[label] = (
            aligned["strategy"] / aligned["benchmark"] - 1
            if not aligned.empty else pd.Series(dtype=float)
        )
        if not benchmark.empty:
            fingerprint = _fingerprint(benchmark)
            if fingerprint not in benchmark_fingerprints:
                bench_label = f"Benchmark · {label}"
                benchmark_fingerprints[fingerprint] = bench_label
                benchmark_columns[bench_label] = benchmark

        metrics = standard_metrics(
            loaded.nav,
            loaded.benchmark_nav,
            turnover=loaded.run.summary.get("年化换手"),
        )
        binding = loaded.run.dataset_binding() or {}
        metric_rows.append({
            "strategy": label,
            "run_id": loaded.run.id,
            "annual_return": metrics.get("年化收益", np.nan),
            "excess_return": metrics.get("年化超额", np.nan),
            "sharpe": metrics.get("夏普比率", np.nan),
            "max_drawdown": metrics.get("最大回撤", np.nan),
            "calmar": metrics.get("卡玛比率", np.nan),
            "turnover": metrics.get("年化换手", np.nan),
            "dataset_version": binding.get("dataset_version", "legacy_unbound"),
            "universe_id": binding.get("universe_id", "legacy_unbound"),
            "universe_version": binding.get("universe_version", "legacy_unbound"),
        })

    return StrategyCompareResult(
        metrics=pd.DataFrame(metric_rows),
        nav=pd.concat(nav_columns, axis=1),
        benchmark=pd.concat(benchmark_columns, axis=1) if benchmark_columns else pd.DataFrame(),
        excess=pd.concat(excess_columns, axis=1),
        drawdown=pd.concat(drawdown_columns, axis=1),
        labels=labels,
    )
