"""按存档参数重放一次研究运行，并比对参数与结果。

第 1 项要求实验「可以重新打开、比较和复现」。这里把"复现"变成可执行的函数而不是界面技巧：

- `replay_saved_run` 用存档 config 重建因子合成与组合回测（不碰界面控件）；
- `factor_version_drift` 在因子定义已变化时明确告警 —— 这正是记录因子版本的意义；
- `compare_run_configs` / `compare_run_metrics` 给出参数与指标的并排差异表。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from qfm.factors import get_factor

# config 里不存在时的兜底默认值（兼容 v1 存档与手工编辑过的记录）
DEFAULTS: dict[str, Any] = {
    "weight_mode": "equal",
    "signal_horizon": 20,
    "weight_lookback": 252,
    "orthogonalize": False,
    "ortho_controls": (),
    "top_n": 30,
    "start_date": None,
    "rebalance": "ME",
    "benchmark": "equal",
    "costs": {"commission": 0.0003, "stamp": 0.0005, "impact": 0.001},
    "max_participation": 0.05,
    "initial_capital": 1_000_000.0,
    "portfolio_constraints": {},
}

CONFIG_LABELS = {
    "weight_mode": "权重模式",
    "signal_horizon": "前瞻天数",
    "weight_lookback": "估权回看窗口",
    "orthogonalize": "启用正交化",
    "ortho_controls": "剥离暴露",
    "top_n": "最多持仓数量",
    "start_date": "回测起点",
    "rebalance": "调仓频率",
    "benchmark": "基准",
    "max_participation": "参与率上限",
    "initial_capital": "初始资金",
}


@dataclass
class ReplayResult:
    """一次重放的结果与可信度提示。"""

    names: list[str]
    weights: dict[str, float]
    backtest: Any
    score: pd.DataFrame
    config: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


def factor_names_of(config: dict[str, Any]) -> list[str]:
    """兼容 v1（裸名字列表）与 v2（因子定义对象列表）。"""
    factors = config.get("factors")
    if not isinstance(factors, (list, tuple)):
        return []
    names: list[str] = []
    for item in factors:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict) and item.get("name"):
            names.append(str(item["name"]))
    return names


def factor_version_drift(factor_versions: Any) -> list[str]:
    """比对存档里的因子版本与当前注册表，给出漂移告警。

    这是"可复现"的关键一环：因子定义被改过以后，即使参数一模一样，结果也不该相同。
    """
    warnings: list[str] = []
    if not isinstance(factor_versions, (list, tuple)):
        return warnings
    for item in factor_versions:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        name = str(item["name"])
        saved_version = item.get("version")
        saved_hash = item.get("source_hash")
        current = get_factor(name)
        if current is None:
            warnings.append(f"因子 {name} 已不在因子库中，无法复现该运行。")
            continue
        if saved_version is None:
            warnings.append(f"因子 {name} 的存档未记录版本（旧格式），无法确认定义是否一致。")
            continue
        if current.version != saved_version:
            warnings.append(
                f"因子 {name} 已从 v{saved_version} 更新到 v{current.version}，"
                "即使参数相同，重跑结果也会不同。"
            )
        elif saved_hash and current.source_hash != saved_hash:
            warnings.append(f"因子 {name} 的定义指纹与存档不一致（同为 v{current.version}），请核对因子库。")
    return warnings


def replay_saved_run(panel, config: dict[str, Any], factor_versions: Any = None) -> ReplayResult:
    """按存档参数重建策略回测。

    不重新推导任何参数：所有取值都来自存档 config，缺失项用 DEFAULTS 兜底并记录告警。
    """
    from qfm.portfolio import PortfolioConstraints, run_backtest, synthesize

    names = factor_names_of(config)
    if not names:
        raise ValueError("存档参数缺少因子列表，无法重放")
    missing = [name for name in names if get_factor(name) is None]
    if missing:
        raise ValueError(f"存档引用了已不存在的因子: {missing}")

    warnings = factor_version_drift(factor_versions if factor_versions is not None else config.get("factor_versions"))
    merged: dict[str, Any] = {}
    for key, default in DEFAULTS.items():
        if key in config and config[key] is not None:
            merged[key] = config[key]
        else:
            merged[key] = default
            warnings.append(f"存档缺少「{CONFIG_LABELS.get(key, key)}」，已用默认值 {default!r} 重放。")

    constraints_raw = merged["portfolio_constraints"] or {}
    constraints = PortfolioConstraints(
        max_stock_weight=float(constraints_raw.get("max_stock_weight", 1.0)),
        max_industry_weight=float(constraints_raw.get("max_industry_weight", 1.0)),
        max_rebalance_turnover=constraints_raw.get("max_rebalance_turnover"),
    )
    ortho = bool(merged["orthogonalize"])
    weights_mode = str(merged["weight_mode"])
    rebalance = str(merged["rebalance"])
    score, weights = synthesize(
        panel, names, mode=weights_mode, horizon=int(merged["signal_horizon"]),
        orthogonalize=ortho,
        ortho_controls=tuple(merged["ortho_controls"]) if ortho else None,
        weight_lookback=int(merged["weight_lookback"]),
        weight_rebalance=rebalance,
    )
    backtest = run_backtest(
        panel, score,
        top_n=int(merged["top_n"]),
        start=merged["start_date"],
        rebalance=rebalance,
        bench_mode=str(merged["benchmark"]),
        initial_capital=float(merged["initial_capital"]),
        max_participation=float(merged["max_participation"]),
        cost=dict(merged["costs"]),
        constraints=constraints,
    )
    return ReplayResult(
        names=names, weights=dict(weights), backtest=backtest, score=score,
        config=dict(config), warnings=warnings,
    )


def _flatten(value: Any) -> str:
    if isinstance(value, dict):
        return ", ".join(f"{key}={_flatten(item)}" for key, item in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return " · ".join(str(item) for item in value)
    if isinstance(value, bool):
        return "是" if value else "否"
    if value is None:
        return "—"
    return str(value)


def compare_run_configs(runs: list[Any]) -> pd.DataFrame:
    """参数差异表：只列出各运行之间**不一致**的字段，一致项折叠为一行。

    直接并排全部参数会淹没差异，而复现出问题时首先要看的就是"到底哪里不一样"。
    """
    if len(runs) < 2:
        return pd.DataFrame()
    keys: list[str] = []
    for run in runs:
        for key in run.config:
            if key not in keys and key not in {"factors", "code_version", "factor_versions"}:
                keys.append(key)
    rows = []
    identical: list[str] = []
    for key in sorted(keys):
        values = [_flatten(run.config.get(key)) for run in runs]
        label = CONFIG_LABELS.get(key, key)
        if len(set(values)) == 1:
            identical.append(f"{label}={values[0]}")
            continue
        row = {"参数": label, "存档键": key}
        for run, value in zip(runs, values):
            row[run.name] = value
        rows.append(row)
    table = pd.DataFrame(rows)
    if identical:
        table = pd.concat(
            [table, pd.DataFrame([{"参数": "（所有运行一致）", "存档键": "—",
                                   **{run.name: "；".join(identical) for run in runs}}])],
            ignore_index=True,
        )
    return table


def compare_run_metrics(runs: list[Any], navs: list[pd.Series]) -> pd.DataFrame:
    """指标并排表 + 两两净值差异，用于确认"复现值是否与存档一致"。"""
    from qfm.portfolio import perf_stats

    rows = []
    for run, nav in zip(runs, navs):
        stats = perf_stats(nav)
        rows.append({
            "运行": run.name,
            "状态": run.status,
            "因子": " · ".join(run.factor_names()),
            "年化收益": stats.get("年化收益"),
            "夏普": stats.get("夏普比率"),
            "最大回撤": stats.get("最大回撤"),
            "卡玛比率": stats.get("卡玛比率"),
            "年化换手": run.summary.get("年化换手"),
            "累计成本": run.summary.get("累计交易成本"),
            "净值终值": float(nav.iloc[-1]) if len(nav) else np.nan,
        })
    return pd.DataFrame(rows)


def nav_difference(left: pd.Series, right: pd.Series) -> dict[str, float]:
    """两个净值序列的差异摘要（复现校验用）。"""
    aligned = pd.concat([left.rename("left"), right.rename("right")], axis=1, join="inner").dropna()
    if aligned.empty:
        return {"重叠天数": 0, "最大绝对差": float("nan"), "末值差": float("nan")}
    diff = (aligned["left"] - aligned["right"]).abs()
    return {
        "重叠天数": int(len(aligned)),
        "最大绝对差": float(diff.max()),
        "末值差": float(abs(aligned["left"].iloc[-1] - aligned["right"].iloc[-1])),
        "完全一致": bool(np.allclose(aligned["left"], aligned["right"], rtol=1e-12, atol=1e-15)),
    }
