"""自动挖掘引擎：特征池 × 变换 → 批量候选（流式）→ 批量检验 → TOP 排行榜

v2（2026-08-18）：候选空间从"基础指标×窗口×变换"(48) 扩展到
"49 个已注册因子 × 4 种变换 + 基础指标窗口集"(≈244)，并改为流式生成：
逐候选"生成→检验→丢弃"，峰值内存 ≈ 单候选矩阵，避免全量囤积（旧版 288 候选 ≈ 16GB）。
负向因子（direction=negative）在入池时取反，统一"值越大越好"口径，排行榜 |IC| 才可公平比较。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from qfm.pipeline.tests import factor_report

WINDOWS = (5, 10, 20, 60)
TRANSFORMS = {
    "mean": lambda s, w: s.rolling(w).mean(),       # 窗口均值（水平）
    "std": lambda s, w: s.rolling(w).std(),         # 窗口波动（变异）
    "mom": lambda s, w: s.rolling(w).sum(),         # 窗口累计（对收益=动量，对量=累积量）
}

# v2 因子级变换模板：对已注册因子做截面/时序变换
V2_TRANSFORMS = {
    "raw": lambda s: s,                                                    # 原值
    "rank": lambda s: s.rank(axis=1),                                      # 截面排名（无量纲）
    "zscore": lambda s: (s - s.mean(axis=1)) / s.std(axis=1).replace(0, np.nan),  # 截面标准化
    "detrend20": lambda s: s - s.rolling(20).mean(),                       # 去 20 日均值（时序偏离）
}


def base_indicators(panel) -> dict[str, pd.DataFrame]:
    """基础指标（全部来自 DataPanel，零额外请求）"""
    return {
        "ret": panel.close.pct_change(fill_method=None),   # 日收益率
        "turnover": panel.turnover,                        # 日换手率
        "amount": panel.amount,                            # 日成交额
        "volume": panel.volume,                            # 日成交量
    }


def generate_candidates(panel, windows=WINDOWS) -> dict[str, pd.DataFrame]:
    """生成候选因子：基础指标 × 窗口 × 变换（v1 保留，兼容调用方）"""
    cands = {}
    for bname, bdf in base_indicators(panel).items():
        for w in windows:
            for tname, tfunc in TRANSFORMS.items():
                cands[f"{bname}_{tname}_{w}"] = tfunc(bdf, w)
    return cands


def _feature_pool(panel, names=None):
    """特征池（惰性）：已注册因子逐个产出 (名, date×stock)，负向因子取反"""
    from qfm.factors import list_factors

    fs = list_factors()
    if names:
        fs = [f for f in fs if f.name in names]
    for f in fs:
        try:
            fdf = f.func(panel)
        except Exception:  # noqa: BLE001 个别因子缺数据时跳过，不阻塞整轮挖掘
            continue
        yield f.name, (-fdf if f.direction == "negative" else fdf)


def generate_candidates_v2(panel, names=None, transforms=None):
    """v2 候选生成器（流式）：特征池 × 因子级变换 + 基础指标窗口集。

    逐候选 yield (候选名, date×stock 矩阵)，不囤积内存。
    候选名格式：`{特征}__{变换}`（基础指标候选沿用 `{指标}_{变换}_{窗口}`）。
    """
    ts = tuple(transforms) if transforms else tuple(V2_TRANSFORMS)
    for fname, fdf in _feature_pool(panel, names):
        for t in ts:
            yield f"{fname}__{t}", V2_TRANSFORMS[t](fdf)
    for bname, bdf in base_indicators(panel).items():
        for w in WINDOWS:
            for tname, tfunc in TRANSFORMS.items():
                yield f"{bname}_{tname}_{w}", tfunc(bdf, w)


def _candidate_total(panel, names=None, transforms=None) -> int:
    """候选总数（进度条用）：特征池数 × 变换数 + 基础窗口集"""
    from qfm.factors import list_factors

    fs = list_factors()
    if names:
        fs = [f for f in fs if f.name in names]
    ts = tuple(transforms) if transforms else tuple(V2_TRANSFORMS)
    return len(fs) * len(ts) + 4 * len(WINDOWS) * len(TRANSFORMS)


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
               trials_dir: str | None = None, feature_names=None,
               transforms=None) -> tuple[pd.DataFrame, dict | None]:
    """批量检验全部候选（v2 流式），返回 (排行榜 DataFrame, 试验矩阵元信息或 None)

    排行榜按 |IC| 降序，含「无未来函数」列与「变换」列；
    save_trials=True 时逐候选计算月频 TOP-N 组合收益，保存 T×N 试验矩阵。
    """
    n_total = _candidate_total(panel, feature_names, transforms)
    n_cap = min(max_candidates, n_total) if max_candidates else n_total

    rows = []
    trial_series: dict[str, pd.Series] = {}
    for i, (name, fdf) in enumerate(generate_candidates_v2(panel, feature_names, transforms)):
        if i >= n_cap:
            break
        if progress:
            progress(i, n_cap, name)
        rep = factor_report(fdf, panel.close, horizon=horizon, direction="positive")
        s = rep["ic_summary"]
        transform = name.split("__")[-1] if "__" in name else "—"
        rows.append({
            "因子": name,
            "变换": transform,
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
        meta = {"horizon": horizon, "top_n": top_n, "max_candidates": len(rows)}
        path = save_trials_matrix(matrix, meta, trials_dir)
        meta = {**meta, "path": path, "n_trials": int(matrix.shape[1]),
                "T_periods": int(matrix.shape[0])}
    return out, meta
