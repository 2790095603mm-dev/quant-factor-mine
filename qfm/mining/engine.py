"""自动挖掘引擎：基础指标 × 窗口 × 变换 → 批量候选 → 批量检验 → TOP 排行榜"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from qfm.pipeline.tests import factor_report

WINDOWS = (5, 10, 20, 60)
TRANSFORMS = {
    "mean": lambda s, w: s.rolling(w).mean(),       # 窗口均值（水平）
    "std": lambda s, w: s.rolling(w).std(),         # 窗口波动（变异）
    "mom": lambda s, w: s.rolling(w).sum(),         # 窗口累计（对收益=动量，对量=累积量）
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
    """生成候选因子：基础指标 × 窗口 × 变换"""
    cands = {}
    for bname, bdf in base_indicators(panel).items():
        for w in windows:
            for tname, tfunc in TRANSFORMS.items():
                cands[f"{bname}_{tname}_{w}"] = tfunc(bdf, w)
    return cands


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
