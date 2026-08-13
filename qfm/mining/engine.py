"""自动挖掘引擎：基础指标 × 窗口 × 变换 → 批量候选 → 批量检验 → TOP 排行榜"""

from __future__ import annotations

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


def run_mining(panel, horizon: int = 20, max_candidates: int | None = None,
               progress=None) -> pd.DataFrame:
    """批量检验全部候选，返回按 |IC| 降序的排行榜 DataFrame"""
    cands = generate_candidates(panel)
    if max_candidates:
        cands = dict(list(cands.items())[:max_candidates])

    rows = []
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
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["|IC|"] = out["IC"].abs()
        out = out.sort_values("|IC|", ascending=False).drop(columns="|IC|").reset_index(drop=True)
    return out
