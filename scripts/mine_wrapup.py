"""挖掘战役收尾：排行榜 → TOP 因子正交化诊断 + 过拟合检验 → 中文挖掘报告

内存/性能优化方案（与挖掘引擎 v2 同思路）：
- 面板加载一次，逐候选流式重建，仅匹配 TOP-N（同源去重后）才进入深度检验
- 正交化用 numpy 批量逐日截面 OLS（≈100× 快于逐日 pandas 版，结果口径一致）
- 原始 IC 直接复用排行榜（不重算）；试验矩阵复用已落盘 parquet（不重算月频收益）
- 正交化后只算 IC（轻量），不做分层/换手

运行：cd 项目根目录 && /opt/anaconda3/bin/python3 scripts/mine_wrapup.py
"""

import os
# 逐日小矩阵 OLS 必须单线程 BLAS（多线程同步开销 >> 计算）；须在 numpy import 前设置
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import socket
import sys
import time
from pathlib import Path

ROOT = Path("/Users/zxt/.zcode/workspace/default/quant-factor-mine")
os.chdir(ROOT)  # 保证相对路径缓存命中（data_cache 等）
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from qfm.data import DataLoader, build_panel, get_universe
from qfm.mining.engine import V2_TRANSFORMS, TRANSFORMS, base_indicators, latest_trials
from qfm.orthogonalize import controls_from_panel
from qfm.overfit import overfit_report
from qfm.pipeline.tests import compute_ic, ic_summary, forward_returns
from qfm.factors import list_factors

socket.setdefaulttimeout(20)  # 兜底：任何网络请求 20s 超时，绝不永久挂起

TOPN = 20          # 深度检验的 TOP 因子数
HORIZON = 20
LEADERBOARD = ROOT / "data_cache" / "mine_leaderboard_20260818.csv"

t0 = time.time()
out_lines: list[str] = []


def say(line: str):
    print(line, flush=True)
    out_lines.append(line)


def _winsor_zs(y: np.ndarray) -> np.ndarray:
    """逐日 MAD 截尾 + z-score（与 qfm.orthogonalize.winsorize_zscore 口径一致）"""
    med = float(np.nanmedian(y))
    mad = float(np.nanmedian(np.abs(y - med)))
    if not np.isfinite(mad) or mad == 0:
        return np.full_like(y, np.nan)
    clipped = np.clip(y, med - 5 * 1.4826 * mad, med + 5 * 1.4826 * mad)
    std = float(np.nanstd(clipped))
    if not np.isfinite(std) or std == 0:
        return np.full_like(y, np.nan)
    return (clipped - float(np.nanmean(clipped))) / std


def ortho_fast(Y_s: np.ndarray, C_s: np.ndarray, cols_idx: list[int],
               min_samples: int = 30, ridge: float = 1e-8) -> np.ndarray:
    """numpy 批量截面 OLS 正交化（正规方程 + 小 Ridge，≈100× 快于逐日 pandas 版）。

    Y_s: 采样截面×stock 因子信号数组；C_s: 采样截面×stock×K 控制数组（**只构建一次**）；
    cols_idx: 本组使用的控制列索引。
    每个采样截面：y = [1 X]β + resid → 残差 MAD 截尾 z-score（对齐 qfm.orthogonalize）。
    采样在调用方完成（月频每 20 交易日 1 截面，435 个），内存降 20 倍、诊断同口径。
    """
    N, S = Y_s.shape
    Cg = C_s[:, :, cols_idx]
    k = Cg.shape[2]
    resid = np.full_like(Y_s, np.nan)
    for t in range(N):
        mask = np.isfinite(Y_s[t]) & np.isfinite(Cg[t]).all(axis=1)
        idx = np.where(mask)[0]
        n = len(idx)
        if n < max(min_samples, k + 3):
            continue
        yy = Y_s[t, idx]
        xx = np.empty((n, k + 1))
        xx[:, 0] = 1.0
        xx[:, 1:] = Cg[t, idx]
        xtx = xx.T @ xx
        xtx.flat[:: k + 2] += ridge  # 小 Ridge 防病态（行业共线已被 drop_first 处理）
        beta = np.linalg.solve(xtx, xx.T @ yy)
        resid[t, idx] = _winsor_zs(yy - xx @ beta)
    return resid


if __name__ == "__main__":


    say(f"# 挖掘战役报告（index800 全量 × 244 候选 × 前瞻 {HORIZON} 日）")
    say("")
    say(f"生成时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── 1. 加载面板（一次，供流式重建因子；只取有本地缓存的股票，保证离线）──
    say("\n## 1. 数据加载")
    dl = DataLoader(cache_dir=str(ROOT / "data_cache"))
    codes = get_universe("index800")
    cached_codes = [c for c in codes if (ROOT / "data_cache" / "bars" / f"{c}.parquet").exists()]
    say(f"- 股票池 {len(codes)} 只 → 本地缓存 {len(cached_codes)} 只（跳过无缓存 {len(codes)-len(cached_codes)} 只，离线运行）")
    bars = dl.load_bars(cached_codes)
    ind = dl.load_indicators()
    panel = build_panel(bars, ind)
    say(f"- 面板：{panel.close.shape[0]} 日 × {panel.close.shape[1]} 只（耗时 {time.time()-t0:.0f}s）")

    # ── 2. 读排行榜：同源候选去重（IC/IC_IR/正占比/换手率全同 = 同一数据），取 TOP-N ──
    lb = pd.read_csv(LEADERBOARD)
    lb_dedup = lb.drop_duplicates(subset=["IC", "IC_IR", "正占比", "换手率"])
    top = lb_dedup.head(TOPN)
    top_names = top["因子"].tolist()
    ic0_map = dict(zip(top["因子"], top["IC"]))
    say(f"\n## 2. 排行榜共 {len(lb)} 候选；同源去重后 {len(lb_dedup)}；深度检验 TOP {len(top_names)}")

    # ── 3. 流式重建 + 正交化诊断 + 过拟合检验 ────────────────────────
    say("\n## 3. 正交化诊断（原始 IC vs 剥离行业/市值/风格后 IC）")
    say("> 口径：正交化在月频截面（每 20 交易日 1 截面）做截面 OLS，剥离前后同口径比较。")
    say("")
    say("| 候选因子 | 原始 IC | 剥离行业 | +市值 | +风格 | IC保留 | 结论 |")
    say("|---|---|---|---|---|---|---|")

    controls = controls_from_panel(panel, ("industry", "size", "style"))
    # 逐步剥离的分组：全部行业 one-hot 为一组（~30 键），再叠加市值，最后叠加风格(beta/vol)
    ind_keys = [k for k in controls if k.startswith("industry_")]
    size_keys = [k for k in controls if k == "size"]
    style_keys = [k for k in controls if k in ("beta", "vol")]
    GROUPS = [ind_keys, ind_keys + size_keys, ind_keys + size_keys + style_keys]
    ctrl_dfs_all = {k: controls[k] for k in controls}
    ctrl_order = list(controls)  # 列名顺序（C_all 构建后 controls 即释放，此处留序）
    STEP = 20  # 月频采样：每 20 交易日取 1 截面（435 个），内存降 20 倍
    C_all = np.stack([c.to_numpy(dtype=float)[::STEP] for c in ctrl_dfs_all.values()],
                     axis=2)  # 采样截面×S×K 控制数组，只构建一次，3 组剥离按列切片复用
    del ctrl_dfs_all, controls  # 释放控制面板（~1.7GB），只保留 C_all
    import gc as _gc
    _gc.collect()
    fwd = forward_returns(panel.close, HORIZON)
    sample_idx = panel.close.index[::STEP]  # 采样日期（IC 口径对齐）

    trials_matrix, manifest = latest_trials()
    n_trials_all = len(lb)  # 战役实际尝试的候选数（DSR/PBO 的试验次数口径）

    rows_ortho: list[dict] = []
    rows_overfit: dict[str, dict] = {}
    matched = 0
    _factor_map = {f.name: f for f in list_factors()}


    def _rebuild_candidate(name: str, panel):
        """按候选名直取重建因子矩阵（只算目标，不遍历全候选池）"""
        if "__" in name:                       # {特征}__{变换}：已注册因子 + 因子级变换
            base, tf = name.rsplit("__", 1)
            return V2_TRANSFORMS[tf](_factor_map[base].func(panel))
        base, tf, w = name.rsplit("_", 2)      # {指标}_{变换}_{窗口}：基础指标窗口集
        return TRANSFORMS[tf](base_indicators(panel)[base], int(w))


    for name in top_names:
        fdf = _rebuild_candidate(name, panel)
        matched += 1
        ic0 = ic0_map[name]  # 原始 IC 直接复用排行榜，不重算

        # 逐组剥离（行业 → +市值 → +风格）：采样截面数组组间复用，每步只算轻量 IC
        t_fac = time.time()
        Y_s = fdf.to_numpy(dtype=float)[::STEP]
        ic_steps = [ic0]
        for group in GROUPS:
            cols_idx = [ctrl_order.index(k) for k in group]
            resid = ortho_fast(Y_s, C_all, cols_idx)
            rdf = pd.DataFrame(resid, index=sample_idx, columns=fdf.columns)
            ic_steps.append(ic_summary(
                compute_ic(rdf, fwd.loc[rdf.index]).dropna())["ic_mean"])
        say(f"  · {name}: 3 组剥离+IC 耗时 {time.time()-t_fac:.1f}s")
        decay = (ic_steps[-1] / ic0) if abs(ic0) > 1e-9 else float("nan")
        verdict = "真信号" if decay > 0.7 else ("半真" if decay > 0.4 else "暴露为主")
        rows_ortho.append({"候选因子": name, "原始IC": ic0, "剥离行业": ic_steps[1],
                           "+市值": ic_steps[2], "+风格": ic_steps[3], "IC保留": decay,
                           "结论": verdict})
        say(f"| {name} | {ic0:.4f} | {ic_steps[1]:.4f} | {ic_steps[2]:.4f} | {ic_steps[3]:.4f} "
            f"| {decay*100:.0f}% | {verdict} |")

        # 过拟合检验：收益序列直接取试验矩阵（战役已算好），n_trials=全候选数
        if trials_matrix is not None and name in trials_matrix.columns:
            rets = trials_matrix[name].dropna()
            rep_o = overfit_report(rets, n_trials=n_trials_all, trials_matrix=trials_matrix)
            rows_overfit[name] = {
                "DSR": rep_o.dsr, "PBO": rep_o.pbo,
                "年化SR": rep_o.sr_annual, "年化SR修正": rep_o.sr_annual_adjusted,
                "MinTRL(年)": rep_o.minimum_track_record_length, "判定": rep_o.verdict,
            }

    say(f"\n匹配到 {matched}/{len(top_names)} 个 TOP 候选（试验矩阵可用: {trials_matrix is not None}）")

    # ── 4. 过拟合检验表 ──────────────────────────────────────────────
    say("\n## 4. 过拟合检验（DSR / PBO / Haircut，n_trials=全 244 候选）")
    say("")
    say("| 候选因子 | DSR | PBO | 年化SR | 修正后SR | MinTRL(年) | 判定 |")
    say("|---|---|---|---|---|---|---|")
    for o in rows_ortho:
        f = rows_overfit.get(o["候选因子"])
        if f is None:
            continue
        say(f"| {o['候选因子']} | {f['DSR']:.3f} | {f['PBO']:.3f} | {f['年化SR']:.2f} | {f['年化SR修正']:.2f} | {f['MinTRL(年)']:.1f} | {f['判定']} |")

    # ── 5. 入库建议 ──────────────────────────────────────────────────
    say("\n## 5. 入库建议")
    say("")
    say("**建议入库（正交化后 IC 保留 >70% 且过拟合检验 PASS）**：")
    say("")
    for o in rows_ortho:
        f = rows_overfit.get(o["候选因子"])
        if o["结论"] == "真信号" and f and f["判定"] == "PASS":
            say(f"- **{o['候选因子']}**：原始 IC {o['原始IC']:.4f} → 全剥离后 {o['+风格']:.4f}")
        elif o["结论"] == "真信号" and (f is None or f["判定"] != "PASS"):
            say(f"- {o['候选因子']}：IC 稳健但过拟合检验未 PASS（{f['判定'] if f else '无试验数据'}），可观察")

    say("\n**不建议（暴露为主）**：")
    say("")
    for o in rows_ortho:
        if o["结论"] != "真信号":
            keep = (o["+风格"] / o["原始IC"] * 100) if abs(o["原始IC"]) > 1e-9 else float("nan")
            say(f"- {o['候选因子']}（IC 保留 {keep:.0f}%，{o['结论']}）")

    # ── 写盘 ─────────────────────────────────────────────────────────
    out = "\n".join(out_lines)
    md_path = ROOT / "reports" / "mine_report_20260818.md"
    md_path.parent.mkdir(exist_ok=True)
    md_path.write_text(out, encoding="utf-8")
    say(f"\n📄 报告已保存: {md_path}")
    say(f"总耗时 {time.time()-t0:.0f}s")
