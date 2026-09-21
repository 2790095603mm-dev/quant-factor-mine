"""因子检验：IC / IC_IR / 分层回测 / 单调性 / 换手率（自研，不依赖 alphalens）

性能说明：compute_ic / layer_test 用 numpy 行级向量化（截面排名 + Pearson），
比逐日 pandas.corr / qcut 快 5-10 倍，全池 800 只 × 8600 日单因子 < 1s。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

MIN_STOCKS = 20  # 截面最少股票数，低于则跳过该日


def forward_returns(close: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    """未来 horizon 日收益率：r_{t→t+h} = close[t+h]/close[t] - 1

    注意：末 horizon 行必然为 NaN（未来价格不存在），由下游的 NaN 掩码剔除。
    """
    if not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError(f"前瞻天数必须为正整数，收到: {horizon!r}")
    return close.shift(-horizon) / close - 1


def align_to_factor(factor: pd.DataFrame, fwd_ret: pd.DataFrame) -> pd.DataFrame:
    """把未来收益按**标签**对齐到因子矩阵。

    compute_ic / layer_test 内部按位置逐行计算（numpy 向量化，800 只 × 8600 日 < 1s），
    一旦两个矩阵的日期或股票轴不一致就会静默错位——历史上只有 factor 被重索引，fwd 从不校验。
    统一在这里对齐：缺失的轴补 NaN（自然被掩码排除），多余的轴丢弃。
    """
    if factor.index.equals(fwd_ret.index) and factor.columns.equals(fwd_ret.columns):
        return fwd_ret
    return fwd_ret.reindex(index=factor.index, columns=factor.columns)


def _pearson_by_row(x: pd.DataFrame, y: pd.DataFrame) -> pd.Series:
    """逐行（逐日截面）Pearson 相关，numpy 向量化行循环"""
    xv, yv = x.values, y.values
    n = xv.shape[0]
    out = np.full(n, np.nan)
    for i in range(n):
        xi, yi = xv[i], yv[i]
        mask = ~(np.isnan(xi) | np.isnan(yi))
        if mask.sum() < MIN_STOCKS:
            continue
        xi, yi = xi[mask], yi[mask]
        xc = xi - xi.mean()
        yc = yi - yi.mean()
        denom = np.sqrt((xc * xc).sum() * (yc * yc).sum())
        if denom == 0:
            continue
        out[i] = (xc * yc).sum() / denom
    return pd.Series(out, index=x.index)


def compute_ic(factor: pd.DataFrame, fwd_ret: pd.DataFrame, method: str = "spearman") -> pd.Series:
    """每日截面因子值与未来收益的相关性（IC 序列，index=日期）

    spearman：全矩阵截面排名（axis=1，NaN 保留）后按行 Pearson
    pearson：直接按行 Pearson
    """
    if method not in ("spearman", "pearson"):
        raise ValueError(f"未知 IC 方法: {method}")
    fwd_ret = align_to_factor(factor, fwd_ret)
    if method == "spearman":
        return _pearson_by_row(factor.rank(axis=1), fwd_ret.rank(axis=1))
    return _pearson_by_row(factor, fwd_ret)


def ic_summary(ic: pd.Series) -> dict:
    """IC 序列汇总：均值 / IC_IR / t 值 / 正占比"""
    ic = ic.dropna()
    if len(ic) == 0:
        return {"ic_mean": np.nan, "ic_ir": np.nan, "ic_t": np.nan, "pos_ratio": np.nan, "n_days": 0}
    mu, sd = ic.mean(), ic.std(ddof=1)
    return {
        "ic_mean": mu,
        "ic_ir": mu / sd if sd > 0 else np.nan,
        "ic_t": mu / (sd / np.sqrt(len(ic))) if sd > 0 else np.nan,
        "pos_ratio": (ic > 0).mean(),
        "n_days": len(ic),
    }


def layer_test(factor: pd.DataFrame, fwd_ret: pd.DataFrame, n_layers: int = 5) -> pd.DataFrame:
    """分层回测：按因子值分 n 层（等频），统计各层未来收益均值

    返回 DataFrame：index=层号(1最低~n最高)，列=mean_ret / n_days

    这里是**预测统计口径**（各层未来收益均值，重叠窗口、不复利），不是可交易净值曲线；
    可交易的复利分层净值见 qfm.pipeline.tearsheet。
    """
    if n_layers < 2:
        raise ValueError("分层数至少为 2")
    fwd_ret = align_to_factor(factor, fwd_ret)
    fv, rv = factor.values, fwd_ret.values
    rows = []
    for i in range(factor.shape[0]):
        fi, ri = fv[i], rv[i]
        mask = ~(np.isnan(fi) | np.isnan(ri))
        if mask.sum() < MIN_STOCKS * n_layers:
            continue
        fi, ri = fi[mask], ri[mask]
        # 等频分层：ordinal 排名 → 均分 n_layers 桶（等价 qcut）
        ranks = rankdata(fi, method="ordinal") - 1
        layer = (ranks * n_layers // len(fi)).astype(int)
        means = np.bincount(layer, weights=ri) / np.bincount(layer)
        rows.append(means)
    if not rows:
        return pd.DataFrame(columns=["mean_ret", "n_days"])
    out = pd.DataFrame(rows, columns=range(1, n_layers + 1))
    return pd.DataFrame({"mean_ret": out.mean(), "n_days": out.count()}).rename_axis("layer")


def monotonicity(layer: pd.DataFrame, direction: str = "positive") -> dict:
    """分层单调性：正向因子期望 层收益随层号递增；负向因子期望递减"""
    if len(layer) < 2:
        return {"monotonic": False, "spread": np.nan, "top_minus_bottom": np.nan}
    vals = layer["mean_ret"].values
    top_minus_bottom = vals[-1] - vals[0]
    corr = pd.Series(vals).corr(pd.Series(range(1, len(vals) + 1)))
    expected = 1.0 if direction == "positive" else -1.0
    monotonic = (corr * expected) > 0.5
    return {
        "monotonic": monotonic,
        "spread": top_minus_bottom,
        "top_minus_bottom": top_minus_bottom,
        "corr": corr,
    }


def turnover_ratio(factor: pd.DataFrame, top_pct: float = 0.1, rebalance: int = 20) -> float:
    """因子组合换手率：每 rebalance 日取前 top_pct，计算名单重叠率（1 - 交集/并集）"""
    dates = factor.index[::rebalance]
    prev = None
    swaps = []
    for dt in dates:
        f = factor.loc[dt].dropna()
        if len(f) < MIN_STOCKS:
            continue
        n = max(1, int(len(f) * top_pct))
        cur = set(f.nlargest(n).index)
        if prev is not None and prev:
            swaps.append(1 - len(prev & cur) / len(prev | cur))
        prev = cur
    return float(np.mean(swaps)) if swaps else np.nan


def _report_coverage(signal_result, pipeline, oriented: pd.DataFrame) -> dict:
    """覆盖率来源优先级：本次跑出的管线留痕 > 调用方传入的管线留痕 > 只报信号覆盖率。

    第三步是诚实的降级：拿不到各阶段留痕时只报 signal，不伪造中间阶段的数值。
    """
    if signal_result is not None:
        return dict(signal_result.coverage)
    if pipeline is not None:
        return dict(pipeline.coverage)
    return {"signal": float(oriented.notna().to_numpy().mean())}


def factor_report(factor_df: pd.DataFrame, close: pd.DataFrame, horizon: int = 20,
                  direction: str = "positive", lookahead: dict | None = None,
                  preprocessed: bool = False, panel=None, config=None,
                  pipeline=None) -> dict:
    """单因子完整检验：清洗 → 方向统一 → IC → 分层 → 换手 → 汇总。

    报告口径统一为“调整后因子越高越好”。这让负向因子的 IC、分层与
    多空价差可直接和正向因子比较，不再要求读者自行取反。

    信号处理统一走 qfm.pipeline.run_pipeline（缺失处理 → 去极值 → 中性化 →
    标准化 → 方向 → Decay → Lag → Signal）：
    - preprocessed=False（默认）：按 config（默认等价 clean_factor）在管内处理因子；
    - preprocessed=True：调用方已跑过管线（如模拟页传入的就是 SignalResult.signal），
      此处不再重复截尾，否则 IC 口径会与持仓排序不一致。
    """
    from qfm.pipeline.pipeline import PipelineConfig, run_pipeline

    if preprocessed:
        # 已由调用方处理过：只做轴对齐，不再变换
        if not factor_df.index.equals(close.index) or not factor_df.columns.equals(close.columns):
            factor_df = factor_df.reindex(index=close.index, columns=close.columns)
        signal_result = None
        oriented = factor_df.astype(float)
        raw_cleaned = oriented
        direction_sign = 1
    else:
        signal_result = run_pipeline(
            factor_df, panel=panel, config=config or PipelineConfig(),
            direction=direction, close=close,
        )
        oriented = signal_result.signal
        raw_cleaned = signal_result.stages["standardize"]
        direction_sign = signal_result.direction_sign

    fwd = forward_returns(close, horizon)

    ic = compute_ic(oriented, fwd)
    summary = ic_summary(ic)
    layer = layer_test(oriented, fwd, n_layers=5)
    mono = monotonicity(layer, "positive")
    turn = turnover_ratio(oriented, top_pct=0.1, rebalance=horizon)

    # 滚动 IC 衰减监控：120 日均线 + 近期 IC（近 60 日）与全期对比
    ic_rolling = ic.rolling(120, min_periods=30).mean() if len(ic) else pd.Series(dtype=float)
    ic_recent = ic.tail(60).mean() if len(ic) >= 30 else np.nan
    ic_decay = (ic_recent - summary["ic_mean"]) if pd.notna(ic_recent) else np.nan

    # 分年 IC
    ic_by_year = ic.groupby(ic.index.year).mean() if len(ic) else pd.Series(dtype=float)

    report = {
        "ic_series": ic,
        "ic_summary": summary,
        "layer": layer,
        "monotonicity": mono,
        "turnover": turn,
        "ic_by_year": ic_by_year,
        "ic_rolling": ic_rolling,
        "ic_recent": ic_recent,
        "ic_decay": ic_decay,
        "cleaned": oriented,
        "raw_cleaned": raw_cleaned,
        "direction_adjusted": direction_sign < 0,
        "fwd": fwd,
        "horizon": horizon,
        # 统一管线的完整留痕（阶段中间帧、各阶段覆盖率、中性化诊断）
        "pipeline": signal_result,
        "coverage": _report_coverage(signal_result, pipeline, oriented),
        **({"lookahead": lookahead} if lookahead is not None else {}),
    }
    return report
