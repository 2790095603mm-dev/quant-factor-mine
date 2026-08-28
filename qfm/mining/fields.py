"""技术指标字段扩充（Qlib Alpha158 思路借鉴，独立实现，零外部依赖）

全部字段只用 t 及之前的数据（rolling/shift），结构性无未来函数。
字段缺失属性时跳过（如只有 close 的测试面板只产出 close 类字段）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def tech_fields(panel, fields: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """技术指标字段池：{字段名: date×stock DataFrame}"""
    out: dict[str, pd.DataFrame] = {}

    def _get(name: str) -> pd.DataFrame | None:
        try:
            return getattr(panel, name)
        except AttributeError:
            return None

    close, high, low, volume = _get("close"), _get("high"), _get("low"), _get("volume")

    # ── RSI（相对强弱，14 日）──────────────────────────────
    if close is not None:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        out["rsi_14"] = 100 - 100 / (1 + rs)
        out["rsi_14"] = out["rsi_14"].where(loss > 0, 100.0)

    # ── 乖离率 BIAS（收盘价偏离均线幅度）──────────────────
    if close is not None:
        for n in (5, 20):
            ma = close.rolling(n).mean()
            out[f"bias_{n}"] = (close - ma) / ma

    # ── 量比（当日量 / N 日均量）───────────────────────────
    if volume is not None:
        out["vol_ratio_5"] = volume / volume.rolling(5).mean()

    # ── ATR（真实波幅均值）────────────────────────────────
    if close is not None and high is not None and low is not None:
        pc = close.shift(1)
        tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
        out["atr_14"] = tr.rolling(14).mean() / close  # 归一化到价格

    # ── 振幅（当日 (high-low)/close）──────────────────────
    if close is not None and high is not None and low is not None:
        out["amp_ratio"] = (high - low) / close

    # ── 收盘位置（close 在 N 日高低区间的位置 0~1）────────
    if close is not None and high is not None and low is not None:
        lo = low.rolling(20).min()
        hi = high.rolling(20).max()
        out["close_pos_20"] = (close - lo) / (hi - lo)

    # ── 上涨/下跌量占比（10 日上涨日成交量 / 下跌日成交量）─
    if close is not None and volume is not None:
        up = close.diff().gt(0)
        up_vol = volume.where(up).rolling(10).sum()
        dn_vol = volume.where(~up).rolling(10).sum()
        out["updown_ratio_10"] = up_vol / dn_vol.replace(0, np.nan)

    # ── 资金流代理（收益 × 成交量，价格涨跌带量）──────────
    if close is not None and volume is not None:
        out["money_flow"] = close.diff() * volume

    # ── 20 日高低区间幅度 ─────────────────────────────────
    if high is not None and low is not None:
        out["hl_range_20"] = (high.rolling(20).max() - low.rolling(20).min()) / close

    if fields is not None:
        return {k: out[k] for k in fields if k in out}
    return out
