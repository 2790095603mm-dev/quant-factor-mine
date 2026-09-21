"""股票池：中证指数成分股 / 全市场（本地缓存 30 天，避免每次启动都依赖网络）

时点性说明（重要）：`get_universe` 取的是**当前**指数成分股快照，用今天的成分股回测
历史会引入幸存者偏差与成分股调整前视。价格数据本身能屏蔽"尚未上市"的股票
（未上市期间透视表为 NaN，IC/分层/选股都自然排除），但**无法**还原历史成分股变动。

因此本模块提供显式的成员区间机制：若 `data_cache/membership_<pool>.json` 存在，
则按其中每个标的的生效区间做掩码；否则退回当前快照，并由
`qfm.research.snapshot` 在运行记录里标注 `membership_source`，使偏差可审计。
"""

import json
import os
import time
from dataclasses import fields

import akshare as ak
import numpy as np
import pandas as pd

INDEX_POOLS = {
    "hs300": ("000300", "沪深300"),
    "zz500": ("000905", "中证500"),
    "zz1000": ("000852", "中证1000"),
}

POOL_ALIASES = {
    "cn_hs300": "hs300",
    "cn_zz500": "zz500",
    "cn_zz1000": "zz1000",
    "cn_all_a": "full",
}

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data_cache")
_CACHE_FILE = os.path.join(_DATA_DIR, "universe.json")
_CACHE_TTL = 30 * 24 * 3600  # 30 天

CURRENT_SNAPSHOT_SOURCE = "current-constituents-snapshot"
EXPLICIT_MEMBERSHIP_SOURCE = "explicit-membership-file"


def get_index_cons(symbol: str) -> list:
    """中证指数官网成分股（已验证列名：成分券代码）"""
    df = ak.index_stock_cons_csindex(symbol=symbol)
    codes = df["成分券代码"].astype(str).str.zfill(6).tolist()
    return codes


def membership_path(pool: str) -> str:
    """显式成员区间文件路径。"""
    return os.path.join(_DATA_DIR, f"membership_{pool}.json")


def load_membership(pool: str) -> dict | None:
    """读取显式成员区间文件，不存在或非法时返回 None。

    格式：
        {"source": "csindex-cons-history", "as_of": "2026-09-20",
         "members": {"600519": ["2001-08-27", "2026-09-18"], ...}}

    `end` 为 null 表示"截至最新"。该文件需要真实历史成分股数据，目前仓库内没有。
    """
    path = membership_path(pool)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    members = data.get("members")
    if not isinstance(members, dict) or not members:
        return None
    parsed: dict[str, tuple[pd.Timestamp, pd.Timestamp | None]] = {}
    for code, window in members.items():
        if not isinstance(window, (list, tuple)) or len(window) != 2:
            continue
        start, end = window
        if start is None:
            continue
        try:
            parsed[str(code).zfill(6)] = (
                pd.Timestamp(start),
                None if end is None else pd.Timestamp(end),
            )
        except (TypeError, ValueError):
            continue
    if not parsed:
        return None
    return {"source": str(data.get("source") or EXPLICIT_MEMBERSHIP_SOURCE), "members": parsed}


def apply_membership(panel, membership: dict):
    """把成员生效区间外的数据置为 NaN（返回新面板）。

    区间外 = 该标的当时不在股票池内，不应参与截面统计，也不应被买入。只处理成员文件里
    出现的代码；未出现的代码保持原样（保守，避免因文件不完整而误删样本）。
    """
    members = membership.get("members") or {}
    if not members:
        return panel
    dates = panel.close.index
    data = {}
    for field in fields(panel):
        value = getattr(panel, field.name)
        if field.name == "fund" and isinstance(value, dict):
            data[field.name] = {
                key: _mask_frame(frame, dates, members) for key, frame in value.items()
            }
        elif isinstance(value, pd.DataFrame):
            data[field.name] = _mask_frame(value, dates, members)
        else:
            data[field.name] = value
    return type(panel)(**data)


def _mask_frame(frame: pd.DataFrame, dates, members: dict) -> pd.DataFrame:
    """区间外置 NaN；不改动列集合，保持与闭市价面板同轴。"""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return frame
    out = frame.copy()
    for column in out.columns:
        window = members.get(str(column).zfill(6))
        if window is None:
            continue
        start, end = window
        outside = out.index < start
        if end is not None:
            outside |= out.index > end
        out.loc[outside, column] = np.nan
    return out


def _load_cache() -> dict:
    try:
        if os.path.exists(_CACHE_FILE):
            with open(_CACHE_FILE) as f:
                data = json.load(f)
            if time.time() - data.get("ts", 0) < _CACHE_TTL:
                return data.get("pools", {})
    except (OSError, ValueError):
        pass
    return {}


def _save_cache(pools: dict):
    try:
        os.makedirs(os.path.dirname(_CACHE_FILE), exist_ok=True)
        with open(_CACHE_FILE, "w") as f:
            json.dump({"ts": time.time(), "pools": pools}, f)
    except OSError:
        pass


def get_universe(pool: str = "index800") -> list:
    """解析内置、兼容别名与自定义股票池。"""
    if pool.startswith("custom_"):
        from qfm.data.catalog import DataCatalog

        return list(DataCatalog().get_universe(pool).symbols)

    cached = _load_cache()
    if pool in cached:
        return cached[pool]

    resolved = POOL_ALIASES.get(pool, pool)
    if resolved in INDEX_POOLS:
        symbol, _ = INDEX_POOLS[resolved]
        codes = get_index_cons(symbol)
    elif resolved == "index800":
        codes = []
        for key in ("hs300", "zz500"):
            sym, _ = INDEX_POOLS[key]
            codes += get_index_cons(sym)
        codes = sorted(set(codes))
    elif resolved == "full":
        # 全市场：用业绩报表里的全部股票代码（业绩报表是全市场季度数据）
        import qfm.data.loader as loader

        ind = loader._load_indicators_raw()
        codes = sorted(ind["stock"].unique().tolist())
    else:
        raise ValueError(f"未知股票池: {pool}")

    codes = sorted(set(str(code).zfill(6) for code in codes))
    _save_cache({**cached, pool: codes})
    return codes


def resolve_universe(pool: str, panel) -> tuple[list[str], str]:
    """返回 (实际生效的代码列表, 时点性来源标签)。

    有显式成员文件时按其成员过滤代码，并按区间掩码面板；否则返回当前快照的全部代码。
    调用方负责把来源标签记入运行快照，使"这次回测用的是哪种池"可审计。
    """
    codes = get_universe(pool)
    membership = load_membership(pool)
    if membership is None:
        return codes, CURRENT_SNAPSHOT_SOURCE
    members = membership["members"]
    resolved = [code for code in codes if str(code).zfill(6) in members]
    return (resolved or codes), str(membership["source"])
