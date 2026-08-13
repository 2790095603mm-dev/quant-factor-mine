"""股票池：中证指数成分股 / 全市场（本地缓存 30 天，避免每次启动都依赖网络）"""

import json
import os
import time

import akshare as ak

INDEX_POOLS = {
    "hs300": ("000300", "沪深300"),
    "zz500": ("000905", "中证500"),
}

_CACHE_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data_cache", "universe.json")
_CACHE_TTL = 30 * 24 * 3600  # 30 天


def get_index_cons(symbol: str) -> list:
    """中证指数官网成分股（已验证列名：成分券代码）"""
    df = ak.index_stock_cons_csindex(symbol=symbol)
    codes = df["成分券代码"].astype(str).str.zfill(6).tolist()
    return codes


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
    """股票池：index800 = 沪深300+中证500；full = 全市场（从业绩报表代码集合推导）"""
    cached = _load_cache()
    if pool in cached:
        return cached[pool]

    if pool == "index800":
        codes = []
        for key in ("hs300", "zz500"):
            sym, _ = INDEX_POOLS[key]
            codes += get_index_cons(sym)
        codes = sorted(set(codes))
    elif pool == "full":
        # 全市场：用业绩报表里的全部股票代码（业绩报表是全市场季度数据）
        import qfm.data.loader as loader

        ind = loader._load_indicators_raw()
        codes = sorted(ind["stock"].unique().tolist())
    else:
        raise ValueError(f"未知股票池: {pool}")

    _save_cache({**cached, pool: codes})
    return codes
