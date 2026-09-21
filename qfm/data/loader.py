"""数据层：akshare 拉取 + parquet 本地缓存

- 日线行情：新浪 qfq + 不复权双拉（东财接口在本环境被拦截，新浪已验证可用）
- 复权口径：OHLC 存前复权价，另存精确复权因子 `factor = close_qfq / close_raw`；
  volume / amount / outstanding_share / turnover 两个接口返回值完全一致，均为真实现值。
  日内 OHLC 共享同一 factor，因此真实价一律由 `价 / factor` 还原（见 panel.raw_price）。
- 财务指标：东财业绩报表（季度全市场，按公告日对齐防未来函数）
- 缓存：data_cache/ 下按股票缓存 parquet，二次加载秒级
"""

from __future__ import annotations

import json
import os
import time
from datetime import date, datetime

import akshare as ak
import pandas as pd

SINA_COLS = {
    "date": "date",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "amount": "amount",
    "outstanding_share": "outstanding_share",
    "turnover": "turnover",
}

# 缓存 parquet 的必需列；缺 factor 视为旧版缓存，需重拉
CACHE_COLUMNS = [*SINA_COLS.values(), "factor"]

# 业绩报表列 → 标准字段
YJBB_COLS = {
    "股票代码": "stock",
    "每股收益": "eps",
    "每股净资产": "bvps",
    "净资产收益率": "roe",
    "销售毛利率": "gross_margin",
    "每股经营现金流量": "ocfps",
    "营业总收入-同比增长": "rev_yoy",
    "净利润-同比增长": "profit_yoy",
    "营业总收入-季度环比增长": "rev_qoq",
    "净利润-季度环比增长": "profit_qoq",
    "所处行业": "industry",
    "最新公告日期": "ann_date",
}


def _sina_symbol(code: str) -> str:
    """新浪代码格式：sh600519 / sz000001"""
    if code.startswith(("6", "9")):
        return f"sh{code}"
    return f"sz{code}"


def quarter_ends(start: str = "20200101", end: str | None = None) -> list[str]:
    """季度末日期列表 YYYYMMDD，从 start 到最近完整季度"""
    end_dt = date.today() if end is None else datetime.strptime(end, "%Y%m%d").date()
    start_dt = datetime.strptime(start, "%Y%m%d").date()
    qs = []
    y, m = start_dt.year, ((start_dt.month - 1) // 3) * 3 + 1
    while True:
        qm = {1: "0331", 4: "0630", 7: "0930", 10: "1231"}[m]
        d = datetime(y, int(qm[0:2]), int(qm[2:4])).date()
        if d > end_dt:
            break
        if d >= start_dt:
            qs.append(d.strftime("%Y%m%d"))
        y, m = (y + 1, 1) if m == 10 else (y, m + 3)
    return qs


class DataLoader:
    def __init__(self, cache_dir: str = "data_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.bars_dir = os.path.join(cache_dir, "bars")
        os.makedirs(self.bars_dir, exist_ok=True)

    # ---------- 日线行情 ----------
    def load_bars(self, codes: list, progress=None, refresh: bool = False) -> pd.DataFrame:
        """拉取/读取多只股票日线，返回长表

        [date, stock, open, high, low, close, volume, amount, outstanding_share, turnover, factor]

        refresh=False（默认）命中缓存即返回；refresh=True 重新拉取并与缓存合并
        （按日期去重、保留本次结果），用于增量更新到最新交易日。
        旧版缓存（缺 factor 列）无论 refresh 取值都会重拉，以完成口径迁移。
        """
        frames = []
        n = len(codes)
        for i, code in enumerate(codes):
            if progress:
                progress(i, n, code)
            path = os.path.join(self.bars_dir, f"{code}.parquet")
            cached = None if refresh else self._read_cache(path)
            if cached is not None:
                df = cached
            else:
                fetched = self._fetch_one(code)
                if fetched is not None and len(fetched):
                    df = self._merge_cache(path, fetched)
                    df.to_parquet(path)
                else:
                    df = pd.DataFrame(columns=CACHE_COLUMNS)
            if df.empty:
                continue
            frames.append(df.assign(stock=code))
            if (i + 1) % 50 == 0:
                time.sleep(0.2)  # 温和限速，避免被源站限流
        if not frames:
            return pd.DataFrame(columns=[*CACHE_COLUMNS, "stock"])
        out = pd.concat(frames, ignore_index=True)
        out["date"] = pd.to_datetime(out["date"])
        return out.sort_values(["date", "stock"]).reset_index(drop=True)

    @staticmethod
    def _read_cache(path: str) -> pd.DataFrame | None:
        """读取缓存；缺 factor 列的旧版缓存返回 None（触发重拉迁移）。"""
        if not os.path.exists(path):
            return None
        try:
            df = pd.read_parquet(path)
        except (OSError, ValueError):
            return None
        return df if "factor" in df.columns else None

    @staticmethod
    def _merge_cache(path: str, fetched: pd.DataFrame) -> pd.DataFrame:
        """增量合并：已有缓存 + 本次拉取，按日期去重且以本次结果为准。"""
        cached = None
        if os.path.exists(path):
            try:
                old = pd.read_parquet(path)
                if "factor" in old.columns:
                    cached = old
            except (OSError, ValueError):
                cached = None
        if cached is None or cached.empty:
            return fetched
        merged = pd.concat([cached, fetched], ignore_index=True)
        merged["date"] = pd.to_datetime(merged["date"])
        return (
            merged.drop_duplicates(subset=["date"], keep="last")
            .sort_values("date")
            .reset_index(drop=True)
        )

    def _fetch_one(self, code: str, retries: int = 3) -> pd.DataFrame | None:
        """单只股票日线：前复权 + 不复权双拉，算精确复权因子

        factor = 前复权收盘价 / 真实收盘价。日内 OHLC 共享同一因子，
        故真实价 = 前复权价 / factor。volume 等字段两接口一致，取前复权侧即可。
        失败重试，最终失败返回 None。
        """
        symbol = _sina_symbol(code)
        for attempt in range(retries):
            try:
                qfq = ak.stock_zh_a_daily(symbol=symbol, adjust="qfq")
                if qfq is None or qfq.empty:
                    return None
                raw = ak.stock_zh_a_daily(symbol=symbol, adjust="")
                if raw is None or raw.empty:
                    return None
                qfq = qfq.rename(columns=SINA_COLS)[list(SINA_COLS.values())]
                raw = raw[["date", "close"]].rename(columns={"close": "close_raw"})
                df = qfq.merge(raw, on="date", how="left")
                raw_close = df["close_raw"].replace(0, pd.NA)
                df["factor"] = (df["close"] / raw_close).astype(float)
                df = df.drop(columns=["close_raw"])
                return df[CACHE_COLUMNS]
            except Exception as e:  # noqa: BLE001
                if attempt == retries - 1:
                    print(f"[loader] {code} 拉取失败: {e}")
                    return None
                time.sleep(1 + attempt)

    # ---------- 财务指标（业绩报表） ----------
    def load_indicators(self, progress=None) -> pd.DataFrame:
        """季度业绩报表全量，返回长表 [ann_date, report_date, stock, roe, gross_margin, bvps, eps, rev_yoy, profit_yoy, ocfps]"""
        cache = os.path.join(self.cache_dir, "indicators.parquet")
        if os.path.exists(cache):
            return pd.read_parquet(cache)

        qs = quarter_ends()
        frames = []
        for i, q in enumerate(qs):
            if progress:
                progress(i, len(qs), q)
            try:
                df = ak.stock_yjbb_em(date=q)
                if df is None or df.empty:
                    continue
                keep = [c for c in YJBB_COLS if c in df.columns]
                df = df[keep].rename(columns=YJBB_COLS)
                df["report_date"] = pd.to_datetime(q)
                df["ann_date"] = pd.to_datetime(df["ann_date"])
                frames.append(df)
                time.sleep(0.3)
            except Exception as e:  # noqa: BLE001
                print(f"[loader] 业绩报表 {q} 拉取失败: {e}")
        if not frames:
            raise RuntimeError("业绩报表全部拉取失败，请检查网络")
        out = pd.concat(frames, ignore_index=True)
        out = out.dropna(subset=["stock"])
        out["stock"] = out["stock"].astype(str).str.zfill(6)
        out.to_parquet(cache)
        return out

    # ---------- 数据状态 ----------
    def status(self) -> dict:
        bars = [f for f in os.listdir(self.bars_dir) if f.endswith(".parquet")]
        legacy = 0
        try:
            from pyarrow.parquet import ParquetFile
        except ImportError:  # pandas 也可使用 fastparquet；保留兼容回退
            ParquetFile = None
        for name in bars:
            path = os.path.join(self.bars_dir, name)
            try:
                # 这里只校验 schema，不把 799 份多年行情读进内存。
                cols = ParquetFile(path).schema.names if ParquetFile else pd.read_parquet(path).columns
            except (OSError, ValueError):
                legacy += 1
                continue
            if "factor" not in cols:
                legacy += 1
        ind = os.path.exists(os.path.join(self.cache_dir, "indicators.parquet"))
        return {
            "stocks_cached": len(bars),
            "indicators_cached": ind,
            # 旧版缓存（无复权因子）数量；>0 时下次加载会自动重拉完成口径迁移
            "stocks_without_factor": legacy,
        }

    def clear_bars(self):
        for f in os.listdir(self.bars_dir):
            os.remove(os.path.join(self.bars_dir, f))


def _load_indicators_raw() -> pd.DataFrame:
    """供 universe.full 使用：不经过 DataLoader 实例"""
    loader = DataLoader()
    return loader.load_indicators()
