"""研究运行的数据审计快照：实际样本、覆盖率与确定性内容指纹。"""

from __future__ import annotations

from hashlib import sha256
from typing import Any

import numpy as np
import pandas as pd

from qfm.data.panel import DataPanel


SNAPSHOT_VERSION = 2
MARKET_FIELDS = ("open", "high", "low", "close", "volume", "amount", "turnover", "mv_float")
MARKET_SOURCE = "akshare:sina-qfq"
FUNDAMENTAL_SOURCE = "akshare:eastmoney-yjbb"


def _normalise_code(value: object) -> str:
    """将常见 A 股数字代码稳定表示为六位字符串。"""
    code = str(value).strip()
    return code.zfill(6) if code.isdigit() and len(code) <= 6 else code


def _update_token(hasher: Any, value: object) -> None:
    """使用长度前缀写入哈希，避免不同字段拼接产生歧义。"""
    raw = str(value).encode("utf-8")
    hasher.update(len(raw).to_bytes(8, byteorder="big"))
    hasher.update(raw)


def _canonical_frame(
    raw: object,
    dates: pd.DatetimeIndex,
    codes: list[str],
) -> tuple[pd.DataFrame, bool]:
    """将可选面板对齐到收盘价网格，并显式标记来源字段是否存在。"""
    present = isinstance(raw, pd.DataFrame) and not raw.empty
    if not present:
        return pd.DataFrame(index=dates, columns=codes), False

    frame = raw.copy()
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    frame.columns = [_normalise_code(column) for column in frame.columns]
    if frame.index.has_duplicates:
        raise ValueError("无法构建数据快照：面板日期存在重复值")
    if frame.columns.has_duplicates:
        raise ValueError("无法构建数据快照：面板证券代码存在重复值")
    return frame.reindex(index=dates, columns=codes), True


def _coverage(frame: pd.DataFrame) -> float:
    cells = frame.shape[0] * frame.shape[1]
    return float(frame.notna().to_numpy().sum() / cells) if cells else 0.0


def _new_hasher(kind: str, dates: pd.DatetimeIndex, codes: list[str]) -> Any:
    hasher = sha256()
    _update_token(hasher, f"qfm-research-snapshot:{SNAPSHOT_VERSION}")
    _update_token(hasher, kind)
    _update_token(hasher, "dates")
    for date in dates:
        _update_token(hasher, pd.Timestamp(date).isoformat())
    _update_token(hasher, "codes")
    for code in codes:
        _update_token(hasher, code)
    return hasher


def _update_frame(hasher: Any, field: str, frame: pd.DataFrame, present: bool) -> None:
    _update_token(hasher, field)
    _update_token(hasher, "present" if present else "absent")
    if not present:
        return
    row_hashes = pd.util.hash_pandas_object(frame, index=True, categorize=True)
    values = row_hashes.to_numpy(dtype="<u8", copy=False)
    hasher.update(values.tobytes())


def _digest(hasher: Any) -> str:
    return f"sha256:{hasher.hexdigest()}"


def _quality_warnings(coverage: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    labels = {"close": "收盘价", "open": "开盘价", "amount": "成交额"}
    for key, label in labels.items():
        value = float(coverage[key])
        if value < 0.95:
            warnings.append(f"{label}覆盖率为 {value:.1%}，低于 95%；回测结果应结合缺失样本检查。")
    if float(coverage["industry"]) == 0.0:
        warnings.append("未记录可用行业归属；行业暴露审核与行业约束无法复现。")
    if not coverage["fundamentals"]:
        warnings.append("未记录可用财务字段；基本面因子的输入无法复核。")
    return warnings


def build_data_snapshot(panel: DataPanel, pool: str) -> dict[str, object]:
    """从当次回测面板构造可持久化、无需网络的审计快照。"""
    if panel.close.empty:
        raise ValueError("无法构建数据快照：收盘价面板为空")

    dates = pd.DatetimeIndex(pd.to_datetime(panel.close.index)).sort_values()
    codes = sorted(_normalise_code(column) for column in panel.close.columns)
    if len(set(codes)) != len(codes):
        raise ValueError("无法构建数据快照：收盘价证券代码存在重复值")

    market_frames: dict[str, tuple[pd.DataFrame, bool]] = {
        field: _canonical_frame(getattr(panel, field, None), dates, codes)
        for field in MARKET_FIELDS
    }
    fund_raw = panel.fund if isinstance(panel.fund, dict) else {}
    fund_names = sorted(str(name) for name in fund_raw)
    fund_frames = {
        name: _canonical_frame(fund_raw[name], dates, codes)
        for name in fund_names
    }
    industry, industry_present = _canonical_frame(getattr(panel, "industry", None), dates, codes)

    coverage: dict[str, Any] = {
        field: _coverage(frame)
        for field, (frame, _present) in market_frames.items()
    }
    coverage["industry"] = _coverage(industry)
    coverage["fundamentals"] = {
        name: _coverage(frame)
        for name, (frame, _present) in fund_frames.items()
    }

    market_hasher = _new_hasher("market", dates, codes)
    for field in MARKET_FIELDS:
        frame, present = market_frames[field]
        _update_frame(market_hasher, field, frame, present)

    fundamentals_hasher = _new_hasher("fundamentals", dates, codes)
    _update_frame(fundamentals_hasher, "industry", industry, industry_present)
    for name in fund_names:
        frame, present = fund_frames[name]
        _update_frame(fundamentals_hasher, f"fund:{name}", frame, present)

    universe_hasher = _new_hasher("universe", dates, codes)
    return {
        "snapshot_version": SNAPSHOT_VERSION,
        "pool": str(pool),
        "stocks": len(codes),
        "trading_days": len(dates),
        "data_start": dates.min().isoformat(),
        "data_end": dates.max().isoformat(),
        "stock_codes": codes,
        "universe_fingerprint": _digest(universe_hasher),
        "sources": {"market": MARKET_SOURCE, "fundamentals": FUNDAMENTAL_SOURCE},
        "coverage": coverage,
        "fingerprints": {
            "market": _digest(market_hasher),
            "fundamentals": _digest(fundamentals_hasher),
        },
        "quality_warnings": _quality_warnings(coverage),
    }
