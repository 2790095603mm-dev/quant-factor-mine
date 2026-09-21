"""复权口径与股票池时点性的离线测试（零网络）。

背景：日线 OHLC 为前复权价，而 volume/amount/outstanding_share/turnover 是真实现值。
直接用「前复权价 × 流通股本」会得到错误市值（历史被系统性低估，且逐股低估幅度不同，
是横截面污染而非整体偏移）；用「真实每股指标 / 前复权价」算估值因子则分子分母量纲不一致。
本文件锁定修复后的口径。
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from qfm.data.loader import CACHE_COLUMNS, DataLoader
from qfm.data.panel import DataPanel, build_panel, raw_price, valuation_price
from qfm.data.universe import apply_membership, get_universe, load_membership
from qfm.factors import compute_factor

DATES = pd.bdate_range("2024-01-02", periods=6)
CODES = ("600000", "600001")


def _bars(factor: float | None = 1.0, close: float = 10.0, shares: float = 1e9) -> pd.DataFrame:
    """长表行情；factor=None 表示不产出 factor 列（旧版缓存/合成数据）。"""
    rows = []
    for code in CODES:
        for date in DATES:
            row = {
                "date": date, "stock": code,
                "open": close, "high": close * 1.1, "low": close * 0.9, "close": close,
                "volume": 1e6, "amount": 1e7,
                "outstanding_share": shares, "turnover": 0.001,
            }
            if factor is not None:
                row["factor"] = factor
            rows.append(row)
    return pd.DataFrame(rows)


def _indicators(bvps: float = 10.0, eps: float = 1.5) -> pd.DataFrame:
    """季度业绩报表：公告日早于所有交易日，确保 merge_asof 能对齐。"""
    ann = DATES[0] - pd.Timedelta(days=30)
    return pd.DataFrame(
        {
            "stock": list(CODES),
            "report_date": [ann] * len(CODES),
            "ann_date": [ann] * len(CODES),
            "bvps": [bvps] * len(CODES),
            "eps": [eps] * len(CODES),
        }
    )


# ---------- build_panel 口径 ----------

def test_market_cap_uses_raw_price_not_adjusted_price():
    """factor=0.5 表示前复权价只有真实价的一半：市值必须按真实价算（2 倍）。"""
    adjusted = build_panel(_bars(factor=0.5), _indicators())
    unadjusted = build_panel(_bars(factor=1.0), _indicators())

    assert adjusted.factor.iloc[0, 0] == pytest.approx(0.5)
    assert adjusted.close_raw.iloc[0, 0] == pytest.approx(20.0)
    assert adjusted.mv_float.iloc[0, 0] == pytest.approx(20.0 * 1e9)
    # 前复权价本身仍用于收益率类因子，必须保持原值
    assert adjusted.close.iloc[0, 0] == pytest.approx(10.0)
    # 同一交易日同一股票，真实市值是不复权口径的 2 倍
    assert adjusted.mv_float.iloc[0, 0] == pytest.approx(2 * unadjusted.mv_float.iloc[0, 0])


def test_missing_factor_column_defaults_to_one():
    """旧版缓存/合成数据没有 factor 列时视为 1.0，价格本身就是真实价。"""
    panel = build_panel(_bars(factor=None), _indicators())

    assert (panel.factor == 1.0).all().all()
    assert panel.close_raw.equals(panel.close)
    assert panel.mv_float.iloc[0, 0] == pytest.approx(10.0 * 1e9)


def test_non_positive_factor_becomes_nan_instead_of_poisoning_prices():
    bars = _bars(factor=1.0)
    bars.loc[0, "factor"] = 0.0

    panel = build_panel(bars, _indicators())

    assert np.isnan(panel.factor.iloc[0, 0])
    assert np.isnan(panel.close_raw.iloc[0, 0])
    assert np.isnan(panel.mv_float.iloc[0, 0])


def test_build_panel_tolerates_missing_fundamentals():
    """无财务表时不应崩溃（历史实现对空表会 NameError），且行情口径仍然正确。"""
    panel = build_panel(_bars(factor=0.5), pd.DataFrame())

    assert panel.fund == {}
    assert panel.industry.empty
    assert panel.close_raw.iloc[0, 0] == pytest.approx(20.0)


# ---------- raw_price / valuation_price ----------

def test_raw_price_reconstructs_unadjusted_ohlc():
    """日内 OHLC 共享同一复权因子，故任一价格都能用 价 / factor 还原。」"""
    panel = build_panel(_bars(factor=0.25), _indicators())

    assert raw_price(panel, "close").iloc[0, 0] == pytest.approx(40.0)
    assert raw_price(panel, "open").iloc[0, 0] == pytest.approx(40.0)
    assert raw_price(panel, "high").iloc[0, 0] == pytest.approx(44.0)
    assert raw_price(panel, "low").iloc[0, 0] == pytest.approx(36.0)


def test_raw_price_without_factor_returns_price_unchanged():
    panel = DataPanel(close=pd.DataFrame({"600000": [10.0]}))

    assert raw_price(panel, "close").equals(panel.close)


def test_valuation_price_falls_back_to_close_for_synthetic_panels():
    synthetic = DataPanel(close=pd.DataFrame({"600000": [10.0]}))
    assert valuation_price(synthetic).equals(synthetic.close)

    real = build_panel(_bars(factor=0.5), _indicators())
    assert valuation_price(real).iloc[0, 0] == pytest.approx(20.0)


# ---------- 估值因子改用真实价 ----------

def test_valuation_factors_use_raw_price():
    """bvps=10、真实价 20 → bp 应为 0.5；若误用前复权价(10) 会得到 1.0。"""
    panel = build_panel(_bars(factor=0.5, close=10.0), _indicators(bvps=10.0, eps=1.0))

    assert compute_factor("bp", panel).iloc[0, 0] == pytest.approx(0.5)
    assert compute_factor("ep", panel).iloc[0, 0] == pytest.approx(0.05)


def test_valuation_factors_unchanged_when_factor_is_one():
    panel = build_panel(_bars(factor=1.0, close=10.0), _indicators(bvps=10.0, eps=1.0))

    assert compute_factor("bp", panel).iloc[0, 0] == pytest.approx(1.0)
    assert compute_factor("ep", panel).iloc[0, 0] == pytest.approx(0.1)


# ---------- 缓存迁移与增量合并 ----------

def test_cache_without_factor_is_rejected_so_it_gets_refetched(tmp_path):
    """旧版缓存缺 factor 列，必须被判为失效，否则口径永远无法迁移。"""
    legacy = tmp_path / "600000.parquet"
    _bars(factor=None).drop(columns=["factor", "stock"], errors="ignore").to_parquet(legacy)

    assert DataLoader._read_cache(str(legacy)) is None


def test_cache_with_factor_is_honoured(tmp_path):
    fresh = tmp_path / "600000.parquet"
    single = _bars(factor=1.0)
    single[single["stock"] == "600000"].drop(columns=["stock"]).to_parquet(fresh)

    cached = DataLoader._read_cache(str(fresh))

    assert cached is not None
    assert "factor" in cached.columns
    assert len(cached) == len(DATES)


def test_merge_cache_appends_new_dates_and_keeps_latest_value(tmp_path):
    path = tmp_path / "600000.parquet"
    single = _bars(factor=1.0)
    cached = single[single["stock"] == "600000"].drop(columns=["stock"])
    cached.to_parquet(path)

    # 本次拉取：覆盖最后一天并追加一天
    fetched = cached.tail(2).copy()
    fetched["close"] = 99.0
    fetched["factor"] = 2.0
    extra = cached.tail(1).copy()
    extra["date"] = DATES[-1] + pd.Timedelta(days=1)
    fetched = pd.concat([fetched, extra], ignore_index=True)

    merged = DataLoader._merge_cache(str(path), fetched)

    assert len(merged) == len(DATES) + 1
    assert not merged["date"].duplicated().any()
    last_cached_date = merged[merged["date"] == DATES[-1]]
    assert last_cached_date["close"].iloc[0] == 99.0, "同日应保留本次拉取结果"
    assert merged["date"].is_monotonic_increasing


def test_loader_status_reports_legacy_cache_count(tmp_path):
    (tmp_path / "bars").mkdir()
    legacy = _bars(factor=None).drop(columns=["factor", "stock"], errors="ignore")
    legacy.to_parquet(tmp_path / "bars" / "600000.parquet")
    _bars(factor=1.0).drop(columns=["stock"]).to_parquet(tmp_path / "bars" / "600001.parquet")

    status = DataLoader(cache_dir=str(tmp_path)).status()

    assert status["stocks_cached"] == 2
    assert status["stocks_without_factor"] == 1


def test_load_bars_skips_failed_symbols_without_concat_warning(tmp_path, monkeypatch):
    """单只数据源失败只应降级跳过，不能污染其余缓存或反复产生 concat 警告。"""
    loader = DataLoader(cache_dir=str(tmp_path))
    good = _bars(factor=1.0)
    good[good["stock"] == "600000"].drop(columns=["stock"]).to_parquet(
        tmp_path / "bars" / "600000.parquet"
    )
    monkeypatch.setattr(loader, "_fetch_one", lambda _code: None)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        loaded = loader.load_bars(["600000", "600999"])

    assert not captured
    assert set(loaded["stock"]) == {"600000"}


def test_load_bars_returns_typed_empty_frame_when_every_symbol_fails(tmp_path, monkeypatch):
    loader = DataLoader(cache_dir=str(tmp_path))
    monkeypatch.setattr(loader, "_fetch_one", lambda _code: None)

    loaded = loader.load_bars(["600999"])

    assert loaded.empty
    assert list(loaded.columns) == [*CACHE_COLUMNS, "stock"]


def test_cache_columns_contract():
    assert "factor" in CACHE_COLUMNS
    assert set(CACHE_COLUMNS) >= {"date", "close", "outstanding_share", "turnover"}


# ---------- 股票池时点性 ----------

def test_get_universe_uses_fresh_local_cache_without_network(tmp_path, monkeypatch):
    """冷启动必须能读取股票池缓存；这是 Streamlit 数据页的首条真实路径。"""
    cache = tmp_path / "universe.json"
    cache.write_text(
        json.dumps({"ts": 2_000_000_000, "pools": {"index800": ["600000", "000001"]}}),
        encoding="utf-8",
    )
    monkeypatch.setattr("qfm.data.universe._CACHE_FILE", str(cache))
    monkeypatch.setattr("qfm.data.universe.time.time", lambda: 2_000_000_001)
    monkeypatch.setattr(
        "qfm.data.universe.get_index_cons",
        lambda _symbol: pytest.fail("命中新鲜缓存时不应访问网络"),
    )

    assert get_universe("index800") == ["600000", "000001"]


def test_absent_membership_file_is_reported_as_none(tmp_path, monkeypatch):
    monkeypatch.setattr("qfm.data.universe._DATA_DIR", str(tmp_path))

    assert load_membership("index800") is None


def test_load_membership_parses_windows_and_skips_malformed(tmp_path, monkeypatch):
    monkeypatch.setattr("qfm.data.universe._DATA_DIR", str(tmp_path))
    (tmp_path / "membership_index800.json").write_text(
        json.dumps(
            {
                "source": "csindex-cons-history",
                "members": {
                    "600000": ["2024-01-01", "2024-01-03"],
                    "600001": ["2024-01-04", None],
                    "600002": ["bad-window", "2024-01-05"],
                    "600003": "not-a-window",
                },
            }
        ),
        encoding="utf-8",
    )

    membership = load_membership("index800")

    assert membership is not None
    assert membership["source"] == "csindex-cons-history"
    assert set(membership["members"]) == {"600000", "600001"}
    start, end = membership["members"]["600001"]
    assert start == pd.Timestamp("2024-01-04") and end is None


def test_apply_membership_masks_values_outside_effective_window():
    panel = build_panel(_bars(factor=1.0), _indicators())
    membership = {
        "source": "explicit-membership-file",
        "members": {"600000": (DATES[1], DATES[3])},
    }

    masked = apply_membership(panel, membership)

    column = masked.close["600000"]
    assert column.isna().tolist() == [True, False, False, False, True, True]
    # 未出现在成员文件里的代码保持原样
    assert masked.close["600001"].notna().all()
    # 市值与真实价必须一起被屏蔽，否则会留下"区间外仍可交易"的漏洞
    assert masked.mv_float["600000"].isna().tolist() == [True, False, False, False, True, True]
    assert masked.close_raw["600000"].isna().tolist() == [True, False, False, False, True, True]


def test_apply_membership_is_noop_without_members():
    panel = build_panel(_bars(factor=1.0), _indicators())

    assert apply_membership(panel, {"members": {}}).close.equals(panel.close)
