"""研究运行数据快照：样本、覆盖率和内容指纹的离线测试。"""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from qfm.research.snapshot import build_data_snapshot


def test_snapshot_is_deterministic_and_records_actual_sample(panel):
    first = build_data_snapshot(panel, "index800")
    second = build_data_snapshot(panel, "index800")

    assert first == second
    assert first["snapshot_version"] == 2
    assert first["stocks"] == panel.close.shape[1]
    assert first["stock_codes"] == sorted(panel.close.columns.tolist())
    assert first["coverage"]["close"] == 1.0
    assert first["universe_fingerprint"].startswith("sha256:")
    assert first["fingerprints"]["market"].startswith("sha256:")


def test_snapshot_fingerprints_change_only_for_relevant_data_family(panel):
    base = build_data_snapshot(panel, "index800")
    market_changed = deepcopy(panel)
    market_changed.close.iloc[0, 0] += 0.01
    market = build_data_snapshot(market_changed, "index800")
    fund_changed = deepcopy(panel)
    fund_changed.fund["roe"].iloc[0, 0] += 0.01
    fundamentals = build_data_snapshot(fund_changed, "index800")

    assert market["fingerprints"]["market"] != base["fingerprints"]["market"]
    assert market["fingerprints"]["fundamentals"] == base["fingerprints"]["fundamentals"]
    assert fundamentals["fingerprints"]["market"] == base["fingerprints"]["market"]
    assert fundamentals["fingerprints"]["fundamentals"] != base["fingerprints"]["fundamentals"]


def test_snapshot_reports_missing_optional_data_without_network(panel):
    sparse = deepcopy(panel)
    sparse.open.iloc[:, :] = np.nan
    sparse.amount.iloc[:, :] = np.nan
    sparse.industry = sparse.industry.iloc[:0, :0]
    sparse.fund = {}

    snapshot = build_data_snapshot(sparse, "index800")

    assert snapshot["coverage"]["open"] == 0.0
    assert snapshot["coverage"]["amount"] == 0.0
    assert snapshot["coverage"]["industry"] == 0.0
    assert snapshot["coverage"]["fundamentals"] == {}
    assert any("开盘价" in warning for warning in snapshot["quality_warnings"])
    assert any("财务" in warning for warning in snapshot["quality_warnings"])
