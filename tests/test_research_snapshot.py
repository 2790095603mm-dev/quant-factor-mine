"""研究运行数据快照：样本、覆盖率和内容指纹的离线测试。"""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from qfm.data.universe import CURRENT_SNAPSHOT_SOURCE
from qfm.research.snapshot import build_data_snapshot


def test_snapshot_is_deterministic_and_records_actual_sample(panel):
    first = build_data_snapshot(panel, "index800")
    second = build_data_snapshot(panel, "index800")

    assert first == second
    assert first["snapshot_version"] == 3
    assert first["stocks"] == panel.close.shape[1]
    assert first["stock_codes"] == sorted(panel.close.columns.tolist())
    assert first["coverage"]["close"] == 1.0
    assert first["universe_fingerprint"].startswith("sha256:")
    assert first["fingerprints"]["market"].startswith("sha256:")
    # 时点性来源必须可审计：默认即"当前成分股快照"，并给出偏差警告
    assert first["membership_source"] == CURRENT_SNAPSHOT_SOURCE
    assert any("历史成分股变动" in w for w in first["quality_warnings"])


def test_snapshot_records_effective_cross_section_and_warns_on_thin_history(panel):
    """早期截面样本远少于总池时必须告警（未上市股票不参与历史截面）。"""
    dense = build_data_snapshot(panel, "index800")
    assert dense["effective_cross_section"]["min"] == panel.close.shape[1]
    assert dense["effective_cross_section"]["max"] == panel.close.shape[1]
    assert not any("有效股票数" in w for w in dense["quality_warnings"])

    # 让一半以上股票只有最后 60 个交易日有数据，模拟"近年才入池"
    thinning = deepcopy(panel)
    thinning.close.iloc[:-60, :40] = np.nan
    sparse = build_data_snapshot(thinning, "index800")
    assert sparse["effective_cross_section"]["min"] == panel.close.shape[1] - 40
    assert sparse["effective_cross_section"]["first_date_count"] == panel.close.shape[1] - 40
    assert any("有效股票数" in w for w in sparse["quality_warnings"])


def test_snapshot_accepts_explicit_membership_source(panel):
    snapshot = build_data_snapshot(panel, "index800", membership_source="explicit-membership-file")

    assert snapshot["membership_source"] == "explicit-membership-file"
    assert not any("历史成分股变动" in w for w in snapshot["quality_warnings"])


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
