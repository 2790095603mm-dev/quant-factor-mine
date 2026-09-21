"""研究运行与 Dataset / Universe 版本绑定测试。"""

from __future__ import annotations

import pandas as pd
import pytest

from qfm.data.catalog import REQUIRED_BINDING_FIELDS
from qfm.portfolio import BacktestResult
from qfm.research.models import ResearchRun
from qfm.research.payload import build_strategy_run_payload
from qfm.research.store import ResearchStore


def _payload(panel, tmp_path):
    index = panel.close.index[:3]
    nav = pd.Series([1.0, 1.01, 1.02], index=index)
    result = BacktestResult(
        nav=nav,
        bench_nav=pd.Series([1.0, 1.005, 1.01], index=index),
        holdings=pd.DataFrame(0.0, index=index, columns=panel.close.columns),
        trades=pd.DataFrame(columns=["date", "stock", "side"]),
        turnover=0.0,
        cost_pct=0.0,
        cost_total=0.0,
        cash_weight=pd.Series(1.0, index=index),
        params={"execution": "next_open", "constraints": {}},
    )
    return build_strategy_run_payload(
        panel=panel,
        pool="cn_hs300",
        names=["ep_ttm"],
        mode="equal",
        horizon=20,
        weight_lookback=252,
        orthogonalize=False,
        ortho_controls=(),
        top_n=30,
        start_date="2021-01-01",
        rebalance="ME",
        bench_mode="equal",
        costs={"commission": 0.0003, "stamp": 0.0005, "impact": 0.001},
        max_participation=0.05,
        initial_capital=1_000_000,
        weights={"ep_ttm": 1.0},
        backtest=result,
        catalog_root=tmp_path / "catalog",
    )


def test_new_payload_binds_exact_dataset_and_universe_versions(panel, tmp_path):
    payload = _payload(panel, tmp_path)

    assert all(payload["data_snapshot"].get(key) for key in REQUIRED_BINDING_FIELDS)
    assert payload["data_snapshot"]["universe_id"] == "cn_hs300"
    assert not ResearchRun.create(
        "project_x", "bound", {}, payload["data_snapshot"], {}, {"nav": "nav.csv"}
    ).legacy_unbound


def test_old_run_without_binding_remains_readable_and_is_marked_legacy():
    run = ResearchRun.create(
        "project_x", "legacy", {}, {"pool": "index800"}, {}, {"nav": "nav.csv"}
    )

    restored = ResearchRun.from_dict({**run.to_dict(), "format_version": 2})

    assert restored.legacy_unbound
    assert restored.dataset_binding() is None


def test_store_can_require_binding_for_user_facing_saves(panel, tmp_path):
    payload = _payload(panel, tmp_path)
    store = ResearchStore(tmp_path / "research")
    project = store.create_project("绑定测试")

    with pytest.raises(ValueError, match="dataset_version"):
        store.save_run(
            project.id,
            "unbound",
            payload["config"],
            {"pool": "cn_hs300"},
            payload["summary"],
            payload["nav"],
            payload["benchmark_nav"],
            payload["weights"],
            payload["yearly_performance"],
            payload["trades"],
            require_binding=True,
        )


def test_loaded_config_contains_version_binding(panel, tmp_path):
    payload = _payload(panel, tmp_path)
    store = ResearchStore(tmp_path / "research")
    project = store.create_project("绑定测试")
    run = store.save_run(
        project.id,
        "bound",
        payload["config"],
        payload["data_snapshot"],
        payload["summary"],
        payload["nav"],
        payload["benchmark_nav"],
        payload["weights"],
        payload["yearly_performance"],
        payload["trades"],
        require_binding=True,
    )

    restored = store.load_run_config(run.id)

    assert restored["_data_snapshot"]["dataset_version"] == payload["data_snapshot"]["dataset_version"]
    assert restored["_data_snapshot"]["universe_version"] == payload["data_snapshot"]["universe_version"]
