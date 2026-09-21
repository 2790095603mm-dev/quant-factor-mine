"""实验系统测试：状态/标签/版本快照、重新打开、按存档参数复现。

核心诉求来自第 1 项「所有因子分析和回测都保存完整参数、结果和状态，
可以重新打开、比较和复现」。这里把"可复现"做成可执行断言。
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from qfm.factors import factor_definitions
from qfm.portfolio import PortfolioConstraints, run_backtest, synthesize
from qfm.research.models import (
    FORMAT_VERSION,
    STATUS_COMPLETED,
    STATUS_FAILED,
    ResearchRun,
)
from qfm.research.payload import CODE_VERSION, build_strategy_run_payload
from qfm.research.replay import (
    compare_run_configs,
    compare_run_metrics,
    factor_names_of,
    factor_version_drift,
    nav_difference,
    replay_saved_run,
)
from qfm.research.store import ResearchStore


@pytest.fixture
def store(tmp_path):
    return ResearchStore(tmp_path / "research")


@pytest.fixture
def project(store):
    return store.create_project("复现测试", "验证可复现性")


def _payload(panel, names=("ep_ttm", "mom_20"), mode="equal", top_n=10):
    """忠实复刻 app.py 的调用方式：synthesize 与 run_backtest 必须用同一套参数。

    这里刻意把 weight_lookback / weight_rebalance 显式传给 synthesize —— 它们既影响
    因子权重也写进 config，只有两边一致，"按存档重跑"才可能逐值复现。
    """
    score, weights = synthesize(panel, list(names), mode=mode, horizon=5,
                                weight_lookback=126, weight_rebalance="ME")
    bt = run_backtest(
        panel, score, top_n=top_n, start="2024-02-01", rebalance="ME",
        constraints=PortfolioConstraints(max_stock_weight=0.3),
    )
    payload = build_strategy_run_payload(
        panel=panel, pool="index800", names=list(names), mode=mode, horizon=5,
        weight_lookback=126, orthogonalize=False, ortho_controls=(), top_n=top_n,
        start_date="2024-02-01", rebalance="ME", bench_mode="equal",
        costs={"commission": 0.0003, "stamp": 0.0005, "impact": 0.001},
        max_participation=0.05, initial_capital=1_000_000.0,
        weights=weights, backtest=bt,
        pipeline_config={"neutralize": [], "standardize": "zscore"},
    )
    return payload, bt


def _save(store, project, payload, name="运行 A", tags=()):
    return store.save_run(
        project.id, name, payload["config"], payload["data_snapshot"], payload["summary"],
        payload["nav"], payload["benchmark_nav"], payload["weights"],
        payload["yearly_performance"], payload["trades"],
        payload.get("constraint_history"),
        tags=tags, code_version=payload["config"].get("code_version"),
        factor_definitions=payload.get("factor_definitions"),
    )


# ---------- 载荷与因子版本 ----------

def test_payload_records_factor_definitions_not_just_names(panel):
    payload, _ = _payload(panel)

    factors = payload["config"]["factors"]
    assert all(isinstance(item, dict) for item in factors), "必须存定义对象而非裸名字"
    assert {item["name"] for item in factors} == {"ep_ttm", "mom_20"}
    for item in factors:
        assert item["version"] >= 1
        assert item["source_hash"].startswith("sha256:")
        assert item["formula"]
    assert payload["config"]["code_version"] == CODE_VERSION


def test_payload_accepts_v1_style_name_list_via_helper(panel):
    """旧存档里 factors 是字符串列表，读取侧必须兼容。"""
    assert factor_names_of({"factors": ["bp", "roe"]}) == ["bp", "roe"]
    assert factor_names_of({"factors": [{"name": "bp", "version": 2}]}) == ["bp"]
    assert factor_names_of({}) == []


# ---------- 保存：状态 / 标签 / 版本快照 ----------

def test_saved_run_records_status_tags_and_factor_versions(store, project, panel):
    payload, _ = _payload(panel)

    run = _save(store, project, payload, tags=("基线", "已复现"))

    assert run.status == STATUS_COMPLETED
    assert run.factor_versions and {item["name"] for item in run.factor_versions} == {"ep_ttm", "mom_20"}
    manifest = json.loads((store.runs_dir / run.id / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == STATUS_COMPLETED
    assert manifest["tags"] == ["基线", "已复现"]
    assert manifest["format_version"] == FORMAT_VERSION
    assert manifest["finished_at"]
    assert manifest["code_version"] == CODE_VERSION
    assert manifest["factor_versions"][0]["version"] >= 1


def test_factor_definitions_artifact_is_written(store, project, panel):
    payload, _ = _payload(panel)

    run = _save(store, project, payload)

    assert "factor_definitions" in run.artifacts
    definitions = store.load_factor_definitions(run.id)
    assert {item["name"] for item in definitions} == {"ep_ttm", "mom_20"}
    # 与当前因子库的定义一致（本次运行就是用它算的）
    current = {item["name"]: item for item in factor_definitions(["ep_ttm", "mom_20"])}
    assert definitions[0]["source_hash"] == current[definitions[0]["name"]]["source_hash"]


def test_failed_run_is_persisted_with_reason(store, project, panel):
    """失败运行也要落盘，否则"这组参数跑不出来"这类信息会永久丢失。"""
    payload, _ = _payload(panel)

    run = store.save_failed_run(project.id, "失败尝试", payload["config"],
                                "前置数据不足：Top30 需要至少 30 只有效股票", tags=("失败",))

    assert run.status == STATUS_FAILED
    assert "前置数据不足" in run.error
    assert (store.runs_dir / run.id / "error.txt").read_text(encoding="utf-8").strip() == run.error
    runs, warnings = store.list_runs(project.id)
    assert warnings == []
    assert any(item.id == run.id and item.status == STATUS_FAILED for item in runs)


def test_delete_run_removes_artifacts(store, project, panel):
    payload, _ = _payload(panel)
    run = _save(store, project, payload)

    store.delete_run(run.id)

    assert not (store.runs_dir / run.id).exists()
    with pytest.raises(KeyError):
        store.load_run(run.id)


def test_delete_unknown_run_raises(store):
    with pytest.raises(KeyError):
        store.delete_run("run_missing")


# ---------- 向后兼容 ----------

def test_v1_manifest_is_still_readable(store, project, panel):
    """旧格式记录（因子名字符串、无状态/标签）必须仍可打开。"""
    payload, _ = _payload(panel)
    run = _save(store, project, payload)
    manifest_path = store.runs_dir / run.id / "manifest.json"
    legacy = json.loads(manifest_path.read_text(encoding="utf-8"))
    legacy["format_version"] = 1
    legacy["config"]["factors"] = ["ep_ttm", "mom_20"]
    for key in ("status", "tags", "code_version", "factor_versions", "finished_at", "error"):
        legacy.pop(key, None)
    manifest_path.write_text(json.dumps(legacy, ensure_ascii=False), encoding="utf-8")

    restored = store.load_run(run.id).run

    assert restored.format_version == 1
    assert restored.status == STATUS_COMPLETED
    assert restored.tags == ()
    assert restored.factor_versions == ()
    assert restored.factor_names() == ["ep_ttm", "mom_20"]


def test_unsupported_format_version_is_rejected():
    with pytest.raises(ValueError, match="不支持的研究记录格式版本"):
        ResearchRun.from_dict({
            "format_version": 99, "id": "run_x", "project_id": "project_x", "name": "x",
            "created_at": "2026-01-01T00:00:00+00:00", "config": {}, "data_snapshot": {},
            "summary": {}, "artifacts": {"nav": "nav.csv"},
        })


def test_v1_config_without_factor_versions_cannot_be_verified(store, project, panel):
    """旧记录没存因子版本，必须明确告警"无法确认定义是否一致"，而不是假装能复现。"""
    payload, _ = _payload(panel)
    run = _save(store, project, payload)

    warnings = factor_version_drift(run.factor_versions)

    assert warnings == []  # v2 记录应当干净
    assert any("未记录版本" in w for w in factor_version_drift([{"name": "ep_ttm"}]))
    assert any("已不在因子库" in w for w in factor_version_drift([{"name": "ghost", "version": 1}]))


# ---------- 重新打开与复现 ----------

def test_load_run_config_round_trips_every_parameter(store, project, panel):
    payload, _ = _payload(panel, names=("bp", "mom_60"), mode="ic", top_n=15)

    run = _save(store, project, payload)
    restored = store.load_run_config(run.id)

    assert restored["top_n"] == 15
    assert restored["weight_mode"] == "ic"
    assert restored["signal_horizon"] == 5
    assert restored["start_date"] == "2024-02-01"
    assert restored["portfolio_constraints"]["max_stock_weight"] == pytest.approx(0.3)
    assert restored["_run_id"] == run.id
    assert restored["_run_name"] == run.name
    assert {item["name"] for item in restored["factors"]} == {"bp", "mom_60"}


def test_replay_reproduces_the_saved_nav_exactly(store, project, panel):
    """核心不变量：按存档参数重跑，净值必须与存档逐值一致。"""
    payload, _ = _payload(panel, names=("ep_ttm", "mom_20"))
    run = _save(store, project, payload)
    loaded = store.load_run(run.id)

    replayed = replay_saved_run(panel, loaded.run.config, loaded.run.factor_versions)

    difference = nav_difference(replayed.backtest.nav, loaded.nav)
    assert difference["完全一致"], f"复现失败：{difference}"
    assert difference["最大绝对差"] < 1e-12
    assert replayed.warnings == []
    assert replayed.names == ["ep_ttm", "mom_20"]


def test_replay_matches_weights_and_turnover(store, project, panel):
    payload, _ = _payload(panel, mode="ic")
    run = _save(store, project, payload)
    loaded = store.load_run(run.id)

    replayed = replay_saved_run(panel, loaded.run.config, loaded.run.factor_versions)

    assert set(replayed.weights) == set(loaded.weights.index)
    for name, weight in replayed.weights.items():
        assert weight == pytest.approx(float(loaded.weights.loc[name, "weight"]), abs=1e-12)
    assert replayed.backtest.turnover == pytest.approx(loaded.run.summary["年化换手"], rel=1e-9)


def test_replay_without_data_panel_is_impossible(store, project, panel):
    """数据面板不可得时必须明确报错，而不是给出一个看似成功的空结果。"""
    from qfm.data.panel import DataPanel

    payload, _ = _payload(panel)
    run = _save(store, project, payload)

    with pytest.raises(ValueError, match="因子列表"):
        replay_saved_run(DataPanel(), {"factors": []})


def test_replay_reports_factor_version_drift(store, project, panel):
    """因子定义改版后，重跑必须告警（否则"复现不一致"会被误当成数据问题）。"""
    payload, _ = _payload(panel)
    run = _save(store, project, payload)

    replayed = replay_saved_run(
        panel, run.config,
        [{"name": "ep_ttm", "version": 99, "source_hash": "sha256:deadbeef"}],
    )

    assert any("已从 v99 更新" in w for w in replayed.warnings)


def test_replay_fills_defaults_for_edited_manifests_and_warns(store, project, panel):
    """手工编辑过的存档缺参数时，用默认值兜底但必须告警。"""
    payload, _ = _payload(panel)
    run = _save(store, project, payload)
    trimmed = dict(run.config)
    for key in ("rebalance", "top_n", "max_participation"):
        trimmed.pop(key, None)

    replayed = replay_saved_run(panel, trimmed)

    assert len(replayed.warnings) == 3
    assert any("调仓频率" in w for w in replayed.warnings)


def test_replay_rejects_unknown_factor(store, project, panel):
    payload, _ = _payload(panel)
    run = _save(store, project, payload)

    with pytest.raises(ValueError, match="已不存在的因子"):
        replay_saved_run(panel, {**run.config, "factors": [{"name": "ghost_factor"}]})


# ---------- 比较 ----------

def test_compare_run_configs_shows_only_differences(store, project, panel):
    first_payload, _ = _payload(panel, top_n=10)
    second_payload, _ = _payload(panel, top_n=25)
    first = _save(store, project, first_payload, name="N10")
    second = _save(store, project, second_payload, name="N25")

    table = compare_run_configs([first, second])

    differing = table.loc[table["存档键"] != "—", "存档键"].tolist()
    assert "top_n" in differing
    assert "costs" not in differing, "相同的参数不应出现在差异区"
    assert "top_n" in table["存档键"].tolist()
    assert set(table.columns) >= {"参数", "存档键", "N10", "N25"}
    assert table.loc[table["存档键"] == "top_n", "N10"].iloc[0] == "10"


def test_compare_run_configs_needs_two_runs(store, project, panel):
    payload, _ = _payload(panel)
    run = _save(store, project, payload)

    assert compare_run_configs([run]).empty


def test_compare_run_metrics_reports_status_and_factors(store, project, panel):
    payload, _ = _payload(panel)
    first = _save(store, project, payload, name="成功")
    second = store.save_failed_run(project.id, "失败", payload["config"], "boom")
    navs = [store.load_run(first.id).nav, pd.Series(dtype=float)]

    table = compare_run_metrics([first, second], navs)

    assert table["状态"].tolist() == [STATUS_COMPLETED, STATUS_FAILED]
    assert table["因子"].iloc[0] == "ep_ttm · mom_20"
    assert pd.isna(table["年化收益"].iloc[1])


def test_nav_difference_detects_mismatch(panel):
    left = pd.Series([1.0, 1.1, 1.2], index=pd.bdate_range("2024-01-02", periods=3))
    same = left.copy()
    off = left * 1.01

    assert nav_difference(left, same)["完全一致"] is True
    result = nav_difference(left, off)
    assert result["完全一致"] is False
    assert result["最大绝对差"] > 0


def test_nav_difference_handles_disjoint_series(panel):
    left = pd.Series([1.0], index=pd.bdate_range("2024-01-02", periods=1))
    right = pd.Series([1.0], index=pd.bdate_range("2025-01-02", periods=1))

    result = nav_difference(left, right)

    assert result["重叠天数"] == 0
