"""历史策略跨项目统一比较测试。"""

from __future__ import annotations

import pandas as pd
import pytest

from qfm.analysis.strategy_compare import compare_strategies
from qfm.research.store import ResearchStore


def _save(store, project, name, values, benchmark, turnover):
    index = pd.bdate_range("2025-01-02", periods=len(values), name="date")
    return store.save_run(
        project.id,
        name,
        {"factors": ["mom_20"]},
        {"pool": "index800"},
        {"年化换手": turnover},
        pd.Series(values, index=index, name="nav"),
        pd.Series(benchmark, index=index, name="benchmark_nav"),
        pd.DataFrame({"weight": [1.0]}, index=pd.Index(["mom_20"], name="factor")),
        pd.DataFrame({"年份": [2025], "收益": [values[-1] - 1]}),
        pd.DataFrame(columns=["date", "stock", "side"]),
    )


def test_strategy_compare_has_required_metrics_and_curves(tmp_path):
    store = ResearchStore(tmp_path / "research")
    project_a = store.create_project("项目 A")
    project_b = store.create_project("项目 B")
    periods = 80
    first = _save(
        store, project_a, "动量", [1 + index * 0.002 for index in range(periods)],
        [1 + index * 0.001 for index in range(periods)], 3.2,
    )
    second = _save(
        store, project_b, "价值", [1 + index * 0.0015 for index in range(periods)],
        [1 + index * 0.0008 for index in range(periods)], 2.4,
    )

    result = compare_strategies([store.load_run(first.id), store.load_run(second.id)])

    assert set(result.metrics.columns) >= {
        "strategy", "annual_return", "excess_return", "sharpe",
        "max_drawdown", "calmar", "turnover",
    }
    assert set(result.nav.columns) == set(result.drawdown.columns)
    assert set(result.excess.columns) == set(result.nav.columns)
    assert len(result.benchmark.columns) == 2


def test_strategy_names_are_disambiguated_by_run_id(tmp_path):
    store = ResearchStore(tmp_path / "research")
    project = store.create_project("项目")
    values = [1 + index * 0.001 for index in range(40)]
    benchmark = [1 + index * 0.0005 for index in range(40)]
    first = _save(store, project, "同名运行", values, benchmark, 1.0)
    second = _save(store, project, "同名运行", values, benchmark, 1.0)

    result = compare_strategies([store.load_run(first.id), store.load_run(second.id)])

    assert len(set(result.metrics["strategy"])) == 2


def test_compare_requires_two_completed_runs(tmp_path):
    store = ResearchStore(tmp_path / "research")
    project = store.create_project("项目")
    run = _save(store, project, "单个", [1 + i * 0.001 for i in range(40)],
                [1 + i * 0.0005 for i in range(40)], 1.0)

    with pytest.raises(ValueError, match="至少选择 2 个"):
        compare_strategies([store.load_run(run.id)])


def test_store_lists_runs_across_projects(tmp_path):
    store = ResearchStore(tmp_path / "research")
    first_project = store.create_project("A")
    second_project = store.create_project("B")
    values = [1 + i * 0.001 for i in range(40)]
    benchmark = [1 + i * 0.0005 for i in range(40)]
    _save(store, first_project, "A1", values, benchmark, 1.0)
    _save(store, second_project, "B1", values, benchmark, 1.0)

    pairs = store.list_all_runs()

    assert {(project.name, run.name) for project, run in pairs} == {("A", "A1"), ("B", "B1")}
