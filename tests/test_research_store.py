"""本地研究账本的写入、读取与故障隔离。"""

from __future__ import annotations

import pandas as pd
import pytest

from qfm.research.store import ResearchStore


@pytest.fixture
def artifacts():
    index = pd.date_range("2026-01-01", periods=2, freq="D", name="date")
    nav = pd.Series([1.0, 1.02], index=index, name="nav")
    benchmark_nav = pd.Series([1.0, 1.01], index=index, name="benchmark_nav")
    weights = pd.DataFrame(
        {"weight": [0.5, 0.5]}, index=pd.Index(["ep_ttm", "roe"], name="factor")
    )
    yearly = pd.DataFrame({"年份": [2026], "收益": [0.02]})
    trades = pd.DataFrame({"date": index[:1], "stock": ["600000"], "side": ["BUY"]})
    return nav, benchmark_nav, weights, yearly, trades


def test_store_round_trips_project_and_run(tmp_path, artifacts):
    nav, benchmark_nav, weights, yearly, trades = artifacts
    store = ResearchStore(tmp_path / "research")
    project = store.create_project("质量组合", "成本后月度回测")

    run = store.save_run(
        project.id,
        "2026-09-05",
        {"top_n": 30},
        {"pool": "index800", "stocks": 799},
        {"夏普比率": 1.1},
        nav,
        benchmark_nav,
        weights,
        yearly,
        trades,
    )
    loaded = store.load_run(run.id)

    assert loaded.run.project_id == project.id
    pd.testing.assert_series_equal(loaded.nav, nav, check_freq=False)
    pd.testing.assert_series_equal(loaded.benchmark_nav, benchmark_nav, check_freq=False)
    pd.testing.assert_frame_equal(loaded.weights, weights)
    pd.testing.assert_frame_equal(loaded.yearly_performance, yearly)
    pd.testing.assert_frame_equal(loaded.trades, trades)


def test_store_lists_newest_run_first(tmp_path, artifacts):
    nav, benchmark_nav, weights, yearly, trades = artifacts
    store = ResearchStore(tmp_path / "research")
    project = store.create_project("测试")
    earlier = store.save_run(project.id, "较早", {}, {}, {}, nav, benchmark_nav, weights, yearly, trades)
    later = store.save_run(project.id, "较晚", {}, {}, {}, nav, benchmark_nav, weights, yearly, trades)

    runs, warnings = store.list_runs(project.id)

    assert warnings == []
    assert [run.id for run in runs] == [later.id, earlier.id]


def test_store_skips_corrupt_manifest_and_returns_warning(tmp_path):
    store = ResearchStore(tmp_path / "research")
    project = store.create_project("测试")
    broken = store.runs_dir / "run_broken"
    broken.mkdir(parents=True)
    (broken / "manifest.json").write_text("{broken", encoding="utf-8")

    runs, warnings = store.list_runs(project.id)

    assert runs == []
    assert any("run_broken" in warning for warning in warnings)


def test_failed_write_never_exposes_final_run_directory(tmp_path, monkeypatch, artifacts):
    nav, benchmark_nav, weights, yearly, trades = artifacts
    store = ResearchStore(tmp_path / "research")
    project = store.create_project("测试")

    def raise_os_error(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(store, "_write_csv", raise_os_error)

    with pytest.raises(OSError, match="disk full"):
        store.save_run(project.id, "失败", {}, {}, {}, nav, benchmark_nav, weights, yearly, trades)

    assert list(store.runs_dir.glob("run_*")) == []


def test_save_rejects_unknown_project(tmp_path, artifacts):
    nav, benchmark_nav, weights, yearly, trades = artifacts
    store = ResearchStore(tmp_path / "research")

    with pytest.raises(KeyError, match="不存在"):
        store.save_run("project_missing", "x", {}, {}, {}, nav, benchmark_nav, weights, yearly, trades)
