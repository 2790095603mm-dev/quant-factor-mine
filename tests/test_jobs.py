"""统一 Job 生命周期与确定性缓存。"""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from qfm.jobs import JobService, JobStatus, JobStore, build_cache_key, canonicalise


def _request(**overrides):
    base = {
        "factor_version": [{"name": "mom_20", "version": 1, "source_hash": "sha256:a"}],
        "dataset_version": "dataset_a",
        "universe_id": "cn_hs300",
        "universe_version": "universe_a",
        "date_range": ["2024-01-01", "2025-01-01"],
        "pipeline_config": {"neutralize": ["size"], "decay": 0},
        "backtest_config": {"rebalance": "ME", "top_n": 30},
    }
    base.update(overrides)
    return base


def test_canonicalise_normalises_nested_runtime_types():
    value = canonicalise(
        {
            "when": pd.Timestamp("2024-01-02"),
            "day": date(2024, 1, 3),
            "number": np.int64(3),
            "items": ("b", "a"),
            "flags": {"z", "a"},
        }
    )

    assert value == {
        "day": "2024-01-03",
        "flags": ["a", "z"],
        "items": ["b", "a"],
        "number": 3,
        "when": "2024-01-02T00:00:00",
    }


def test_cache_key_is_stable_for_reordered_dicts():
    left = build_cache_key("FACTOR_ANALYSIS", {"pipeline_config": {"b": 2, "a": 1}})
    right = build_cache_key("FACTOR_ANALYSIS", {"pipeline_config": {"a": 1, "b": 2}})

    assert left == right
    assert left.startswith("cache_")


@pytest.mark.parametrize(
    "changed",
    [
        {"dataset_version": "dataset_changed"},
        {"universe_id": "cn_zz500"},
        {"date_range": ["2024-02-01", "2025-01-01"]},
        {"pipeline_config": {"neutralize": [], "decay": 0}},
        {"backtest_config": {"rebalance": "W-FRI", "top_n": 30}},
        {"factor_version": [{"name": "mom_20", "version": 2, "source_hash": "sha256:b"}]},
    ],
)
def test_cache_key_changes_when_any_reproducibility_input_changes(changed):
    assert build_cache_key("BACKTEST", _request()) != build_cache_key("BACKTEST", _request(**changed))


def test_job_type_is_part_of_cache_identity():
    assert build_cache_key("FACTOR_COMPUTE", _request()) != build_cache_key("BACKTEST", _request())


def test_success_then_cache_hit_records_two_success_jobs(tmp_path):
    service = JobService(tmp_path)
    calls = []

    first = service.run("FACTOR_COMPUTE", _request(), lambda: calls.append(1) or {"value": 1})
    second = service.run("FACTOR_COMPUTE", _request(), lambda: calls.append(2) or {"value": 2})

    assert first.value == second.value == {"value": 1}
    assert calls == [1]
    assert first.job.status == second.job.status == JobStatus.SUCCESS.value
    assert first.job.cache_hit is False
    assert second.job.cache_hit is True
    jobs = JobStore(tmp_path).list_jobs()
    assert len(jobs) == 2
    assert all(job.status == JobStatus.SUCCESS.value for job in jobs)


def test_failure_persists_failed_job_and_does_not_create_cache(tmp_path):
    service = JobService(tmp_path)

    with pytest.raises(RuntimeError, match="boom"):
        service.run(
            "BACKTEST",
            _request(),
            lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        )

    jobs = JobStore(tmp_path).list_jobs()
    assert len(jobs) == 1
    assert jobs[0].status == JobStatus.FAILED.value
    assert "boom" in jobs[0].error
    assert not list((tmp_path / "results").glob("*.pkl"))


def test_corrupt_cache_is_recomputed_and_replaced(tmp_path):
    request = _request()
    key = build_cache_key("MULTI_FACTOR", request)
    results = tmp_path / "results"
    results.mkdir(parents=True)
    (results / f"{key}.pkl").write_bytes(b"not a pickle")
    calls = []

    run = JobService(tmp_path).run("MULTI_FACTOR", request, lambda: calls.append(1) or [1, 2, 3])
    cached = JobService(tmp_path).run("MULTI_FACTOR", request, lambda: calls.append(2) or [])

    assert run.value == cached.value == [1, 2, 3]
    assert calls == [1]
    assert run.job.cache_hit is False and cached.job.cache_hit is True


def test_job_manifest_round_trip_preserves_request(tmp_path):
    run = JobService(tmp_path).run("FACTOR_ANALYSIS", _request(), lambda: {"ok": True})
    loaded = JobStore(tmp_path).get_job(run.job.job_id)

    assert loaded.to_dict() == run.job.to_dict()
    assert loaded.started_at and loaded.finished_at
    assert datetime.fromisoformat(loaded.finished_at) >= datetime.fromisoformat(loaded.started_at)
