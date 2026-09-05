"""研究账本领域模型的序列化与输入校验。"""

from __future__ import annotations

import pytest

from qfm.research.models import ResearchProject, ResearchRun


def test_project_round_trip_preserves_metadata():
    project = ResearchProject.create(name="价值+质量", description="月频选股")

    restored = ResearchProject.from_dict(project.to_dict())

    assert restored == project


def test_project_rejects_blank_or_oversized_name():
    with pytest.raises(ValueError, match="项目名称"):
        ResearchProject.create(name="   ")
    with pytest.raises(ValueError, match="80"):
        ResearchProject.create(name="x" * 81)


def test_run_round_trip_preserves_config_and_summary():
    run = ResearchRun.create(
        project_id="project_1",
        name="2024 月频",
        config={"top_n": 30},
        data_snapshot={"pool": "index800"},
        summary={"夏普比率": 1.2},
        artifacts={"nav": "nav.csv"},
    )

    assert ResearchRun.from_dict(run.to_dict()) == run


def test_run_rejects_unknown_format_version():
    data = ResearchRun.create(
        project_id="project_1",
        name="run",
        config={},
        data_snapshot={},
        summary={},
        artifacts={"nav": "nav.csv"},
    ).to_dict()
    data["format_version"] = 999

    with pytest.raises(ValueError, match="格式版本"):
        ResearchRun.from_dict(data)
