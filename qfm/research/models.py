"""研究项目与运行记录的纯数据模型。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import uuid4

import pandas as pd


FORMAT_VERSION = 1
MAX_NAME_LENGTH = 80


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _text(data: Mapping[str, object], key: str, *, allow_empty: bool = False) -> str:
    value = data.get(key)
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(f"缺少或非法字段: {key}")
    return value


def _mapping(data: Mapping[str, object], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"缺少或非法字段: {key}")
    return dict(value)


@dataclass(frozen=True)
class ResearchProject:
    """用户命名的研究容器。"""

    id: str
    name: str
    description: str
    created_at: str
    updated_at: str

    @classmethod
    def create(cls, name: str, description: str = "") -> "ResearchProject":
        cleaned = name.strip()
        if not cleaned or len(cleaned) > MAX_NAME_LENGTH:
            raise ValueError(f"项目名称不能为空且最长 {MAX_NAME_LENGTH} 个字符")
        now = _now()
        return cls(
            id=f"project_{uuid4().hex}",
            name=cleaned,
            description=description.strip(),
            created_at=now,
            updated_at=now,
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ResearchProject":
        return cls(
            id=_text(data, "id"),
            name=_text(data, "name"),
            description=_text(data, "description", allow_empty=True),
            created_at=_text(data, "created_at"),
            updated_at=_text(data, "updated_at"),
        )


@dataclass(frozen=True)
class ResearchRun:
    """一次已完成回测的不可变描述和产物索引。"""

    id: str
    project_id: str
    name: str
    created_at: str
    config: dict[str, Any]
    data_snapshot: dict[str, Any]
    summary: dict[str, Any]
    artifacts: dict[str, str]
    format_version: int = FORMAT_VERSION

    @classmethod
    def create(
        cls,
        project_id: str,
        name: str,
        config: dict[str, Any],
        data_snapshot: dict[str, Any],
        summary: dict[str, Any],
        artifacts: dict[str, str],
    ) -> "ResearchRun":
        cleaned_project_id = project_id.strip()
        if not cleaned_project_id:
            raise ValueError("项目 ID 不能为空")
        cleaned_name = name.strip() or f"运行 {_now()}"
        if len(cleaned_name) > MAX_NAME_LENGTH:
            raise ValueError(f"运行名称最长 {MAX_NAME_LENGTH} 个字符")
        cls._validate_artifacts(artifacts)
        return cls(
            id=f"run_{uuid4().hex}",
            project_id=cleaned_project_id,
            name=cleaned_name,
            created_at=_now(),
            config=dict(config),
            data_snapshot=dict(data_snapshot),
            summary=dict(summary),
            artifacts=dict(artifacts),
        )

    @staticmethod
    def _validate_artifacts(artifacts: Mapping[str, object]) -> None:
        if not artifacts:
            raise ValueError("运行产物不能为空")
        for label, filename in artifacts.items():
            if not isinstance(label, str) or not label or not isinstance(filename, str) or not filename:
                raise ValueError("运行产物文件名非法")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "id": self.id,
            "project_id": self.project_id,
            "name": self.name,
            "created_at": self.created_at,
            "config": dict(self.config),
            "data_snapshot": dict(self.data_snapshot),
            "summary": dict(self.summary),
            "artifacts": dict(self.artifacts),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ResearchRun":
        version = data.get("format_version")
        if version != FORMAT_VERSION:
            raise ValueError(f"不支持的研究记录格式版本: {version}")
        artifacts = _mapping(data, "artifacts")
        cls._validate_artifacts(artifacts)
        return cls(
            id=_text(data, "id"),
            project_id=_text(data, "project_id"),
            name=_text(data, "name"),
            created_at=_text(data, "created_at"),
            config=_mapping(data, "config"),
            data_snapshot=_mapping(data, "data_snapshot"),
            summary=_mapping(data, "summary"),
            artifacts={key: str(value) for key, value in artifacts.items()},
            format_version=version,
        )


@dataclass(frozen=True)
class LoadedResearchRun:
    """已读取且可在页面直接呈现的运行产物。"""

    run: ResearchRun
    nav: pd.Series
    benchmark_nav: pd.Series
    weights: pd.DataFrame
    yearly_performance: pd.DataFrame
    trades: pd.DataFrame
