"""研究项目与运行记录的纯数据模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import uuid4

import pandas as pd

from qfm.data.catalog import REQUIRED_BINDING_FIELDS


FORMAT_VERSION = 3
# v1 只存因子名字符串；v2 有完整运行快照但未强制绑定 Dataset / Universe 版本。
SUPPORTED_FORMAT_VERSIONS = (1, 2, 3)
MAX_NAME_LENGTH = 80

STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUSES = (STATUS_COMPLETED, STATUS_FAILED)
STATUS_LABELS = {STATUS_COMPLETED: "已完成", STATUS_FAILED: "失败"}


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
    """一次回测运行的不可变描述和产物索引。

    除参数（config）与结果（summary + 产物 CSV）外，还记录可复现性所需的元信息：
    status（完成/失败）、tags、finished_at、code_version（代码侧口径版本）、
    factor_versions（当次所用因子的确切版本与源码指纹）。
    """

    id: str
    project_id: str
    name: str
    created_at: str
    config: dict[str, Any]
    data_snapshot: dict[str, Any]
    summary: dict[str, Any]
    artifacts: dict[str, str]
    format_version: int = FORMAT_VERSION
    status: str = STATUS_COMPLETED
    finished_at: str = ""
    tags: tuple[str, ...] = ()
    code_version: dict[str, Any] = field(default_factory=dict)
    factor_versions: tuple[dict[str, Any], ...] = ()
    error: str = ""

    @classmethod
    def create(
        cls,
        project_id: str,
        name: str,
        config: dict[str, Any],
        data_snapshot: dict[str, Any],
        summary: dict[str, Any],
        artifacts: dict[str, str],
        status: str = STATUS_COMPLETED,
        tags: tuple[str, ...] | list[str] = (),
        code_version: dict[str, Any] | None = None,
        factor_versions: list[dict[str, Any]] | None = None,
        error: str = "",
    ) -> "ResearchRun":
        cleaned_project_id = project_id.strip()
        if not cleaned_project_id:
            raise ValueError("项目 ID 不能为空")
        cleaned_name = name.strip() or f"运行 {_now()}"
        if len(cleaned_name) > MAX_NAME_LENGTH:
            raise ValueError(f"运行名称最长 {MAX_NAME_LENGTH} 个字符")
        if status not in STATUSES:
            raise ValueError(f"未知运行状态: {status}")
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
            status=status,
            finished_at=_now(),
            tags=tuple(str(tag) for tag in tags),
            code_version=dict(code_version or {}),
            factor_versions=tuple(dict(item) for item in (factor_versions or ())),
            error=error.strip(),
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
            "status": self.status,
            "finished_at": self.finished_at,
            "tags": list(self.tags),
            "code_version": dict(self.code_version),
            "factor_versions": [dict(item) for item in self.factor_versions],
            "error": self.error,
            "config": dict(self.config),
            "data_snapshot": dict(self.data_snapshot),
            "summary": dict(self.summary),
            "artifacts": dict(self.artifacts),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ResearchRun":
        version = data.get("format_version")
        if version not in SUPPORTED_FORMAT_VERSIONS:
            raise ValueError(f"不支持的研究记录格式版本: {version}")
        artifacts = _mapping(data, "artifacts")
        cls._validate_artifacts(artifacts)
        tags = data.get("tags")
        factor_versions = data.get("factor_versions")
        code_version = data.get("code_version")
        status = data.get("status") or STATUS_COMPLETED
        if status not in STATUSES:
            raise ValueError(f"未知运行状态: {status}")
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
            status=status,
            finished_at=str(data.get("finished_at") or data.get("created_at") or ""),
            tags=tuple(str(tag) for tag in tags) if isinstance(tags, (list, tuple)) else (),
            code_version=dict(code_version) if isinstance(code_version, dict) else {},
            factor_versions=(
                tuple(dict(item) for item in factor_versions)
                if isinstance(factor_versions, (list, tuple))
                else ()
            ),
            error=str(data.get("error") or ""),
        )

    def factor_names(self) -> list[str]:
        """当次运行的因子名（兼容 v1 的字符串列表与 v2 的定义对象列表）。"""
        factors = self.config.get("factors")
        if not isinstance(factors, (list, tuple)):
            return []
        names: list[str] = []
        for item in factors:
            if isinstance(item, str):
                names.append(item)
            elif isinstance(item, Mapping) and item.get("name"):
                names.append(str(item["name"]))
        return names

    def dataset_binding(self) -> dict[str, str] | None:
        """返回可复现的数据绑定；历史记录缺字段时明确返回 ``None``。"""
        if any(not self.data_snapshot.get(key) for key in REQUIRED_BINDING_FIELDS):
            return None
        return {key: str(self.data_snapshot[key]) for key in REQUIRED_BINDING_FIELDS}

    @property
    def legacy_unbound(self) -> bool:
        """该运行是否属于未绑定明确 Dataset / Universe 版本的历史记录。"""
        return self.dataset_binding() is None


@dataclass(frozen=True)
class LoadedResearchRun:
    """已读取且可在页面直接呈现的运行产物。"""

    run: ResearchRun
    nav: pd.Series
    benchmark_nav: pd.Series
    weights: pd.DataFrame
    yearly_performance: pd.DataFrame
    trades: pd.DataFrame
    constraint_history: pd.DataFrame | None = None
