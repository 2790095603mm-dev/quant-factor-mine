"""统一任务模型与确定性缓存键。"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4


CACHE_SCHEMA_VERSION = 1
JOB_TYPES = ("FACTOR_COMPUTE", "FACTOR_ANALYSIS", "MULTI_FACTOR", "BACKTEST")


class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonicalise(value: Any) -> Any:
    """把常见运行时类型转换为稳定、可排序的 JSON 值。"""
    if isinstance(value, Mapping):
        return {str(key): canonicalise(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [canonicalise(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalised = [canonicalise(item) for item in value]
        return sorted(normalised, key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True))
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(value.item):
        try:
            return canonicalise(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return canonicalise(value.to_dict())
    raise TypeError(f"缓存请求包含不可序列化类型: {type(value).__name__}")


def build_cache_key(job_type: str, request: Mapping[str, Any]) -> str:
    if job_type not in JOB_TYPES:
        raise ValueError(f"未知 Job 类型: {job_type}")
    payload = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "job_type": job_type,
        "request": canonicalise(request),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "cache_" + sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    job_type: str
    status: str
    cache_key: str
    created_at: str
    request: dict[str, Any]
    started_at: str = ""
    finished_at: str = ""
    result_path: str = ""
    cache_hit: bool = False
    error: str = ""
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def create(cls, job_type: str, request: Mapping[str, Any], cache_key: str) -> "JobRecord":
        if job_type not in JOB_TYPES:
            raise ValueError(f"未知 Job 类型: {job_type}")
        return cls(
            job_id=f"job_{uuid4().hex}",
            job_type=job_type,
            status=JobStatus.PENDING.value,
            cache_key=cache_key,
            created_at=utc_now(),
            request=canonicalise(request),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["warnings"] = list(self.warnings)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "JobRecord":
        status = str(data.get("status") or "")
        if status not in {item.value for item in JobStatus}:
            raise ValueError(f"未知 Job 状态: {status}")
        job_type = str(data.get("job_type") or "")
        if job_type not in JOB_TYPES:
            raise ValueError(f"未知 Job 类型: {job_type}")
        request = data.get("request")
        if not isinstance(request, dict):
            raise ValueError("Job request 必须是对象")
        return cls(
            job_id=str(data["job_id"]),
            job_type=job_type,
            status=status,
            cache_key=str(data["cache_key"]),
            created_at=str(data["created_at"]),
            request=dict(request),
            started_at=str(data.get("started_at") or ""),
            finished_at=str(data.get("finished_at") or ""),
            result_path=str(data.get("result_path") or ""),
            cache_hit=bool(data.get("cache_hit", False)),
            error=str(data.get("error") or ""),
            warnings=tuple(str(item) for item in data.get("warnings") or ()),
        )

