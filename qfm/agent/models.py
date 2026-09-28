"""Agent 层数据模型：工具、调用、任务、轨迹、运行记录。

设计原则是「一切可落盘、一切可重放」——每个对象都能 `to_dict()` 成
JSON 安全的结构，因为演示和事后复盘都依赖这份轨迹：

- `ToolSpec`   说明「Agent 能做什么」，同时也是给 LLM 的工具目录；
- `ToolCall`   记录「Agent 决定做什么」以及**为什么**（`reason` 字段）；
- `ToolResult` 记录「实际发生了什么」（耗时、尝试次数、是否命中缓存）；
- `TaskRecord` 是任务状态机的载体（PENDING→RUNNING→…→SUCCESS/FAILED/DENIED）；
- `AgentRun`   是一次完整研究的汇总，可直接渲染成 Markdown 报告。

状态词表的取舍：现有仓库里有两套互不映射的状态枚举
（`qfm.jobs.models.JobStatus` 用 SUCCESS/FAILED，`qfm.research.models` 用
completed/failed）。Agent 层需要表达更多中间态（重试中、被拒绝、被跳过），
所以这里定义自己的 `TaskStatus`，并显式提供 `to_job_status()` 做映射，
而不是去改动既有枚举、破坏已落盘的台账格式。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping

__all__ = [
    "TaskStatus",
    "ToolKind",
    "RunStatus",
    "utc_now",
    "new_id",
    "ToolSpec",
    "ToolCall",
    "ToolResult",
    "TaskRecord",
    "PlanStep",
    "Plan",
    "AgentRun",
]


def utc_now() -> str:
    """ISO8601 时间戳（本地时区偏移可见，便于和实验台账对齐）。"""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class TaskStatus(str, Enum):
    """任务状态机。

    合法流转（`qfm.agent.runtime` 是唯一写入方）::

        PENDING ──► RUNNING ──► SUCCESS
                       │  ▲
                       │  └── RETRYING ──► FAILED
                       ├──► FAILED      （重试耗尽或确定性错误）
                       ├──► DENIED      （权限策略拒绝，未执行）
                       └──► SKIPPED     （前置步骤失败导致跳过）
    """

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    RETRYING = "RETRYING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    DENIED = "DENIED"
    SKIPPED = "SKIPPED"

    @property
    def terminal(self) -> bool:
        return self in _TERMINAL_STATUSES

    def to_job_status(self) -> str:
        """映射到 `qfm.jobs.models.JobStatus` 的取值（未接入 Job 时为 PENDING）。"""
        return _TO_JOB_STATUS.get(self, "PENDING")


_TERMINAL_STATUSES = frozenset(
    {TaskStatus.SUCCESS, TaskStatus.FAILED, TaskStatus.DENIED, TaskStatus.SKIPPED}
)

_TO_JOB_STATUS: Mapping["TaskStatus", str] = {
    TaskStatus.PENDING: "PENDING",
    TaskStatus.RUNNING: "RUNNING",
    TaskStatus.RETRYING: "RUNNING",
    TaskStatus.SUCCESS: "SUCCESS",
    TaskStatus.FAILED: "FAILED",
    TaskStatus.DENIED: "FAILED",
    TaskStatus.SKIPPED: "FAILED",
}


class ToolKind(str, Enum):
    """工具性质，权限策略按此分级。"""

    READ = "read"        # 只读：查询元数据、列因子、看历史
    COMPUTE = "compute"  # 计算：要花时间与内存，但不改外部状态
    WRITE = "write"      # 落盘：写报告、写实验台账（唯一有副作用的类别）


class RunStatus(str, Enum):
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"  # 计划完成但有步骤失败/被拒 → 如实标注
    FAILED = "FAILED"


@dataclass(frozen=True)
class ToolSpec:
    """一个工具的自描述。既是注册表元数据，也是给 LLM 的工具目录条目。"""

    name: str
    kind: ToolKind
    description: str
    parameters: Mapping[str, str] = field(default_factory=dict)  # 参数名 → "类型，说明"
    required: tuple[str, ...] = ()
    returns: str = ""
    cost: float = 1.0          # 预算计价单位，expensive 步骤可设更大值
    retryable: bool = True     # 确定性工具（如纯计算）可置 False
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError(f"工具名非法: {self.name!r}")
        unknown = set(self.required) - set(self.parameters)
        if unknown:
            raise ValueError(f"必需参数未在 parameters 中声明: {sorted(unknown)}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "description": self.description,
            "parameters": dict(self.parameters),
            "required": list(self.required),
            "returns": self.returns,
            "cost": self.cost,
            "retryable": self.retryable,
            "tags": list(self.tags),
        }

    def catalog_line(self) -> str:
        """工具目录里的一行紧凑描述，用于 LLM prompt。"""
        params = ", ".join(f"{k}" for k in self.parameters)
        required = f" [必填: {', '.join(self.required)}]" if self.required else ""
        return f"- {self.name}({params}){required} — {self.description}"


@dataclass
class ToolCall:
    """一次工具调用意图。`reason` 让轨迹能回答「Agent 为什么要这么调」。"""

    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    call_id: str = ""

    def __post_init__(self) -> None:
        if not self.call_id:
            self.call_id = new_id("call")

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "tool": self.tool,
            "arguments": _json_safe(self.arguments),
            "reason": self.reason,
        }


@dataclass
class ToolResult:
    """一次工具调用的结果。`value` 保留原对象供后续步骤使用，不进上下文。"""

    call_id: str
    tool: str
    ok: bool
    value: Any = None
    error: str = ""
    error_type: str = ""
    duration_s: float = 0.0
    attempts: int = 0
    cache_hit: bool = False
    summary: str = ""
    warnings: tuple[str, ...] = ()

    def to_dict(self, *, include_value: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "call_id": self.call_id,
            "tool": self.tool,
            "ok": self.ok,
            "error": self.error,
            "error_type": self.error_type,
            "duration_s": round(self.duration_s, 3),
            "attempts": self.attempts,
            "cache_hit": self.cache_hit,
            "summary": self.summary,
            "warnings": list(self.warnings),
        }
        if include_value:
            payload["value"] = _json_safe(self.value)
        return payload


@dataclass
class TaskRecord:
    """任务状态机的一条记录，是 trace 报表的基本单元。"""

    task_id: str
    index: int
    goal: str
    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    attempts: int = 0
    error: str = ""
    result_summary: str = ""
    cache_hit: bool = False
    started_at: str = ""
    finished_at: str = ""
    duration_s: float = 0.0
    artifacts: tuple[str, ...] = ()
    retry_log: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "index": self.index,
            "goal": self.goal,
            "tool": self.tool,
            "arguments": _json_safe(self.arguments),
            "status": self.status.value,
            "attempts": self.attempts,
            "error": self.error,
            "result_summary": self.result_summary,
            "cache_hit": self.cache_hit,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": round(self.duration_s, 3),
            "artifacts": list(self.artifacts),
            "retry_log": [dict(item) for item in self.retry_log],
        }


@dataclass(frozen=True)
class PlanStep:
    """计划中的一步。

    `on_failure` 决定「这一步失败之后怎么办」，是让 Agent 表现得像研究员而不是
    脚本的关键：

    - ``abort``    前置依赖失败 → 后续步骤没有意义，整体中止；
    - ``skip``     可选步骤（例如附带的一致性审计）失败不影响主结论；
    - ``continue`` 尽力而为，记录失败但继续走完计划。
    """

    tool: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    goal: str = ""
    on_failure: str = "abort"
    optional: bool = False

    def __post_init__(self) -> None:
        if self.on_failure not in ("abort", "skip", "continue"):
            raise ValueError(f"非法 on_failure: {self.on_failure!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "arguments": _json_safe(dict(self.arguments)),
            "goal": self.goal,
            "on_failure": self.on_failure,
            "optional": self.optional,
        }


@dataclass
class Plan:
    """一份执行计划。`source` 标明是规则生成还是 LLM 生成——报告里会如实标注。"""

    request: str
    steps: list[PlanStep] = field(default_factory=list)
    source: str = "rule"
    notes: tuple[str, ...] = ()
    round: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request,
            "source": self.source,
            "notes": list(self.notes),
            "round": self.round,
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass
class AgentRun:
    """一次完整研究运行的汇总，可直接渲染成报告。"""

    run_id: str
    request: str
    status: RunStatus = RunStatus.SUCCESS
    created_at: str = ""
    finished_at: str = ""
    planner: str = "rule"
    data_source: str = ""
    data_note: str = ""
    plans: list[Plan] = field(default_factory=list)
    tasks: list[TaskRecord] = field(default_factory=list)
    findings: dict[str, Any] = field(default_factory=dict)
    anomalies: list[dict[str, Any]] = field(default_factory=list)
    warnings: tuple[str, ...] = ()
    artifacts: tuple[str, ...] = ()
    context_stats: dict[str, Any] = field(default_factory=dict)
    permission_events: tuple[dict[str, Any], ...] = ()
    report_markdown: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = utc_now()

    @property
    def duration_s(self) -> float:
        return sum(task.duration_s for task in self.tasks)

    def counts(self) -> dict[str, int]:
        tally: dict[str, int] = {status.value: 0 for status in TaskStatus}
        for task in self.tasks:
            tally[task.status.value] += 1
        return {key: value for key, value in tally.items() if value}

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "request": self.request,
            "status": self.status.value,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "planner": self.planner,
            "data_source": self.data_source,
            "data_note": self.data_note,
            "task_counts": self.counts(),
            "duration_s": round(self.duration_s, 3),
            "plans": [plan.to_dict() for plan in self.plans],
            "tasks": [task.to_dict() for task in self.tasks],
            "findings": _json_safe(self.findings),
            "anomalies": _json_safe(self.anomalies),
            "warnings": list(self.warnings),
            "artifacts": list(self.artifacts),
            "context_stats": _json_safe(self.context_stats),
            "permission_events": [_json_safe(event) for event in self.permission_events],
        }


def _json_safe(value: Any) -> Any:
    """把 numpy / pandas / datetime / 嵌套容器递归转成 JSON 可序列化结构。

    pandas 对象**必须先于鸭子类型分支判定**：`DataFrame` 和 `Series` 都有
    `to_dict()`，若先走 `hasattr(value, "to_dict")` 分支，一个 2000×800 的因子
    矩阵会被逐值序列化进 artifact —— 恰好是上下文管理要避免的那种膨胀。
    所以这里按类型名提前拦截，只保留形状、列名与少量样本值。
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else None
    if isinstance(value, datetime):
        return value.isoformat()

    type_name = type(value).__name__
    if type_name == "DataFrame":
        return {
            "shape": [int(value.shape[0]), int(value.shape[1])],
            "columns_head": [str(column) for column in list(value.columns)[:8]],
        }
    if type_name == "Series":
        head: list[Any] = []
        for item in list(value.head(5)):
            head.append(_json_safe(item))
        return {
            "name": str(value.name) if value.name is not None else "",
            "size": int(value.size),
            "head": head,
        }
    if type_name == "Index":
        return [str(item) for item in list(value)[:8]]

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]

    # numpy 标量：必须在 pandas 分支之后，避免和 Series.item() 混淆
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_safe(value.item())
        except Exception:  # noqa: BLE001 - 多元素数组取 .item() 会抛，按未识别类型处理
            pass
    if type_name == "ndarray":
        return {
            "shape": [int(dim) for dim in value.shape],
            "head": [_json_safe(item) for item in list(value.flatten()[:5])],
        }
    if hasattr(value, "isoformat"):  # date / pd.Timestamp
        try:
            return value.isoformat()
        except Exception:  # noqa: BLE001
            return str(value)
    if hasattr(value, "to_dict"):  # 其他 dataclass / 自定义模型
        try:
            return _json_safe(value.to_dict())
        except Exception:  # noqa: BLE001
            return str(value)
    return str(value)
