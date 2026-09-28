"""工具注册表：Agent 与量化平台之间的唯一接口层。

## 为什么要有这一层

仓库里已有的算法函数（`run_factor_simulation`、`compare_factors`、
`build_tear_sheet`、`replay_saved_run` …）都不依赖 Streamlit，可以直接调用，
但它们**不是为 Agent 设计的**：

- 参数是 Python 对象（`DataPanel`、`PipelineConfig`），不是可序列化的原语，
  进不了 job 的 `request` 字典，也就拿不到内容寻址缓存；
- 返回值是重量级对象（整个 `BacktestResult`、几千行的 DataFrame），
  直接塞进上下文会瞬间撑爆预算；
- 没有参数校验，传错一个 key 会一路穿透到 pandas 内部才炸出难懂的报错。

所以注册表负责四件事：

1. **契约化**：每个工具声明 `ToolSpec`（名称、类别、参数、返回、成本），
   这份声明同时是给 LLM 的工具目录；
2. **参数校验**：缺参、未知参数、类型不符一律抛 `InvalidArgumentsError`
   （PermanentError 子类）——**不重试**，直接让 Agent 重新规划；
3. **结果收敛**：工具的返回值统一为 `(payload, summary)` 二元组：
   `payload` 是完整结果（留在 `ToolContext` 或落盘 artifact，供后续步骤使用），
   `summary` 是给上下文看的紧凑摘要。大对象与小摘要的分离是上下文管理的根基。
4. **大对象外置**：`ToolContext` 持有 panel、因子矩阵、回测结果等重对象，
   Agent 上下文里只出现它们的摘要与句柄。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from qfm.agent.errors import InvalidArgumentsError, ToolNotFoundError
from qfm.agent.models import ToolKind, ToolResult, ToolSpec

__all__ = ["Tool", "ToolRegistry", "ToolContext", "ToolOutput", "summarize_payload"]

ToolOutput = tuple[Any, str]
"""工具函数返回 `(payload, summary)`：完整结果 + 给上下文看的紧凑摘要。"""

_JSON_TYPES: Mapping[str, tuple[type, ...]] = {
    "str": (str,),
    "int": (int,),
    "float": (int, float),
    "bool": (bool,),
    "list": (list, tuple),
    "dict": (dict,),
    "any": (object,),
}


@dataclass
class ToolContext:
    """一次 Agent 运行内的共享状态。

    命名的用意：Agent 的**上下文**（进 LLM 的文本）与工具的**工作集**
    （DataFrame、回测对象）是两件事。把重对象放在这里，上下文只拿摘要，
    才能在「6 年 × 800 只股票」这种数据规模下保持上下文不爆。
    """

    workdir: Path
    data_dir: Path
    provider: Any = None
    panel: Any = None
    panel_meta: dict[str, Any] = field(default_factory=dict)
    store: Any = None            # qfm.research.store.ResearchStore | None
    jobs: Any = None             # qfm.jobs.service.JobService | None
    catalog_root: Any = None
    ledger: dict[str, Any] = field(default_factory=dict)  # 步骤之间的产物句柄
    artifacts: list[str] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)   # 供报告使用的结构化发现

    def put(self, key: str, value: Any) -> None:
        self.ledger[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.ledger.get(key, default)

    def require(self, key: str, hint: str = "") -> Any:
        """取一个前置步骤应该已产出的对象；缺失说明计划编排有误。"""
        if key not in self.ledger:
            tail = f"（{hint}）" if hint else ""
            raise InvalidArgumentsError(
                f"缺少前置产物 {key!r}{tail}；请先执行产出它的步骤"
            )
        return self.ledger[key]

    def artifact_path(self, name: str) -> Path:
        target = Path(self.workdir) / "artifacts" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        return target


@dataclass(frozen=True)
class Tool:
    spec: ToolSpec
    fn: Callable[..., ToolOutput]

    @property
    def name(self) -> str:
        return self.spec.name


class ToolRegistry:
    """工具注册表 + 参数校验 + 调度。"""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    # ---- 注册 -------------------------------------------------------------
    def register(
        self,
        name: str,
        kind: ToolKind,
        description: str,
        *,
        parameters: Mapping[str, str] | None = None,
        required: tuple[str, ...] = (),
        returns: str = "",
        cost: float = 1.0,
        retryable: bool = True,
        tags: tuple[str, ...] = (),
    ) -> Callable[[Callable[..., ToolOutput]], Callable[..., ToolOutput]]:
        """装饰器形式注册工具。"""

        def decorator(fn: Callable[..., ToolOutput]) -> Callable[..., ToolOutput]:
            spec = ToolSpec(
                name=name,
                kind=kind,
                description=description,
                parameters=dict(parameters or {}),
                required=required,
                returns=returns,
                cost=cost,
                retryable=retryable,
                tags=tags,
            )
            self.add(spec, fn)
            return fn

        return decorator

    def add(self, spec: ToolSpec, fn: Callable[..., ToolOutput]) -> Tool:
        if spec.name in self._tools:
            raise ValueError(f"工具重复注册: {spec.name}")
        tool = Tool(spec=spec, fn=fn)
        self._tools[spec.name] = tool
        return tool

    def replace(self, name: str, fn: Callable[..., ToolOutput]) -> Tool:
        """替换某个工具的实现，保留其契约声明。

        用于加包装（埋点、限流、注入故障）而不改动声明 —— 契约不变、
        实现可换，这正是工具层与具体实现解耦的价值。
        """
        existing = self.get(name)
        tool = Tool(spec=existing.spec, fn=fn)
        self._tools[name] = tool
        return tool

    # ---- 查询 -------------------------------------------------------------
    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolNotFoundError(
                f"未注册的工具 {name!r}；可用: {', '.join(sorted(self._tools))}"
            ) from None

    def specs(self) -> list[ToolSpec]:
        return [self._tools[key].spec for key in sorted(self._tools)]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def by_kind(self, kind: ToolKind) -> list[ToolSpec]:
        return [spec for spec in self.specs() if spec.kind == kind]

    def catalog(self) -> str:
        """工具目录文本，直接进 LLM prompt。"""
        lines = ["可用工具："]
        for spec in self.specs():
            lines.append(spec.catalog_line())
            for param, doc in spec.parameters.items():
                marker = "必填" if param in spec.required else "可选"
                lines.append(f"    - {param} ({marker}): {doc}")
        return "\n".join(lines)

    # ---- 校验与调度 -------------------------------------------------------
    def validate(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """校验并归一化参数。

        Raises:
            ToolNotFoundError: 工具未注册。
            InvalidArgumentsError: 缺少必填参数 / 含未知参数 / 类型不符。
        """
        tool = self.get(name)
        spec = tool.spec
        given = dict(arguments)

        missing = [key for key in spec.required if key not in given or given[key] is None]
        if missing:
            raise InvalidArgumentsError(
                f"{name} 缺少必填参数 {missing}；该工具参数为 {sorted(spec.parameters)}"
            )

        unknown = sorted(set(given) - set(spec.parameters))
        if unknown:
            raise InvalidArgumentsError(
                f"{name} 收到未知参数 {unknown}；该工具参数为 {sorted(spec.parameters)}"
            )

        for key, value in given.items():
            if value is None:
                continue
            declared = spec.parameters[key].split("，")[0].strip()
            expected = _JSON_TYPES.get(declared)
            if expected is None or expected is _JSON_TYPES["any"]:
                continue
            # bool 是 int 的子类，需排除以免 True 被当成合法 int
            if isinstance(value, bool) and declared != "bool":
                raise InvalidArgumentsError(f"{name}.{key} 期望 {declared}，得到 bool")
            if not isinstance(value, expected):
                raise InvalidArgumentsError(
                    f"{name}.{key} 期望 {declared}，得到 {type(value).__name__}"
                )
        return given

    def invoke(
        self,
        name: str,
        arguments: Mapping[str, Any],
        context: ToolContext,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> ToolResult:
        """执行一次工具调用并封装成 `ToolResult`（本方法不抛异常）。

        适合「一次调用、只要结果」的场景（测试、脚本）。需要重试机制的场景
        应当用 `call()`，因为它让异常向上传播，重试层才能按错误性质决定是否重试。
        """
        started = clock()
        try:
            payload, summary, _ = self.call(name, arguments, context, clock=clock)
        except Exception as exc:  # noqa: BLE001 - 本方法的契约是不抛异常
            return ToolResult(
                call_id="",
                tool=name,
                ok=False,
                error=str(exc),
                error_type=type(exc).__name__,
                duration_s=clock() - started,
                attempts=0,
            )
        warnings = ()
        if isinstance(payload, dict):
            raw_warnings = payload.get("warnings")
            if isinstance(raw_warnings, (list, tuple)):
                warnings = tuple(str(item) for item in raw_warnings)
        return ToolResult(
            call_id="",
            tool=name,
            ok=True,
            value=payload,
            duration_s=clock() - started,
            attempts=1,
            summary=summary,
            warnings=warnings,
        )

    def call(
        self,
        name: str,
        arguments: Mapping[str, Any],
        context: ToolContext,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> tuple[Any, str, float]:
        """校验并执行工具，返回 `(payload, summary, duration_s)`。

        与 `invoke` 的区别：**异常向上传播**。参数错误抛
        `InvalidArgumentsError`（PermanentError，不重试），工具内部的瞬时故障
        原样抛出（由 `qfm.agent.retry` 分类后决定是否重试）。
        """
        started = clock()
        tool = self.get(name)
        params = self.validate(name, arguments)
        payload, summary = tool.fn(context=context, **params)
        return payload, summary, clock() - started


def summarize_payload(payload: Any, max_chars: int = 600) -> str:
    """把任意工具返回值压成一行到几行的紧凑摘要。

    摘要的目标是「够 LLM 判断下一步」，而不是「完整复现结果」——
    后者交给 artifact 文件。所以这里优先抽取**结构信息**（键名、形状、
    覆盖率、关键指标），而不是罗列数据。
    """
    if payload is None:
        return "（无返回）"
    if isinstance(payload, str):
        text = payload.strip()
        return text if len(text) <= max_chars else text[: max_chars - 1] + "…"
    if isinstance(payload, (int, float, bool)):
        return str(payload)
    if isinstance(payload, dict):
        parts: list[str] = []
        for key, value in payload.items():
            if key in ("_frame", "_object"):
                continue
            parts.append(f"{key}={_compact(value)}")
            if sum(len(part) for part in parts) > max_chars:
                parts.append("…")
                break
        return ", ".join(parts)
    if isinstance(payload, (list, tuple)):
        head = ", ".join(_compact(item) for item in payload[:5])
        more = f" …共 {len(payload)} 项" if len(payload) > 5 else ""
        return f"[{head}]{more}"
    return _compact(payload)


def _compact(value: Any, limit: int = 60) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (int, str)):
        text = str(value)
        return text if len(text) <= limit else text[: limit - 1] + "…"
    if isinstance(value, dict):
        return "{" + ", ".join(list(value)[:4]) + ("…}" if len(value) > 4 else "}")
    if isinstance(value, (list, tuple)):
        return f"[{len(value)} 项]"
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            return f"<{type(value).__name__} {tuple(int(dim) for dim in shape)}>"
        except Exception:  # noqa: BLE001
            return f"<{type(value).__name__}>"
    return f"<{type(value).__name__}>"
