"""Agent 层异常分类：把「重试有用」和「重试无用」分开。

重试机制的价值完全取决于错误分类的准确度：

- 把参数错误重试三遍 → 白等 3 秒并污染日志；
- 把一次网络抖动直接判死 → Agent 因为可恢复故障而放弃整个研究请求。

所以这里不做「捕获全部 Exception 然后 sleep 重试」这种廉价实现，
而是显式区分三类：

1. `TransientError` —— 瞬时故障（网络抖动、超时、数据源限流、上游临时不可用），
   重试有意义的概率高；
2. `PermanentError` —— 确定性错误（参数非法、工具不存在、数据形状不对、
   权限不足），重试只会浪费预算；
3. 未分类的裸异常 —— 默认按 permanent 处理（保守：宁可快速失败也不要空转）。

另有一个容易被忽略的场景：**静默降级**。现有 `qfm.data.loader` 在单只股票
拉取失败时是 `print` 后返回 `None`、把股票悄悄丢掉，最终仍然返回一个「看起来
成功」的面板。对 Agent 而言这比抛异常更危险——它不知道自己拿的是残缺数据。
因此 `classify_degradation()` 提供「结果不合格也当作瞬时故障」的能力，
让重试机制能覆盖这种无异常失败。
"""

from __future__ import annotations

import socket
from typing import Any, Iterable

__all__ = [
    "AgentError",
    "TransientError",
    "PermanentError",
    "PermissionDeniedError",
    "BudgetExceededError",
    "ToolNotFoundError",
    "InvalidArgumentsError",
    "DegradedResultError",
    "TRANSIENT_TYPES",
    "TRANSIENT_HINTS",
    "classify",
    "is_transient",
    "to_permanent",
    "describe_exception",
]


class AgentError(Exception):
    """Agent 层异常基类。"""


class TransientError(AgentError):
    """瞬时故障：重试可能成功。"""


class PermanentError(AgentError):
    """确定性故障：重试无用，应直接反馈给上层重新规划或如实报告。"""


class PermissionDeniedError(PermanentError):
    """工具权限策略拒绝本次调用。"""


class BudgetExceededError(PermanentError):
    """调用次数或成本预算耗尽。"""


class ToolNotFoundError(PermanentError):
    """请求的工具未注册。"""


class InvalidArgumentsError(PermanentError):
    """工具参数缺失、类型不符或含未知字段。"""


class DegradedResultError(TransientError):
    """无异常但结果不合格（例如数据覆盖率过低）。

    继承 `TransientError` 是刻意设计：静默降级与瞬时故障在处置上是同一类
    ——都应该再试一次，而不是当作成功结果继续往下算。
    """


# 标准库里的瞬时故障类型。
TRANSIENT_TYPES: tuple[type[BaseException], ...] = (
    ConnectionError,      # 含 ConnectionResetError / BrokenPipeError
    ConnectionAbortedError,
    ConnectionRefusedError,
    ConnectionResetError,
    TimeoutError,         # Python 3.10+ 内置；socket.timeout 是其别名
    socket.timeout,
    socket.gaierror,
    BlockingIOError,
    InterruptedError,
)

# 第三方库（requests / urllib3 / akshare 底层）常抛自己的异常类型，
# 它们不一定继承上面的标准库类型，故用消息关键词兜底。
TRANSIENT_HINTS: tuple[str, ...] = (
    "timeout",
    "timed out",
    "connection",
    "connectionreset",
    "connection aborted",
    "temporarily unavailable",
    "temporary failure",
    "name resolution",
    "remote end closed",
    "read timed out",
    "too many requests",
    "rate limit",
    "server error",
    "bad gateway",
    "service unavailable",
    "502",
    "503",
    "504",
    "429",
    "超时",
    "网络",
    "连接",
    "限流",
    "重试",
    "服务不可用",
)


def _message_of(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}".lower()


def classify(exc: BaseException) -> str:
    """返回 ``"transient"`` 或 ``"permanent"``。

    判定顺序（先精确后模糊）：

    1. 显式继承关系优先——`TransientError` / `PermanentError` 直接定调；
    2. 标准库瞬时类型（含 `OSError` 家族里明确可恢复的几个）；
    3. 消息关键词兜底（覆盖 requests / akshare 的非类型化异常）；
    4. 其余按 permanent 处理。
    """
    if isinstance(exc, DegradedResultError):
        return "transient"
    if isinstance(exc, TransientError):
        return "transient"
    if isinstance(exc, PermanentError):
        return "permanent"
    if isinstance(exc, TRANSIENT_TYPES):
        return "transient"

    message = _message_of(exc)
    if any(hint in message for hint in TRANSIENT_HINTS):
        return "transient"
    return "permanent"


def is_transient(exc: BaseException) -> bool:
    return classify(exc) == "transient"


def to_permanent(exc: BaseException, context: str = "") -> PermanentError:
    """把任意异常收敛成 `PermanentError`，保留原始信息便于上报。"""
    prefix = f"{context}： " if context else ""
    if isinstance(exc, PermanentError):
        return exc
    return PermanentError(f"{prefix}{describe_exception(exc)}")


def describe_exception(exc: BaseException) -> str:
    """给人类看的单行错误描述。"""
    return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__


def classify_degradation(
    coverage: float,
    *,
    min_coverage: float,
    subject: str,
    detail: dict[str, Any] | None = None,
) -> DegradedResultError | None:
    """覆盖率不达标时返回 `DegradedResultError`，否则返回 `None`。

    这是「无异常失败」的统一入口：上游把股票悄悄丢掉、或某个字段大面积缺失时，
    调用方用本函数把静默降级翻译成一个可被重试机制识别的错误。
    """
    if min_coverage <= 0 or coverage >= min_coverage:
        return None
    pct = f"{coverage:.1%}"
    floor = f"{min_coverage:.1%}"
    extra = ""
    if detail:
        items: Iterable[str] = (f"{k}={v}" for k, v in sorted(detail.items()))
        extra = "（" + ", ".join(items) + "）"
    return DegradedResultError(
        f"{subject} 覆盖率 {pct} 低于下限 {floor}{extra}；"
        "可能上游静默丢弃了部分标的，结果不可信"
    )
