"""重试策略：指数退避 + 抖动 + 错误分类 + 结果校验。

## 为什么需要单独一层

现有仓库有两处与失败打交道的地方，都不足以支撑 Agent：

1. `qfm.jobs.service.JobService.run` —— **零重试**。`compute()` 抛异常即写
   FAILED 并原样向上抛，没有尝试计数、没有退避。
2. `qfm.data.loader.DataLoader._fetch_one` —— 有重试，但是抓数专用的：
   固定 3 次、线性退避（`sleep(1 + attempt)`）、捕获裸 `Exception` 后
   吞掉并返回 `None`，最终失败对上层完全不可见。

Agent 需要的是第三种：**可观测、可分类、可配置、且能覆盖静默降级**的重试。

## 四个关键设计点

**① 错误分类决定要不要重试。** 见 `qfm.agent.errors`。参数错误重试三次只是
浪费 3 秒预算；网络抖动直接判死则会因为一次可恢复故障放弃整个研究请求。

**② 指数退避 + 全抖动（full jitter）。** 固定间隔重试在多个调用并发时会产生
「惊群」——所有调用同时醒来再次撞上同一个限流的服务端。抖动把重试时间打散：

    delay = min(max_delay, base * multiplier^(attempt-1)) * uniform(1-j, 1+j)

**③ 结果校验（validate）覆盖「无异常的失败」。** 这是本模块最有价值的部分。
`DataLoader.load_bars` 在单只股票失败时只 `print` 一行然后把它丢掉，最终仍然
返回一个看似成功的面板。如果只按异常重试，Agent 会拿着残缺数据算出「看起来
合理」的结论。传入 `validate=` 后，调用方可以断言「覆盖率必须 ≥ 80%」，
不达标即抛 `DegradedResultError`（TransientError 子类）→ 触发重试。

**④ 每次尝试都留痕。** `RetryOutcome.attempts` 记录每次失败的类型、消息、
退避时长，直接进任务轨迹。面试官能一眼看到「这步重试了 2 次，因为覆盖率
只有 41%」，而不是只有一个笼统的 FAILED。
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from qfm.agent.errors import (
    TransientError,
    classify,
    describe_exception,
)

__all__ = [
    "RetryPolicy",
    "Attempt",
    "RetryOutcome",
    "call_with_retry",
]

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    """重试策略。

    Args:
        max_attempts: 总尝试次数上限（含首次）。1 表示不重试。
        base_delay: 首次失败后的等待秒数。
        multiplier: 退避倍数，逐次相乘。
        max_delay: 单次等待上限，防止指数爆炸。
        jitter: 抖动比例 [0, 1]，0 表示固定间隔。
        only_transient: 为真时只有瞬时故障才重试；为假则任何异常都重试
            （仅在明确知道工具幂等且故障可恢复时使用）。
    """

    max_attempts: int = 3
    base_delay: float = 0.5
    multiplier: float = 2.0
    max_delay: float = 8.0
    jitter: float = 0.15
    only_transient: bool = True

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts 必须 >= 1，得到 {self.max_attempts}")
        if self.base_delay < 0 or self.max_delay < 0:
            raise ValueError("退避时长不能为负")
        if self.multiplier < 1:
            raise ValueError(f"multiplier 必须 >= 1，得到 {self.multiplier}")
        if not 0.0 <= self.jitter <= 1.0:
            raise ValueError(f"jitter 必须落在 [0, 1]，得到 {self.jitter}")

    @property
    def enabled(self) -> bool:
        return self.max_attempts > 1

    def should_retry(self, attempt: int, exc: BaseException) -> bool:
        """`attempt` 为已完成的尝试次数（1-based）。"""
        if attempt >= self.max_attempts:
            return False
        if not self.only_transient:
            return True
        return classify(exc) == "transient"

    def raw_delay(self, attempt: int) -> float:
        """第 `attempt` 次失败后的理论退避时长（不含抖动，`attempt` 从 1 开始）。"""
        return min(self.base_delay * (self.multiplier ** (attempt - 1)), self.max_delay)

    def delay_for(self, attempt: int, rng: random.Random | None = None) -> float:
        """第 `attempt` 次失败后的实际等待时长（含抖动）。"""
        raw = self.raw_delay(attempt)
        if self.jitter <= 0 or raw <= 0:
            return raw
        picker = rng or random
        factor = picker.uniform(1.0 - self.jitter, 1.0 + self.jitter)
        return max(0.0, raw * factor)

    def schedule(self) -> list[float]:
        """确定性退避序列（不含抖动），用于文档与测试展示。"""
        return [self.raw_delay(attempt) for attempt in range(1, self.max_attempts)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_attempts": self.max_attempts,
            "base_delay": self.base_delay,
            "multiplier": self.multiplier,
            "max_delay": self.max_delay,
            "jitter": self.jitter,
            "only_transient": self.only_transient,
        }


@dataclass
class Attempt:
    """一次失败尝试的记录。"""

    index: int
    error_type: str
    error: str
    classification: str
    delay_s: float = 0.0
    will_retry: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.index,
            "error_type": self.error_type,
            "error": self.error,
            "classification": self.classification,
            "delay_s": round(self.delay_s, 3),
            "will_retry": self.will_retry,
        }


@dataclass
class RetryOutcome:
    """重试执行结果。`ok=False` 时 `error` 是最后一次失败的异常。"""

    ok: bool
    value: Any = None
    attempts: list[Attempt] = field(default_factory=list)
    error: BaseException | None = None

    @property
    def attempt_count(self) -> int:
        return len(self.attempts) + (1 if self.ok else 0)

    @property
    def retried(self) -> bool:
        return bool(self.attempts)

    @property
    def error_message(self) -> str:
        return describe_exception(self.error) if self.error is not None else ""

    @property
    def error_type(self) -> str:
        return type(self.error).__name__ if self.error is not None else ""

    def log(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self.attempts]


def call_with_retry(
    fn: Callable[[], T],
    policy: RetryPolicy | None = None,
    *,
    validate: Callable[[T], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    on_attempt: Callable[[Attempt], None] | None = None,
    rng: random.Random | None = None,
    deadline_s: float | None = None,
) -> RetryOutcome:
    """执行 `fn`，按 `policy` 重试，返回 `RetryOutcome`（本函数不抛异常）。

    Args:
        fn: 无参可调用对象，通常会包一层 lambda 绑定工具参数。
        policy: 重试策略；``None`` 表示默认策略（3 次尝试）。
        validate: 结果校验器。接收 `fn()` 的返回值，不满足要求时抛
            `TransientError`（例如 `DegradedResultError`）即触发重试。
            这是覆盖「静默降级」的关键钩子。
        sleep: 休眠函数，测试时注入替身即可零等待。
        on_attempt: 每次失败尝试的回调（用于实时进度展示）。
        rng: 随机源，测试时注入固定种子使退避时长可预测。
        deadline_s: 可选的总时长上限；超时后不再重试，直接以
            `TransientError` 结束。

    Returns:
        `RetryOutcome`。成功时 `ok=True` 且 `value` 为 `fn()` 结果；
        失败时 `ok=False`、`error` 为最后一次异常、`attempts` 为完整失败轨迹。
    """
    active = policy or RetryPolicy()
    attempts: list[Attempt] = []
    started = time.monotonic()

    for attempt in range(1, active.max_attempts + 1):
        try:
            value = fn()
            if validate is not None:
                validate(value)
            return RetryOutcome(ok=True, value=value, attempts=list(attempts))
        except BaseException as exc:  # noqa: BLE001 - 分类后决定去留，需捕获全部
            if not isinstance(exc, Exception):
                raise  # KeyboardInterrupt / SystemExit 不参与重试
            classification = classify(exc)
            will_retry = active.should_retry(attempt, exc)

            if will_retry and deadline_s is not None:
                elapsed = time.monotonic() - started
                if elapsed >= deadline_s:
                    will_retry = False

            delay = active.delay_for(attempt, rng) if will_retry else 0.0
            record = Attempt(
                index=attempt,
                error_type=type(exc).__name__,
                error=str(exc) or type(exc).__name__,
                classification=classification,
                delay_s=delay,
                will_retry=will_retry,
            )
            attempts.append(record)
            if on_attempt is not None:
                on_attempt(record)

            if not will_retry:
                return RetryOutcome(ok=False, attempts=list(attempts), error=exc)
            if delay > 0:
                sleep(delay)

    # 理论上不可达：最后一次尝试的 will_retry 必为 False。
    last = attempts[-1] if attempts else None
    error = TransientError("重试次数耗尽，但未捕获到异常")
    return RetryOutcome(
        ok=False,
        attempts=list(attempts),
        error=error if last is None else TransientError(f"重试 {last.index} 次后仍失败：{last.error}"),
    )
