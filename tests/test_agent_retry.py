"""重试层测试：退避、错误分类、静默降级的校验钩子、尝试轨迹。"""

from __future__ import annotations

import random

import pytest

from qfm.agent.errors import (
    DegradedResultError,
    PermanentError,
    TransientError,
    classify,
    classify_degradation,
    describe_exception,
    to_permanent,
)
from qfm.agent.retry import RetryPolicy, call_with_retry


# ---------------------------------------------------------------------------
# 策略本身
# ---------------------------------------------------------------------------
def test_backoff_schedule_is_exponential_and_capped():
    policy = RetryPolicy(max_attempts=5, base_delay=1.0, multiplier=2.0, max_delay=5.0)
    # 1, 2, 4, 8→截断到 5
    assert policy.schedule() == [1.0, 2.0, 4.0, 5.0]


def test_jitter_stays_within_declared_bounds():
    policy = RetryPolicy(max_attempts=4, base_delay=2.0, jitter=0.25)
    rng = random.Random(7)
    for attempt in (1, 2, 3):
        raw = policy.raw_delay(attempt)
        for _ in range(20):
            delay = policy.delay_for(attempt, rng)
            assert raw * 0.75 <= delay <= raw * 1.25


def test_zero_jitter_is_deterministic():
    policy = RetryPolicy(base_delay=3.0, jitter=0.0)
    assert policy.delay_for(1) == 3.0


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"max_attempts": 0}, "max_attempts"),
        ({"base_delay": -1}, "退避时长"),
        ({"multiplier": 0.5}, "multiplier"),
        ({"jitter": 1.5}, "jitter"),
    ],
)
def test_invalid_policy_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        RetryPolicy(**kwargs)


def test_only_transient_gate_controls_retry():
    strict = RetryPolicy(max_attempts=3, only_transient=True)
    loose = RetryPolicy(max_attempts=3, only_transient=False)
    assert strict.should_retry(1, ConnectionError("x")) is True
    assert strict.should_retry(1, ValueError("x")) is False
    assert loose.should_retry(1, ValueError("x")) is True
    # 尝试次数用尽后一律不再重试
    assert strict.should_retry(3, ConnectionError("x")) is False


# ---------------------------------------------------------------------------
# 错误分类
# ---------------------------------------------------------------------------
def test_classify_separates_transient_from_permanent():
    assert classify(ConnectionError("reset")) == "transient"
    assert classify(TimeoutError("timed out")) == "transient"
    assert classify(TransientError("抖动")) == "transient"
    assert classify(ValueError("参数非法")) == "permanent"
    assert classify(KeyError("缺字段")) == "permanent"
    assert classify(PermanentError("确定性")) == "permanent"


def test_classify_uses_message_hints_for_untyped_errors():
    """akshare / requests 常抛非类型化异常，只能靠消息判断。"""

    class OpaqueError(Exception):
        pass

    assert classify(OpaqueError("HTTPSConnectionPool: Read timed out")) == "transient"
    assert classify(OpaqueError("网络连接被重置")) == "transient"
    assert classify(OpaqueError("429 Too Many Requests")) == "transient"
    assert classify(OpaqueError("字段名拼错了")) == "permanent"


def test_degraded_result_is_retryable():
    """静默降级必须被当成可重试，否则残缺数据会被当作成功结果继续算。"""
    assert classify(DegradedResultError("覆盖率 15%")) == "transient"


def test_classify_degradation_builds_error_with_evidence():
    assert classify_degradation(0.95, min_coverage=0.8, subject="面板") is None
    error = classify_degradation(
        0.41, min_coverage=0.8, subject="面板", detail={"请求": 800, "实际": 328}
    )
    assert isinstance(error, DegradedResultError)
    assert "41.0%" in str(error) and "80.0%" in str(error)
    assert "请求=800" in str(error) and "实际=328" in str(error)


def test_classify_degradation_can_be_disabled():
    assert classify_degradation(0.0, min_coverage=0.0, subject="某物") is None


def test_to_permanent_wraps_and_preserves_context():
    wrapped = to_permanent(ValueError("原始原因"), context="解析股票池")
    assert isinstance(wrapped, PermanentError)
    assert "解析股票池" in str(wrapped) and "ValueError" in str(wrapped)
    original = PermanentError("已经是永久错误")
    assert to_permanent(original) is original


def test_describe_exception_handles_empty_message():
    assert describe_exception(ValueError()) == "ValueError"
    assert describe_exception(ValueError("原因")) == "ValueError: 原因"


# ---------------------------------------------------------------------------
# 执行器
# ---------------------------------------------------------------------------
def test_retries_transient_then_succeeds():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("boom")
        return "ok"

    outcome = call_with_retry(flaky, RetryPolicy(max_attempts=3), sleep=lambda _: None)
    assert outcome.ok and outcome.value == "ok"
    assert len(calls) == 3
    assert outcome.attempt_count == 3
    assert [item.classification for item in outcome.attempts] == ["transient", "transient"]
    assert outcome.attempts[0].will_retry is True


def test_permanent_error_is_not_retried():
    calls = []

    def bad():
        calls.append(1)
        raise ValueError("参数非法")

    outcome = call_with_retry(bad, RetryPolicy(max_attempts=5), sleep=lambda _: None)
    assert not outcome.ok
    assert len(calls) == 1, "确定性错误不应重试"
    assert outcome.error_type == "ValueError"
    assert outcome.attempts[0].will_retry is False


def test_exhausted_transient_returns_last_error():
    calls = []

    def always_fail():
        calls.append(1)
        raise ConnectionError(f"第 {len(calls)} 次失败")

    outcome = call_with_retry(always_fail, RetryPolicy(max_attempts=3), sleep=lambda _: None)
    assert not outcome.ok
    assert len(calls) == 3
    assert "第 3 次失败" in outcome.error_message
    assert len(outcome.log()) == 3


def test_validate_hook_turns_silent_degradation_into_retry():
    """本模块最关键的用例：没有异常，但结果不合格也必须重试。

    真实场景是 DataLoader 在股票拉取失败时静默丢弃、仍然返回「成功」的面板。
    只按异常重试的实现会拿着残缺数据算出看起来合理的结论。
    """
    attempts = []

    def degraded_panel():
        attempts.append(1)
        coverage = 0.3 if len(attempts) < 3 else 0.97
        return {"coverage": coverage, "stocks": int(coverage * 800)}

    def require_coverage(payload):
        if payload["coverage"] < 0.8:
            raise DegradedResultError(f"覆盖率仅 {payload['coverage']:.0%}")

    outcome = call_with_retry(
        degraded_panel,
        RetryPolicy(max_attempts=3),
        validate=require_coverage,
        sleep=lambda _: None,
    )
    assert outcome.ok
    assert outcome.value["coverage"] == 0.97
    assert len(attempts) == 3
    assert outcome.attempts[0].error_type == "DegradedResultError"


def test_validate_rejecting_forever_ends_as_failure():
    def always_bad():
        return {"coverage": 0.1}

    def require(payload):
        raise DegradedResultError("覆盖率过低")

    outcome = call_with_retry(
        always_bad, RetryPolicy(max_attempts=2), validate=require, sleep=lambda _: None
    )
    assert not outcome.ok
    assert outcome.attempts[-1].will_retry is False


def test_on_attempt_callback_sees_every_failure():
    seen = []

    def flaky():
        raise ConnectionError("x")

    call_with_retry(
        flaky,
        RetryPolicy(max_attempts=3),
        sleep=lambda _: None,
        on_attempt=lambda attempt: seen.append(attempt.index),
    )
    assert seen == [1, 2, 3]


def test_sleep_receives_computed_backoff():
    slept = []

    def flaky():
        raise ConnectionError("x")

    call_with_retry(
        flaky,
        RetryPolicy(max_attempts=3, base_delay=1.0, multiplier=2.0, jitter=0.0, max_delay=10.0),
        sleep=slept.append,
    )
    assert slept == [1.0, 2.0]


def test_deadline_stops_retrying(monkeypatch):
    """总时长超限后不再重试，避免一次研究卡在无限退避里。"""
    import qfm.agent.retry as retry_module

    now = {"value": 0.0}

    def fake_monotonic():
        return now["value"]

    monkeypatch.setattr(retry_module.time, "monotonic", fake_monotonic)
    calls = []

    def slow_fail():
        calls.append(1)
        now["value"] += 10.0  # 每次尝试消耗 10 秒
        raise ConnectionError("x")

    outcome = call_with_retry(
        slow_fail, RetryPolicy(max_attempts=5), sleep=lambda _: None, deadline_s=5.0
    )
    assert not outcome.ok
    assert len(calls) == 1, "超过总时长上限后不应继续重试"
    assert outcome.attempts[0].will_retry is False


def test_deadline_allows_retry_within_budget(monkeypatch):
    import qfm.agent.retry as retry_module

    now = {"value": 0.0}
    monkeypatch.setattr(retry_module.time, "monotonic", lambda: now["value"])
    calls = []

    def quick_fail():
        calls.append(1)
        now["value"] += 1.0
        raise ConnectionError("x")

    outcome = call_with_retry(
        quick_fail, RetryPolicy(max_attempts=3), sleep=lambda _: None, deadline_s=60.0
    )
    assert not outcome.ok
    assert len(calls) == 3


def test_keyboard_interrupt_is_not_swallowed():
    def interrupted():
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        call_with_retry(interrupted, RetryPolicy(max_attempts=3), sleep=lambda _: None)


def test_policy_dict_is_json_ready():
    payload = RetryPolicy(max_attempts=4).to_dict()
    assert payload["max_attempts"] == 4
    assert isinstance(payload["jitter"], float)
