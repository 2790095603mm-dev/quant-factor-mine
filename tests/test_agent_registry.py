"""工具注册表测试：契约声明、参数校验、调度、摘要压缩。"""

from __future__ import annotations

import pytest

from qfm.agent.errors import InvalidArgumentsError, ToolNotFoundError
from qfm.agent.models import ToolKind, ToolSpec
from qfm.agent.registry import ToolRegistry, summarize_payload


@pytest.fixture()
def registry():
    reg = ToolRegistry()

    @reg.register(
        "echo",
        kind=ToolKind.READ,
        description="回显",
        parameters={"text": "str，要回显的文本", "times": "int，重复次数"},
        required=("text",),
    )
    def echo(context, text, times=1):
        return {"text": text * times, "times": times}, f"回显 {text!r} {times} 次"

    @reg.register(
        "boom",
        kind=ToolKind.COMPUTE,
        description="总是抛瞬时故障",
        parameters={},
        required=(),
    )
    def boom(context):
        raise ConnectionError("网络抖动")

    return reg


# ---------------------------------------------------------------------------
# 注册与查询
# ---------------------------------------------------------------------------
def test_registry_lists_tools_sorted(registry):
    assert registry.names() == ["boom", "echo"]
    assert len(registry) == 2
    assert "echo" in registry


def test_duplicate_registration_rejected(registry):
    with pytest.raises(ValueError, match="重复注册"):
        registry.add(ToolSpec(name="echo", kind=ToolKind.READ, description="x"), lambda **_: None)


def test_tool_spec_validates_name_and_required_params():
    with pytest.raises(ValueError, match="工具名非法"):
        ToolSpec(name="bad name!", kind=ToolKind.READ, description="x")
    with pytest.raises(ValueError, match="必需参数未在 parameters"):
        ToolSpec(
            name="ok", kind=ToolKind.READ, description="x",
            parameters={"a": "str"}, required=("b",),
        )


def test_unknown_tool_raises_tool_not_found(registry):
    with pytest.raises(ToolNotFoundError, match="未注册的工具"):
        registry.get("nope")


def test_specs_serialise_and_filter_by_kind(registry):
    payloads = [spec.to_dict() for spec in registry.specs()]
    assert payloads[0]["kind"] == "compute"
    assert len(registry.by_kind(ToolKind.READ)) == 1


# ---------------------------------------------------------------------------
# 参数校验
# ---------------------------------------------------------------------------
def test_missing_required_argument_is_permanent_error(registry):
    with pytest.raises(InvalidArgumentsError, match="缺少必填参数"):
        registry.validate("echo", {})


def test_none_for_required_argument_counts_as_missing(registry):
    with pytest.raises(InvalidArgumentsError, match="缺少必填参数"):
        registry.validate("echo", {"text": None})


def test_unknown_argument_is_rejected(registry):
    """拼错参数名必须立刻失败，而不是穿透到工具内部炸出难懂的报错。"""
    with pytest.raises(InvalidArgumentsError, match="未知参数"):
        registry.validate("echo", {"text": "a", "repeat": 3})


def test_type_mismatch_is_rejected(registry):
    with pytest.raises(InvalidArgumentsError, match="期望 int"):
        registry.validate("echo", {"text": "a", "times": "三"})


def test_bool_is_not_accepted_as_int(registry):
    """bool 是 int 的子类，必须显式拒绝，否则 True 会被当成合法次数。"""
    with pytest.raises(InvalidArgumentsError, match="得到 bool"):
        registry.validate("echo", {"text": "a", "times": True})


def test_int_is_accepted_for_float_parameter():
    reg = ToolRegistry()
    reg.add(
        ToolSpec(name="f", kind=ToolKind.READ, description="x",
                 parameters={"ratio": "float，比例"}, required=("ratio",)),
        lambda context, ratio: ({"ratio": ratio}, "ok"),
    )
    assert reg.validate("f", {"ratio": 1}) == {"ratio": 1}


def test_invalid_arguments_are_permanent_so_never_retried():
    from qfm.agent.errors import classify

    error = InvalidArgumentsError("缺参")
    assert classify(error) == "permanent"


# ---------------------------------------------------------------------------
# 调度
# ---------------------------------------------------------------------------
def test_call_returns_payload_and_summary(registry, tool_context):
    payload, summary, duration = registry.call("echo", {"text": "ab", "times": 2}, tool_context)
    assert payload["text"] == "abab"
    assert "回显" in summary
    assert duration >= 0


def test_call_propagates_tool_exceptions_for_retry_layer(registry, tool_context):
    """call() 的契约是异常向上传播——否则重试层无法按错误性质分类。"""
    with pytest.raises(ConnectionError):
        registry.call("boom", {}, tool_context)


def test_invoke_wraps_errors_without_raising(registry, tool_context):
    result = registry.invoke("boom", {}, tool_context)
    assert result.ok is False
    assert result.error_type == "ConnectionError"


def test_invoke_reports_validation_failure(registry, tool_context):
    result = registry.invoke("echo", {}, tool_context)
    assert result.ok is False
    assert result.error_type == "InvalidArgumentsError"
    assert result.attempts == 0


def test_invoke_success_collects_warnings(registry, tool_context):
    reg = ToolRegistry()
    reg.add(
        ToolSpec(name="warn", kind=ToolKind.COMPUTE, description="x"),
        lambda context: ({"warnings": ["覆盖率偏低", "样本不足"]}, "完成"),
    )
    result = reg.invoke("warn", {}, tool_context)
    assert result.ok
    assert result.warnings == ("覆盖率偏低", "样本不足")


# ---------------------------------------------------------------------------
# 工作集（ToolContext）
# ---------------------------------------------------------------------------
def test_context_ledger_roundtrip(tool_context):
    tool_context.put("a", 1)
    assert tool_context.get("a") == 1
    assert tool_context.get("missing", "default") == "default"


def test_context_require_fails_with_actionable_message(tool_context):
    with pytest.raises(InvalidArgumentsError, match="缺少前置产物"):
        tool_context.require("panel", "请先执行 load_panel")


def test_context_artifact_path_creates_directory(tool_context):
    target = tool_context.artifact_path("sub/report.md")
    assert target.parent.exists()


# ---------------------------------------------------------------------------
# 摘要压缩
# ---------------------------------------------------------------------------
def test_summarize_dict_keeps_keys_and_compacts_values():
    text = summarize_payload({
        "ic_mean": 0.03462912,
        "n_symbols": 42,
        "note": "ok",
        "frame": __import__("pandas").DataFrame({"a": [1, 2]}),
    })
    assert "ic_mean" in text and "n_symbols=42" in text
    # 浮点不应出现在摘要里满精度
    assert "0.03462912" not in text


def test_summarize_truncates_long_lists():
    text = summarize_payload(list(range(100)))
    assert "共 100 项" in text


def test_summarize_handles_scalars_and_none():
    assert summarize_payload(None) == "（无返回）"
    assert summarize_payload(3.5) == "3.5"
    assert summarize_payload(True) == "True"


def test_summarize_respects_char_limit():
    text = summarize_payload({f"key_{index}": "值" * 20 for index in range(50)}, max_chars=200)
    assert len(text) <= 250


def test_summarize_skips_internal_frame_keys():
    text = summarize_payload({"_frame": "x", "ic": 0.05})
    assert "_frame" not in text and "ic" in text
