"""权限门测试：白/黑名单、只读模式、人工确认、预算，以及 fail-closed 语义。"""

from __future__ import annotations

import pytest

from qfm.agent.models import ToolKind, ToolSpec
from qfm.agent.permissions import (
    OPEN_POLICY,
    READ_ONLY_POLICY,
    PermissionGate,
    PermissionPolicy,
)

READ_SPEC = ToolSpec(name="list_factors", kind=ToolKind.READ, description="列因子")
COMPUTE_SPEC = ToolSpec(name="run_experiment", kind=ToolKind.COMPUTE, description="跑实验", cost=4.0)
WRITE_SPEC = ToolSpec(name="generate_report", kind=ToolKind.WRITE, description="写报告")


def _gate(**kwargs):
    policy = PermissionPolicy(**kwargs)
    return PermissionGate(policy), policy


# ---------------------------------------------------------------------------
# 名单
# ---------------------------------------------------------------------------
def test_default_policy_allows_ordinary_tools():
    gate, _ = _gate()
    assert gate.check(READ_SPEC).allowed
    assert gate.check(COMPUTE_SPEC).allowed


def test_denied_list_wins_over_everything():
    gate, _ = _gate(denied=frozenset({"run_experiment"}), allowed=None)
    decision = gate.check(COMPUTE_SPEC)
    assert not decision.allowed
    assert decision.rule == "denied_list"
    assert "拒绝名单" in decision.reason


def test_allowlist_blocks_everything_outside_it():
    gate, _ = _gate(allowed=frozenset({"list_factors"}))
    assert gate.check(READ_SPEC).allowed
    blocked = gate.check(COMPUTE_SPEC)
    assert not blocked.allowed
    assert blocked.rule == "not_in_allowlist"
    assert "run_experiment" in blocked.reason


def test_denied_beats_allowlist():
    gate, _ = _gate(
        allowed=frozenset({"run_experiment"}), denied=frozenset({"run_experiment"})
    )
    assert not gate.check(COMPUTE_SPEC).allowed


# ---------------------------------------------------------------------------
# 只读模式
# ---------------------------------------------------------------------------
def test_read_only_blocks_only_write_tools():
    gate, _ = _gate(read_only=True)
    assert gate.check(READ_SPEC).allowed
    assert gate.check(COMPUTE_SPEC).allowed, "计算不是副作用，只读模式不应拦它"
    blocked = gate.check(WRITE_SPEC)
    assert not blocked.allowed
    assert blocked.rule == "read_only"


def test_read_only_preset_is_usable():
    gate = PermissionGate(READ_ONLY_POLICY)
    assert gate.check(READ_SPEC).allowed
    assert not gate.check(WRITE_SPEC).allowed


# ---------------------------------------------------------------------------
# 人工确认：fail-closed
# ---------------------------------------------------------------------------
def test_confirmation_denied_when_no_approver():
    """没有审批人时必须拒绝，而不是「没人审批所以放行」。"""
    gate, _ = _gate(confirm_required=frozenset({"generate_report"}))
    decision = gate.check(WRITE_SPEC)
    assert not decision.allowed
    assert decision.needs_confirmation
    assert decision.rule == "confirmation_required"
    assert "没有审批人" in decision.reason


def test_confirmation_passes_with_approving_approver():
    gate = PermissionGate(
        PermissionPolicy(confirm_required=frozenset({"generate_report"})),
        approver=lambda spec, args: True,
    )
    assert gate.check(WRITE_SPEC).allowed


def test_confirmation_denied_when_approver_rejects():
    gate = PermissionGate(
        PermissionPolicy(confirm_required=frozenset({"generate_report"})),
        approver=lambda spec, args: False,
    )
    assert not gate.check(WRITE_SPEC).allowed


def test_approver_exception_is_treated_as_rejection():
    """审批回调本身出错也必须当拒绝处理，不能因为异常而放行。"""

    def broken(spec, args):
        raise RuntimeError("审批服务不可用")

    gate = PermissionGate(
        PermissionPolicy(confirm_required=frozenset({"generate_report"})),
        approver=broken,
    )
    assert not gate.check(WRITE_SPEC).allowed
    assert any(event.get("decision") == "approver_error" for event in gate.events)


def test_approver_receives_arguments():
    captured = {}

    def approver(spec, args):
        captured["spec"] = spec.name
        captured["args"] = dict(args)
        return True

    gate = PermissionGate(
        PermissionPolicy(confirm_required=frozenset({"generate_report"})), approver=approver
    )
    gate.check(WRITE_SPEC, {"title": "报告"})
    assert captured == {"spec": "generate_report", "args": {"title": "报告"}}


# ---------------------------------------------------------------------------
# 预算
# ---------------------------------------------------------------------------
def test_total_call_budget_stops_further_calls():
    gate, policy = _gate(max_total_calls=2, max_cost=1e9)
    for _ in range(2):
        assert gate.check(READ_SPEC).allowed
        gate.record(READ_SPEC)
    blocked = gate.check(READ_SPEC)
    assert not blocked.allowed
    assert blocked.rule == "total_calls_budget"


def test_per_tool_budget_is_independent():
    gate, _ = _gate(max_total_calls=100, max_calls_per_tool={"run_experiment": 1}, max_cost=1e9)
    assert gate.check(COMPUTE_SPEC).allowed
    gate.record(COMPUTE_SPEC)
    assert not gate.check(COMPUTE_SPEC).allowed
    assert gate.check(READ_SPEC).allowed, "单工具上限不应影响其他工具"


def test_cost_budget_uses_tool_cost():
    gate, _ = _gate(max_total_calls=100, max_cost=5.0)
    assert gate.check(COMPUTE_SPEC).allowed  # cost 4.0
    gate.record(COMPUTE_SPEC)
    blocked = gate.check(COMPUTE_SPEC)  # 再来一次就 8.0 > 5.0
    assert not blocked.allowed
    assert blocked.rule == "cost_budget"
    assert "8.0" in blocked.reason


def test_budget_snapshot_reports_usage():
    gate = PermissionGate(OPEN_POLICY)
    gate.record(READ_SPEC)
    gate.record(COMPUTE_SPEC)
    snapshot = gate.snapshot()
    assert snapshot["budget"]["calls"] == 2
    assert snapshot["budget"]["cost"] == pytest.approx(5.0)
    assert snapshot["budget"]["per_tool"]["list_factors"] == 1


# ---------------------------------------------------------------------------
# 事件留痕
# ---------------------------------------------------------------------------
def test_denials_are_recorded_as_events():
    gate, _ = _gate(denied=frozenset({"run_experiment"}))
    gate.check(COMPUTE_SPEC, {"name": "bp", "start": "2025-01-01", "end": "2026-01-01"})
    assert len(gate.events) == 1
    event = gate.events[0]
    assert event["tool"] == "run_experiment"
    assert event["decision"] == "denied"
    assert event["rule"] == "denied_list"
    assert "name" in event["arguments"]


def test_allowed_calls_are_not_recorded_as_events():
    gate, _ = _gate()
    gate.check(READ_SPEC)
    assert gate.events == []


def test_decision_serialises_for_the_run_report():
    gate, _ = _gate(denied=frozenset({"run_experiment"}))
    payload = gate.check(COMPUTE_SPEC).to_dict()
    assert payload["allowed"] is False
    assert payload["rule"] == "denied_list"
    assert isinstance(payload["reason"], str)


# ---------------------------------------------------------------------------
# 策略校验
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"max_total_calls": 0}, "max_total_calls"),
        ({"max_cost": 0}, "max_cost"),
        ({"max_calls_per_tool": {"x": 0}}, "max_calls_per_tool"),
    ],
)
def test_invalid_policy_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        PermissionPolicy(**kwargs)


def test_policy_with_replaces_fields():
    policy = PermissionPolicy()
    updated = policy.with_(read_only=True, max_total_calls=5)
    assert updated.read_only and updated.max_total_calls == 5
    assert policy.read_only is False, "原策略不应被修改"
