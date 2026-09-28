"""工具权限门：谁能在什么条件下调用什么工具。

## 为什么 Agent 需要权限层

给 Agent 挂工具，本质上是把「执行任意操作的能力」交给一个可能被自然语言
误导的规划器。真实生产里的风险很具体：

- 用户说「把结果覆盖掉重跑」，Agent 就调了写盘工具，覆盖掉别人三个月的实验台账；
- 规划器进入循环，反复调用昂贵的回测工具，把配额烧光；
- 演示时想让 Agent 只读不写，但没有开关。

所以权限不是「安全检查表」，而是**运行时的不变式**：在调用发生之前拦截，
并把拒绝原因变成一个 Agent 能看见、能适应的观察结果。这正是它与 `if` 判断的
区别——被拒绝的调用会进入轨迹，Agent 必须据此重新规划或如实报告，
而不是静默失败。

## 四个维度的约束

| 维度 | 字段 | 拦截什么 |
| --- | --- | --- |
| 白名单/黑名单 | `allowed` / `denied` | 指定工具一律不可用 |
| 只读模式 | `read_only` | 一切 `write` 类工具（唯一有副作用的类别） |
| 人工确认 | `confirm_required` | 高风险工具需审批，无审批人时默认拒绝 |
| 预算 | `max_total_calls` / `max_calls_per_tool` / `max_cost` | 防循环、防烧配额 |

设计取向是**默认拒绝**（fail-closed）：`approver` 为 `None` 时，
需要确认的调用一律拒绝并说明原因，绝不「因为没人审批所以放行」。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from qfm.agent.models import ToolKind, ToolSpec

__all__ = [
    "PermissionPolicy",
    "PermissionDecision",
    "PermissionGate",
    "BudgetState",
    "READ_ONLY_POLICY",
    "OPEN_POLICY",
]


@dataclass(frozen=True)
class PermissionPolicy:
    """权限与预算策略。

    Args:
        allowed: 白名单。``None`` 表示不限制；给出集合则集合外全部拒绝。
        denied: 黑名单，优先级高于白名单。
        read_only: 为真时拒绝所有 `write` 类工具。
        confirm_required: 需要审批的工具集合。
        max_total_calls: 总调用次数上限（含失败与被拒前的检查）。
        max_calls_per_tool: 单工具调用次数上限。
        max_cost: 累计成本上限，按 `ToolSpec.cost` 计价。
    """

    allowed: frozenset[str] | None = None
    denied: frozenset[str] = frozenset()
    read_only: bool = False
    confirm_required: frozenset[str] = frozenset()
    max_total_calls: int = 40
    max_calls_per_tool: Mapping[str, int] = field(default_factory=dict)
    max_cost: float = 200.0

    def __post_init__(self) -> None:
        if self.max_total_calls < 1:
            raise ValueError(f"max_total_calls 必须 >= 1，得到 {self.max_total_calls}")
        if self.max_cost <= 0:
            raise ValueError(f"max_cost 必须 > 0，得到 {self.max_cost}")
        for name, cap in self.max_calls_per_tool.items():
            if cap < 1:
                raise ValueError(f"{name} 的 max_calls_per_tool 必须 >= 1，得到 {cap}")

    def with_(self, **changes: Any) -> "PermissionPolicy":
        from dataclasses import replace

        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": sorted(self.allowed) if self.allowed is not None else None,
            "denied": sorted(self.denied),
            "read_only": self.read_only,
            "confirm_required": sorted(self.confirm_required),
            "max_total_calls": self.max_total_calls,
            "max_calls_per_tool": dict(self.max_calls_per_tool),
            "max_cost": self.max_cost,
        }


#: 只读演示模式：禁掉一切落盘，用于「只看不写」的评审场景。
READ_ONLY_POLICY = PermissionPolicy(read_only=True, max_total_calls=30)

#: 宽松模式：不限制工具，只保留预算兜底防死循环。
OPEN_POLICY = PermissionPolicy(max_total_calls=60)


@dataclass
class BudgetState:
    """预算消耗状态。"""

    calls: int = 0
    cost: float = 0.0
    per_tool: dict[str, int] = field(default_factory=dict)

    def record(self, spec: ToolSpec) -> None:
        self.calls += 1
        self.cost += spec.cost
        self.per_tool[spec.name] = self.per_tool.get(spec.name, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "cost": round(self.cost, 3),
            "per_tool": dict(self.per_tool),
        }


@dataclass(frozen=True)
class PermissionDecision:
    """权限判定结果。`reason` 会原样进入轨迹与报告，必须写清「为什么」。"""

    allowed: bool
    reason: str = ""
    needs_confirmation: bool = False
    rule: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "needs_confirmation": self.needs_confirmation,
            "rule": self.rule,
        }


class PermissionGate:
    """权限门。持有策略与预算状态；每次调用前 `check()`，通过后 `record()`。

    Args:
        policy: 权限与预算策略。
        approver: 审批回调。接收 `(spec, arguments)`，返回是否批准。
            ``None`` 表示无审批人 —— 此时需要确认的调用**一律拒绝**。
    """

    def __init__(
        self,
        policy: PermissionPolicy | None = None,
        approver: Callable[[ToolSpec, Mapping[str, Any]], bool] | None = None,
    ) -> None:
        self.policy = policy or PermissionPolicy()
        self.approver = approver
        self.budget = BudgetState()
        self.events: list[dict[str, Any]] = []

    # ---- 判定 -------------------------------------------------------------
    def check(self, spec: ToolSpec, arguments: Mapping[str, Any] | None = None) -> PermissionDecision:
        """判定一次调用是否允许。不修改预算状态（由 `record` 负责）。"""
        args = dict(arguments or {})
        decision = self._evaluate(spec, args)
        if not decision.allowed:
            self.events.append({
                "tool": spec.name,
                "kind": spec.kind.value,
                "decision": "denied",
                "rule": decision.rule,
                "reason": decision.reason,
                "arguments": {key: str(value)[:80] for key, value in args.items()},
            })
        return decision

    def _evaluate(self, spec: ToolSpec, args: Mapping[str, Any]) -> PermissionDecision:
        policy = self.policy

        if spec.name in policy.denied:
            return PermissionDecision(
                allowed=False,
                reason=f"工具 {spec.name} 在拒绝名单中，本次运行不可用",
                rule="denied_list",
            )

        if policy.allowed is not None and spec.name not in policy.allowed:
            return PermissionDecision(
                allowed=False,
                reason=(
                    f"工具 {spec.name} 不在允许名单内；"
                    f"本次运行只允许 {sorted(policy.allowed)}"
                ),
                rule="not_in_allowlist",
            )

        if policy.read_only and spec.kind is ToolKind.WRITE:
            return PermissionDecision(
                allowed=False,
                reason=f"当前为只读模式，{spec.name} 属于写操作（{spec.kind.value}），被拒绝",
                rule="read_only",
            )

        if spec.name in policy.confirm_required and not self._confirm(spec, args):
            return PermissionDecision(
                allowed=False,
                reason=(
                    f"工具 {spec.name} 需要人工确认"
                    + ("，但当前运行没有审批人（默认拒绝）" if self.approver is None else "，审批未通过")
                ),
                needs_confirmation=True,
                rule="confirmation_required",
            )

        cap = policy.max_calls_per_tool.get(spec.name)
        if cap is not None and self.budget.per_tool.get(spec.name, 0) >= cap:
            return PermissionDecision(
                allowed=False,
                reason=f"工具 {spec.name} 已达到单工具调用上限 {cap} 次",
                rule="per_tool_budget",
            )

        if self.budget.calls >= policy.max_total_calls:
            return PermissionDecision(
                allowed=False,
                reason=f"已达总调用上限 {policy.max_total_calls} 次，停止继续调用工具",
                rule="total_calls_budget",
            )

        if self.budget.cost + spec.cost > policy.max_cost:
            return PermissionDecision(
                allowed=False,
                reason=(
                    f"调用 {spec.name} 会使累计成本达到 "
                    f"{self.budget.cost + spec.cost:.1f}，超过上限 {policy.max_cost:.1f}"
                ),
                rule="cost_budget",
            )

        return PermissionDecision(allowed=True, reason="通过", rule="allow")

    def _confirm(self, spec: ToolSpec, args: Mapping[str, Any]) -> bool:
        if self.approver is None:
            return False
        try:
            return bool(self.approver(spec, args))
        except Exception as exc:  # noqa: BLE001 - 审批异常按拒绝处理（fail-closed）
            self.events.append({
                "tool": spec.name,
                "decision": "approver_error",
                "reason": f"{type(exc).__name__}: {exc}",
            })
            return False

    # ---- 记账 -------------------------------------------------------------
    def record(self, spec: ToolSpec) -> None:
        self.budget.record(spec)

    def snapshot(self) -> dict[str, Any]:
        return {
            "policy": self.policy.to_dict(),
            "budget": self.budget.to_dict(),
            "events": [dict(event) for event in self.events],
        }
