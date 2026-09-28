"""Quant Research Agent：在既有量化研究平台上加的一层自主研究能力。

## 这一层解决什么

`qfm` 已经有完整的研究底座：数据加载与版本绑定、49 个因子、统一处理管线、
单因子实验引擎、内容寻址的作业缓存、实验台账与重放。缺的不是算法，而是
**把一串研究动作组织起来、并在遇到问题时自己调整**的那一层：

    用户说「分析最近一年银行股低估值因子的表现」
      → 解析出股票池、区间、因子（规划器）
      → 依次调用取数 / 因子计算 / 实验 / 异常检查 / 报告（工具层）
      → 中途遇到瞬时故障会退避重试，数据覆盖率不达标会像故障一样重试
      → 检查出「截面只有 42 只、分层检验不可用」后，自动补跑宽基对照
      → 全程每一步都有状态、耗时、重试、权限事件的留痕

## 目录结构

| 模块 | 职责 |
| --- | --- |
| `errors` | 异常分类：区分「重试有用」与「重试无用」，含静默降级的表达 |
| `retry` | 指数退避 + 抖动 + 结果校验（覆盖无异常的失败） |
| `models` | 工具/调用/任务/轨迹/运行记录的数据模型与状态机 |
| `registry` | 工具注册表：契约声明、参数校验、调度 |
| `permissions` | 工具权限门：白/黑名单、只读模式、人工确认、预算 |
| `context` | 上下文管理：摘要代替原文、超限外置、超预算折叠最旧 |
| `providers` | 数据源抽象：本地缓存 / 离线合成面板 |
| `synthetic` | 确定性合成面板（零网络自举，含复权口径对齐） |
| `vocab` | 研究语义词表：行业、股票池、因子主题、日期区间 |
| `planner` | 规划器：规则规划（离线可用）/ LLM 规划（可选增强） |
| `tools/*` | 12 个工具：股票池、面板、因子、实验、异常检查、台账、报告 |
| `runtime` | 主循环：权限 → 重试 → 观察 → 异常驱动的重新规划 → 报告 |
| `reporting` | 两类报告：研究结论（给研究员）/ 执行轨迹（给工程评审） |

## 快速使用

```bash
# 命令行（无缓存数据时自动回落到离线合成面板）
python -m qfm.agent "分析最近一年银行股低估值因子的表现"

# 只读模式：禁掉一切写盘工具，看 Agent 如何适应
python -m qfm.agent "分析最近一年白酒股动量因子" --read-only

# 禁掉某个工具，观察 Agent 如实报告而不是编造结论
python -m qfm.agent "分析最近一年银行股低估值因子" --deny generate_report
```

```python
from qfm.agent import AgentRuntime, AgentConfig, RulePlanner, build_default_registry, resolve_provider

runtime = AgentRuntime(
    build_default_registry(),
    RulePlanner(),
    AgentConfig(),
    provider=resolve_provider("auto"),
    runs_root="reports/agent_runs",
)
run = runtime.run("分析最近一年银行股低估值因子的表现")
print(run.status.value, run.counts())
```

## 关于「默认规划器不是 LLM」

这是刻意的设计，不是能力缺失。Agent 架构里可复用的骨架是**工具契约、权限门、
重试、上下文管理、异常驱动的重新规划**；规划器是可插拔的一层。默认用确定性的
规则规划器，保证了：

- clone 下来无需 API Key、无需外网即可复现完整流程；
- 演示结果稳定可复现（同输入必然同计划）；
- 规划失败时行为可预测。

需要真实模型规划时设置 `QFM_AGENT_LLM_BASE_URL` / `QFM_AGENT_LLM_API_KEY` /
`QFM_AGENT_LLM_MODEL` 即可切换到 `LLMPlanner`，它会用同一份工具目录规划，
并在任何失败时回落到规则规划器。
"""

from __future__ import annotations

from qfm.agent.context import ContextBudget, ContextManager
from qfm.agent.errors import (
    AgentError,
    BudgetExceededError,
    DegradedResultError,
    InvalidArgumentsError,
    PermanentError,
    PermissionDeniedError,
    ToolNotFoundError,
    TransientError,
)
from qfm.agent.llm import LLMClient, NullClient, OpenAICompatibleClient, build_llm_client
from qfm.agent.models import (
    AgentRun,
    Plan,
    PlanStep,
    RunStatus,
    TaskRecord,
    TaskStatus,
    ToolCall,
    ToolKind,
    ToolResult,
    ToolSpec,
)
from qfm.agent.permissions import (
    OPEN_POLICY,
    READ_ONLY_POLICY,
    PermissionDecision,
    PermissionGate,
    PermissionPolicy,
)
from qfm.agent.planner import LLMPlanner, PlanRequest, Planner, RulePlanner
from qfm.agent.providers import (
    LocalCacheProvider,
    PanelBundle,
    SyntheticProvider,
    resolve_provider,
)
from qfm.agent.registry import ToolContext, ToolRegistry
from qfm.agent.reporting import render_findings_markdown, render_run_markdown
from qfm.agent.retry import Attempt, RetryOutcome, RetryPolicy, call_with_retry
from qfm.agent.runtime import AgentConfig, AgentRuntime, Session
from qfm.agent.tools import build_default_registry

__version__ = "1.0.0"

__all__ = [
    "__version__",
    # 运行时
    "AgentRuntime",
    "AgentConfig",
    "Session",
    # 规划
    "Planner",
    "RulePlanner",
    "LLMPlanner",
    "PlanRequest",
    # 工具
    "ToolRegistry",
    "ToolContext",
    "build_default_registry",
    # 权限
    "PermissionPolicy",
    "PermissionGate",
    "PermissionDecision",
    "READ_ONLY_POLICY",
    "OPEN_POLICY",
    # 重试
    "RetryPolicy",
    "RetryOutcome",
    "Attempt",
    "call_with_retry",
    # 异常
    "AgentError",
    "TransientError",
    "PermanentError",
    "DegradedResultError",
    "PermissionDeniedError",
    "BudgetExceededError",
    "ToolNotFoundError",
    "InvalidArgumentsError",
    # 上下文
    "ContextBudget",
    "ContextManager",
    # 数据
    "resolve_provider",
    "PanelBundle",
    "LocalCacheProvider",
    "SyntheticProvider",
    # 模型
    "AgentRun",
    "TaskRecord",
    "TaskStatus",
    "RunStatus",
    "ToolSpec",
    "ToolCall",
    "ToolResult",
    "ToolKind",
    "Plan",
    "PlanStep",
    # LLM
    "LLMClient",
    "NullClient",
    "OpenAICompatibleClient",
    "build_llm_client",
    # 报告
    "render_run_markdown",
    "render_findings_markdown",
]
