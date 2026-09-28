"""工具集装配：把所有工具注册进一个注册表。

`build_default_registry()` 是唯一入口，工具按类别分模块，注册顺序无关紧要
（注册表内部按名字排序）。这样加新工具只需新增一个 `register_*_tools`，
不必改动运行时。
"""

from __future__ import annotations

from qfm.agent.registry import ToolRegistry
from qfm.agent.tools.data_tools import register_data_tools
from qfm.agent.tools.experiment_tools import register_experiment_tools
from qfm.agent.tools.factor_tools import register_factor_tools
from qfm.agent.tools.ledger_tools import register_ledger_tools

__all__ = ["build_default_registry", "EXPERIMENT_TOOLS", "DATA_TOOLS", "FACTOR_TOOLS"]

#: 各步骤会引用到的工具名，集中在这里避免拼写漂移。
DATA_TOOLS = ("resolve_universe", "load_panel", "describe_panel")
FACTOR_TOOLS = ("list_factors", "describe_factor", "compute_factor", "audit_lookahead")
EXPERIMENT_TOOLS = ("run_experiment", "check_anomalies", "compare_factors", "generate_report")


def build_default_registry() -> ToolRegistry:
    """构建包含全部内置工具的注册表。"""
    registry = ToolRegistry()
    register_data_tools(registry)
    register_factor_tools(registry)
    register_experiment_tools(registry)
    register_ledger_tools(registry)
    return registry
