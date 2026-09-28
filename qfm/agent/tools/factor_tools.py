"""因子类工具：列因子、查因子定义、计算因子值、无未来函数审计。

## 上下文管理在这里的体现

`list_factors` 返回 49 个因子的元数据（几 KB），适合进上下文；
`compute_factor` 返回 2000 × 800 的浮点矩阵（内存里 10 MB 量级），
**绝不能**进上下文。所以它走 `(payload, summary)` 约定：

- 完整矩阵留在 `ToolContext`（键 `factor:<name>`），供 `run_experiment` 复用，
  避免重复计算；
- 上下文只拿到覆盖率、截面宽度、缺失比例这些**用于判断下一步是否可行**的统计量。

## 无未来函数审计为什么可以复用现成实现

`qfm/pipeline/lookahead.py` 的 `scan_source` 是纯正则静态扫描，永不抛异常，
返回 `{leaks, warnings}`。它本来是被 Streamlit 自定义因子页调用的；Agent 直接包一层
即可获得同样的能力。审计失败被设计成 `on_failure="skip"` 的可选步骤——
审计工具本身出问题不应该让整个研究请求失败，但审计结论必须如实进报告。
"""

from __future__ import annotations

import inspect
import sys

import pandas as pd

from qfm.agent.errors import InvalidArgumentsError, PermanentError
from qfm.agent.models import ToolKind
from qfm.agent.registry import ToolContext

__all__ = ["register_factor_tools"]


def register_factor_tools(registry) -> None:
    """把因子类工具注册进 `registry`。"""

    @registry.register(
        "list_factors",
        kind=ToolKind.READ,
        description="列出因子库中的因子及其家族、方向、中文名。用于在不知道具体因子名时先摸清可用选项",
        parameters={
            "family": "str，按家族过滤，如「价值」「动量」",
            "keyword": "str，按名称/中文名/描述模糊过滤",
            "limit": "int，最多返回多少个（默认 30）",
        },
        required=(),
        returns="payload: {n_total, n_returned, families, factors:[{name, label, family, direction, description}]}",
        cost=0.5,
        tags=("factor", "discovery"),
    )
    def list_factors(
        context: ToolContext,
        family: str | None = None,
        keyword: str | None = None,
        limit: int = 30,
    ):
        from qfm.factors import list_factors as all_factors

        factors = all_factors(family) if family else all_factors()
        if keyword:
            needle = keyword.lower()
            factors = [
                factor
                for factor in factors
                if needle in factor.name.lower()
                or needle in (factor.description or "").lower()
                or needle in _label(factor.name).lower()
                or needle in factor.family
            ]
        total = len(factors)
        shown = factors[: max(1, int(limit))]
        payload = {
            "n_total": total,
            "n_returned": len(shown),
            "families": sorted({factor.family for factor in all_factors()}),
            "factors": [
                {
                    "name": factor.name,
                    "label": _label(factor.name),
                    "family": factor.family,
                    "direction": factor.direction,
                    "description": factor.description,
                }
                for factor in shown
            ],
        }
        summary = (
            f"因子库共 {payload['n_total']} 个"
            + (f"（过滤后 {total} 个）" if (family or keyword) else "")
            + f"；返回 {len(shown)} 个: "
            + ", ".join(f"{item['label']}({item['name']})" for item in payload["factors"][:6])
        )
        return payload, summary

    @registry.register(
        "describe_factor",
        kind=ToolKind.READ,
        description="查看单个因子定义：家族、定义方向、公式、标签、版本与定义指纹",
        parameters={"name": "str，因子名，如 bp"},
        required=("name",),
        returns="payload: {name, label, family, direction, description, formula, tags, version, source_hash, params}",
        cost=0.3,
        tags=("factor",),
    )
    def describe_factor(context: ToolContext, name: str):
        factor = _require_factor(name)
        payload = {
            "name": factor.name,
            "label": _label(factor.name),
            "family": factor.family,
            "direction": factor.direction,
            "description": factor.description,
            "formula": factor.formula,
            "tags": list(factor.tags),
            "version": factor.version,
            "source_hash": factor.source_hash,
            "params": dict(factor.params or {}),
            "version_index": {"name": factor.name, "version": factor.version,
                              "source_hash": factor.source_hash},
        }
        summary = (
            f"{payload['label']}({name}) 家族={factor.family} 方向={factor.direction} "
            f"版本={factor.version}；{factor.description}"
        )
        return payload, summary

    @registry.register(
        "compute_factor",
        kind=ToolKind.COMPUTE,
        description="在已加载面板上计算因子值（date×stock 矩阵），返回覆盖率等统计量；矩阵本身留在运行上下文，不进对话上下文",
        parameters={
            "name": "str，因子名，如 bp",
            "min_coverage": "float，因子覆盖率下限，低于则重试（默认 0.30）",
        },
        required=("name",),
        returns="payload: {name, n_days, n_symbols, coverage, mean, std, latest_valid, layer_test_available}",
        cost=1.5,
        tags=("factor",),
    )
    def compute_factor(context: ToolContext, name: str, min_coverage: float | None = None):
        factor = _require_factor(name)
        panel = context.panel
        if panel is None or getattr(panel.close, "empty", True):
            raise InvalidArgumentsError("尚未加载面板；请先执行 load_panel")

        values = factor.func(panel)
        if not isinstance(values, pd.DataFrame):
            raise PermanentError(f"因子 {name} 返回了 {type(values).__name__}，应为 DataFrame")
        values = values.reindex(index=panel.close.index, columns=panel.close.columns)

        coverage = float(values.notna().to_numpy().mean())
        floor = 0.30 if min_coverage is None else float(min_coverage)
        if floor > 0 and coverage < floor:
            raise _degraded(
                name, coverage, floor, panel
            )

        latest = values.iloc[-1].dropna()
        context.put(f"factor:{name}", values)
        cross_section = float(values.notna().sum(axis=1).mean())
        payload = {
            "name": name,
            "n_days": int(values.shape[0]),
            "n_symbols": int(values.shape[1]),
            "coverage": round(coverage, 4),
            "mean": _finite(float(values.stack().mean()) if coverage else float("nan")),
            "std": _finite(float(values.stack().std()) if coverage else float("nan")),
            "latest_valid": int(latest.size),
            "avg_cross_section": round(cross_section, 1),
            "layer_test_available": bool(cross_section >= 100),
        }
        summary = (
            f"{_label(name)}({name}) 已计算：{payload['n_days']}×{payload['n_symbols']}，"
            f"覆盖率 {coverage:.1%}，均值 {payload['mean'] if payload['mean'] is None else format(payload['mean'], '.4f')}，"
            f"平均截面 {payload['avg_cross_section']:.0f} 只"
        )
        return payload, summary

    @registry.register(
        "audit_lookahead",
        kind=ToolKind.READ,
        description="对因子实现做静态无未来函数审计（扫描负向 shift/rolling、未来收益字段等泄漏模式）",
        parameters={"name": "str，因子名；省略则审计整个因子注册模块"},
        required=(),
        returns="payload: {name, leaks, warnings, structural, verdict}",
        cost=0.5,
        tags=("factor", "audit"),
    )
    def audit_lookahead(context: ToolContext, name: str | None = None):
        from qfm.pipeline.lookahead import check_structural, scan_source

        if name:
            factor = _require_factor(name)
            module = sys.modules.get(getattr(factor.func, "__module__", "") or "")
            source = inspect.getsource(module) if module is not None else ""
            target = name
        else:
            source = ""
            for module_name in ("qfm.factors.blogger", "qfm.factors.practical"):
                module = sys.modules.get(module_name)
                if module is not None:
                    source += inspect.getsource(module)
            target = "因子库模块"
        if not source:
            raise PermanentError(f"无法获取 {target} 的源码用于审计")

        result = scan_source(source)
        leaks = list(result.get("leaks") or [])
        warnings = list(result.get("warnings") or [])
        payload = {
            "name": target,
            "leaks": leaks,
            "warnings": warnings,
            "structural": check_structural("factor"),
            "verdict": "命中泄漏模式" if leaks else ("有提示" if warnings else "通过"),
        }
        summary = (
            f"{target} 无未来函数审计：{payload['verdict']}"
            + (f"；泄漏 {len(leaks)} 处" if leaks else "")
            + (f"；提示 {len(warnings)} 处" if warnings else "")
        )
        return payload, summary


def _require_factor(name: str):
    from qfm.factors import get_factor

    factor = get_factor(name)
    if factor is None:
        from qfm.factors import list_factors as all_factors

        near = [item.name for item in all_factors() if name.lower() in item.name.lower()][:5]
        hint = f"；相近的因子: {near}" if near else ""
        raise PermanentError(f"未知因子 {name!r}{hint}")
    return factor


def _label(name: str) -> str:
    from qfm.factors import factor_label

    return factor_label(name)


def _finite(value: float) -> float | None:
    return value if value == value and abs(value) != float("inf") else None


def _degraded(name: str, coverage: float, floor: float, panel):
    """构造因子覆盖率过低的降级异常，附带诊断信息。"""
    from qfm.agent.errors import DegradedResultError

    available = sorted((getattr(panel, "fund", {}) or {}).keys())
    return DegradedResultError(
        f"因子 {name} 覆盖率仅 {coverage:.1%}，低于下限 {floor:.0%}。"
        f"该因子可能依赖面板中缺失的财务字段；当前可用财务字段: {available or '无'}"
    )
