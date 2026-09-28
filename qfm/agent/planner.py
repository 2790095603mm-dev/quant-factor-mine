"""规划器：把自然语言研究请求翻译成工具调用序列。

## 为什么默认是规则规划器

一个可复现的 Demo 不能依赖「此刻 API 配额是否可用」。所以：

- `RulePlanner` 是**默认**且**始终可用**的——它把「分析最近一年银行股低估值因子
  的表现」这类请求确定性地解析成计划，零外网依赖。面试官 clone 下来就能跑。
- `LLMPlanner` 是**可选增强**——配置了 `QFM_AGENT_LLM_*` 环境变量才启用，
  用真实的模型做规划；任何失败（无网络、超时、输出不是合法 JSON、工具名不存在）
  都自动回落到 `RulePlanner`，并在计划里留下 `notes` 说明回落原因。

这个取舍写在代码里而不只是文档里：**Agent 架构的价值不取决于规划器是不是 LLM**。
工具契约、权限门、重试、上下文管理、异常驱动的重新规划——这些才是可复用的骨架，
规划器只是可插拔的一层。

## 规则规划器怎么做到「看起来像理解」

它并不真的理解语言，而是把请求拆成四个**可验证的槽位**，逐槽位解析：

| 槽位 | 解析方式 | 失败时 |
| --- | --- | --- |
| 股票池 | 行业关键词 → `qfm.agent.vocab.match_industry`；指数说法 → `resolve_pool` | 回落全市场 |
| 区间 | `parse_period` 解析「最近一年」「2023年到2025年」 | 回落最近三年 |
| 因子 | `match_factor_themes` 匹配主题词与因子名 | 回落价值族（`bp`/`ep_ttm`） |
| 动作 | 关键词判断是否需要对比、审计、多空 | 默认单因子完整检验 |

每个槽位的解析结果都会写进计划的 `notes`，所以「Agent 为什么算了这个」是可追溯的。
未识别的部分不会被悄悄补全成默认值——而是明确记录「未识别，按默认处理」。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from qfm.agent.models import Plan, PlanStep
from qfm.agent.vocab import match_factor_themes, parse_period, resolve_pool

__all__ = [
    "Planner",
    "RulePlanner",
    "LLMPlanner",
    "PlanRequest",
    "DEFAULT_FACTORS",
    "BUILD_WIDE_FALLBACK",
    "DEFAULT_POOL",
]

#: 未识别到任何因子主题时的默认因子（价值族，最常见的低估值研究入口）。
DEFAULT_FACTORS = ("bp", "ep_ttm")

#: 需要宽基对照时使用的股票池。
BUILD_WIDE_FALLBACK = "index800"

#: 请求未指定股票池时的默认池，与 BUILD_WIDE_FALLBACK 保持同一个值。
DEFAULT_POOL = "index800"

#: 触发「多空研究模拟」的表述。
_LONG_SHORT_HINTS = ("多空", "对冲", "long_short", "long-short")
#: 触发「因子对比」的表述。
_COMPARE_HINTS = ("对比", "比较", "哪个好", "对照", "vs", "VS")
#: 触发「换手/中性化」加码设置的表述。
_NEUTRAL_HINTS = ("中性", "剥离", "行业暴露", "风格暴露", "正交")


@dataclass
class PlanRequest:
    """规划器的输入。

    `as_of` 是**必需**的：相对区间（「最近一年」）必须先知道数据到哪天，
    才能落成具体日期。由调用方在规划前向 provider 索取（`DataProvider.as_of()`），
    这样计划里的日期始终是确定值，而不是「运行到哪算哪」。
    """

    text: str
    as_of: str
    pool: str | None = None
    factors: tuple[str, ...] = ()
    universe_symbols: tuple[str, ...] = ()
    max_stocks: int | None = None
    horizon: int = 20
    top_n: int = 30


class Planner(Protocol):
    """规划器接口。"""

    name: str

    def plan(self, request: PlanRequest, catalog: str) -> Plan: ...


class RulePlanner:
    """确定性规则规划器：零外网依赖，始终可用。"""

    name = "rule"

    def __init__(self, wide_pool: str = BUILD_WIDE_FALLBACK) -> None:
        self.wide_pool = wide_pool

    def parse(self, request: PlanRequest) -> dict[str, Any]:
        """把请求解析成槽位字典。独立出来便于单独测试与展示。"""
        text = request.text.strip()
        as_of = request.as_of

        # ---- 股票池槽位 ----
        keyword = None
        if not request.pool:
            keyword = _industry_keyword(text)
        pool = request.pool or resolve_pool(text)
        if request.universe_symbols:
            pool = pool or "explicit"
            keyword = None

        # ---- 区间槽位 ----
        start, end, period_label = parse_period(text, as_of)
        if not start:
            start, end, period_label = parse_period("最近三年", as_of)

        # ---- 因子槽位 ----
        themes, explicit = match_factor_themes(text)
        if request.factors:
            factors = list(request.factors)
            factor_source = "显式指定"
        elif explicit:
            factors = explicit[:3]
            factor_source = "从请求中识别"
        else:
            factors = list(DEFAULT_FACTORS)
            factor_source = "未识别到因子主题，按价值族默认"

        # ---- 动作槽位 ----
        mode = "long_short" if any(hint in text for hint in _LONG_SHORT_HINTS) else "long_only"
        want_compare = any(hint in text for hint in _COMPARE_HINTS) and len(factors) > 1
        want_neutral = any(hint in text for hint in _NEUTRAL_HINTS)

        notes: list[str] = []
        if keyword:
            notes.append(f"行业关键词：{keyword!r}")
        if pool:
            notes.append(f"股票池：{pool}")
        if request.universe_symbols:
            notes.append(f"股票池为显式指定的 {len(request.universe_symbols)} 只标的")
        notes.append(f"区间：{period_label or '未指定，按数据最近三年'}")
        if themes:
            notes.append(f"因子主题：{'、'.join(themes)}")
        notes.append(f"因子：{', '.join(factors)}（{factor_source}）")
        if mode == "long_short":
            notes.append("检测到「多空/对冲」表述，按多空研究模拟配置")
        if want_compare:
            notes.append("检测到「对比」表述，追加因子对比步骤")
        if want_neutral:
            notes.append("检测到「中性化」表述，信号做行业+市值中性")

        return {
            "text": text,
            "keyword": keyword,
            "pool": pool,
            "start": start,
            "end": end,
            "period_label": period_label,
            "factors": factors,
            "themes": themes,
            "mode": mode,
            "want_compare": want_compare,
            "want_neutral": want_neutral,
            "notes": notes,
        }

    def plan(self, request: PlanRequest, catalog: str) -> Plan:
        slots = self.parse(request)
        steps = self._steps(slots, request)
        return Plan(
            request=request.text,
            steps=steps,
            source=self.name,
            notes=tuple(slots["notes"]),
            round=0,
        )

    def _steps(self, slots: dict[str, Any], request: PlanRequest) -> list[PlanStep]:
        factors: list[str] = slots["factors"]
        primary = factors[0]
        neutralization = "industry_size" if slots["want_neutral"] else "none"
        decay = 5 if slots["want_neutral"] else 0

        steps: list[PlanStep] = []

        # ① 解析股票池
        if request.universe_symbols:
            # 标的列表由调用方通过上下文预设（preset_universe），
            # 不写进计划参数里——否则 800 个代码会灌满上下文。
            universe_args: dict[str, Any] = {}
        elif slots["keyword"]:
            universe_args = {"keyword": slots["keyword"], "max_stocks": request.max_stocks}
        else:
            # 未提到股票池时回落 index800（沪深300+中证500）：800 只足以跨过
            # 分层检验的 100 只门槛，又比全市场（A 股 5000+ 只）快一个数量级。
            universe_args = {
                "pool": slots["pool"] or DEFAULT_POOL,
                "max_stocks": request.max_stocks,
            }
        steps.append(PlanStep(
            tool="resolve_universe",
            arguments=universe_args,
            goal="把研究请求里的股票池说法解析成具体标的",
            on_failure="abort",
        ))

        # ② 取数 + 数据体检
        steps.append(PlanStep(
            tool="load_panel",
            arguments={
                "max_stocks": request.max_stocks,
                # 把分析窗口交给取数步骤，让它裁剪掉几十年的无关历史：
                # 既避免估值类因子的覆盖率被历史区间稀释而误判为数据缺失，
                # 也把逐日截面计算量降低一个数量级。
                "start": slots["start"],
                "end": slots["end"],
            },
            goal="加载行情、财务与行业数据，裁剪到分析窗口并核对覆盖率",
            on_failure="abort",
        ))
        steps.append(PlanStep(
            tool="describe_panel",
            arguments={},
            goal="数据体检：确认截面宽度与财务字段可用性",
            on_failure="continue",
        ))

        # ③ 因子发现（仅当因子是默认推断出来的，才值得先列一遍库）
        if len(factors) == 1 and factors == list(DEFAULT_FACTORS)[:1]:
            steps.append(PlanStep(
                tool="list_factors",
                arguments={"family": "价值", "limit": 10},
                goal="列出价值族可选因子，确认研究对象",
                on_failure="skip",
                optional=True,
            ))

        # ④ 逐因子：定义 → 计算 → 实验
        for factor in factors:
            steps.append(PlanStep(
                tool="describe_factor",
                arguments={"name": factor},
                goal=f"确认 {factor} 的定义方向与版本",
                on_failure="abort",
            ))
            steps.append(PlanStep(
                tool="compute_factor",
                arguments={"name": factor},
                goal=f"计算 {factor} 因子值并核对覆盖率",
                on_failure="abort",
            ))
            steps.append(PlanStep(
                tool="run_experiment",
                arguments={
                    "name": factor,
                    "start": slots["start"],
                    "end": slots["end"],
                    "horizon": request.horizon,
                    "top_n": request.top_n,
                    "mode": slots["mode"],
                    "neutralization": neutralization,
                    "decay": decay,
                },
                goal=f"运行 {factor} 的 IC/分层检验与成本后组合回测",
                on_failure="abort" if factor == primary else "continue",
            ))

        # ⑤ 异常检查（必须在实验之后，且是重新规划的触发点）
        steps.append(PlanStep(
            tool="check_anomalies",
            arguments={},
            goal="检查样本宽度、IC 显著性、换手与单调性等异常",
            on_failure="continue",
        ))

        # ⑥ 无未来函数审计（可选，失败不影响主结论）
        steps.append(PlanStep(
            tool="audit_lookahead",
            arguments={"name": primary},
            goal="静态审计因子实现是否引入未来函数",
            on_failure="skip",
            optional=True,
        ))

        # ⑦ 多因子对比（仅当请求提到对比）
        if slots["want_compare"]:
            steps.append(PlanStep(
                tool="compare_factors",
                arguments={"names": factors},
                goal="对比候选因子并检查是否互相重复",
                on_failure="skip",
                optional=True,
            ))

        # ⑧ 报告
        steps.append(PlanStep(
            tool="generate_report",
            arguments={"title": _report_title(slots, request)},
            goal="把研究结论与执行轨迹写成 Markdown 报告",
            on_failure="continue",
        ))
        return steps


class LLMPlanner:
    """LLM 规划器：可用时用模型规划，任何失败都回落到规则规划器。

    回落不是为了「优雅降级」好看，而是为了让 Agent 的输出**始终是可执行计划**。
    一个规划器如果可能返回不可执行的结果，上层就必须再写一层校验；
    这里把校验收在规划器内部：解析出的工具名必须在 `catalog` 里出现，
    否则整份计划作废、回落到规则规划。
    """

    name = "llm"

    def __init__(self, client, fallback: Planner | None = None, max_tokens: int = 1200) -> None:
        self.client = client
        self.fallback = fallback or RulePlanner()
        # 提示词里要给出「本地槽位解析结果」，必须与回落计划来自同一个实例，
        # 否则两处对同一请求的解析可能不一致。
        self.rule = self.fallback if isinstance(self.fallback, RulePlanner) else RulePlanner()
        self.max_tokens = max_tokens
        self.last_fallback_reason = ""

    def plan(self, request: PlanRequest, catalog: str) -> Plan:
        rule_plan = self.fallback.plan(request, catalog)
        prompt = self._prompt(request, catalog, rule_plan)
        try:
            raw = self.client.complete(
                [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.max_tokens,
            )
            steps = self._parse_steps(raw, catalog)
        except Exception as exc:  # noqa: BLE001 - 规划失败必须回落，不能中断研究
            self.last_fallback_reason = f"{type(exc).__name__}: {exc}"
            return self._fallback_plan(rule_plan, self.last_fallback_reason)

        if not steps:
            self.last_fallback_reason = "LLM 未返回可执行的步骤"
            return self._fallback_plan(rule_plan, self.last_fallback_reason)

        return Plan(
            request=request.text,
            steps=steps,
            source=self.name,
            notes=(
                f"由 {getattr(self.client, 'name', 'llm')} 规划，共 {len(steps)} 步",
                "规则规划器同时产出了一份计划作为对照",
            ),
            round=0,
        )

    def _fallback_plan(self, plan: Plan, reason: str) -> Plan:
        return Plan(
            request=plan.request,
            steps=list(plan.steps),
            source=f"{self.name}→{self.fallback.name}",
            notes=tuple(plan.notes) + (f"LLM 规划不可用，已回落到规则规划：{reason}",),
            round=plan.round,
        )

    def _prompt(self, request: PlanRequest, catalog: str, reference: Plan) -> str:
        slots = self.rule.parse(request)
        reference_lines = "\n".join(
            f"{index}. {step.tool}({json.dumps(dict(step.arguments), ensure_ascii=False)})"
            for index, step in enumerate(reference.steps, start=1)
        )
        return "\n".join([
            f"研究请求：{request.text}",
            f"数据可用区间终点：{request.as_of or '未知'}",
            f"本地解析到的槽位：{json.dumps({k: v for k, v in slots.items() if k != 'notes'}, ensure_ascii=False, default=str)}",
            "",
            catalog,
            "",
            "规则规划器给出的参考计划（可以改进它，也可以直接采用）：",
            reference_lines,
            "",
            "请输出 JSON：{\"steps\": [{\"tool\": \"工具名\", \"arguments\": {...},"
            " \"goal\": \"这一步的目的\", \"on_failure\": \"abort|skip|continue\"}]}",
            "约束：只使用上面列出的工具名；arguments 的键必须是该工具声明的参数；"
            "start/end 使用 YYYY-MM-DD。",
        ])

    def _parse_steps(self, raw: str, catalog: str) -> list[PlanStep]:
        payload = _extract_json(raw)
        if not isinstance(payload, dict):
            raise ValueError("LLM 输出不是 JSON 对象")
        raw_steps = payload.get("steps")
        if not isinstance(raw_steps, list):
            raise ValueError("LLM 输出缺少 steps 列表")

        known = _known_tools(catalog)
        steps: list[PlanStep] = []
        for item in raw_steps:
            if not isinstance(item, dict):
                raise ValueError("steps 中存在非对象元素")
            tool = str(item.get("tool", "")).strip()
            if tool not in known:
                raise ValueError(f"LLM 计划引用了不存在的工具 {tool!r}")
            arguments = item.get("arguments") or {}
            if not isinstance(arguments, dict):
                raise ValueError(f"{tool} 的 arguments 不是对象")
            steps.append(PlanStep(
                tool=tool,
                arguments=arguments,
                goal=str(item.get("goal", ""))[:120],
                on_failure=str(item.get("on_failure", "continue")),
            ))
        return steps


_SYSTEM_PROMPT = (
    "你是一个量化研究助理的计划生成器。你的任务是把研究员的一句话请求，"
    "翻译成对现有量化平台工具的调用序列。你只输出 JSON，不要解释。"
    "计划要覆盖：确定股票池 → 取数 → 计算因子 → 运行实验 → 检查异常 → 生成报告。"
)


def _known_tools(catalog: str) -> set[str]:
    return set(re.findall(r"^- (\w+)\(", catalog, flags=re.MULTILINE))


def _extract_json(raw: str) -> Any:
    """从模型输出里抽出 JSON：容忍 ```json 代码围栏与前后解释文字。"""
    text = (raw or "").strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("输出中找不到合法 JSON")


def _industry_keyword(text: str) -> str | None:
    """从文本里找出行业关键词。

    先看词表里有没有直接出现的术语（银行、白酒、半导体…），命中即返回；
    否则尝试「XX股 / XX板块 / XX行业」这种显式句式。
    """
    from qfm.agent.vocab import SECTOR_ALIASES, strip_suffix

    for term in sorted(SECTOR_ALIASES, key=len, reverse=True):
        if term in text:
            return strip_suffix(term) or term

    match = re.search(r"([\u4e00-\u9fa5]{2,6}?)(?:股|板块|行业|赛道)", text)
    if match:
        candidate = match.group(1)
        # 排除「沪深300股」这类指数说法与常见非行业词
        if candidate in ("沪深", "中证", "全", "个", "这些", "那些", "该"):
            return None
        if resolve_pool(candidate) is not None:
            return None
        return candidate
    return None


def _report_title(slots: dict[str, Any], request: PlanRequest) -> str:
    label = slots.get("period_label") or "指定区间"
    target = slots.get("keyword") or slots.get("pool") or "全市场"
    return f"{label} {target} 因子表现研究"
