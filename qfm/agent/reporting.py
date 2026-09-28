"""报告渲染：把研究结果与执行轨迹写成 Markdown。

报告分两个层次，因为它们服务的读者不同：

- `render_findings_markdown` —— **研究结论**。给研究员/面试官看的：区间、
  指标、IC 显著性、异常检查、结论与局限。这份由 `generate_report` 工具在
  Agent 运行过程中产出。
- `render_run_markdown` —— **执行过程**。给工程评审看的：计划、每步工具调用
  与耗时、重试轨迹、权限事件、上下文压缩统计、产物清单。这份由运行结束时产出。

分开的理由：把「结论」和「过程」混在一份文档里，两个读者都得在一半内容里找自己
关心的部分。而且执行轨迹里有重试与权限拒绝这类信息，放在研究结论旁边会干扰判断。

## 两个必须坚持的诚实性约束

1. **数据来源必须显著标注。** 合成数据上的 IC 不是实证结论。报告开头就带
   `> ⚠️` 横幅，而不是在文末小字里提一句。
2. **局限必须显式列出。** 未通过的检验、被收窄的参数、被拒绝的工具调用，
   都要出现在「局限与未决项」里，而不是从报告里悄悄消失。
"""

from __future__ import annotations

from typing import Any, Iterable

__all__ = [
    "render_findings_markdown",
    "render_run_markdown",
    "format_metric",
    "SYNTHETIC_BANNER",
]

SYNTHETIC_BANNER = (
    "> ⚠️ **本报告基于合成面板，不是真实行情数据。**\n"
    "> 面板由 `qfm.agent.synthetic` 确定性生成，并人为植入了弱价值效应，"
    "仅用于演示 Agent 的研究流程（规划 → 取数 → 计算 → 检验 → 异常检查 → 报告）。\n"
    "> 其中的 IC、收益、回撤等数字**不构成任何实证结论**，"
    "不能用于推断真实 A 股市场上任何因子的表现。"
)

_SEVERITY_ICON = {"high": "🔴", "medium": "🟠", "low": "🟡"}


def format_metric(value: Any, kind: str = "number") -> str:
    """统一的指标格式化：None/NaN 显示为「—」，避免报告里出现 nan。

    kind 取值：``pct``（带符号百分比，用于收益）/ ``ratio``（无符号百分比，
    用于覆盖率、占比）/ ``ic`` / ``t`` / ``int`` / ``number``。
    """
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number or abs(number) == float("inf"):
        return "—"
    if kind == "pct":
        return f"{number:+.2%}"
    if kind == "ratio":
        return f"{number:.2%}"
    if kind == "ic":
        return f"{number:+.4f}"
    if kind == "int":
        return f"{int(round(number))}"
    if kind == "t":
        return f"{number:+.2f}"
    return f"{number:.4f}"

def render_findings_markdown(
    title: str,
    *,
    experiment: dict | None,
    primary: dict | None = None,
    anomalies: Iterable[dict] | None = None,
    primary_anomalies: Iterable[dict] | None = None,
    panel_meta: dict | None = None,
    compare: dict | None = None,
    include_trace: bool = True,
) -> str:
    """渲染研究结论报告。

    Args:
        experiment: 「当前」实验结果。
        primary: 主实验结果。补救轮跑过宽基对照时，`experiment` 是对照、
            `primary` 才是用户原本问的那个池子的结果。报告以**主实验**为正文，
            对照作为稳健性检验单独成节 —— 因为用户问的是前者。
        anomalies: 当前实验的异常。
        primary_anomalies: 主实验的异常。

    若 `primary` 给出且与 `experiment` 不同，报告会多出「宽基对照」一节。
    """
    control = experiment if (primary is not None and experiment is not primary) else None
    headline = primary or experiment
    # 主实验的异常优先；未单独给出时退回当前实验的异常。
    # 任一参数为 None 都不能直接 list()，否则报告渲染会抛 TypeError。
    if primary is not None and primary_anomalies is not None:
        headline_anomalies = list(primary_anomalies)
    else:
        headline_anomalies = list(anomalies or [])
    panel_meta = panel_meta or {}

    lines: list[str] = [f"# {title}", ""]
    if str(panel_meta.get("source")) == "synthetic" or str(
        (headline or {}).get("data_source")
    ) == "synthetic":
        lines += [SYNTHETIC_BANNER, ""]

    if not headline:
        lines += ["_本次运行没有产出实验结果。_", ""]
        return "\n".join(lines)

    window = headline.get("window") or {}
    lines += [
        "## 一、研究配置",
        "",
        f"- 因子：**{headline.get('label')}**（`{headline.get('name')}`，"
        f"家族 {headline.get('family')}，定义方向 {headline.get('direction')}）",
        f"- 股票池：{headline.get('pool') or panel_meta.get('pool', '—')}"
        f"（{headline.get('n_symbols')} 只标的）",
        f"- 区间：{window.get('start')} ~ {window.get('end')}"
        f"（{window.get('trading_days')} 个交易日）",
        f"- 组合：{headline.get('mode')}，持仓 {headline.get('top_n_effective')} 只，"
        f"调仓 {headline.get('rebalance')}，中性化 {headline.get('neutralization')}"
        + (f"，Decay {headline.get('decay')}" if headline.get("decay") else ""),
        f"- IC 前瞻：{headline.get('horizon')} 个交易日",
        "",
    ]
    if window.get("note"):
        lines += [f"> 注：{window['note']}", ""]
    if headline.get("cache_hit"):
        lines += ["_本次实验命中内容寻址缓存，未重复计算。_", ""]

    lines += _render_metrics_table(headline)
    lines += _render_layer_section(headline)

    yearly = headline.get("yearly") or []
    if yearly:
        lines += [
            "## 四、分年表现",
            "",
            "| 年份 | 收益 | IC | 最大回撤 | 夏普 |",
            "| --- | --- | --- | --- | --- |",
        ]
        for row in yearly:
            lines.append(
                f"| {format_metric(row.get('年份'), 'int')} | "
                f"{format_metric(row.get('收益'), 'pct')} | "
                f"{format_metric(row.get('IC'), 'ic')} | "
                f"{format_metric(row.get('最大回撤'), 'pct')} | "
                f"{format_metric(row.get('夏普'))} |"
            )
        lines.append("")

    if control:
        lines += _render_control_section(headline, control)

    if compare:
        lines += [
            "## 六、因子对比",
            "",
            "| 因子 | IC 均值 | RankIC | ICIR | 多空收益 | 换手 | 覆盖 | 稳定性 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in compare.get("table") or []:
            lines.append(
                f"| {row.get('label')}（`{row.get('name')}`） | "
                f"{format_metric(row.get('ic_mean'), 'ic')} | "
                f"{format_metric(row.get('rank_ic'), 'ic')} | "
                f"{format_metric(row.get('ic_ir'))} | "
                f"{format_metric(row.get('long_short_return'), 'pct')} | "
                f"{format_metric(row.get('turnover'), 'ratio')} | "
                f"{format_metric(row.get('coverage'), 'ratio')} | "
                f"{format_metric(row.get('stability'))} |"
            )
        lines.append("")
        pairs = compare.get("high_correlations") or []
        for item in pairs:
            lines.append(
                f"- 高相关：`{item.get('left')}` 与 `{item.get('right')}` "
                f"相关性 {format_metric(item.get('correlation'))}"
                "（两者可能重复表达同一信息，需择一或正交化）"
            )
        if pairs:
            lines.append("")

    lines += _render_anomalies(headline_anomalies, heading="异常检查（主实验）")
    if control:
        lines += _render_anomalies(
            list(anomalies or []),
            heading=f"异常检查（宽基对照：{control.get('pool')}）",
        )

    lines += ["## 结论与局限", ""]
    lines += _render_conclusion(headline, headline_anomalies, control)

    if include_trace:
        trace = headline.get("trace")
        if trace:
            lines += ["## 附：本次执行轨迹", "", trace, ""]
    return "\n".join(lines)


def _render_metrics_table(experiment: dict) -> list[str]:
    ic = experiment.get("ic") or {}
    metrics = experiment.get("metrics") or {}
    return [
        "## 二、核心指标",
        "",
        "| 维度 | 指标 | 数值 |",
        "| --- | --- | --- |",
        f"| 预测力 | IC 均值 | {format_metric(ic.get('ic_mean'), 'ic')} |",
        f"| 预测力 | IC_IR | {format_metric(ic.get('ic_ir'))} |",
        f"| 预测力 | IC t 值 | {format_metric(ic.get('ic_t'), 't')} |",
        f"| 预测力 | IC 为正占比 | {format_metric(ic.get('pos_ratio'), 'ratio')} |",
        f"| 预测力 | 有效天数 | {format_metric(ic.get('n_days'), 'int')} |",
        f"| 组合 | 年化收益 | {format_metric(metrics.get('年化收益'), 'pct')} |",
        f"| 组合 | 年化超额 | {format_metric(metrics.get('年化超额'), 'pct')} |",
        f"| 组合 | 夏普比率 | {format_metric(metrics.get('夏普比率'))} |",
        f"| 组合 | 最大回撤 | {format_metric(metrics.get('最大回撤'), 'pct')} |",
        f"| 组合 | 卡玛比率 | {format_metric(metrics.get('卡玛比率'))} |",
        f"| 交易 | 因子换手率 | {format_metric(experiment.get('turnover'), 'ratio')} |",
        f"| 交易 | 日均换手 | {format_metric(metrics.get('日均换手'), 'ratio')} |",
        f"| 交易 | 累计交易成本 | {format_metric(metrics.get('累计交易成本'), 'ratio')} |",
        f"| 质量 | 信号覆盖率 | {format_metric(experiment.get('coverage'), 'ratio')} |",
        f"| 质量 | 分层检验可用 | {'是' if experiment.get('layer_available') else '否（样本宽度不足）'} |",
        "",
    ]


def _render_layer_section(experiment: dict) -> list[str]:
    mono = experiment.get("monotonicity") or {}
    if not experiment.get("layer_available"):
        return [
            "## 三、分层与单调性",
            "",
            "**不可用。** 分层检验要求每日至少 100 只有效股票，"
            f"本次股票池为 {experiment.get('n_symbols')} 只，未达门槛。"
            "因此本报告只能依据 IC 判断因子的预测力，无法验证分组区分度。",
            "",
        ]
    return [
        "## 三、分层与单调性",
        "",
        f"- 单调性检验：{'通过' if mono.get('monotonic') else '未通过'}",
        f"- 顶底层收益差：{format_metric(mono.get('top_minus_bottom'), 'pct')}",
        f"- 层序相关系数：{format_metric(mono.get('corr'))}",
        "",
    ]


def _render_control_section(primary: dict, control: dict) -> list[str]:
    """主实验与宽基对照的并排比较表。"""
    p_ic, c_ic = primary.get("ic") or {}, control.get("ic") or {}
    p_m, c_m = primary.get("metrics") or {}, control.get("metrics") or {}
    lines = [
        f"## 五、宽基对照（{control.get('pool')}，{control.get('n_symbols')} 只）",
        "",
        "主实验的股票池截面偏窄，无法进行分层检验。为判断结论是否受样本宽度影响，"
        f"在同一区间、同一组合配置下改用 {control.get('pool')} 重跑了一次。"
        "两次实验**只有股票池不同**。",
        "",
        "| 指标 | 主实验（窄池） | 宽基对照 | 差异 |",
        "| --- | --- | --- | --- |",
    ]

    def row(label: str, left, right, kind: str) -> str:
        delta = "—"
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if left == left and right == right:
                delta = format_metric(right - left, kind)
        return f"| {label} | {format_metric(left, kind)} | {format_metric(right, kind)} | {delta} |"

    lines.append(f"| 股票池标的数 | {primary.get('n_symbols')} | {control.get('n_symbols')} | — |")
    lines.append(row("IC 均值", p_ic.get("ic_mean"), c_ic.get("ic_mean"), "ic"))
    lines.append(row("IC t 值", p_ic.get("ic_t"), c_ic.get("ic_t"), "t"))
    lines.append(row("IC 为正占比", p_ic.get("pos_ratio"), c_ic.get("pos_ratio"), "ratio"))
    lines.append(row("年化收益", p_m.get("年化收益"), c_m.get("年化收益"), "pct"))
    lines.append(row("年化超额", p_m.get("年化超额"), c_m.get("年化超额"), "pct"))
    lines.append(row("夏普比率", p_m.get("夏普比率"), c_m.get("夏普比率"), "number"))
    lines.append(row("最大回撤", p_m.get("最大回撤"), c_m.get("最大回撤"), "pct"))
    lines.append(
        f"| 分层检验可用 | {'是' if primary.get('layer_available') else '否'} | "
        f"{'是' if control.get('layer_available') else '否'} | — |"
    )
    lines.append("")

    p_mean = _num(p_ic.get("ic_mean"))
    c_mean = _num(c_ic.get("ic_mean"))
    if p_mean is not None and c_mean is not None:
        if p_mean * c_mean > 0:
            lines += [
                "**对照结论：** 两次实验的 IC 方向一致，说明该因子在窄池上的表现"
                "不是样本宽度造成的假象；但数值差异反映了行业构成的影响。",
                "",
            ]
        else:
            lines += [
                "**对照结论：** 两次实验的 IC **方向相反**。这说明窄池上的结论"
                "主要由行业构成驱动，不能推广到更宽的市场范围。",
                "",
            ]
    return lines


def _render_anomalies(anomalies: list[dict], heading: str = "异常检查") -> list[str]:
    if not anomalies:
        return [f"## {heading}", "", "未发现样本宽度、显著性、方向、换手、单调性或回撤异常。", ""]
    lines = [f"## {heading}", ""]
    for item in anomalies:
        icon = _SEVERITY_ICON.get(item.get("severity", "low"), "•")
        lines.append(f"### {icon} {item.get('code')}（{item.get('severity')}）")
        lines.append("")
        lines.append(str(item.get("message", "")))
        lines.append("")
        evidence = item.get("evidence") or {}
        if evidence:
            pairs = ", ".join(f"`{key}={value}`" for key, value in evidence.items())
            lines.append(f"- 证据：{pairs}")
        if item.get("remediation"):
            lines.append(f"- 建议：{item['remediation']}")
        lines.append("")
    return lines


def _render_conclusion(experiment: dict, anomalies: list[dict], control: dict | None) -> list[str]:
    """生成结论段落。

    结论必须**由证据推出**，而不是预设「因子有效」。IC 不显著或超额为负时，
    结论就应该说「在该区间该池内未观察到可靠的预测力」。
    """
    lines: list[str] = []
    label = experiment.get("label")
    name = experiment.get("name")
    ic = experiment.get("ic") or {}
    metrics = experiment.get("metrics") or {}
    ic_mean = _num(ic.get("ic_mean"))
    ic_t = _num(ic.get("ic_t"))
    excess = _num(metrics.get("年化超额"))
    pos_ratio = _num(ic.get("pos_ratio"))
    n_symbols = experiment.get("n_symbols") or 0

    if ic_mean is None or ic_t is None:
        return [f"- 未能得到 {label} 的有效 IC 统计，无法判断预测力。"]

    significant = abs(ic_t) >= 2.0
    if significant and ic_mean > 0:
        lines.append(
            f"- **预测力：观察到显著的正向预测力。** {label}（`{name}`）在"
            f"{experiment.get('pool') or '本次池'}上的 IC 均值 {ic_mean:+.4f}"
            f"（t={ic_t:+.2f}），IC 为正的交易日占 {pos_ratio:.1%}，方向与因子定义一致。"
        )
    elif significant and ic_mean < 0:
        lines.append(
            f"- **预测力：IC 显著为负（{ic_mean:+.4f}，t={ic_t:+.2f}）。** "
            "方向与因子定义相反，需先排查方向设置与风格暴露，不应直接采用。"
        )
    else:
        lines.append(
            f"- **预测力：未达显著水平。** IC 均值 {ic_mean:+.4f}（t={ic_t:+.2f}），"
            "绝对值小于 2，无法拒绝「无预测力」的原假设。"
        )

    if excess is not None:
        verb = "跑赢" if excess > 0 else "**未跑赢**"
        lines.append(f"- **组合表现：** 年化超额 {excess:+.2%}，组合{verb}等权基准。")
        if excess <= 0:
            lines.append(
                "  因子有排序能力不等于多头组合能创造相对价值；"
                "差异主要来自持仓集中度与基准构造。"
            )

    if control:
        control_ic = _num((control.get("ic") or {}).get("ic_mean"))
        if control_ic is not None:
            same = control_ic * ic_mean > 0
            lines.append(
                f"- **稳健性（宽基对照）：** 在 {control.get('pool')}"
                f"（{control.get('n_symbols')} 只）上 IC 均值为 {control_ic:+.4f}，"
                + (
                    "方向与主实验一致。"
                    if same
                    else "**方向与主实验相反**，说明主实验结论受行业构成影响，不可外推。"
                )
            )

    lines.append("- **局限：**")
    if n_symbols < 100:
        lines.append(
            f"  - 主实验股票池仅 {n_symbols} 只，截面偏窄，IC 对个股异常表现更敏感；"
            "分层检验因样本不足无法进行。"
        )
    if not experiment.get("layer_available"):
        lines.append("  - 缺少分层单调性证据，无法排除「排序能力只集中在极端层」的可能。")
    if any(item.get("code") == "negative_excess" for item in anomalies):
        lines.append("  - 超额收益为负，说明该区间内按此因子构建的多头组合不具相对价值。")
    lines.append("  - 本结论仅覆盖所测区间，未做样本外验证，也未考虑涨跌停与流动性约束的真实冲击。")
    return lines


def render_run_markdown(run) -> str:
    """渲染执行过程报告（给工程评审看的）。"""
    lines: list[str] = [
        f"# Agent 执行记录 {run.run_id}",
        "",
        f"- 研究请求：`{run.request}`",
        f"- 状态：**{run.status.value}**",
        f"- 规划器：{run.planner}",
        f"- 数据来源：{run.data_source}（{run.data_note}）",
        f"- 开始：{run.created_at}　结束：{run.finished_at or '—'}",
        f"- 工具调用合计耗时：{run.duration_s:.2f}s",
        "",
    ]

    counts = run.counts()
    lines += [
        "## 任务状态汇总",
        "",
        "| 状态 | 数量 |",
        "| --- | --- |",
    ]
    for status in ("SUCCESS", "RETRYING", "FAILED", "DENIED", "SKIPPED", "PENDING", "RUNNING"):
        if status in counts:
            lines.append(f"| {status} | {counts[status]} |")
    lines.append("")

    for plan in run.plans:
        lines += [
            f"## 计划（第 {plan.round + 1} 轮，来源 {plan.source}）",
            "",
        ]
        for note in plan.notes:
            lines.append(f"> {note}")
        if plan.notes:
            lines.append("")
        lines += ["| # | 工具 | 目标 | 失败策略 |", "| --- | --- | --- | --- |"]
        for index, step in enumerate(plan.steps, start=1):
            lines.append(
                f"| {index} | `{step.tool}` | {step.goal or '—'} | "
                f"{step.on_failure}{'（可选）' if step.optional else ''} |"
            )
        lines.append("")

    lines += [
        "## 逐步执行轨迹",
        "",
        "| # | 工具 | 状态 | 尝试 | 耗时 | 缓存 | 结果摘要 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for task in run.tasks:
        icon = {
            "SUCCESS": "✅", "FAILED": "❌", "DENIED": "🚫",
            "SKIPPED": "⏭", "RETRYING": "🔁",
        }.get(task.status.value, "•")
        lines.append(
            f"| {task.index} | `{task.tool}` | {icon} {task.status.value} | "
            f"{task.attempts} | {task.duration_s:.2f}s | "
            f"{'是' if task.cache_hit else '否'} | {_escape(task.result_summary or task.error or '—')} |"
        )
    lines.append("")

    retried = [task for task in run.tasks if task.retry_log]
    if retried:
        lines += ["## 重试轨迹", ""]
        for task in retried:
            lines.append(f"### {task.index}. `{task.tool}` — {task.goal or ''}")
            lines.append("")
            lines.append("| 尝试 | 错误类型 | 分类 | 退避 | 是否重试 | 错误 |")
            lines.append("| --- | --- | --- | --- | --- | --- |")
            for item in task.retry_log:
                lines.append(
                    f"| {item.get('attempt')} | {item.get('error_type')} | "
                    f"{item.get('classification')} | {item.get('delay_s')}s | "
                    f"{'是' if item.get('will_retry') else '否'} | "
                    f"{_escape(str(item.get('error', ''))[:120])} |"
                )
            lines.append("")

    if run.permission_events:
        lines += ["## 权限事件", "", "| 工具 | 判定 | 规则 | 原因 |", "| --- | --- | --- | --- |"]
        for event in run.permission_events:
            lines.append(
                f"| `{event.get('tool')}` | {event.get('decision')} | "
                f"{event.get('rule', '—')} | {_escape(str(event.get('reason', '')))} |"
            )
        lines.append("")

    if run.context_stats:
        stats = run.context_stats
        lines += [
            "## 上下文管理",
            "",
            f"- 条目数：{stats.get('items')}",
            f"- 当前字符数：{stats.get('chars')}（原始 {stats.get('original_chars')}）",
            f"- 压缩比：{stats.get('compression_ratio')}",
            f"- 外置到 artifact：{stats.get('offloaded_items')} 条",
            f"- 折叠：{stats.get('folded_items')} 条",
            "",
        ]

    if run.artifacts:
        lines += ["## 产物", ""]
        lines += [f"- `{path}`" for path in run.artifacts]
        lines.append("")

    if run.warnings:
        lines += ["## 运行告警", ""]
        lines += [f"- {warning}" for warning in run.warnings]
        lines.append("")
    return "\n".join(lines)


def _num(value) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def _escape(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")
