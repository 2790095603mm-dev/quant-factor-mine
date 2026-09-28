"""研究助理页面：把 Agent 的一次运行做成可交互、可复盘的界面。

## 页面设计的三个取向

**① 过程比结论更重要。** 这个页面存在的主要目的是让「Agent 做了什么」可见：
计划、每一步的状态/耗时/重试次数/缓存命中、权限拒绝、异常检查结果、上下文压缩
统计。只展示最终结论的话，读者无法判断结论可不可信。

**② 页面的数据规模不能取决于侧边栏的股票池。** 「研究助理」出现在不加载面板的
那一组导航里：Agent 自己解析股票池（可能是一个行业），侧边栏选的 `index800`
与它无关。否则用户想分析银行股却被强制先加载 800 只股票。

**③ 权限开关必须在界面上。** 只读模式、禁用工具、预算上限都是可点的控件而不是
配置文件里的选项——权限这件事如果不能当场演示，就很难让人相信它真的生效。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from qfm.agent.llm import build_llm_client
from qfm.agent.permissions import PermissionPolicy
from qfm.agent.planner import LLMPlanner, RulePlanner
from qfm.agent.providers import resolve_provider
from qfm.agent.retry import RetryPolicy
from qfm.agent.runtime import AgentConfig, AgentRuntime
from qfm.agent.tools import build_default_registry

EXAMPLES = (
    "分析最近一年银行股低估值因子的表现",
    "最近三年动量因子在中证500的表现",
    "对比最近两年白酒股的 roe 与 gross_margin 因子",
    "分析最近一年科技板块的波动率因子，做行业中性",
)

_STATUS_ICON = {
    "SUCCESS": "✅", "FAILED": "❌", "DENIED": "🚫",
    "SKIPPED": "⏭", "RETRYING": "🔁", "PENDING": "·", "RUNNING": "▶",
}
_SEVERITY_ICON = {"high": "🔴", "medium": "🟠", "low": "🟡"}


REQUEST_KEY = "agent_request_text"
_PENDING_KEY = "_agent_pending_request"
_AUTORUN_KEY = "_agent_autorun"


def render_agent_page(
    *,
    runs_root: str = "reports/agent_runs",
    data_dir: str = "data_cache",
    store=None,
    jobs=None,
    catalog_root=None,
) -> None:
    """渲染研究助理页面。"""
    # 必须先于 text_area 的创建：widget 一旦实例化，它的 session_state 键
    # 在本轮就不能再被修改（Streamlit 会抛异常），所以示例按钮走「待应用值」。
    pending = st.session_state.pop(_PENDING_KEY, None)
    if pending is not None:
        st.session_state[REQUEST_KEY] = pending

    st.markdown(
        '<div class="qfm-sig"><h1>研究助理</h1>'
        '<div class="sub">一句话研究请求 → 自动规划、取数、计算、检验、异常检查与报告</div></div>',
        unsafe_allow_html=True,
    )

    _render_intro()
    _render_examples()

    with st.form("agent_request"):
        request = st.text_area(
            "研究请求",
            value=st.session_state.get(REQUEST_KEY, EXAMPLES[0]),
            height=80,
            key=REQUEST_KEY,
            help="用自然语言描述：股票池（行业或指数）、区间、因子主题，例如「最近一年银行股低估值因子」。",
        )
        cols = st.columns(4)
        data_mode = cols[0].selectbox(
            "数据源", ["auto", "cache", "synthetic"], key="agent_data_mode",
            help="auto 优先本地缓存，缺失时回落到离线合成面板（会显著标注）。",
        )
        planner_mode = cols[1].selectbox(
            "规划器", ["auto", "rule", "llm"], key="agent_planner",
            help="rule 为确定性规则规划（离线可用）；llm 需配置 QFM_AGENT_LLM_* 环境变量。",
        )
        top_n = cols[2].number_input("持仓数", 1, 200, 30, key="agent_top_n")
        horizon = cols[3].number_input("IC 前瞻", 1, 120, 20, key="agent_horizon")

        with st.expander("权限与预算（可当场验证 Agent 如何适应拒绝）", expanded=False):
            pcols = st.columns(3)
            read_only = pcols[0].checkbox(
                "只读模式", key="agent_read_only",
                help="禁掉一切写盘工具（generate_report），研究结论仍会产出。",
            )
            max_calls = pcols[1].number_input("调用次数上限", 1, 200, 40, key="agent_max_calls")
            max_attempts = pcols[2].number_input("单步重试上限", 1, 6, 3, key="agent_max_attempts")
            denied = st.multiselect(
                "禁用工具",
                ["load_panel", "compute_factor", "run_experiment", "check_anomalies",
                 "generate_report", "compare_factors", "audit_lookahead"],
                key="agent_denied",
                help="选中的工具一律不可调用，用于观察 Agent 是否如实报告而不是编造结论。",
            )
            max_replan = st.number_input(
                "异常驱动的补救轮数上限", 0, 3, 1, key="agent_max_replan",
                help="1 表示：检查出截面过窄时自动补跑一次宽基对照。",
            )

        submitted = st.form_submit_button("▶ 运行研究", type="primary", use_container_width=True)

    autorun = bool(st.session_state.pop(_AUTORUN_KEY, False))
    _render_last_run_summary()

    if not (submitted or autorun):
        st.info(
            "填写研究请求后点击「运行研究」，或直接点上方示例一键运行。"
            "默认使用离线合成面板与确定性规则规划器，因此无需 API Key、无需外网即可复现完整流程。"
        )
        return

    text = (request or "").strip()
    if not text:
        st.warning("请先填写研究请求。")
        return

    _run_agent(
        text=text,
        data_mode=data_mode,
        planner_mode=planner_mode,
        top_n=int(top_n),
        horizon=int(horizon),
        read_only=bool(read_only),
        denied=tuple(denied),
        max_calls=int(max_calls),
        max_attempts=int(max_attempts),
        max_replan=int(max_replan),
        runs_root=runs_root,
        data_dir=data_dir,
        store=store,
        jobs=jobs,
        catalog_root=catalog_root,
    )


def _render_examples() -> None:
    """示例请求按钮：一键填入并直接运行。

    刻意放在 `st.form` **之外**：表单内的按钮无法安全地改写同一轮里已实例化的
    widget 状态，走「待应用值 + 重跑」既绕开了这个限制，也让示例变成一键可跑。
    """
    st.caption("示例请求（点击即运行）：")
    columns = st.columns(len(EXAMPLES))
    for index, (column, example) in enumerate(zip(columns, EXAMPLES)):
        if column.button(f"示例 {index + 1}", use_container_width=True, help=example):
            st.session_state[_PENDING_KEY] = example
            st.session_state[_AUTORUN_KEY] = True
            st.rerun()


# ---------------------------------------------------------------------------
# 执行
# ---------------------------------------------------------------------------
def _run_agent(
    *,
    text: str,
    data_mode: str,
    planner_mode: str,
    top_n: int,
    horizon: int,
    read_only: bool,
    denied: tuple[str, ...],
    max_calls: int,
    max_attempts: int,
    max_replan: int,
    runs_root: str,
    data_dir: str,
    store,
    jobs,
    catalog_root,
) -> None:
    try:
        provider = resolve_provider(data_mode, cache_dir=data_dir)
    except Exception as exc:  # noqa: BLE001 - 数据源不可用时给出可读提示
        st.error(f"数据源不可用：{exc}")
        return

    planner, planner_label = _build_planner(planner_mode)
    policy = PermissionPolicy(
        denied=frozenset(denied),
        read_only=read_only,
        max_total_calls=int(max_calls),
    )
    config = AgentConfig(
        retry=RetryPolicy(max_attempts=int(max_attempts)),
        permissions=policy,
        max_replan_rounds=int(max_replan),
    )

    st.caption(
        f"数据源：{getattr(provider, 'describe', lambda: provider.name)()}　·　"
        f"规划器：{planner_label}　·　"
        f"权限：{'只读，' if read_only else ''}"
        + (f"禁用 {sorted(denied)}，" if denied else "")
        + f"调用上限 {max_calls}"
    )

    timeline = st.container()
    runtime = AgentRuntime(
        build_default_registry(),
        planner,
        config,
        provider=provider,
        runs_root=runs_root,
        store=store,
        jobs=jobs,
        catalog_root=catalog_root,
        on_event=_stream_printer(timeline),
    )

    with st.status("运行研究中…", expanded=True) as status:
        run = runtime.run(
            text, factors=(), max_stocks=None, horizon=int(horizon), top_n=int(top_n)
        )
        status.update(
            label=f"运行结束：{run.status.value}（{run.counts()}）",
            state="complete" if run.status.value == "SUCCESS" else "error",
        )

    st.session_state["agent_last_run"] = run
    _render_run(run)


def _build_planner(mode: str):
    rule = RulePlanner()
    if mode == "rule":
        return rule, "规则规划器（离线、确定性）"
    client = build_llm_client()
    if mode == "llm":
        if not client.available:
            st.warning("未配置 QFM_AGENT_LLM_*，已回落到规则规划器。")
            return rule, "规则规划器（LLM 未配置）"
        return LLMPlanner(client, fallback=rule), f"LLM 规划器（{client.config.model}）"
    if client.available:
        return LLMPlanner(client, fallback=rule), f"LLM 规划器（{client.config.model}）"
    return rule, "规则规划器（离线、确定性）"


def _stream_printer(container):
    """把运行事件实时写进页面。刻意只渲染「需要人注意」的事件。"""
    lines: list[str] = []

    def printer(event: str, payload: dict) -> None:
        if event == "planned":
            lines.append(
                f"**规划**（{payload.get('source')}，{payload.get('steps')} 步）"
            )
            for note in payload.get("notes") or []:
                lines.append(f"- {note}")
        elif event == "step_started":
            lines.append(f"{payload.get('index'):>2}. ▶ `{payload.get('tool')}` {payload.get('goal') or ''}")
        elif event == "step_finished":
            icon = _STATUS_ICON.get(payload.get("status", ""), "•")
            extra = []
            if (payload.get("attempts") or 1) > 1:
                extra.append(f"尝试 {payload['attempts']} 次")
            if payload.get("cache_hit"):
                extra.append("命中缓存")
            if payload.get("error"):
                extra.append(str(payload["error"])[:100])
            suffix = ("　—　" + "，".join(extra)) if extra else ""
            lines.append(
                f"　　{icon} {payload.get('status')}　{payload.get('duration_s', 0):.2f}s　"
                f"{payload.get('summary') or ''}{suffix}"
            )
        elif event == "step_retrying":
            lines.append(
                f"　　🔁 第 {payload.get('attempt')} 次失败（{payload.get('classification')}）："
                f"{str(payload.get('error'))[:110]}　{payload.get('delay_s')}s 后重试"
            )
        elif event == "step_denied":
            lines.append(
                f"　　🚫 被权限门拒绝（{payload.get('rule')}）：{payload.get('reason')}"
            )
        elif event == "replanned":
            lines.append(f"\n**重新规划**（第 {payload.get('round')} 轮）　{payload.get('reason')}")
        container.markdown("\n\n".join(lines[-60:]))

    return printer


# ---------------------------------------------------------------------------
# 结果展示
# ---------------------------------------------------------------------------
def _render_run(run) -> None:
    st.divider()
    _render_verdict(run)
    _render_experiments(run)
    _render_trace(run)
    _render_anomalies(run)
    _render_process_details(run)
    _render_artifacts(run)


def _render_verdict(run) -> None:
    cols = st.columns(4)
    cols[0].metric("运行状态", run.status.value)
    cols[1].metric("任务数", sum(run.counts().values()))
    cols[2].metric("工具耗时", f"{run.duration_s:.1f}s")
    stats = run.context_stats or {}
    cols[3].metric(
        "上下文", f"{stats.get('chars', 0)} 字符",
        help=f"{stats.get('items', 0)} 条；外置 {stats.get('offloaded_items', 0)} 条；"
             f"折叠 {stats.get('folded_items', 0)} 条；"
             f"压缩比 {stats.get('compression_ratio', 1)}",
    )
    if run.warnings:
        for warning in run.warnings:
            st.warning(warning)


def _render_experiments(run) -> None:
    primary = run.findings.get("primary")
    control = run.findings.get("control")
    if not primary:
        st.error("本次运行没有产出实验结果。")
        return

    if control:
        st.markdown("#### 窄池主实验 vs 宽基对照")
        st.caption(
            "异常检查发现主实验截面过窄（无法做分层检验），已自动在同一区间、"
            f"同一组合配置下改用 {control.get('pool')} 重跑一次作为对照。"
        )
        st.dataframe(
            _comparison_frame(primary, control),
            use_container_width=True, hide_index=True,
        )
    st.markdown("#### 主实验指标")
    st.dataframe(_metrics_frame(primary), use_container_width=True, hide_index=True)

    yearly = primary.get("yearly") or []
    if yearly:
        with st.expander("分年表现", expanded=False):
            st.dataframe(pd.DataFrame(yearly), use_container_width=True, hide_index=True)


def _metrics_frame(experiment: dict) -> pd.DataFrame:
    ic = experiment.get("ic") or {}
    metrics = experiment.get("metrics") or {}

    def fmt(value, pattern="{:.4f}"):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "—"
        if number != number:
            return "—"
        return pattern.format(number)

    rows = [
        ("股票池", experiment.get("pool"), ""),
        ("标的数", experiment.get("n_symbols"), ""),
        ("区间", f"{experiment.get('window', {}).get('start')} ~ "
                 f"{experiment.get('window', {}).get('end')}", ""),
        ("IC 均值", fmt(ic.get("ic_mean"), "{:+.4f}"), ""),
        ("IC t 值", fmt(ic.get("ic_t"), "{:+.2f}"), ""),
        ("IC_IR", fmt(ic.get("ic_ir")), ""),
        ("IC 为正占比", fmt(ic.get("pos_ratio"), "{:.1%}"), ""),
        ("年化收益", fmt(metrics.get("年化收益"), "{:+.2%}"), ""),
        ("年化超额", fmt(metrics.get("年化超额"), "{:+.2%}"), ""),
        ("夏普比率", fmt(metrics.get("夏普比率")), ""),
        ("最大回撤", fmt(metrics.get("最大回撤"), "{:.2%}"), ""),
        ("因子换手率", fmt(experiment.get("turnover"), "{:.1%}"), ""),
        ("信号覆盖率", fmt(experiment.get("coverage"), "{:.1%}"), ""),
        ("分层检验", "可用" if experiment.get("layer_available") else "不可用（截面不足 100 只）", ""),
    ]
    return pd.DataFrame(rows, columns=["指标", "数值", ""]).drop(columns=[""])


def _comparison_frame(primary: dict, control: dict) -> pd.DataFrame:
    def get(payload, section, key):
        return (payload.get(section) or {}).get(key)

    rows = [
        ("标的数", primary.get("n_symbols"), control.get("n_symbols"), "{:.0f}"),
        ("IC 均值", get(primary, "ic", "ic_mean"), get(control, "ic", "ic_mean"), "{:+.4f}"),
        ("IC t 值", get(primary, "ic", "ic_t"), get(control, "ic", "ic_t"), "{:+.2f}"),
        ("年化收益", get(primary, "metrics", "年化收益"),
         get(control, "metrics", "年化收益"), "{:+.2%}"),
        ("年化超额", get(primary, "metrics", "年化超额"),
         get(control, "metrics", "年化超额"), "{:+.2%}"),
        ("夏普比率", get(primary, "metrics", "夏普比率"),
         get(control, "metrics", "夏普比率"), "{:.4f}"),
        ("最大回撤", get(primary, "metrics", "最大回撤"),
         get(control, "metrics", "最大回撤"), "{:.2%}"),
    ]
    data = []
    for label, left, right, pattern in rows:
        data.append({
            "指标": label,
            f"主实验（{primary.get('pool')}）": _fmt(left, pattern),
            f"对照（{control.get('pool')}）": _fmt(right, pattern),
        })
    return pd.DataFrame(data)


def _fmt(value, pattern: str) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if number != number:
        return "—"
    return pattern.format(number)


def _render_trace(run) -> None:
    st.markdown("#### 逐步执行轨迹")
    rows = []
    for task in run.tasks:
        rows.append({
            "步骤": task.index,
            "工具": task.tool,
            "目标": task.goal,
            "状态": f"{_STATUS_ICON.get(task.status.value, '•')} {task.status.value}",
            "尝试": task.attempts,
            "耗时(s)": round(task.duration_s, 2),
            "缓存": "是" if task.cache_hit else "",
            "结果 / 错误": (task.result_summary or task.error)[:150],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    retried = [task for task in run.tasks if task.retry_log]
    if retried:
        with st.expander(f"重试轨迹（{len(retried)} 个步骤发生重试）", expanded=True):
            for task in retried:
                st.markdown(f"**{task.index}. `{task.tool}`** — {task.goal}")
                st.dataframe(
                    pd.DataFrame(task.retry_log), use_container_width=True, hide_index=True
                )


def _render_anomalies(run) -> None:
    by_experiment = run.findings.get("anomalies_by_experiment") or {}
    if not by_experiment:
        return
    st.markdown("#### 异常检查")
    labels = list(by_experiment)
    key = st.selectbox(
        "查看哪个实验的异常", labels, key="agent_anomaly_key",
        format_func=lambda name: f"{name}（{len(by_experiment[name]['anomalies'])} 项）",
    )
    entry = by_experiment[key]
    anomalies = entry.get("anomalies") or []
    if not anomalies:
        st.success("未发现样本宽度、显著性、方向、换手、单调性或回撤异常。")
        return
    for item in anomalies:
        icon = _SEVERITY_ICON.get(item.get("severity"), "•")
        with st.expander(f"{icon} {item.get('code')}（{item.get('severity')}）", expanded=True):
            st.markdown(str(item.get("message")))
            evidence = item.get("evidence") or {}
            if evidence:
                st.caption("证据：" + "，".join(f"{key}={value}" for key, value in evidence.items()))
            if item.get("remediation"):
                st.info(f"建议：{item['remediation']}")


def _render_process_details(run) -> None:
    with st.expander("规划与权限（工程视角）", expanded=False):
        for plan in run.plans:
            st.markdown(f"**第 {plan.round + 1} 轮计划**（来源 `{plan.source}`）")
            for note in plan.notes:
                st.caption(f"· {note}")
            st.dataframe(
                pd.DataFrame([{
                    "工具": step.tool,
                    "参数": json.dumps(dict(step.arguments), ensure_ascii=False),
                    "目标": step.goal,
                    "失败策略": step.on_failure + ("（可选）" if step.optional else ""),
                } for step in plan.steps]),
                use_container_width=True, hide_index=True,
            )
        if run.permission_events:
            st.markdown("**权限事件**")
            st.dataframe(
                pd.DataFrame([dict(event) for event in run.permission_events]),
                use_container_width=True, hide_index=True,
            )
        else:
            st.caption("本次运行没有被拒绝的调用。可在上方开启「只读模式」或勾选「禁用工具」后再跑一次。")


def _render_artifacts(run) -> None:
    st.markdown("#### 产物")
    report_path = run.findings.get("research_report")
    cols = st.columns(2)
    if report_path and Path(report_path).exists():
        body = Path(report_path).read_text(encoding="utf-8")
        cols[0].download_button(
            "下载研究报告（Markdown）", body,
            file_name=f"research_report_{run.run_id}.md", mime="text/markdown",
            use_container_width=True,
        )
    run_dir = Path("reports/agent_runs") / run.run_id
    payload = json.dumps(run.to_dict(), ensure_ascii=False, indent=2)
    cols[1].download_button(
        "下载运行记录（JSON）", payload,
        file_name=f"{run.run_id}.json", mime="application/json",
        use_container_width=True,
    )
    st.caption(
        f"完整运行记录已落盘：`{run_dir}/`（report.md 执行轨迹、run.json 机器可读、"
        "artifacts/ 完整中间结果、context_digest.txt 上下文目录）"
    )
    with st.expander("研究报告全文", expanded=False):
        if report_path and Path(report_path).exists():
            st.markdown(Path(report_path).read_text(encoding="utf-8"))
    with st.expander("执行轨迹全文（给工程评审）", expanded=False):
        st.markdown(run.report_markdown)


def _render_intro() -> None:
    with st.expander("这个页面在做什么？（先读这段）", expanded=False):
        st.markdown(
            """
这一层是加在既有量化平台之上的 **研究助理**：把一句自然语言请求跑成完整研究流程，
并在遇到问题时自己调整。

| 环节 | 具体行为 |
| --- | --- |
| 解析 | 从请求里抽出股票池（行业关键词 / 指数）、区间、因子主题与组合设置 |
| 取数 | 加载面板、裁剪到分析窗口、核对覆盖率 |
| 计算 | 计算因子值、核对覆盖率，重对象留在工作集不进对话上下文 |
| 检验 | 统一管线 + IC / 分层 / 换手 + 成本后组合回测，结果进内容寻址缓存 |
| 异常检查 | 截面宽度、IC 显著性与方向、换手侵蚀、单调性、超额、回撤 |
| 重新规划 | 报出高优异常时自动补跑宽基对照，让「样本是否够宽」有对照证据 |
| 报告 | 研究结论（给研究员）+ 执行轨迹（给工程评审），落盘可复盘 |

**三道闸门**：每次工具调用都要先过权限门（允许/拒绝/需确认/预算），再过重试层
（瞬时故障退避重试、确定性错误立即失败），最后由工具自身校验结果
（覆盖率不达标也按瞬时故障重试，避免静默降级被当成成功）。

**默认离线可复现**：默认用确定性规则规划器 + 离线合成面板，无需 API Key 与外网。
配置 `QFM_AGENT_LLM_*` 后可用真实模型规划（仍是同一个工具集与同一套闸门）。
            """
        )


def _render_last_run_summary() -> None:
    run = st.session_state.get("agent_last_run")
    if run is None:
        return
    st.caption(
        f"上次运行：{run.status.value}　任务 {run.counts()}　"
        f"耗时 {run.duration_s:.1f}s　请求「{run.request}」"
    )
