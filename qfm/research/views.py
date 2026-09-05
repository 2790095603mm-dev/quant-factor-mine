"""研究账本的 Streamlit 页面与显式保存控件。"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from qfm.research.store import ResearchStore
from qfm.research.ui_state import can_save_strategy_run, set_active_project


def _format_metric(value: Any, template: str) -> str:
    """将缺失或非有限的归档指标显示为占位符，而不是令页面格式化失败。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return template.format(number) if math.isfinite(number) else "—"


def _project_label(project_id: str, projects: dict[str, Any]) -> str:
    project = projects[project_id]
    return f"{project.name} · {project.id[-8:]}"


def active_project_name(store: ResearchStore) -> str | None:
    """返回当前会话项目名称，并清除失效选择。"""
    project_id = st.session_state.get("active_project_id")
    if not project_id:
        return None
    try:
        return store.get_project(str(project_id)).name
    except (KeyError, ValueError):
        set_active_project(st.session_state, None)
        return None


def _run_table(runs: list[Any]) -> pd.DataFrame:
    rows = []
    for run in runs:
        summary = run.summary
        config = run.config
        rows.append(
            {
                "运行": run.name,
                "创建时间": run.created_at.replace("T", " ")[:19],
                "因子": " · ".join(config.get("factors", [])),
                "调仓": config.get("rebalance", "—"),
                "年化收益": summary.get("年化收益"),
                "夏普": summary.get("夏普比率"),
                "最大回撤": summary.get("最大回撤"),
                "年化换手": summary.get("年化换手"),
                "累计成本": summary.get("累计交易成本"),
                "数据区间": f"{run.data_snapshot.get('data_start', '—')[:10]} → "
                f"{run.data_snapshot.get('data_end', '—')[:10]}",
                "运行 ID": run.id,
            }
        )
    return pd.DataFrame(rows)


def render_strategy_save_panel(store: ResearchStore) -> None:
    """在策略结果生成后提供明确、可重试的保存动作。"""
    payload = st.session_state.get("latest_strategy_run")
    if not payload:
        return

    st.divider()
    st.markdown("**③ 归档到研究项目**")
    project_id = st.session_state.get("active_project_id")
    if not can_save_strategy_run(project_id, payload):
        st.info("先在「研究项目」创建并选择一个项目，再将本次已完成回测保存为研究记录。")
        return

    project_name = active_project_name(store)
    if not project_name:
        st.info("当前研究项目已不存在；请到「研究项目」重新选择。")
        return

    st.caption(f"当前项目：**{project_name}**。保存会记录参数、数据快照、净值、权重、分年绩效与成交，不会重新运行回测。")
    col_name, col_save, col_open = st.columns([2, 1, 1])
    run_name = col_name.text_input("研究运行名称", key="research_run_name", placeholder="例如：价值质量 · 月频 · 成本后")
    if col_save.button("保存至当前项目", key="save_research_run", use_container_width=True):
        try:
            run = store.save_run(
                str(project_id),
                run_name,
                payload["config"],
                payload["data_snapshot"],
                payload["summary"],
                payload["nav"],
                payload["benchmark_nav"],
                payload["weights"],
                payload["yearly_performance"],
                payload["trades"],
                payload.get("constraint_history"),
            )
        except (OSError, ValueError, KeyError) as exc:
            st.error(f"保存失败：{exc}")
        else:
            st.session_state["last_saved_research_run"] = run.id
            st.success(f"已保存研究运行：{run.name} · {run.id[-8:]}")
    if col_open.button("打开研究项目", key="open_research_project", use_container_width=True):
        st.session_state["section"] = "研究项目"
        st.rerun()


def _ratio(snapshot: dict[str, Any], key: str) -> float:
    """不完整或手工编辑的旧 manifest 也能安全展示。"""
    coverage = snapshot.get("coverage")
    if not isinstance(coverage, dict):
        return 0.0
    try:
        return float(coverage.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _short_fingerprint(value: object) -> str:
    text = str(value)
    return f"{text[:20]}…" if len(text) > 20 else text


def _render_data_snapshot(snapshot: dict[str, Any]) -> None:
    """以紧凑审核卡展示新版快照，旧运行则保持完整 JSON 可见。"""
    if snapshot.get("snapshot_version") != 2:
        st.info("该运行未记录扩展数据审计信息；下次重新运行并保存后可查看覆盖率与数据指纹。")
        st.json(snapshot)
        return

    sources = snapshot.get("sources") if isinstance(snapshot.get("sources"), dict) else {}
    fingerprints = snapshot.get("fingerprints") if isinstance(snapshot.get("fingerprints"), dict) else {}
    col_stocks, col_days, col_close, col_amount = st.columns(4)
    col_stocks.metric("样本证券", snapshot.get("stocks", "—"))
    col_days.metric("交易日", snapshot.get("trading_days", "—"))
    col_close.metric("收盘覆盖", f"{_ratio(snapshot, 'close'):.1%}")
    col_amount.metric("成交额覆盖", f"{_ratio(snapshot, 'amount'):.1%}")
    st.caption(
        f"数据区间：{str(snapshot.get('data_start', '—'))[:10]} → {str(snapshot.get('data_end', '—'))[:10]} · "
        f"行情：{sources.get('market', '未记录')} · 财务：{sources.get('fundamentals', '未记录')}"
    )
    st.caption(
        f"样本指纹：{_short_fingerprint(snapshot.get('universe_fingerprint', '未记录'))} · "
        f"行情指纹：{_short_fingerprint(fingerprints.get('market', '未记录'))} · "
        f"财务/行业指纹：{_short_fingerprint(fingerprints.get('fundamentals', '未记录'))}"
    )
    warnings = snapshot.get("quality_warnings", [])
    if isinstance(warnings, list):
        for warning in warnings:
            st.warning(str(warning))
    st.markdown("**完整数据快照**")
    st.json(snapshot)


def page_research(store: ResearchStore) -> None:
    """显示项目创建、运行比较和单次运行详情。"""
    st.markdown(
        '<div class="qfm-sig"><h1>研究项目</h1>'
        '<div class="sub">保存数据快照、策略参数与回测产物；比较实验，而不是只记住一个好看的夏普率。</div></div>',
        unsafe_allow_html=True,
    )
    with st.expander("新建研究项目", expanded=not store.list_projects()):
        with st.form("create_research_project", clear_on_submit=True):
            name = st.text_input("项目名称", placeholder="例如：价值质量月频组合")
            description = st.text_area("研究说明（可选）", placeholder="假设、基准与待验证问题", height=84)
            submitted = st.form_submit_button("创建并设为当前项目", use_container_width=True)
        if submitted:
            try:
                project = store.create_project(name, description)
            except ValueError as exc:
                st.error(str(exc))
            else:
                set_active_project(st.session_state, project.id)
                st.success(f"已创建并选中：{project.name}")
                st.rerun()

    projects = store.list_projects()
    if not projects:
        st.info("还没有研究项目。创建一个项目后，再到「策略回测」运行并保存研究记录。")
        return

    project_map = {project.id: project for project in projects}
    current_id = st.session_state.get("active_project_id")
    if current_id not in project_map:
        current_id = projects[0].id
        set_active_project(st.session_state, current_id)
    selected_id = st.selectbox(
        "当前研究项目",
        list(project_map),
        index=list(project_map).index(current_id),
        format_func=lambda project_id: _project_label(project_id, project_map),
    )
    if selected_id != current_id:
        set_active_project(st.session_state, selected_id)
        st.rerun()

    project = project_map[selected_id]
    st.caption(project.description or "尚未填写研究说明。建议记录待验证假设、基准与风险边界。")
    runs, warnings = store.list_runs(project.id)
    for warning in warnings:
        st.warning(warning)
    if not runs:
        st.info("该项目还没有已保存运行。到「策略回测」完成一次回测后，点击“保存至当前项目”。")
        return

    table = _run_table(runs)
    st.markdown("**研究运行**")
    st.dataframe(
        table.style.format(
            {
                "年化收益": lambda value: _format_metric(value, "{:+.1%}"),
                "夏普": lambda value: _format_metric(value, "{:.2f}"),
                "最大回撤": lambda value: _format_metric(value, "{:.1%}"),
                "年化换手": lambda value: _format_metric(value, "{:.1f}"),
                "累计成本": lambda value: _format_metric(value, "{:.2%}"),
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    run_map = {run.id: run for run in runs}
    default_compare = list(run_map)[: min(2, len(run_map))]
    selected_runs = st.multiselect(
        "比较净值（最多 8 个运行）",
        list(run_map),
        default=default_compare,
        format_func=lambda run_id: run_map[run_id].name,
    )
    if len(selected_runs) > 8:
        st.warning("一次最多比较 8 个运行；仅绘制前 8 个。")
        selected_runs = selected_runs[:8]
    if selected_runs:
        chart = go.Figure()
        for run_id in selected_runs:
            try:
                loaded = store.load_run(run_id)
            except ValueError as exc:
                st.warning(str(exc))
                continue
            chart.add_trace(
                go.Scatter(
                    x=loaded.nav.index,
                    y=loaded.nav.values,
                    name=loaded.run.name,
                    line={"width": 2},
                )
            )
        chart.update_layout(
            title="保存运行的策略净值对比",
            height=360,
            hovermode="x unified",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#3B2E5E"},
            legend={"orientation": "h", "y": 1.1},
        )
        st.plotly_chart(chart, use_container_width=True)

    detail_id = st.selectbox(
        "查看运行详情",
        list(run_map),
        format_func=lambda run_id: run_map[run_id].name,
        key="research_run_detail",
    )
    try:
        loaded = store.load_run(detail_id)
    except ValueError as exc:
        st.error(str(exc))
        return

    with st.expander("配置与数据快照", expanded=True):
        st.markdown("**数据快照审核**")
        _render_data_snapshot(loaded.run.data_snapshot)
        st.markdown("**完整运行记录**")
        st.json(loaded.run.to_dict())
    col_weights, col_yearly = st.columns(2)
    col_weights.markdown("**因子权重**")
    col_weights.dataframe(loaded.weights.style.format({"weight": "{:.1%}"}), use_container_width=True)
    col_yearly.markdown("**分年绩效**")
    col_yearly.dataframe(loaded.yearly_performance, hide_index=True, use_container_width=True)
    with st.expander(f"成交流水（{len(loaded.trades)} 笔）"):
        st.dataframe(loaded.trades, hide_index=True, use_container_width=True)
    if loaded.constraint_history is not None:
        with st.expander(f"组合约束执行审计（{len(loaded.constraint_history)} 次）"):
            st.dataframe(loaded.constraint_history, hide_index=True, use_container_width=True)
    st.markdown("**导出保存的研究产物**")
    export_columns = st.columns(6 if loaded.constraint_history is not None else 5)
    export_columns[0].download_button(
        "净值 CSV", loaded.nav.to_csv().encode("utf-8-sig"), "research_nav.csv", "text/csv", use_container_width=True
    )
    export_columns[1].download_button(
        "基准净值 CSV", loaded.benchmark_nav.to_csv().encode("utf-8-sig"), "research_benchmark_nav.csv", "text/csv", use_container_width=True
    )
    export_columns[2].download_button(
        "权重 CSV", loaded.weights.to_csv().encode("utf-8-sig"), "research_weights.csv", "text/csv", use_container_width=True
    )
    export_columns[3].download_button(
        "分年绩效 CSV", loaded.yearly_performance.to_csv(index=False).encode("utf-8-sig"), "research_yearly.csv", "text/csv", use_container_width=True
    )
    export_columns[4].download_button(
        "成交 CSV", loaded.trades.to_csv(index=False).encode("utf-8-sig"), "research_trades.csv", "text/csv", use_container_width=True
    )
    if loaded.constraint_history is not None:
        export_columns[5].download_button(
            "约束审计 CSV",
            loaded.constraint_history.to_csv(index=False).encode("utf-8-sig"),
            "research_constraint_history.csv",
            "text/csv",
            use_container_width=True,
        )
