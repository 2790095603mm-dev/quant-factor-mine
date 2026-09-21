"""Strategy Compare 页面。"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from qfm.analysis.strategy_compare import compare_strategies
from qfm.research.models import STATUS_COMPLETED
from qfm.research.store import ResearchStore


def _line_chart(frame: pd.DataFrame, title: str, *, zero_line: bool = False) -> go.Figure:
    figure = go.Figure()
    palette = ["#7C4DFF", "#FF6BA9", "#2F6FED", "#16A085", "#F39C12", "#8E44AD", "#E74C3C", "#34495E"]
    for index, column in enumerate(frame.columns):
        figure.add_trace(go.Scatter(
            x=frame.index, y=frame[column], name=str(column),
            line={"width": 2, "color": palette[index % len(palette)]},
        ))
    if zero_line:
        figure.add_hline(y=0, line={"color": "#9B82D6", "width": 1, "dash": "dash"})
    figure.update_layout(
        title=title, height=380, hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#3B2E5E"}, legend={"orientation": "h", "y": 1.12},
    )
    return figure


def render_strategy_compare(store: ResearchStore) -> None:
    st.markdown(
        '<div class="qfm-sig"><h1>策略对比</h1>'
        '<div class="sub">跨项目选择历史 Backtest Experiment，统一比较收益、风险、换手和相对基准表现。</div></div>',
        unsafe_allow_html=True,
    )
    pairs = [(project, run) for project, run in store.list_all_runs() if run.status == STATUS_COMPLETED]
    if len(pairs) < 2:
        st.info("至少需要两个已完成并保存的历史回测。请先在策略回测或单因子模拟中保存实验。")
        return
    pair_map = {run.id: (project, run) for project, run in pairs}
    defaults = list(pair_map)[:2]
    selected = st.multiselect(
        "选择 2–8 个历史回测",
        list(pair_map),
        default=defaults,
        format_func=lambda run_id: f"{pair_map[run_id][0].name} / {pair_map[run_id][1].name} · {run_id[-8:]}",
        key="strategy_compare_runs",
    )
    if st.button("生成策略对比", key="strategy_compare_run", type="primary", use_container_width=True):
        if not 2 <= len(selected) <= 8:
            st.error("请选择 2–8 个已完成回测。")
        else:
            try:
                result = compare_strategies([store.load_run(run_id) for run_id in selected])
            except (KeyError, ValueError) as exc:
                st.error(f"对比失败：{exc}")
            else:
                st.session_state["strategy_compare_result"] = result
    result = st.session_state.get("strategy_compare_result")
    if result is None:
        return
    metrics = result.metrics.rename(columns={
        "strategy": "策略", "annual_return": "Annual Return", "excess_return": "Excess Return",
        "sharpe": "Sharpe", "max_drawdown": "Max Drawdown", "calmar": "Calmar",
        "turnover": "Turnover", "dataset_version": "Dataset Version",
        "universe_id": "Universe", "universe_version": "Universe Version",
    }).drop(columns=["run_id"])
    st.dataframe(
        metrics.style.format({
            "Annual Return": "{:+.1%}", "Excess Return": "{:+.1%}", "Sharpe": "{:.2f}",
            "Max Drawdown": "{:.1%}", "Calmar": "{:.2f}", "Turnover": "{:.1f}",
        }, na_rep="—"),
        hide_index=True,
        use_container_width=True,
    )
    combined = pd.concat([result.nav, result.benchmark], axis=1)
    st.plotly_chart(_line_chart(combined, "策略累计净值与 Benchmark"), use_container_width=True)
    left, right = st.columns(2)
    left.plotly_chart(_line_chart(result.excess, "Excess Return Curve", zero_line=True), use_container_width=True)
    right.plotly_chart(_line_chart(result.drawdown, "Drawdown", zero_line=True), use_container_width=True)
