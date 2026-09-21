"""Tear Sheet 的可视化：图表构造函数（可离线测试）+ Streamlit 渲染。

图表函数全部返回 plotly Figure，不依赖 Streamlit，便于单测；
`render_tear_sheet` 负责页面排版。

页面刻意把**预测统计口径**与**可交易净值口径**分在两组，避免把"分层收益均值"
误读成"照着做的业绩"。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 与 pipeline/report.py 保持同一套配色
BG = "#F6F2FF"
CARD = "#FFFFFF"
BORDER = "#E5DCF5"
TEXT = "#3B2E5E"
AMBER = "#7C4DFF"
CYAN = "#2F6FED"
RED = "#E5484D"
GREEN = "#2E9E6B"
MUTED = "#6E5C93"

LAYER_COLORS = ["#2F6FED", "#4C8BF5", "#7C4DFF", "#C86BD8", "#E5484D"]


def _style(fig, title: str, height: int = 320):
    fig.update_layout(
        title=title, height=height, template=None,
        paper_bgcolor=CARD, plot_bgcolor=BG,
        font=dict(color=TEXT, size=12, family="PingFang SC, Microsoft YaHei, sans-serif"),
        margin=dict(l=50, r=20, t=44, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED)),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    return fig


def ic_figure(sheet) -> object:
    """IC 时间序列 + 滚动均值 + 累计 IC（双轴）。"""
    import plotly.graph_objects as go

    ic = sheet.ic.dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ic.index, y=ic.values, name="日度 IC",
                             line=dict(color=CYAN, width=1), opacity=0.55))
    if len(sheet.ic_rolling):
        fig.add_trace(go.Scatter(x=sheet.ic_rolling.index, y=sheet.ic_rolling.values,
                                 name="IC 120 日均值", line=dict(color=AMBER, width=2)))
    if len(sheet.ic_cumulative):
        fig.add_trace(go.Scatter(x=sheet.ic_cumulative.index, y=sheet.ic_cumulative.values,
                                 name="累计 IC", line=dict(color=GREEN, width=2, dash="dot"),
                                 yaxis="y2"))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1, dash="dash"))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False,
                                  title="累计 IC", color=GREEN))
    return _style(fig, f"IC / RankIC 序列（前瞻 {sheet.horizon} 日，已统一方向）", 340)


def rolling_icir_figure(sheet) -> object:
    import plotly.graph_objects as go

    fig = go.Figure()
    series = sheet.ic_rolling_ir.dropna()
    if len(series):
        fig.add_trace(go.Scatter(x=series.index, y=series.values, name="滚动 ICIR（120 日）",
                                 line=dict(color=AMBER, width=2)))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1, dash="dash"))
    fig.add_hline(y=0.5, line=dict(color=GREEN, width=1, dash="dot"))
    fig.add_hline(y=-0.5, line=dict(color=RED, width=1, dash="dot"))
    return _style(fig, "滚动 ICIR（虚线为 ±0.5 参考）", 300)


def layer_stat_figure(sheet) -> object:
    """预测统计口径：各层未来收益均值。"""
    import plotly.graph_objects as go

    layer = sheet.layer_stat
    if layer.empty:
        return _style(go.Figure(), "分层收益（样本不足，无法计算）", 300)
    colors = [RED if value < 0 else CYAN for value in layer["mean_ret"].values]
    fig = go.Figure(go.Bar(
        x=[f"L{i}" for i in layer.index], y=layer["mean_ret"].values,
        marker_color=colors, text=np.round(layer["mean_ret"].values, 4),
        textposition="outside",
    ))
    return _style(
        fig,
        f"分层平均收益（未来 {sheet.horizon} 日，L1 低分 → L{len(layer)} 高分）· 统计口径：重叠窗口、不复利",
        320,
    )


def layer_nav_figure(sheet) -> object:
    """可交易净值口径：各层复利净值 + 多空净值。"""
    import plotly.graph_objects as go

    fig = go.Figure()
    if not sheet.layer_nav.empty:
        for position, column in enumerate(sheet.layer_nav.columns):
            fig.add_trace(go.Scatter(
                x=sheet.layer_nav.index, y=sheet.layer_nav.values[:, position],
                name=f"L{column} 净值",
                line=dict(color=LAYER_COLORS[position % len(LAYER_COLORS)],
                          width=3 if position in (0, len(sheet.layer_nav.columns) - 1) else 1.4),
            ))
    if len(sheet.long_short_nav):
        fig.add_trace(go.Scatter(
            x=sheet.long_short_nav.index, y=sheet.long_short_nav.values,
            name="多空净值（高分 − 低分）",
            line=dict(color=TEXT, width=2, dash="dash"),
        ))
    return _style(
        fig, "分层累计净值与多空净值 · 可交易口径：非重叠调仓、层内等权、逐期复利", 360
    )


def turnover_figure(sheet) -> object:
    import plotly.graph_objects as go

    fig = go.Figure()
    if len(sheet.turnover_series):
        fig.add_trace(go.Scatter(x=sheet.turnover_series.index, y=sheet.turnover_series.values,
                                 name="名单换手（前 10%）", line=dict(color=AMBER, width=1.6)))
    return _style(fig, "组合换手（相邻调仓期前 10% 名单的 Jaccard 距离）", 280)


def distribution_figure(sheet) -> object:
    import plotly.graph_objects as go

    dist = sheet.distribution
    fig = go.Figure()
    if not dist.empty:
        fig.add_trace(go.Scatter(x=dist.index, y=dist["P95"], name="P95",
                                 line=dict(color=BORDER, width=0)))
        fig.add_trace(go.Scatter(x=dist.index, y=dist["P5"], name="P5", fill="tonexty",
                                 fillcolor="rgba(47,111,237,0.12)",
                                 line=dict(color=BORDER, width=0)))
        for column, color, width in (("P50", CYAN, 2), ("均值", AMBER, 1.6)):
            fig.add_trace(go.Scatter(x=dist.index, y=dist[column], name=column,
                                     line=dict(color=color, width=width)))
        fig.add_hline(y=0, line=dict(color=MUTED, width=1, dash="dash"))
    return _style(fig, "截面分布：均值 / 中位 / 5%~95% 区间", 300)


def histogram_figure(sheet) -> object:
    import plotly.graph_objects as go

    histogram = sheet.histogram
    fig = go.Figure()
    if histogram:
        edges = np.asarray(histogram["bin_edges"])
        centers = (edges[:-1] + edges[1:]) / 2
        fig.add_trace(go.Bar(x=centers, y=histogram["counts"], marker_color=CYAN))
        fig.add_vline(x=0, line=dict(color=MUTED, width=1, dash="dash"))
    return _style(
        fig,
        "最新截面因子分布" + (f"（{pd.Timestamp(sheet.histogram['date']).date()}）" if histogram else ""),
        280,
    )


def industry_exposure_figure(sheet) -> object:
    import plotly.graph_objects as go

    exposure = sheet.industry_exposure
    fig = go.Figure()
    if not exposure.empty:
        ordered = exposure.sort_values("平均暴露")
        colors = [RED if value < 0 else CYAN for value in ordered["平均暴露"].values]
        fig.add_trace(go.Bar(x=ordered["平均暴露"].values, y=ordered.index,
                             orientation="h", marker_color=colors))
        fig.add_vline(x=0, line=dict(color=MUTED, width=1, dash="dash"))
    return _style(fig, "行业暴露：各行业内的因子均值（按交易日平均）", max(300, 22 * len(exposure)))


def size_exposure_figure(sheet) -> object:
    import plotly.graph_objects as go

    fig = go.Figure()
    series = sheet.size_exposure
    if len(series):
        fig.add_trace(go.Scatter(x=series.index, y=series.values, name="与 log(流通市值) 的截面相关",
                                 line=dict(color=AMBER, width=1.6)))
        fig.add_hline(y=float(series.mean()), line=dict(color=CYAN, width=2, dash="dot"))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1, dash="dash"))
    fig.add_hline(y=0.3, line=dict(color=RED, width=1, dash="dot"))
    fig.add_hline(y=-0.3, line=dict(color=RED, width=1, dash="dot"))
    return _style(fig, "市值暴露时序（虚线为 ±0.3 预警，点线为均值）", 300)


def coverage_figure(sheet) -> object:
    """各阶段覆盖率：说明信号在哪一步被消耗掉。"""
    import plotly.graph_objects as go

    coverage = {key: value for key, value in sheet.coverage.items() if np.isfinite(value)}
    fig = go.Figure()
    if coverage:
        fig.add_trace(go.Bar(x=list(coverage), y=list(coverage.values()),
                             marker_color=[CYAN if v > 0.8 else RED for v in coverage.values()],
                             text=[f"{v:.1%}" for v in coverage.values()], textposition="outside"))
        fig.update_yaxes(range=[0, 1.05], tickformat=".0%")
    return _style(fig, "统一管线各阶段信号覆盖率", 300)


def _format(value, spec=".4f") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return format(number, spec) if np.isfinite(number) else "—"


def render_tear_sheet(sheet) -> None:
    """页面渲染：KPI → 预测能力 → 分层双口径 → 分布 → 暴露 → 覆盖。"""
    import streamlit as st

    kpis = sheet.kpis()
    st.markdown(
        f"**Tear Sheet · {sheet.name}** · 家族 {sheet.meta.get('family', '—')} · "
        f"方向 {sheet.direction} · 前瞻 {sheet.horizon} 日"
    )
    for warning in sheet.warnings:
        st.warning(warning)

    first = st.columns(6)
    for column, (label, value, spec) in zip(first, [
        ("平均 IC", kpis["平均 IC"], "+.4f"),
        ("IC_IR", kpis["IC_IR"], ".2f"),
        ("IC t 值", kpis["IC t 值"], ".2f"),
        ("IC 正占比", kpis["IC 正占比"], ".1%"),
        ("多空价差（统计）", kpis["多空价差（统计）"], "+.4f"),
        ("信号覆盖率", kpis["信号覆盖率"], ".1%"),
    ]):
        column.metric(label, _format(value, spec))
    second = st.columns(6)
    for column, (label, value, spec) in zip(second, [
        ("多空累计（可交易）", kpis["多空累计（可交易）"], "+.1%"),
        ("组合换手", kpis["组合换手"], ".1%"),
        ("市值暴露", kpis["市值暴露"], "+.2f"),
        ("有效天数", kpis["有效天数"], ".0f"),
        ("IC 均值绝对值", kpis["IC 均值绝对值"], ".4f"),
        ("收益率偏度", float(sheet.distribution["偏度"].mean()) if not sheet.distribution.empty else float("nan"), ".2f"),
    ]):
        column.metric(label, _format(value, spec))

    predictive, tradable, distribution_tab, exposure_tab = st.tabs(
        ["预测能力", "分层与多空（可交易）", "因子分布", "暴露与覆盖"]
    )
    with predictive:
        st.caption("统计口径：因子值与未来收益的相关性和分组均值，重叠窗口、不复利，回答「有没有预测力」。")
        st.plotly_chart(ic_figure(sheet), use_container_width=True)
        left, right = st.columns(2)
        left.plotly_chart(rolling_icir_figure(sheet), use_container_width=True)
        right.plotly_chart(layer_stat_figure(sheet), use_container_width=True)
        if len(sheet.ic_by_year):
            st.markdown("**分年 IC**")
            st.dataframe(
                sheet.ic_by_year.rename("平均 IC").to_frame()
                .style.format({"平均 IC": "{:+.4f}"}),
                use_container_width=True,
            )
    with tradable:
        st.caption(
            "可交易口径：固定间隔非重叠调仓、层内等权、逐期复利。与策略回测的日历调仓不同，"
            "两者不可直接对比；这里只回答「照着这个因子分层持有会怎样」。"
        )
        st.plotly_chart(layer_nav_figure(sheet), use_container_width=True)
        left, right = st.columns(2)
        left.plotly_chart(turnover_figure(sheet), use_container_width=True)
        if not sheet.layer_period_returns.empty:
            right.markdown("**各层逐期收益统计**")
            right.dataframe(
                sheet.layer_period_returns.describe().rename(columns=lambda c: f"L{c}")
                .style.format("{:+.4f}"),
                use_container_width=True,
            )
    with distribution_tab:
        left, right = st.columns(2)
        left.plotly_chart(distribution_figure(sheet), use_container_width=True)
        right.plotly_chart(histogram_figure(sheet), use_container_width=True)
        if not sheet.distribution.empty:
            with st.expander("截面分布明细（最近 30 个交易日）"):
                st.dataframe(
                    sheet.distribution.tail(30).style.format("{:.4f}"),
                    use_container_width=True,
                )
    with exposure_tab:
        left, right = st.columns(2)
        left.plotly_chart(size_exposure_figure(sheet), use_container_width=True)
        right.plotly_chart(coverage_figure(sheet), use_container_width=True)
        st.plotly_chart(industry_exposure_figure(sheet), use_container_width=True)
        if not sheet.industry_exposure.empty:
            st.dataframe(
                sheet.industry_exposure.style.format({
                    "平均暴露": "{:+.4f}", "暴露标准差": "{:.4f}",
                    "平均成分数": "{:.1f}", "样本日数": "{:.0f}",
                }),
                use_container_width=True,
            )
