"""Factor Compare 与 Correlation Analysis 页面。"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from qfm.analysis.factor_compare import compare_factors
from qfm.data.catalog import register_panel_dataset
from qfm.factors import compute_factor, factor_label, get_factor, list_factors
from qfm.jobs import JobService
from qfm.pipeline.pipeline import PipelineConfig, run_pipeline
from qfm.research.snapshot import build_data_snapshot


def _heatmap(matrix: pd.DataFrame, title: str) -> go.Figure:
    figure = go.Figure(go.Heatmap(
        z=matrix.values,
        x=[factor_label(name) for name in matrix.columns],
        y=[factor_label(name) for name in matrix.index],
        zmin=-1,
        zmax=1,
        colorscale=[[0, "#FF6BA9"], [0.5, "#FFFFFF"], [1, "#7C4DFF"]],
        text=matrix.round(2),
        texttemplate="%{text}",
        colorbar={"title": "corr"},
    ))
    figure.update_layout(
        title=title,
        height=max(360, 42 * len(matrix) + 150),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#3B2E5E"},
    )
    return figure


def render_factor_compare(panel, pool: str, jobs: JobService, catalog_root=None) -> None:
    st.markdown(
        '<div class="qfm-sig"><h1>因子对比与相关性</h1>'
        '<div class="sub">统一数据、统一方向、统一管线；同时看预测能力、交易摩擦与因子重复度。</div></div>',
        unsafe_allow_html=True,
    )
    available = [factor.name for factor in list_factors()]
    defaults = [name for name in ("mom_20", "rev_20", "ep_ttm", "vol_20") if name in available]
    names = st.multiselect(
        "选择 2–12 个因子",
        available,
        default=defaults,
        format_func=factor_label,
        key="factor_compare_names",
    )
    c1, c2, c3, c4 = st.columns(4)
    horizon = c1.selectbox("前瞻天数", [5, 10, 20, 60], index=2, key="factor_compare_horizon")
    winsorize = c2.selectbox("去极值", ["mad", "quantile", "none"], key="factor_compare_winsor")
    standardize = c3.selectbox("标准化", ["zscore", "rank", "none"], key="factor_compare_standard")
    threshold = c4.slider("重复阈值 |corr|", 0.50, 0.95, 0.70, 0.05, key="factor_compare_threshold")
    first, last = panel.close.index.min().date(), panel.close.index.max().date()
    d1, d2 = st.columns(2)
    start = d1.date_input("开始日期", first, min_value=first, max_value=last, key="factor_compare_start")
    end = d2.date_input("结束日期", last, min_value=first, max_value=last, key="factor_compare_end")

    if st.button("运行统一对比", key="factor_compare_run", type="primary", use_container_width=True):
        if not 2 <= len(set(names)) <= 12:
            st.error("请选择 2–12 个不同因子。")
        elif start > end:
            st.error("开始日期不能晚于结束日期。")
        else:
            config = PipelineConfig(winsorize=winsorize, standardize=standardize)
            snapshot = build_data_snapshot(panel, pool)
            binding = register_panel_dataset(panel, pool, snapshot, root=catalog_root)
            versions = [
                {"name": name, "version": get_factor(name).version, "source_hash": get_factor(name).source_hash}
                for name in names
            ]
            base_request = {
                **binding,
                "universe": pool,
                "date_range": [str(start), str(end)],
                "factor_versions": versions,
                "pipeline_config": config.to_dict(),
                "backtest_config": {},
            }
            try:
                with st.status("计算因子信号…", expanded=False) as status:
                    computed = jobs.run(
                        "FACTOR_COMPUTE",
                        base_request,
                        lambda: {
                            name: run_pipeline(
                                compute_factor(name, panel),
                                panel=panel,
                                config=config,
                                direction=get_factor(name).direction,
                            ).signal.loc[str(start):str(end)]
                            for name in names
                        },
                    )
                    status.update(label="统一计算指标与相关性…")
                    analysed = jobs.run(
                        "FACTOR_ANALYSIS",
                        {**base_request, "horizon": horizon, "correlation_threshold": threshold},
                        lambda: compare_factors(
                            panel,
                            names,
                            horizon=horizon,
                            start=str(start),
                            end=str(end),
                            pipeline_config=config,
                            correlation_threshold=threshold,
                            precomputed_signals=computed.value,
                        ),
                    )
                    status.update(label="对比完成", state="complete")
                st.session_state["factor_compare_result"] = analysed.value
                st.session_state["factor_compare_jobs"] = (computed.job, analysed.job)
            except (KeyError, ValueError, OSError) as exc:
                st.error(f"对比失败：{exc}")

    result = st.session_state.get("factor_compare_result")
    if result is None:
        st.info("选择因子后运行。系统会先缓存统一处理后的信号，再缓存分析结果。")
        return
    job_pair = st.session_state.get("factor_compare_jobs")
    if job_pair and any(job.cache_hit for job in job_pair):
        st.caption("本次部分或全部结果已从确定性缓存读取，没有重复计算。")
    metrics = result.metrics.rename(columns={
        "factor": "因子", "ic": "IC", "rank_ic": "Rank IC", "icir": "ICIR",
        "long_short_return": "Long Short Return", "turnover": "Turnover",
        "coverage": "Coverage", "stability": "Stability",
    }).copy()
    metrics["因子"] = metrics["因子"].map(factor_label)
    st.markdown("**统一指标**")
    st.dataframe(
        metrics.style.format({
            "IC": "{:+.4f}", "Rank IC": "{:+.4f}", "ICIR": "{:.2f}",
            "Long Short Return": "{:+.2%}", "Turnover": "{:.1%}",
            "Coverage": "{:.1%}", "Stability": "{:.1%}",
        }, na_rep="—"),
        hide_index=True,
        use_container_width=True,
    )
    if result.high_correlations.empty:
        st.success(f"没有发现 |Spearman corr| > {threshold:.2f} 的高度重复因子。")
    else:
        st.warning(f"发现 {len(result.high_correlations)} 组高度重复因子（|corr| > {threshold:.2f}），建议去重或正交化后再组合。")
        duplicates = result.high_correlations.rename(columns={
            "factor_a": "因子 A", "factor_b": "因子 B", "correlation": "相关系数",
            "absolute_correlation": "绝对相关",
        }).copy()
        duplicates["因子 A"] = duplicates["因子 A"].map(factor_label)
        duplicates["因子 B"] = duplicates["因子 B"].map(factor_label)
        st.dataframe(
            duplicates.style.format({"相关系数": "{:+.3f}", "绝对相关": "{:.3f}"})
            .background_gradient(subset=["绝对相关"], cmap="Reds", vmin=threshold, vmax=1),
            hide_index=True,
            use_container_width=True,
        )
    left, right = st.columns(2)
    left.plotly_chart(_heatmap(result.pearson_corr, "Pearson Correlation"), use_container_width=True)
    right.plotly_chart(_heatmap(result.spearman_corr, "Spearman Correlation"), use_container_width=True)
