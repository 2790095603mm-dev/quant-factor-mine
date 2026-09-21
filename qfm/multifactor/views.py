"""Multi-Factor Lab 页面。"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from qfm.data.catalog import register_panel_dataset
from qfm.factors import COMPOSITE_WARNINGS, factor_label, get_factor, list_factors
from qfm.jobs import JobService
from qfm.multifactor import CompositeDefinition, CompositeRegistry
from qfm.portfolio import synthesize
from qfm.research.snapshot import build_data_snapshot


MODE_LABELS = {
    "equal": "Equal Weight",
    "ic": "IC Weight",
    "icir": "ICIR Weight",
    "ic_x_ir": "IC × IR Weight",
}


def render_multifactor_lab(panel, pool: str, jobs: JobService, registry: CompositeRegistry,
                           catalog_root=None) -> None:
    st.markdown(
        '<div class="qfm-sig"><h1>多因子实验室</h1>'
        '<div class="sub">把 Factor Library 中的因子组合成可版本化的新因子，并继续用于检验与回测。</div></div>',
        unsafe_allow_html=True,
    )
    for warning in COMPOSITE_WARNINGS:
        st.warning(warning)
    names_all = [factor.name for factor in list_factors()]
    defaults = [name for name in ("ep_ttm", "roe", "mom_20") if name in names_all]
    names = st.multiselect(
        "选择成分因子",
        names_all,
        default=defaults,
        format_func=factor_label,
        key="multifactor_names",
    )
    c1, c2, c3 = st.columns(3)
    mode = c1.selectbox("权重方法", list(MODE_LABELS), format_func=MODE_LABELS.get, key="multifactor_mode")
    horizon = c2.selectbox("IC 前瞻天数", [5, 10, 20, 60], index=2, key="multifactor_horizon")
    lookback = c3.selectbox("估权回看窗口", [60, 126, 252, 504], index=2, key="multifactor_lookback")
    c4, c5 = st.columns(2)
    rebalance = c4.selectbox("权重更新频率", ["ME", "W-FRI"], format_func=lambda x: "月末" if x == "ME" else "每周五", key="multifactor_rebalance")
    orthogonalize = c5.checkbox("综合后做市值正交化", value=False, key="multifactor_ortho")

    if st.button("生成综合因子", key="multifactor_run", type="primary", use_container_width=True):
        if len(set(names)) < 2:
            st.error("至少选择 2 个不同因子。")
        else:
            snapshot = build_data_snapshot(panel, pool)
            binding = register_panel_dataset(panel, pool, snapshot, root=catalog_root)
            request = {
                **binding,
                "universe": pool,
                "date_range": [str(panel.close.index.min()), str(panel.close.index.max())],
                "factor_versions": [
                    {"name": name, "version": get_factor(name).version, "source_hash": get_factor(name).source_hash}
                    for name in names
                ],
                "pipeline_config": {},
                "backtest_config": {},
                "multi_factor_config": {
                    "mode": mode, "horizon": horizon, "lookback": lookback,
                    "rebalance": rebalance, "orthogonalize": orthogonalize,
                },
            }
            try:
                result = jobs.run(
                    "MULTI_FACTOR",
                    request,
                    lambda: synthesize(
                        panel, names, mode=mode, horizon=horizon,
                        weight_lookback=lookback, weight_rebalance=rebalance,
                        orthogonalize=orthogonalize,
                        ortho_controls=("size",) if orthogonalize else None,
                    ),
                )
            except (KeyError, ValueError, OSError) as exc:
                st.error(f"综合失败：{exc}")
            else:
                st.session_state["multifactor_result"] = result.value
                st.session_state["multifactor_job"] = result.job
                st.session_state["multifactor_definition_config"] = {
                    "names": names, "mode": mode, "horizon": horizon, "lookback": lookback,
                    "rebalance": rebalance, "orthogonalize": orthogonalize,
                }

    cached = st.session_state.get("multifactor_result")
    config = st.session_state.get("multifactor_definition_config")
    if cached is None or config is None:
        st.info("生成后可先查看当前权重和历史权重，再决定是否保存进 Factor Library。")
        return
    score, weights = cached
    job = st.session_state.get("multifactor_job")
    if job and job.cache_hit:
        st.caption("已读取缓存：相同因子版本、数据版本和参数没有重复计算。")
    st.markdown("**当前综合结果**")
    k1, k2, k3 = st.columns(3)
    k1.metric("有效覆盖", f"{score.notna().to_numpy().mean():.1%}")
    k2.metric("成分因子", len(config["names"]))
    k3.metric("权重方法", MODE_LABELS[config["mode"]])
    weight_frame = pd.DataFrame({
        "因子": [factor_label(name) for name in weights],
        "当前权重": list(weights.values()),
    }).sort_values("当前权重", ascending=False)
    left, right = st.columns([1, 2])
    left.dataframe(weight_frame.style.format({"当前权重": "{:.1%}"}), hide_index=True, use_container_width=True)
    history = score.attrs.get("weight_history")
    if isinstance(history, pd.DataFrame):
        right.line_chart(history.rename(columns={name: factor_label(name) for name in history.columns}))

    st.markdown("**保存为 Factor Library 因子**")
    c1, c2 = st.columns([1, 2])
    factor_name = c1.text_input("新因子英文名", key="multifactor_save_name", placeholder="例如：value_quality_momentum")
    description = c2.text_input("说明", key="multifactor_save_desc", placeholder="例如：价值、质量和动量的 ICIR 加权综合因子")
    if st.button("保存进 Factor Library", key="multifactor_save", use_container_width=True):
        try:
            definition = CompositeDefinition.create(
                factor_name,
                config["names"],
                description=description,
                mode=config["mode"],
                horizon=config["horizon"],
                weight_lookback=config["lookback"],
                weight_rebalance=config["rebalance"],
                orthogonalize=config["orthogonalize"],
                ortho_controls=("size",) if config["orthogonalize"] else (),
            )
            saved = registry.save(definition)
            warnings = registry.register_all()
        except (OSError, ValueError) as exc:
            st.error(f"保存失败：{exc}")
        else:
            st.success(f"已保存 {saved.name} v{saved.version}，现在可在因子库、因子对比和策略回测中直接选择。")
            for warning in warnings:
                st.warning(warning)
