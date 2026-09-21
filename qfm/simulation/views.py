"""单因子模拟页面：运行配置、持续保留结果、报告及归档。"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import streamlit as st

from qfm.factors import factor_label, get_factor, list_factors
from qfm.data.catalog import register_panel_dataset
from qfm.jobs import JobService
from qfm.pipeline.report import ic_chart, layer_chart
from qfm.pipeline.tearsheet_view import render_tear_sheet
from qfm.research.payload import build_strategy_run_payload
from qfm.research.snapshot import build_data_snapshot
from qfm.research.views import render_strategy_save_panel
from qfm.simulation.engine import SimulationSettings, run_factor_simulation, slice_panel
from qfm.simulation.report import cost_comparison, drawdown_chart, export_html, nav_chart

MODES = {"long_only": "A 股多头", "long_short": "多空研究模拟"}
REBALANCES = {"ME": "月末", "W-FRI": "每周五", "B": "每个交易日"}
NEUTRALIZATIONS = {"none": "不处理", "industry": "行业", "size": "市值", "industry_size": "行业 + 市值"}


def _format(value, spec=".2f"):
    return format(value, spec) if value is not None and np.isfinite(value) else "—"


def render_single_factor(panel, pool, store, jobs: JobService | None = None, catalog_root=None):
    names = [f.name for f in list_factors()]
    if st.session_state.get("test_factor") not in names:
        st.session_state["test_factor"] = names[0]
    cols = st.columns([2, 1, 1])
    name = cols[0].selectbox("选择因子", names, key="test_factor", format_func=factor_label)
    mode = cols[1].selectbox("组合模式", list(MODES), format_func=MODES.get, key="sim_mode")
    horizon = cols[2].selectbox("IC 前瞻天数", [5, 10, 20, 60], index=2, key="sim_horizon")
    first, last = panel.close.index.min().date(), panel.close.index.max().date()
    default_start = max(first, (pd.Timestamp(last) - pd.DateOffset(years=3)).date())
    with st.expander("模拟设置 · 日期、调仓、Delay、Decay、中性化与费用", expanded=False):
        c1, c2, c3 = st.columns(3)
        start = c1.date_input("开始日期", default_start, min_value=first, max_value=last, key="sim_start")
        end = c2.date_input("结束日期", last, min_value=first, max_value=last, key="sim_end")
        rebalance = c3.selectbox("调仓频率", list(REBALANCES), format_func=REBALANCES.get, key="sim_rebalance")
        c1, c2, c3 = st.columns(3)
        top_n = c1.number_input("每侧最多持仓数" if mode == "long_short" else "最多持仓数", 1, 500, 30, key="sim_top_n")
        delay = c2.number_input("Delay（交易日）", 1, 60, 1, key="sim_delay",
                                help="1 = T 日收盘信号，T+1 开盘成交；2 = 再多等一个交易日。")
        decay = c3.number_input("Decay（平滑天数）", 0, 120, 0, key="sim_decay",
                                help="0/1 不平滑；N 日线性加权，最近一天权重最高。")
        c1, c2, c3 = st.columns(3)
        neutral = c1.selectbox("信号中性化", list(NEUTRALIZATIONS), format_func=NEUTRALIZATIONS.get, key="sim_neutral")
        cap = c2.number_input("单股目标权重上限（%）", 0.1, 100.0, 10.0, step=0.5, key="sim_cap")
        capital = c3.number_input("模拟资金（元）", 100_000, 1_000_000_000, 1_000_000, step=100_000, key="sim_capital")
        c1, c2, c3 = st.columns(3)
        commission = c1.number_input("佣金（bp，双边）", 0.0, 100.0, 3.0, step=0.5, key="sim_commission")
        stamp = c2.number_input("印花税（bp，卖出）", 0.0, 100.0, 5.0, step=0.5, key="sim_stamp")
        impact = c3.number_input("冲击成本（bp，双边）", 0.0, 200.0, 10.0, step=0.5, key="sim_impact")
        c1, c2, c3 = st.columns(3)
        rf = c1.number_input("无风险年利率（%）", 0.0, 30.0, 0.0, step=0.1, key="sim_rf")
        participation = c2.number_input("成交额参与率上限（%）", 1.0, 20.0, 5.0, step=1.0,
                                         key="sim_participation", disabled=mode == "long_short")
        borrow = c3.number_input("空头借券年费率（%）", 0.0, 100.0, 3.0, step=0.5,
                                  key="sim_borrow", disabled=mode == "long_only")
        st.caption("股票池和限制股票数沿用左侧「数据参数」。信号中性化去除行业/市值影响，不等于组合权重中性；权重上限较低时会保留现金或降低多空总敞口。")
    settings = SimulationSettings(start=str(start), end=str(end), horizon=horizon, mode=mode,
        top_n=top_n, rebalance=rebalance, delay=delay, decay=decay, neutralization=neutral,
        max_weight=cap / 100, initial_capital=capital, commission=commission / 10000,
        stamp=stamp / 10000, impact=impact / 10000, risk_free_rate=rf / 100,
        max_participation=participation / 100, borrow_rate=borrow / 100)
    fingerprint = hashlib.sha256(json.dumps({"name": name, "pool": pool, "settings": settings.to_dict(),
        "stocks": list(panel.close.columns), "last": str(last)}, sort_keys=True).encode()).hexdigest()
    st.caption(f"{pool} · {len(panel.close.columns)} 只股票 · {start} → {end} · {REBALANCES[rebalance]}调仓 · Delay {delay} · Decay {decay}")
    if mode == "long_short":
        st.caption("多空研究模拟以多头 50% / 空头 50% 为目标，尚不考虑融券券源、涨跌停与成交额参与率限制。")
    if st.button("运行检验与模拟", key="sim_run", type="primary", use_container_width=True):
        try:
            with st.status("单因子检验与模拟中…", expanded=True) as status:
                if jobs is None:
                    result = run_factor_simulation(
                        panel, name, settings,
                        progress=lambda label: status.update(label=label),
                    )
                    job_pair = ()
                else:
                    snapshot = build_data_snapshot(panel, pool)
                    binding = register_panel_dataset(panel, pool, snapshot, root=catalog_root)
                    factor = get_factor(name)
                    base_request = {
                        **binding,
                        "universe": pool,
                        "date_range": [settings.start, settings.end],
                        "factor_versions": [{
                            "name": name, "version": factor.version,
                            "source_hash": factor.source_hash,
                        }],
                        "pipeline_config": {
                            "delay": settings.delay, "decay": settings.decay,
                            "neutralization": settings.neutralization,
                        },
                        "backtest_config": {},
                    }
                    computed = jobs.run(
                        "FACTOR_COMPUTE", base_request,
                        lambda: factor.func(panel),
                    )
                    analysed = jobs.run(
                        "FACTOR_ANALYSIS",
                        {**base_request, "horizon": settings.horizon,
                         "backtest_config": settings.to_dict()},
                        lambda: run_factor_simulation(
                            panel, name, settings,
                            progress=lambda label: status.update(label=label),
                            raw=computed.value,
                        ),
                    )
                    result = analysed.value
                    job_pair = (computed.job, analysed.job)
                payload = build_factor_payload(panel, pool, result)
                status.update(label="生成 Tear Sheet…")
                sheet = build_factor_tear_sheet(panel, result)
                report_html = export_html(result)
                status.update(label="完成：Tear Sheet 与模拟绩效已生成", state="complete", expanded=False)
            st.session_state["factor_simulation"] = {
                "fingerprint": fingerprint, "result": result, "html": report_html, "sheet": sheet,
                "jobs": job_pair,
            }
            st.session_state["factor_simulation_payload"] = payload
        except Exception as exc:
            st.error(f"本次运行未完成：{exc}")
    cached = st.session_state.get("factor_simulation")
    if not cached:
        st.info("选择一个因子，点击「运行检验与模拟」，即可查看完整 Tear Sheet（IC/RankIC/ICIR/分组收益/"
                "多空净值/换手/覆盖/分布/行业与市值暴露）以及成本后组合绩效。")
        return
    if cached["fingerprint"] != fingerprint:
        st.info("参数已改变；下方保留上次运行结果。点击「运行检验与模拟」后更新。")
    if cached.get("jobs") and any(job.cache_hit for job in cached["jobs"]):
        st.caption("本次部分或全部结果已读取缓存，没有重复计算。")
    render_result(cached["result"], cached["html"], cached.get("sheet"))
    render_strategy_save_panel(store, payload_key="factor_simulation_payload", key_prefix="factor_sim", heading="保存本次单因子研究")


def build_factor_tear_sheet(panel, result):
    """用统一管线的信号构建 Tear Sheet，并复用已算好的报告避免重复计算。"""
    from qfm.pipeline.tearsheet import build_tear_sheet

    factor = get_factor(result.name)
    sample = slice_panel(panel, result.backtest.nav.index)
    signal = result.report["cleaned"]
    return build_tear_sheet(
        sample, signal, horizon=result.settings.horizon, name=result.name,
        direction=factor.direction, report=result.report, pipeline=result.pipeline,
        meta={"family": factor.family, "description": factor.description,
              "version": factor.version, "tags": list(factor.tags)},
    )


def build_factor_payload(panel, pool, result):
    config = result.settings
    sample = slice_panel(panel, result.backtest.nav.index)
    payload = build_strategy_run_payload(panel=sample, pool=pool, names=[result.name], mode="single_factor",
        horizon=config.horizon, weight_lookback=0, orthogonalize=config.neutralization != "none",
        ortho_controls=tuple(config.neutralization.split("_")) if config.neutralization != "none" else (),
        top_n=config.top_n, start_date=config.start, rebalance=config.rebalance, bench_mode="equal",
        costs={k: getattr(config, k) for k in ("commission", "stamp", "impact")},
        max_participation=config.max_participation, initial_capital=config.initial_capital,
        weights={result.name: 1.0}, backtest=result.backtest)
    payload["config"]["single_factor_simulation"] = config.to_dict()
    payload["config"]["factor_direction"] = get_factor(result.name).direction
    payload["config"]["performance_version"] = "daily_sharpe_fitness_v1"
    payload["summary"].update({k: float(v) if np.isfinite(v) else None for k, v in result.stats.items()})
    payload["yearly_performance"] = result.yearly.copy()
    return payload


def render_result(result, html_report, sheet=None):
    bt, rep, cfg = result.backtest, result.report, result.settings
    st.divider()
    st.markdown(f"**{factor_label(result.name)} · {MODES[cfg.mode]}**")
    st.caption(f"本次结果：{bt.nav.index.min():%Y-%m-%d} → {bt.nav.index.max():%Y-%m-%d} · {REBALANCES[cfg.rebalance]} · Delay {cfg.delay} / Decay {cfg.decay} · 中性化：{NEUTRALIZATIONS[cfg.neutralization]}")
    metrics = [("Sharpe", "夏普比率", ".2f"), ("最大回撤", "最大回撤", ".1%"),
               ("Fitness", "Fitness", ".2f"), ("年化收益", "年化收益", "+.1%"),
               ("日均换手", "日均换手", ".1%"), ("平均 IC", None, "+.4f")]
    for col, (label, key, spec) in zip(st.columns(6), metrics):
        value = result.stats.get(key) if key else rep["ic_summary"]["ic_mean"]
        col.metric(label, _format(value, spec))
    for warning in result.warnings:
        st.warning(warning)
    tab_names = (["Tear Sheet"] if sheet is not None else []) + \
        ["收益与回撤", "预测能力", "分年与费用", "持仓与成交"]
    tabs = st.tabs(tab_names)
    cursor = 0
    if sheet is not None:
        with tabs[cursor]:
            render_tear_sheet(sheet)
        cursor += 1
    returns_tab, diagnostic_tab, year_tab, trade_tab = tabs[cursor:cursor + 4]
    with returns_tab:
        st.plotly_chart(nav_chart(result), use_container_width=True)
        st.plotly_chart(drawdown_chart(result), use_container_width=True)
        st.caption("无费用对照按同样规则独立模拟；费用会改变可用资金和成交数量，因此两条曲线不只是相减一个固定手续费。基准是本次股票池等权组合。")
    with diagnostic_tab:
        s = rep["ic_summary"]
        cols = st.columns(4)
        for col, label, value, spec in zip(cols, ["IC_IR", "IC t 值", "IC 正占比", "信号覆盖率"],
                [s["ic_ir"], s["ic_t"], s["pos_ratio"], result.coverage], [".2f", ".2f", ".0%", ".1%"]):
            col.metric(label, _format(value, spec))
        st.plotly_chart(ic_chart(rep), use_container_width=True)
        st.plotly_chart(layer_chart(rep, "positive"), use_container_width=True)
        st.caption(f"IC 检验设置处理后的信号与未来 {cfg.horizon} 日收盘收益；末尾标签不完整日期剔除。分层收益是预测统计，不是可交易的多空净值。t 值未校正重叠收益的自相关。")
        with st.expander("近期 IC 与衰减"):
            st.line_chart(rep["ic_rolling"].rename("IC 120 日均线"))
            st.caption(f"近 60 个信号日 IC：{_format(rep['ic_recent'], '+.4f')}；相对全期变化：{_format(rep['ic_decay'], '+.4f')}")
    with year_tab:
        st.dataframe(result.yearly.style.format({"收益": "{:+.1%}", "最大回撤": "{:.1%}",
                      "日胜率": "{:.0%}", "夏普": "{:.2f}", "IC": "{:+.4f}"}, na_rep="—"), use_container_width=True, hide_index=True)
        st.markdown("**同规则费用对照**")
        st.dataframe(cost_comparison(result).style.format({"年化收益": "{:+.1%}", "夏普比率": "{:.2f}",
                      "最大回撤": "{:.1%}", "累计成本": "{:.2%}"}, na_rep="—"), use_container_width=True, hide_index=True)
        st.caption("累计成本占初始资金比例；多空模式包含借券费。分年样本长度不同，夏普须结合完整区间判断。")
    with trade_tab:
        held = bt.holdings.iloc[-1]
        positions = held[held.abs() > 1e-8].sort_values(ascending=False)
        st.caption(f"期末持仓 {len(positions)} 只 · 多头权重 {held.clip(lower=0).sum():.1%} · 空头权重 {-held.clip(upper=0).sum():.1%}")
        st.dataframe(positions.rename("权重").to_frame().style.format("{:.2%}"), use_container_width=True)
        st.dataframe(bt.trades.tail(500).sort_values("date", ascending=False), hide_index=True, use_container_width=True)
        st.caption(f"显示最近 500 笔，共 {len(bt.trades)} 笔；完整成交流水可下载。")
    with st.expander("指标公式与运行设置"):
        st.markdown("Sharpe = 日超额收益均值 ÷ 日收益标准差 × √252。无风险年利率按复利换成日利率。")
        st.markdown("Fitness = Sharpe × √(|252 × 平均日收益| ÷ max(日均换手, 0.125))。这是本地日频研究分数，不能直接套用 WorldQuant BRAIN 的提交门槛。")
        st.caption("日均换手来自每日买卖成交额之和 / 开盘交易前净资产，包含无交易的收益日。当前指数股票池没有历史成分变动；本页结果属于当前样本的历史模拟。")
        st.json(cfg.to_dict())
    c1, c2, c3 = st.columns(3)
    c1.download_button("下载完整 HTML 报告", html_report, file_name=f"{result.name}_simulation.html", mime="text/html", key="sim_download_html")
    series = pd.DataFrame({"成本后净值": bt.nav, "无费用净值": result.gross.nav,
                           "基准净值": bt.bench_nav, "日换手": bt.daily_turnover})
    c2.download_button("下载净值 CSV", series.to_csv().encode("utf-8-sig"), file_name=f"{result.name}_nav.csv", key="sim_download_nav")
    c3.download_button("下载成交流水", bt.trades.to_csv(index=False).encode("utf-8-sig"), file_name=f"{result.name}_trades.csv", key="sim_download_trades")
