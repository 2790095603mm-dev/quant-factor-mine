"""量化因子挖掘流水线 · Streamlit 操作台（深色金融终端风）

启动: streamlit run app.py
"""

import ast
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import streamlit as st

from qfm.data import DataCatalog, DataLoader, build_panel, get_universe, register_panel_dataset
from qfm.data.catalog_views import render_data_catalog
from qfm.factors import (
    all_tags,
    compute_factor,
    factor_label,
    get_factor,
    list_families,
    list_factor_versions,
    list_factors,
    register_factor,
    save_registry,
)
from qfm.mining import run_mining
from qfm.jobs import JobService
from qfm.jobs.views import render_job_center
from qfm.multifactor import CompositeRegistry
from qfm.multifactor.views import render_multifactor_lab
from qfm.pipeline import factor_report, generate_report
from qfm.pipeline.report import ic_chart, layer_chart
from qfm.research import ResearchStore, build_strategy_run_payload
from qfm.research.snapshot import build_data_snapshot
from qfm.research.views import active_project_name, page_research, render_strategy_save_panel
from qfm.analysis.factor_views import render_factor_compare
from qfm.analysis.strategy_views import render_strategy_compare

st.set_page_config(page_title="量化因子挖掘流水线", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

RESEARCH_STORE = ResearchStore(
    Path(os.environ.get("QFM_RESEARCH_ROOT", Path(__file__).resolve().parent / "data_cache" / "research"))
)
CATALOG = DataCatalog(
    Path(os.environ.get("QFM_CATALOG_ROOT", Path(__file__).resolve().parent / "data_cache" / "catalog"))
)
JOB_SERVICE = JobService(
    Path(os.environ.get("QFM_JOB_ROOT", Path(__file__).resolve().parent / "data_cache" / "jobs"))
)
COMPOSITE_REGISTRY = CompositeRegistry(
    Path(os.environ.get("QFM_COMPOSITE_REGISTRY", Path(__file__).resolve().parent / "data_cache" / "factors"))
)

# ---------------------------------------------------------------------------
# 深色金融终端风：琥珀金签名线 + 发丝边框 + 等宽数字
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
/* 浏览器表面：选区 / 滚动条 / 焦点环（impeccable: 页面被"搭出来"而非"组装"的证据） */
::selection { background: rgba(255, 107, 169, 0.30); }
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: #F3EDFF; }
::-webkit-scrollbar-thumb { background: #C9B8F0; border-radius: 5px; border: 2px solid #F3EDFF; }
::-webkit-scrollbar-thumb:hover { background: #9B82D6; }
:focus-visible { outline: 2px solid #7C4DFF !important; outline-offset: 2px; }

/* 极光背景：紫/粉/淡蓝生动渐变，色彩渗透每幅画格 */
.block-container { padding-top: 1.2rem; }
div[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(1100px 600px at 8% -8%, rgba(124, 77, 255, 0.45), transparent 62%),
    radial-gradient(1200px 700px at 96% 4%, rgba(255, 107, 169, 0.42), transparent 58%),
    radial-gradient(1000px 700px at 82% 92%, rgba(79, 163, 255, 0.40), transparent 60%),
    radial-gradient(900px 600px at 12% 105%, rgba(255, 107, 169, 0.26), transparent 55%),
    linear-gradient(160deg, #E9DFFF 0%, #FFE3F0 50%, #DCEBFF 100%);
}
header[data-testid="stHeader"] { background: transparent; }
h1, h2, h3 { letter-spacing: 0.3px; color: #3B2E5E; }
/* 签名线：紫→粉→蓝 渐变 */
.qfm-sig { position: relative; padding-bottom: 10px; margin: 10px 0 18px; }
.qfm-sig::after { content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: 2px;
  border-radius: 1px; background: linear-gradient(90deg, #7C4DFF, #FF6BA9 55%, #4FA3FF); }
.qfm-sig .sub { color: #6E5C93; font-size: 13px; margin-top: 2px; }

/* 玻璃面板：白色磨砂半透明 + 强背景模糊 + 1px 薄光边(顶部更亮) + 分层柔和阴影 */
section[data-testid="stSidebar"] {
  background: rgba(255, 255, 255, 0.42);
  backdrop-filter: blur(16px) saturate(160%);
  -webkit-backdrop-filter: blur(16px) saturate(160%);
  border-right: 1px solid rgba(255, 255, 255, 0.55);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.75), 0 16px 40px rgba(160, 110, 220, 0.25);
}
section[data-testid="stSidebar"] * { font-size: 14px; }
section[data-testid="stSidebar"] [role="radiogroup"] label { padding: 6px 10px; border-radius: 8px; }
div[data-testid="stMetric"] {
  background: rgba(255, 255, 255, 0.42);
  backdrop-filter: blur(16px) saturate(160%);
  -webkit-backdrop-filter: blur(16px) saturate(160%);
  border: 1px solid rgba(255, 255, 255, 0.55);
  border-radius: 12px;
  padding: 10px 14px;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.75), 0 16px 40px rgba(160, 110, 220, 0.25);
}
div[data-testid="stMetric"] label { color: #6E5C93; }
div[data-testid="stMetricValue"] { font-family: "SF Mono", Menlo, Consolas, monospace; color: #2F6FED; }
[data-testid="stDataFrame"] { font-family: "SF Mono", Menlo, Consolas, monospace; }
div[data-testid="stVerticalBlockBorderWrapper"] {
  background: rgba(255, 255, 255, 0.42);
  backdrop-filter: blur(16px) saturate(160%);
  -webkit-backdrop-filter: blur(16px) saturate(160%);
  border: 1px solid rgba(255, 255, 255, 0.55);
  border-radius: 12px;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.75), 0 16px 40px rgba(160, 110, 220, 0.25);
}

/* 按钮：统一词汇，hover/disabled 状态齐全（玻璃质感，紫色主行动） */
.stButton button {
  border-radius: 8px; border: 1px solid rgba(124, 77, 255, 0.55);
  color: #6A3DF5; background: rgba(255, 255, 255, 0.35);
  backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);
  transition: all 0.18s ease-out;
}
.stButton button:hover { background: rgba(124, 77, 255, 0.14); border-color: #7C4DFF; color: #6A3DF5; }
.stButton button:active { background: rgba(124, 77, 255, 0.22); transform: scale(0.98); }
.stButton button:disabled { opacity: 0.45; border-color: #C9B8F0; color: #9B82D6; }

/* 徽章：状态标签而非图标（amber=紫色强调 / cyan=蓝色数据） */
.qfm-badge { display:inline-block; padding: 2px 10px; border-radius: 20px;
  font-size: 12px; border: 1px solid rgba(80, 40, 160, 0.18); color: #6E5C93; margin-right: 6px;
  background: rgba(255, 255, 255, 0.30); }
.qfm-badge.amber { color: #7C4DFF; border-color: rgba(124, 77, 255, 0.40); }
.qfm-badge.cyan { color: #2F6FED; border-color: rgba(47, 111, 237, 0.40); }
.qfm-desc { color: #6E5C93; font-size: 13px; margin-top: 4px; }
[data-testid="stAppDeployButton"] { display: none; }

/* 无障碍缓冲：用户关闭透明度/动效时，玻璃退化为实底 */
@media (prefers-reduced-transparency: reduce) {
  section[data-testid="stSidebar"],
  div[data-testid="stMetric"],
  div[data-testid="stVerticalBlockBorderWrapper"] {
    backdrop-filter: none; -webkit-backdrop-filter: none; background: #FFFFFF;
  }
}
@media (prefers-reduced-motion: reduce) {
  .stButton button { transition: none; }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 数据加载（st.cache_resource：首次带进度条，之后秒级命中缓存）
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_panel(pool: str, max_stocks: int | None = None):
    with st.status("拉取数据中…", expanded=True) as status:
        codes = get_universe(pool)
        if max_stocks:
            codes = codes[:max_stocks]
        st.write(f"股票池 **{pool}**：{len(codes)} 只（首次拉取较慢，已缓存后秒级）")
        dl = DataLoader()
        prog = st.progress(0.0, text="日线行情（新浪 qfq）")
        bars = dl.load_bars(codes, progress=lambda i, n, c: prog.progress(
            (i + 1) / n, text=f"日线 {i+1}/{n} · {c}"))
        prog2 = st.progress(0.0, text="财务指标（业绩报表）")
        ind = dl.load_indicators(progress=lambda i, n, q: prog2.progress(
            (i + 1) / n, text=f"业绩报表 {i+1}/{n} · {q}"))
        status.update(label="构建数据面板中（首次约 1-3 分钟，此后秒级）…")
        panel = build_panel(bars, ind)
        status.update(label=f"✅ 数据就绪：{panel.close.shape[0]} 个交易日 × {panel.close.shape[1]} 只股票",
                      state="complete")
        return panel


def verdict_of(rep: dict) -> tuple[str, str]:
    s = rep["ic_summary"]
    m = abs(s["ic_mean"]) if pd.notna(s["ic_mean"]) else 0
    t = abs(s["ic_t"]) if pd.notna(s["ic_t"]) else 0
    if m >= 0.03 and t >= 2 and rep["monotonicity"].get("monotonic", False):
        return "✅ 通过核心检验", "#2E9E6B"
    if m >= 0.02 or t >= 2:
        return "⚠️ 边缘有效", "#C56A00"
    return "❌ 未通过", "#F85149"


# ---------------------------------------------------------------------------
# 侧边栏
# ---------------------------------------------------------------------------
st.sidebar.markdown("## 量化因子挖掘流水线")
section = st.sidebar.radio("操作台", [
    "研究项目", "因子库", "因子检验", "因子对比", "多因子实验室",
    "自动挖掘", "自定义因子", "策略回测", "策略对比",
    "数据与股票池", "任务中心", "数据管理",
],
                           label_visibility="collapsed", key="section")

# 自动挖掘子分支：批量挖掘（传统指标穷举）/ ML 合成（LightGBM walk-forward）/ 表达式挖掘（WQ 式）
mine_mode = "批量挖掘"
if section == "自动挖掘":
    mine_mode = st.sidebar.radio("挖掘方式", ["批量挖掘", "ML 合成", "表达式挖掘"], key="mine_mode",
                                 help="批量挖掘=指标×窗口×变换穷举；ML 合成=LightGBM 非线性合成新因子；"
                                      "表达式挖掘=操作符组合候选（WQ 方法论，含 Fitness/去重）")

with st.sidebar.expander("数据参数", expanded=False):
    custom_pools = sorted({
        item.universe_id for item in CATALOG.list_universes()
        if item.universe_id.startswith("custom_")
    })
    # 保持旧版默认值 index800，不改变既有页面首次打开时的数据规模与运行口径。
    pool_options = ["index800", "cn_hs300", "cn_zz500", "cn_zz1000", "cn_all_a", "full", *custom_pools]
    pool_labels = {
        "cn_hs300": "沪深300", "cn_zz500": "中证500", "cn_zz1000": "中证1000",
        "cn_all_a": "全A", "index800": "沪深300 + 中证500（兼容）", "full": "全A（兼容）",
    }
    pool = st.selectbox(
        "股票池", pool_options, key="pool",
        format_func=lambda value: pool_labels.get(value, next(
            (item.name for item in CATALOG.list_universes() if item.universe_id == value), value
        )),
        help="标准股票池与自定义股票池都带 Universe ID；实验保存时会记录具体版本。",
    )
    max_stocks = st.number_input("限制股票数（0=全部）", 0, 6000, 0, key="max_stocks")
    st.caption("首次拉取约 5-15 分钟，之后全部走本地缓存")

dl = DataLoader()
cache_status = dl.status()
st.sidebar.caption(f"缓存状态：日线 {cache_status['stocks_cached']} 只 · "
                   f"财务 {'✅' if cache_status['indicators_cached'] else '❌'}")
active_research_project = active_project_name(RESEARCH_STORE)
if active_research_project:
    st.sidebar.caption(f"当前研究项目：{active_research_project}")

# ---------------------------------------------------------------------------
# ① 因子库
# ---------------------------------------------------------------------------
def _open_factor_test(name):
    st.session_state["test_factor"] = name
    st.session_state["section"] = "因子检验"


def page_library():
    factors_all = list_factors()
    family_count = len({f.family for f in factors_all})
    versioned = sum(1 for f in factors_all if f.version > 1)
    all_tags_local = all_tags()
    st.markdown('<div class="qfm-sig"><h1>因子库</h1>'
                f'<div class="sub">{len(factors_all)} 个内置因子 · {family_count} 大家族 · '
                f'{len(all_tags_local)} 个标签 · 每个因子带版本、公式与定义指纹</div></div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    family = c1.selectbox("家族筛选", ["全部"] + list_families())
    keyword = c2.text_input("关键词搜索", placeholder="如：动量 / roe / 换手 / 估值")
    tag = c3.selectbox("标签筛选", ["全部"] + all_tags_local)
    fs = list_factors(family if family != "全部" else None)
    if tag != "全部":
        fs = [f for f in fs if tag in f.tags]
    if keyword:
        needle = keyword.lower()
        fs = [f for f in fs
              if needle in f.name.lower() or keyword in f.description or needle in (f.formula or "").lower()]
    st.caption(f"共 {len(fs)} 个因子" + (f" · 其中 {versioned} 个因子已有多个版本" if versioned else ""))
    cols = st.columns(3)
    for i, f in enumerate(fs):
        with cols[i % 3].container(border=True):
            badges = "".join(f"<span class='qfm-badge cyan'>{t}</span>" for t in f.tags)
            st.markdown(f"**{factor_label(f.name)}**"
                        f"<span class='qfm-badge amber'>{f.family}</span>"
                        f"<span class='qfm-badge cyan'>{'正向' if f.direction=='positive' else '负向'}</span>"
                        f"<span class='qfm-badge amber'>v{f.version}</span>{badges}",
                        unsafe_allow_html=True)
            st.markdown(f"<div class='qfm-desc'>{f.description}</div>", unsafe_allow_html=True)
            with st.expander("公式 / 参数 / 定义指纹"):
                st.code(f.formula or "（未记录源码）", language="python")
                if f.params:
                    st.caption("注册参数：" + "，".join(f"{k}={v!r}" for k, v in sorted(f.params.items())))
                st.caption(f"定义指纹 {f.source_hash} · 注册于 {f.created_at[:19]}")
                versions = list_factor_versions(f.name)
                if len(versions) > 1:
                    st.markdown("**版本历史**")
                    st.dataframe(
                        pd.DataFrame([
                            {"版本": v.version, "说明": v.description, "方向": v.direction,
                             "定义指纹": v.source_hash[:20], "注册时间": v.created_at[:19]}
                            for v in versions
                        ]),
                        hide_index=True, use_container_width=True,
                    )
            st.button("检验此因子", key=f"test_from_library_{f.name}", use_container_width=True,
                      on_click=_open_factor_test, args=(f.name,))


# ---------------------------------------------------------------------------
# ② 因子检验
# ---------------------------------------------------------------------------
def page_test(panel):
    st.markdown('<div class="qfm-sig"><h1>因子检验</h1>'
                '<div class="sub">一次运行 · 因子预测能力 + 组合模拟 · Sharpe / 回撤 / Fitness</div></div>',
                unsafe_allow_html=True)
    import plotly.graph_objects as go

    tab_single, tab_cmp, tab_ortho = st.tabs(["单因子检验与模拟", "多因子对比", "正交化"])

    # ---------- Tab 1：统一单因子检验与模拟 ----------
    with tab_single:
        from qfm.simulation.views import render_single_factor
        render_single_factor(panel, pool, RESEARCH_STORE, JOB_SERVICE, CATALOG.root)

    # ---------- Tab 2：多因子横向对比 ----------
    with tab_cmp:
        cmp_names = st.multiselect("对比因子（可多选）", [f.name for f in list_factors()], format_func=factor_label,
                                   default=["mom_20", "rev_20", "ep_ttm", "vol_20", "turnover_20"])
        cmp_h = st.selectbox("前瞻天数", [5, 10, 20, 60], index=2, key="cmp_h")
        if st.button("运行对比", use_container_width=True):
            if not cmp_names:
                st.error("至少选择一个因子")
                return
            rows = []
            for nm in cmp_names:
                f = get_factor(nm)
                with st.status(f"检验 {nm}…", expanded=False) as stt:
                    rep = factor_report(compute_factor(nm, panel), panel.close,
                                        horizon=cmp_h, direction=f.direction)
                    stt.update(label=f"✅ {nm}", state="complete")
                s = rep["ic_summary"]
                rows.append({
                    "因子": factor_label(nm), "家族": f.family, "方向": "正" if f.direction == "positive" else "负",
                    "IC": s["ic_mean"], "IC_IR": s["ic_ir"], "t值": s["ic_t"],
                    "近60日IC": rep["ic_recent"], "换手": rep["turnover"],
                })
            cmp_df = pd.DataFrame(rows)
            cmp_df["|IC|"] = cmp_df["IC"].abs()
            cmp_df = cmp_df.sort_values("|IC|", ascending=False).drop(columns="|IC|").reset_index(drop=True)
            st.dataframe(cmp_df.style.format({"IC": "{:+.4f}", "IC_IR": "{:.2f}", "t值": "{:.2f}",
                                              "近60日IC": "{:+.4f}", "换手": "{:.0%}"}),
                         hide_index=True, use_container_width=True)
            fig = go.Figure(go.Bar(
                x=cmp_df["因子"], y=cmp_df["IC"],
                marker_color=["#7C4DFF" if v >= 0 else "#FF6BA9" for v in cmp_df["IC"]],
                text=cmp_df["IC"].round(4), textposition="outside"))
            fig.add_hline(y=0, line=dict(color="#C9B8F0", width=1, dash="dash"))
            fig.update_layout(title=f"因子 IC 横向对比（前瞻 {cmp_h} 日）", height=340, margin=dict(t=50),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="#3B2E5E"))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("所有结果均已按因子方向规范为“高分更好”：紫色为正向有效 IC，粉色提示方向或稳定性需复核。")

    # ---------- Tab 3：正交化（QuantSkills factor-orthogonalize 方法论） ----------
    with tab_ortho:
        st.markdown("逐日截面 OLS 正交化：剥离行业 / 市值 / 风格暴露 → 残差因子"
                    "（方法论源自 QuantSkills factor-orthogonalize）")
        c1, c2, c3 = st.columns([2, 1, 1])
        ortho_name = c1.selectbox("选择因子", [f.name for f in list_factors()], key="ortho_f",
                              format_func=factor_label)
        ortho_h = c2.selectbox("前瞻天数", [5, 10, 20, 60], index=2, key="ortho_h")
        ortho_controls = c3.multiselect("剥离暴露", ["industry", "size", "style"],
                                        default=["industry", "size", "style"],
                                        format_func=lambda c: {"industry": "行业", "size": "市值",
                                                               "style": "风格(beta/波动率)"}[c])
        if st.button("运行正交化", use_container_width=True):
            if not ortho_controls:
                st.error("至少选择一项剥离暴露")
                return
            from qfm.orthogonalize import orthogonalize_factor
            from qfm.portfolio.synthesis import factor_panel
            with st.status("正交化中…", expanded=False) as status:
                fdf = factor_panel(panel, [ortho_name])[ortho_name]
                resid, diag = orthogonalize_factor(panel, fdf, controls=tuple(ortho_controls),
                                                   horizon=ortho_h)
                status.update(label=f"✅ 完成：{ortho_name} → 残差因子（{diag['days']} 日有效）",
                              state="complete")
            k1, k2, k3 = st.columns(3)
            k1.metric("IC 保留率", f"{diag['ic_retention']:.0%}" if pd.notna(diag["ic_retention"]) else "—",
                      help="正交后 IC / 正交前 IC；越低说明信号越依赖被剥离暴露")
            k2.metric("暴露 R²（前→后）", f"{diag['exposure_before']:.3f} → {diag['exposure_after']:.3f}",
                      help="信号对控制变量回归 R²：正交后应≈0")
            k3.metric("覆盖率（前→后）", f"{diag['coverage_before']:.0%} → {diag['coverage_after']:.0%}")
            cmp = pd.DataFrame({
                "指标": ["IC", "暴露 R²", "TOP10% 换手", "覆盖率"],
                "正交前": [diag["ic_before"], diag["exposure_before"],
                           diag["turnover_before"], diag["coverage_before"]],
                "正交后": [diag["ic_after"], diag["exposure_after"],
                           diag["turnover_after"], diag["coverage_after"]],
            })
            st.dataframe(cmp.style.format({"正交前": "{:.4f}", "正交后": "{:.4f}"}),
                         hide_index=True, use_container_width=True)
            if diag["days_skipped"]:
                st.caption(f"跳过 {diag['days_skipped']} 日（截面样本 <30）")
            st.download_button("⬇ 下载残差因子 CSV", resid.to_csv().encode("utf-8-sig"),
                               file_name=f"{ortho_name}_residual.csv", mime="text/csv")


# ---------------------------------------------------------------------------
# ③ 自动挖掘
# ---------------------------------------------------------------------------
def page_mine(panel, mode: str = "批量挖掘"):
    st.markdown('<div class="qfm-sig"><h1>自动挖掘</h1>'
                '<div class="sub">基础指标 × 窗口 × 变换 → 批量生成候选 → 批量检验 → TOP 排行榜</div></div>',
                unsafe_allow_html=True)
    if mode == "ML 合成":
        page_mine_ml(panel)
        return
    if mode == "表达式挖掘":
        page_mine_expr(panel)
        return
    c1, c2, c3 = st.columns([1, 1, 1])
    horizon = c1.selectbox("前瞻天数", [5, 10, 20, 60], index=2, key="mine_h")
    max_c = c2.selectbox("候选数限制", [60, 120, 244, 500], index=2,
                         help="v2 全量 244 = 49 个因子 × 4 种变换 + 基础窗口集；每候选约 2 秒（流式计算，内存占用低）")
    trial_top_n = c3.selectbox("试验组合持仓数", [10, 20, 30, 50], index=2,
                               help="试验矩阵只可用于持仓数与调仓频率一致的策略 PBO / DSR 检验")
    save_trials = st.checkbox("保存试验矩阵（供策略回测页过拟合检验使用）", value=True,
                              help="逐候选计算月频 TOP-30 组合收益，落盘 data_cache/trials/")
    if st.button("开始挖掘", use_container_width=True):
        prog = st.progress(0.0, text="准备…")
        df, trials_meta = run_mining(panel, horizon=horizon, max_candidates=max_c,
                                     save_trials=save_trials, top_n=trial_top_n,
                                     trial_context={"pool": st.session_state.get("pool", "index800")},
                                     progress=lambda i, n, nm: prog.progress((i + 1) / n, text=f"{i+1}/{n} · {nm}"))
        st.success(f"挖掘完成：{len(df)} 个候选因子")
        st.dataframe(df.style.format({"IC": "{:+.4f}", "IC_IR": "{:.2f}", "t值": "{:.2f}",
                                      "正占比": "{:.0%}", "换手率": "{:.0%}"}),
                     use_container_width=True, height=420)
        if trials_meta:
            st.session_state["trial_experiment"] = trials_meta
            st.caption(f"📁 试验矩阵已保存：`{trials_meta['path']}`"
                       f"（{trials_meta['n_trials']} 候选 × {trials_meta['T_periods']} 期月频收益 · "
                       f"TOP{trials_meta['top_n']} 等权）")
        st.download_button("⬇ 下载排行榜 CSV", df.to_csv(index=False).encode("utf-8-sig"),
                           file_name="factor_leaderboard.csv", mime="text/csv")


def page_mine_ml(panel):
    """自动挖掘 · ML 合成子分支：LightGBM walk-forward 非线性合成新因子"""
    import plotly.graph_objects as go

    st.markdown("LightGBM walk-forward 滚动训练：把现有因子库**非线性合成**成一个新因子"
                "（仅输出样本外预测，结构性无未来函数）")
    c1, c2, c3 = st.columns([2, 1, 1])
    ml_names = c1.multiselect("特征因子", [f.name for f in list_factors()], format_func=factor_label,
                              default=[f.name for f in list_factors()][:10], key="ml_names")
    ml_h = c2.selectbox("前瞻天数", [5, 10, 20, 60], index=2, key="ml_h")
    ml_folds = c3.selectbox("fold 数", [2, 4, 6], index=1, key="ml_folds",
                            help="越多训练越充分，耗时线性增加")
    ml_cutoff = st.text_input("训练截止日（此后的日期为样本外预测区间）", "2023-12-31", key="ml_cutoff")
    if st.button("训练并合成", use_container_width=True):
        if not ml_names:
            st.error("至少选择一个特征因子")
            return
        from qfm.ml_synthesizer import synthesize_ml_factor

        prog = st.progress(0.0, text="准备数据…")
        try:
            pred, meta = synthesize_ml_factor(
                panel, names=ml_names, horizon=ml_h, train_cutoff=ml_cutoff,
                n_folds=ml_folds,
                progress=lambda i, n, msg: prog.progress((i + 1) / n, text=msg))
        except Exception as e:  # noqa: BLE001
            st.error(f"训练失败：{e}")
            return
        st.success(f"合成完成：样本外预测 {len(pred)} 个交易日 × {pred.shape[1]} 只股票"
                   f"（fold={meta['n_folds']}，区间 {meta['pred_start']} ~ {meta['pred_end']}）")
        rep = factor_report(pred, panel.close, horizon=ml_h, direction="positive")
        s = rep["ic_summary"]
        k1, k2, k3 = st.columns(3)
        k1.metric("样本外 IC", f"{s['ic_mean']:+.4f}" if pd.notna(s["ic_mean"]) else "—")
        k2.metric("IC_IR", f"{s['ic_ir']:.2f}" if pd.notna(s["ic_ir"]) else "—")
        k3.metric("有效天数", s["n_days"])
        st.plotly_chart(ic_chart(rep), use_container_width=True)
        st.plotly_chart(layer_chart(rep, "positive"), use_container_width=True)

        imp = meta["importance_top"]
        figi = go.Figure(go.Bar(
            x=list(imp.values()), y=[k.replace("f_", "") for k in imp],
            orientation="h", marker_color="#7C4DFF"))
        figi.update_layout(title="特征重要性 TOP10", height=320, margin=dict(t=40),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font=dict(color="#3B2E5E"))
        st.plotly_chart(figi, use_container_width=True)

        from qfm.factors import register_factor

        # lambda 的源码会随 app.py 变动，故用训练参数摘要作为稳定定义指纹，
        # 否则同一份 ML 因子每次都会"升级版本"，实验归档无法核对。
        ml_source_key = (f"lightgbm:horizon={ml_h};folds={ml_folds};cutoff={ml_cutoff};"
                         f"features={'|'.join(sorted(ml_names))}")
        register_factor(name="ml_synth", family="机器学习",
                        description=f"LightGBM walk-forward 合成（{ml_h}日前瞻，训练截止 {ml_cutoff}）",
                        direction="positive", source_key=ml_source_key,
                        params={"horizon": ml_h, "folds": ml_folds, "cutoff": ml_cutoff})(
                            lambda d: pred)
        st.caption(f"✅ 已注册为因子 `ml_synth`，可直接在「策略回测」页选择。"
                   f"预测区间 {meta['pred_start']} ~ {meta['pred_end']}——回测起点请设在此区间内或之后。")


def page_mine_expr(panel):
    """自动挖掘 · 表达式挖掘子分支：WQ 式操作符组合候选 → 批量检验 → Fitness/去重排行榜"""
    from qfm.mining.fields import tech_fields
    from qfm.mining.expressions import EXPR_OPS
    from qfm.mining.engine import run_mining_expr
    from qfm.factors import list_factors, ZH_NAMES

    st.markdown("**WQ 式表达式挖掘**：`操作符(字段)` 组合候选（rank / ts_rank / ts_mean / ts_zscore / "
                "ts_std / decay_linear / delta / abs）→ 批量检验 → 排行榜（**Fitness + 相关性去重**）→ "
                "试验矩阵（供策略回测 PBO）")

    fields_all = ([f.name for f in list_factors()]
                  + ["ret", "turnover", "amount", "volume"]
                  + list(tech_fields(panel)))
    default_fields = ([f.name for f in list_factors()][:6]
                      + list(tech_fields(panel))[:4])
    c1, c2, c3 = st.columns([2, 1, 1])
    fields_sel = c1.multiselect(
        "字段池", fields_all, default=default_fields,
        format_func=lambda n: ZH_NAMES.get(n, n), key="expr_fields")
    ops_sel = c2.multiselect("操作符", EXPR_OPS,
                             default=["rank", "ts_mean", "ts_zscore", "delta"], key="expr_ops")
    n_sel = c3.multiselect("窗口", [5, 10, 20, 60], default=[5, 20], key="expr_n")
    c4, c5, c6 = st.columns(3)
    horizon = c4.selectbox("前瞻天数", [5, 10, 20, 60], index=2, key="expr_h")
    max_c = c5.number_input("候选上限（0=全部）", 0, 5000, 300, key="expr_max")
    top_n = c6.selectbox("TOP-N 组合（试验矩阵）", [10, 20, 30, 50], index=2, key="expr_topn")

    if st.button("挖掘表达式候选", use_container_width=True):
        if not fields_sel or not ops_sel:
            st.error("至少选择一个字段和一个操作符")
            return
        prog = st.progress(0.0, text="准备字段池…")
        t0 = pd.Timestamp.now()
        try:
            lb, meta = run_mining_expr(
                panel, horizon=horizon, max_candidates=int(max_c) or None,
                save_trials=True, top_n=int(top_n),
                fields=fields_sel, ops=ops_sel, n_list=tuple(int(x) for x in n_sel),
                progress=lambda i, n, msg: prog.progress(
                    min((i + 1) / max(n, 1), 1.0), text=f"{i + 1}/{n} {msg}"))
        except Exception as e:  # noqa: BLE001
            st.error(f"挖掘失败：{e}")
            return
        cost = (pd.Timestamp.now() - t0).total_seconds()
        k1, k2, k3 = st.columns(3)
        k1.metric("候选数", len(lb))
        k2.metric("耗时", f"{cost:.0f}s")
        k3.metric("试验矩阵", meta["n_trials"] if meta else "—")
        st.dataframe(lb, use_container_width=True, hide_index=True)
        st.caption("列说明：此处 `Fitness` 是旧版月频收益/名单换手近似分数，与「因子检验」的日频成交口径不同；`去重组`=按月频收益相关贪心去重"
                   "（同组号只保留 Fitness 最高者）。试验矩阵保存后可在「策略回测」页加载做 PBO。")
        st.download_button("⬇ 下载排行榜 CSV", lb.to_csv(index=False).encode("utf-8-sig"),
                           file_name="expr_leaderboard.csv", mime="text/csv")
        if meta and meta.get("path"):
            st.caption(f"📁 试验矩阵：`{meta['path']}`"
                       f"（{meta['n_trials']} 候选 × {meta['T_periods']} 期月频收益）")


# ---------------------------------------------------------------------------
# ④ 自定义因子
# ---------------------------------------------------------------------------
CUSTOM_TEMPLATE = '''# 每个因子 = 一个带装饰器的函数，函数签名: def f(d: DataPanel) -> DataFrame
@register_factor(name="my_factor", family="自定义", description="一句话描述你的想法", direction="positive")
def my_factor(d):
    # d.close / d.volume / d.amount / d.turnover / d.mv_float: 日期×股票 透视表
    # d.fund["roe"] / d.fund["eps_ttm"] / ...: 财务因子（按公告日对齐）
    return d.close.pct_change(20) / d.turnover.rolling(20).mean()  # 例：量比修正动量
'''


def validate_custom_factor_source(code: str) -> list[str]:
    """阻止明显的进程/文件/动态执行入口；这不是隔离容器的替代品。"""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"语法错误：{exc.msg}（第 {exc.lineno} 行）"]
    blocked_nodes = (ast.Import, ast.ImportFrom, ast.With, ast.AsyncWith, ast.ClassDef,
                     ast.Global, ast.Nonlocal, ast.Delete, ast.Try, ast.Raise)
    blocked_calls = {"open", "exec", "eval", "compile", "globals", "locals", "vars",
                     "getattr", "setattr", "delattr", "input", "breakpoint", "__import__"}
    errors = []
    for node in ast.walk(tree):
        if isinstance(node, blocked_nodes):
            errors.append(f"不允许 {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            errors.append("不允许双下划线名称")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            errors.append("不允许访问私有属性")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in blocked_calls:
            errors.append(f"不允许调用 {node.func.id}")
    return list(dict.fromkeys(errors))


def page_custom(panel):
    st.markdown('<div class="qfm-sig"><h1>自定义因子</h1>'
                '<div class="sub">把你的主观交易经验写成因子，立即进入统一检验流程</div></div>',
                unsafe_allow_html=True)
    st.code("从 qfm.factors 导入后，装饰器会自动注册；模板如下（可直接改）：", language=None)
    st.warning("自定义代码会在本机研究进程中运行。仅粘贴你信任的公式；服务默认只监听本机，不应暴露到局域网。")
    code = st.text_area("因子代码", CUSTOM_TEMPLATE, height=260)
    c1, c2, c3 = st.columns([1, 1, 2])
    if c1.button("注册并检验", use_container_width=True):
        try:
            source_errors = validate_custom_factor_source(code)
            if source_errors:
                st.error("代码未执行：" + "；".join(source_errors))
                return
            safe_builtins = {"abs": abs, "min": min, "max": max, "sum": sum,
                             "len": len, "range": range, "float": float, "int": int}
            ns = {"__builtins__": safe_builtins, "pd": pd, "np": np,
                  "register_factor": register_factor}
            from qfm.data.panel import DataPanel
            ns["DataPanel"] = DataPanel
            before = set(f.name for f in list_factors())
            exec(code, ns)
            from qfm.pipeline.lookahead import scan_source
            chk = scan_source(code)
            if chk["leaks"]:
                st.error("⚠️ 检测到未来函数泄漏模式：" + "；".join(chk["leaks"]) +
                         "（因子将引用未来数据，检验结果不可信）")
            elif chk["warnings"]:
                st.warning("提示：" + "；".join(chk["warnings"]))
            st.success("注册成功，已自动进入检验流程")
            # 从注册表差集识别新因子，不依赖名称必须以 my_ 开头。
            new_names = sorted(set(f.name for f in list_factors()) - before)
            if new_names:
                name = new_names[-1]
                f = get_factor(name)
                if f:
                    with st.status("检验中…", expanded=False) as status:
                        rep = factor_report(compute_factor(name, panel), panel.close,
                                            horizon=20, direction=f.direction)
                        status.update(label=f"✅ 完成：{name}", state="complete")
                    s = rep["ic_summary"]
                    st.metric("平均 IC", f"{s['ic_mean']:+.4f}" if pd.notna(s["ic_mean"]) else "—")
                    st.plotly_chart(ic_chart(rep), use_container_width=True)
                    st.plotly_chart(layer_chart(rep, f.direction), use_container_width=True)
        except Exception as e:  # noqa: BLE001
            st.error(f"注册失败：{e}")


# ---------------------------------------------------------------------------
# ⑤ 策略回测（一体化：合成 → 回测 → 绩效）
# ---------------------------------------------------------------------------
def page_strategy(panel):
    st.markdown('<div class="qfm-sig"><h1>策略回测</h1>'
                '<div class="sub">因子合成 → 组合回测 → 绩效分析 · 一体化流水线</div></div>',
                unsafe_allow_html=True)
    st.markdown("**① 因子合成**")
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    names = c1.multiselect("选择因子", [f.name for f in list_factors()], format_func=factor_label,
                           default=["ep_ttm", "roe", "rev_20"])
    mode = c2.selectbox("权重模式", ["equal", "ic", "icir", "ic_x_ir"],
                        format_func=lambda m: {"equal": "等权", "ic": "IC 加权", "icir": "ICIR 加权",
                                               "ic_x_ir": "IC × IR 加权"}[m])
    horizon = c3.selectbox("前瞻天数", [5, 10, 20, 60], index=2, key="strategy_h")
    weight_lookback = c4.selectbox("估权回看窗口", [126, 252, 504], index=1,
                                   disabled=mode == "equal",
                                   help="仅 IC / IC_IR 权重使用；每次估权都会剔除最近前瞻期，防止未来数据泄漏")
    with st.expander("因子正交化（行业 / 市值 / 风格暴露剥离）"):
        ortho = st.checkbox("启用正交化", value=False,
                            help="逐日截面 OLS 残差化，消除所选暴露（方法论源自 QuantSkills factor-orthogonalize）")
        ortho_controls = st.multiselect("剥离暴露", ["industry", "size", "style"], default=["size"],
                                        format_func=lambda c: {"industry": "行业", "size": "市值",
                                                               "style": "风格(beta/波动率)"}[c])

    st.markdown("**② 组合回测**")
    c5, c6, c7, c8 = st.columns([1, 1, 1, 1])
    top_n = c5.slider("最多持仓数量", 10, 100, 30, step=5)
    start_d = c6.selectbox("回测起点", ["2021-01-01", "2022-01-01", "2023-01-01"], index=0)
    bench_mode = c7.selectbox("基准", ["equal", "mv"],
                              format_func=lambda m: {"equal": "全池等权", "mv": "市值加权"}[m])
    rebalance_choice = c8.selectbox(
        "调仓频率", ["ME", "W-FRI", "QE", "B", "每 N 个交易日"],
        format_func=lambda r: {"ME": "月末", "W-FRI": "周五", "QE": "季末",
                               "B": "每日", "每 N 个交易日": "每 N 个交易日"}[r])
    if rebalance_choice == "每 N 个交易日":
        rebalance = st.number_input("调仓间隔（交易日）", 2, 250, 20, step=1)
    else:
        rebalance = rebalance_choice
    with st.expander("成交与成本假设（T 日收盘信号 → T+1 开盘成交）"):
        ca, cb, cc, cd, ce = st.columns(5)
        commission_bp = ca.number_input("佣金（bp，双边）", 0.0, 30.0, 3.0, 0.5)
        stamp_bp = cb.number_input("印花税（bp，卖出）", 0.0, 30.0, 5.0, 0.5)
        impact_bp = cc.number_input("冲击成本（bp，双边）", 0.0, 100.0, 10.0, 1.0)
        slippage_bp = cd.number_input("滑点（bp，单边）", 0.0, 100.0, 0.0, 1.0,
                                      help="直接作用于成交价：买入 = 开盘×(1+滑点)，卖出 = 开盘×(1-滑点)。"
                                           "与冲击成本（按成交额计提的费用）相互独立。")
        max_participation = ce.slider("单日成交额参与率上限", 0.01, 0.20, 0.05, 0.01)
        initial_capital = st.number_input("模拟初始资金（元）", 100_000, 1_000_000_000,
                                          1_000_000, step=100_000)
        st.caption("模型会阻止停牌、开盘涨停买入与开盘跌停卖出；科创/创业板按 20%、北交所按 30% 近似。")
        st.warning(
            "**ST 状态在当前数据源中不存在**，因此回测按普通股票（10%/20%/30% 涨跌幅）处理。"
            "引擎已支持 `st` 布尔面板（ST 按 5% 涨跌幅且不进入选股目标），"
            "接入带 ST 标记的数据源后即可生效 —— 现在无法启用，这是数据缺口而非已实现功能。"
        )
    with st.expander("组合风险约束（目标组合）"):
        cr1, cr2 = st.columns(2)
        max_stock_weight = cr1.slider("单股目标权重上限", 0.01, 1.00, 1.00, 0.01)
        max_industry_weight = cr2.slider("单行业目标权重上限", 0.01, 1.00, 1.00, 0.01)
        use_turnover_budget = st.checkbox("限制单次调仓成交额", value=False)
        max_rebalance_turnover = (
            st.slider("单次调仓成交额上限（组合净值）", 0.05, 2.00, 0.50, 0.05)
            if use_turnover_budget else None
        )
        st.caption("约束在信号目标组合上生效。上限过严或行业候选不足时会保留现金；涨跌停、停牌和成交额参与率限制可能使实际仓位延后收敛。")

    if st.button("运行策略回测", use_container_width=True):
        if not names:
            st.error("至少选择一个因子")
            return
        from qfm.portfolio import (PortfolioConstraints, factor_corr, factor_panel,
                                   run_backtest, standard_metrics, synthesize, yearly_perf)
        constraints = PortfolioConstraints(
            max_stock_weight=max_stock_weight,
            max_industry_weight=max_industry_weight,
            max_rebalance_turnover=max_rebalance_turnover,
        )
        with st.status("策略流水线运行中…", expanded=True) as status:
            costs = {"commission": commission_bp / 10_000,
                     "stamp": stamp_bp / 10_000,
                     "impact": impact_bp / 10_000}
            snapshot = build_data_snapshot(panel, pool)
            binding = register_panel_dataset(panel, pool, snapshot, root=CATALOG.root)
            request = {
                **binding,
                "universe": pool,
                "date_range": [start_d, str(panel.close.index.max().date())],
                "factor_versions": [
                    {"name": name, "version": get_factor(name).version,
                     "source_hash": get_factor(name).source_hash}
                    for name in names
                ],
                "pipeline_config": {
                    "mode": mode, "horizon": horizon, "weight_lookback": weight_lookback,
                    "orthogonalize": ortho, "ortho_controls": ortho_controls,
                },
                "backtest_config": {
                    "top_n": top_n, "start": start_d, "rebalance": rebalance,
                    "benchmark": bench_mode, "initial_capital": float(initial_capital),
                    "max_participation": max_participation, "costs": costs,
                    "slippage": slippage_bp / 10_000,
                    "constraints": {
                        "max_stock_weight": max_stock_weight,
                        "max_industry_weight": max_industry_weight,
                        "max_rebalance_turnover": max_rebalance_turnover,
                    },
                },
            }

            def compute_backtest():
                composite, factor_weights = synthesize(
                    panel, names, mode=mode, horizon=horizon,
                    orthogonalize=ortho,
                    ortho_controls=tuple(ortho_controls) if ortho else None,
                    weight_lookback=weight_lookback,
                    weight_rebalance=rebalance,
                )
                backtest = run_backtest(
                    panel, composite, top_n=top_n, start=start_d, rebalance=rebalance,
                    bench_mode=bench_mode, initial_capital=float(initial_capital),
                    max_participation=max_participation, cost=costs, constraints=constraints,
                    slippage=slippage_bp / 10_000,
                )
                return composite, factor_weights, backtest

            job_result = JOB_SERVICE.run("BACKTEST", request, compute_backtest)
            score, weights, bt = job_result.value
            cache_note = " · 已读取缓存" if job_result.job.cache_hit else ""
            status.update(label=f"✅ 完成（成本后）：策略净值 {bt.nav.iloc[-1]:.2f} vs 基准 {bt.bench_nav.iloc[-1]:.2f}{cache_note}",
                          state="complete")

        st.session_state["latest_strategy_run"] = build_strategy_run_payload(
            panel=panel,
            pool=pool,
            names=names,
            mode=mode,
            horizon=horizon,
            weight_lookback=weight_lookback,
            orthogonalize=ortho,
            ortho_controls=tuple(ortho_controls) if ortho else (),
            top_n=top_n,
            start_date=start_d,
            rebalance=rebalance,
            bench_mode=bench_mode,
            costs=costs,
            max_participation=max_participation,
            initial_capital=float(initial_capital),
            weights=weights,
            backtest=bt,
        )

        st.caption(f"研究口径：{pool} 股票池 · {start_d} 起 · {('月末' if rebalance == 'ME' else '周五')}调仓 · "
                   f"T+1 开盘成交 · 成本后净值 · 单日成交额参与率≤{max_participation:.0%}。")

        # 权重 + 相关性
        wdf = pd.DataFrame({"因子": [factor_label(n) for n in weights], "权重": list(weights.values())})\
    .sort_values("权重", ascending=False)
        st.markdown("**因子权重 & 相关性**")
        cw, cc = st.columns([1, 2])
        cw.dataframe(wdf.style.format({"权重": "{:.1%}"}), hide_index=True, use_container_width=True)
        import plotly.graph_objects as go
        corr = factor_corr(factor_panel(panel, names, align_direction=True))
        fig = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1,
            colorscale="Purples", text=corr.round(2), texttemplate="%{text}"))
        fig.update_layout(title="因子截面相关性", height=280, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#3B2E5E"))
        cc.plotly_chart(fig, use_container_width=True)
        if mode != "equal":
            with st.expander("滚动因子权重（仅使用当时已实现收益）"):
                wh = score.attrs.get("weight_history")
                if wh is not None:
                    st.line_chart(wh.rename(columns={n: factor_label(n) for n in wh.columns}))
                    st.caption(f"估权窗口：{weight_lookback} 个交易日；每次估权排除最近 {horizon} 日前瞻收益。表格展示末期权重。")

        # 净值曲线
        st.markdown("**净值曲线**")
        fig2 = go.Figure()
        bench_label = "基准（市值加权）" if bench_mode == "mv" else "基准（全池等权）"
        fig2.add_trace(go.Scatter(x=bt.nav.index, y=bt.nav.values, name="策略",
                                  line=dict(color="#7C4DFF", width=2.2)))
        fig2.add_trace(go.Scatter(x=bt.bench_nav.index, y=bt.bench_nav.values, name=bench_label,
                                  line=dict(color="#FF6BA9", width=1.5, dash="dash")))
        fig2.update_layout(height=380, hovermode="x unified", paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#3B2E5E"),
                           legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig2, use_container_width=True)

        # 最近持仓明细
        with st.expander("最近调仓日持仓"):
            active = bt.holdings[bt.holdings.sum(axis=1) > 0]
            if len(active):
                last_hold = active.iloc[-1]
                pos = last_hold[last_hold > 0].sort_values(ascending=False)
                st.dataframe(pd.DataFrame({"股票代码": pos.index, "权重": pos.values})
                             .style.format({"权重": "{:.2%}"}), hide_index=True, use_container_width=True)

        # 行业暴露
        if bt.industry_exposure is not None and len(bt.industry_exposure):
            st.markdown("**行业暴露（最近调仓日）**")
            last_expo = bt.industry_exposure.iloc[-1]
            last_expo = last_expo[last_expo > 1e-6].sort_values(ascending=False)
            shown_expo = last_expo.head(10).copy()
            if len(last_expo) > len(shown_expo):
                shown_expo.loc["其他行业"] = last_expo.iloc[len(shown_expo):].sum()
            figx = go.Figure(go.Bar(
                x=shown_expo.index, y=shown_expo.values, marker_color="#7C4DFF",
                text=shown_expo.round(3), texttemplate="%{text:.0%}", textposition="outside"))
            figx.update_layout(height=300, margin=dict(t=30), paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#3B2E5E"),
                               xaxis_tickangle=-30)
            st.plotly_chart(figx, use_container_width=True)
            hhi = float((last_expo ** 2).sum())
            st.caption(f"行业集中度 HHI：**{hhi:.3f}**（1.0=全押一个行业；越低越分散）。"
                       f"图表仅展示非零的前 10 个行业，其余合并；HHI 仍使用全部行业计算。")

        # 绩效指标（统一走 standard_metrics，与 Tear Sheet / 实验归档同一口径）
        st.markdown("**绩效指标**")
        stats = standard_metrics(bt.nav, bt.bench_nav, turnover=bt.turnover)
        kcols = st.columns(6)
        kcols[0].metric("总收益", f"{stats.get('总收益', float('nan')):+.1%}")
        kcols[1].metric("年化收益", f"{stats.get('年化收益', float('nan')):+.1%}")
        kcols[2].metric("夏普比率", f"{stats.get('夏普比率', float('nan')):.2f}")
        kcols[3].metric("最大回撤", f"{stats.get('最大回撤', float('nan')):.1%}")
        kcols[4].metric("年化超额", f"{stats.get('年化超额', float('nan')):+.1%}")
        kcols[5].metric("年化换手", f"{bt.turnover:.1f}")
        kd1, kd2, kd3, kd4 = st.columns(4)
        kd1.metric("索提诺比率", f"{stats.get('索提诺比率', float('nan')):.2f}")
        kd2.metric("卡玛比率", f"{stats.get('卡玛比率', float('nan')):.2f}")
        kd3.metric("信息比率", f"{stats.get('信息比率', float('nan')):.2f}")
        kd4.metric("跟踪误差", f"{stats.get('跟踪误差', float('nan')):.2%}")
        if slippage_bp:
            st.caption(f"成交价已计入单边 {slippage_bp:.0f}bp 滑点（买入抬价、卖出压价），与冲击成本分别记账。")
        kc1, kc2, kc3 = st.columns(3)
        kc1.metric("累计交易成本", f"{bt.cost_total:.2%}", help="相对初始资金的已发生佣金、印花税与冲击成本")
        kc2.metric("成交成本率", f"{bt.cost_pct:.2%}", help="累计成本 / 实际成交额")
        kc3.metric("期末现金权重", f"{bt.cash_weight.iloc[-1]:.1%}",
                   help="涨跌停、停牌或参与率限制可能导致部分订单未成交并保留现金")

        if bt.constraint_history is not None and not bt.constraint_history.empty:
            st.markdown("**组合约束执行**")
            latest_constraint = bt.constraint_history.iloc[-1]
            cc1, cc2, cc3, cc4, cc5, cc6 = st.columns(6)
            cc1.metric("目标现金", f"{latest_constraint['target_cash']:.1%}")
            cc2.metric("实际现金", f"{latest_constraint['actual_cash']:.1%}")
            cc3.metric("目标偏离", f"{latest_constraint['target_tracking_error']:.1%}")
            cc4.metric("实际最大单股", f"{latest_constraint['actual_max_stock_weight']:.1%}")
            cc5.metric("实际最大行业", f"{latest_constraint['actual_max_industry_weight']:.1%}")
            cc6.metric(
                "实际超限",
                "是" if latest_constraint["stock_cap_exceeded"] or latest_constraint["industry_cap_exceeded"] else "否",
            )
            st.caption("“实际超限”是执行日收盘快照；它可能来自涨跌停、停牌、成交参与率限制或开盘到收盘价格变化，并不表示目标构建越过了约束。")
            with st.expander("组合约束执行记录"):
                constraint_columns = [
                    "signal_date", "execution_date", "target_positions", "target_invested", "target_cash",
                    "target_max_stock_weight", "target_max_industry_weight", "gross_turnover",
                    "applied_gross_turnover", "turnover_budget", "budget_binding", "turnover_scale",
                    "actual_invested", "actual_cash", "actual_positions", "actual_max_stock_weight",
                    "actual_max_industry_weight", "target_tracking_error", "stock_cap_exceeded",
                    "industry_cap_exceeded",
                ]
                st.dataframe(
                    bt.constraint_history[constraint_columns].style.format({
                        "target_invested": "{:.1%}", "target_cash": "{:.1%}",
                        "target_max_stock_weight": "{:.1%}", "target_max_industry_weight": "{:.1%}",
                        "gross_turnover": "{:.1%}", "applied_gross_turnover": "{:.1%}",
                        "turnover_budget": "{:.1%}", "turnover_scale": "{:.3f}",
                        "actual_invested": "{:.1%}", "actual_cash": "{:.1%}",
                        "actual_max_stock_weight": "{:.1%}", "actual_max_industry_weight": "{:.1%}",
                        "target_tracking_error": "{:.1%}",
                    }),
                    hide_index=True,
                    use_container_width=True,
                )

        yp = yearly_perf(bt.nav)
        st.markdown("**分年绩效**")
        st.dataframe(yp.style.format({"收益": "{:+.1%}", "最大回撤": "{:.1%}", "日胜率": "{:.0%}"}),
                     hide_index=True, use_container_width=True)

        with st.expander(f"实际成交流水（{len(bt.trades)} 笔）"):
            if bt.trades.empty:
                st.info("回测期内没有产生可成交订单。")
            else:
                st.dataframe(bt.trades.sort_values("date", ascending=False).style.format(
                    {"notional": "{:,.0f}", "price": "{:.3f}", "quantity": "{:,.0f}", "cost": "{:,.2f}"}),
                             hide_index=True, use_container_width=True)
                st.download_button("⬇ 下载成交流水 CSV", bt.trades.to_csv(index=False).encode("utf-8-sig"),
                                   file_name="strategy_trades.csv", mime="text/csv")

        # 过拟合检验（QuantSkills skill-backtest-overfit 方法论）
        st.markdown("**过拟合检验**（DSR / PBO / Haircut / MinTRL）")
        monthly = bt.nav.resample("ME").last().pct_change().dropna()
        if len(monthly) < 6:
            st.warning("回测期过短（<6 个月），无法做统计显著性检验")
        else:
            from qfm.mining import latest_trials
            from qfm.overfit import overfit_report

            tl = latest_trials()
            trials_df, trials_meta = (tl[0], tl[1]) if tl else (None, None)
            default_n = int((trials_meta or {}).get("n_trials", 36))
            n_trials = st.number_input("试验次数 n_trials（诚实申报：得到该结果前试过的全部参数/候选配置数）",
                                       min_value=1, max_value=100000, value=default_n,
                                       help="来自最近一次自动挖掘的候选数；少报 = 自欺，DSR/PBO 会偏乐观")
            use_matrix = False
            if trials_df is not None:
                use_matrix = st.checkbox(
                    f"使用最近一次挖掘试验矩阵（{trials_df.shape[1]} 候选 × {trials_df.shape[0]} 期，"
                    f"{trials_meta['created_at'][:10]} 生成，horizon={trials_meta['horizon']}）",
                    value=True)
            rep = overfit_report(monthly.values, n_trials=int(n_trials),
                                 trials_matrix=trials_df.values if (use_matrix and trials_df is not None) else None,
                                 periods_per_year=12, haircut_method="holm")
            vcolor = {"PASS": "#2E9E6B", "FAIL": "#E5484D", "INSUFFICIENT": "#C56A00"}[rep.verdict]
            st.markdown(
                f"<span style='color:{vcolor};font-weight:600;font-size:15px'>结论：{rep.verdict}"
                f" — {'统计上可区分于多重检验噪声' if rep.passed else '存在过拟合 / 数据挖掘嫌疑'}</span>",
                unsafe_allow_html=True)
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("DSR 削减夏普", f"{rep.dsr:.2f}" if pd.notna(rep.dsr) else "—",
                      help="P[真实 SR > 期望最大 SR_N]；≥0.95 才可信")
            k2.metric("PBO 过拟合概率", f"{rep.pbo:.2f}" if pd.notna(rep.pbo) else "—",
                      help="IS 最优策略在 OOS 落入下半区的概率；<0.5 才安全；无试验矩阵时为 —")
            k3.metric("Haircut 夏普（年化）", f"{rep.sr_annual:.2f} → {rep.sr_annual_adjusted:.2f}",
                      help=f"多重检验打折（{rep.haircut_method}，n_trials={rep.n_trials}）")
            k4.metric("MinTRL 最小样本期数", f"{rep.minimum_track_record_length:.0f}"
                      if np.isfinite(rep.minimum_track_record_length) else "∞",
                      help="该夏普达到统计显著（PSR≥95%）所需的最少期数")
            if rep.reasons:
                st.warning("未通过项：" + "；".join(rep.reasons))
            if rep.dsr_degraded:
                st.caption("⚠️ 未提供试验矩阵，DSR 为退化估计（偏宽松）。精确 PBO 需先在「自动挖掘」页跑一次并勾选保存试验矩阵。")

        st.download_button("⬇ 下载净值 CSV", bt.nav.to_csv().encode("utf-8-sig"),
                           file_name="strategy_nav.csv", mime="text/csv")

    render_strategy_save_panel(RESEARCH_STORE)


# ---------------------------------------------------------------------------
# ⑥ 数据管理
# ---------------------------------------------------------------------------
def page_data():
    st.markdown('<div class="qfm-sig"><h1>数据管理</h1>'
                '<div class="sub">本地缓存位于 data_cache/（parquet），二次加载秒级</div></div>',
                unsafe_allow_html=True)
    s = dl.status()
    c1, c2, c3 = st.columns(3)
    c1.metric("已缓存日线股票", s["stocks_cached"])
    c2.metric("财务指标缓存", "✅ 已缓存" if s["indicators_cached"] else "❌ 未缓存")
    c3.metric("缓存大小", f"{sum(os.path.getsize(os.path.join(dl.bars_dir, f)) for f in os.listdir(dl.bars_dir) if f.endswith('.parquet')) / 1e6:.0f} MB")
    c4, c5 = st.columns(2)
    legacy = s.get("stocks_without_factor", 0)
    c4.metric("待迁移复权因子", legacy, help="旧版缓存缺少精确复权因子；下次加载会自动重拉")
    c5.metric("因子库版本快照", "已落盘" if os.path.exists(os.path.join(os.path.dirname(dl.cache_dir), "data_cache", "factors", "registry.json")) else "未落盘")
    if legacy:
        st.warning(
            f"有 {legacy} 只缓存是旧版（缺精确复权因子）。这些股票的真实价与流通市值会偏低，"
            "规模/价值族因子不可信；加载股票池时会自动重拉完成迁移。"
        )
    st.divider()
    col_snap, col_clear_hist = st.columns(2)
    if col_snap.button("导出因子库版本快照", use_container_width=True,
                       help="把当前所有因子的版本、公式与定义指纹写到 data_cache/factors/registry.json"):
        st.success(f"已写入 {save_registry()}")
    if col_clear_hist.button("清空内存中的研究缓存", use_container_width=True):
        load_panel.clear()
        st.success("已清空数据面板缓存（不影响已保存的实验记录）。")
    if st.button("清除日线缓存并重拉", use_container_width=True):
        dl.clear_bars()
        load_panel.clear()
        st.rerun()
    if st.button("清除财务缓存并重拉", use_container_width=True):
        os.remove(os.path.join(dl.cache_dir, "indicators.parquet"))
        load_panel.clear()
        st.rerun()
    st.caption("提示：修改了 loader/因子代码后，用右上角菜单的 Rerun 或上述按钮刷新缓存。")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
if section == "研究项目":
    page_research(RESEARCH_STORE,
                  panel_provider=lambda: load_panel(pool, max_stocks if max_stocks > 0 else None))
elif section == "因子库":
    page_library()
elif section == "策略对比":
    render_strategy_compare(RESEARCH_STORE)
elif section == "数据与股票池":
    render_data_catalog(CATALOG)
elif section == "任务中心":
    render_job_center(JOB_SERVICE.store)
elif section == "数据管理":
    page_data()
else:
    if pool == "full" and max_stocks == 0:
        st.warning("全市场池需要几分钟到几十分钟，建议先设「限制股票数」或改用 index800。")
    panel = load_panel(pool, max_stocks if max_stocks > 0 else None)
    if section == "因子检验":
        page_test(panel)
    elif section == "因子对比":
        render_factor_compare(panel, pool, JOB_SERVICE, CATALOG.root)
    elif section == "多因子实验室":
        render_multifactor_lab(panel, pool, JOB_SERVICE, COMPOSITE_REGISTRY, CATALOG.root)
    elif section == "自动挖掘":
        page_mine(panel, mine_mode)
    elif section == "自定义因子":
        page_custom(panel)
    elif section == "策略回测":
        page_strategy(panel)
