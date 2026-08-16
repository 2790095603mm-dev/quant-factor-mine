"""量化因子挖掘流水线 · Streamlit 操作台（深色金融终端风）

启动: streamlit run app.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import streamlit as st

from qfm.data import DataLoader, build_panel, get_universe
from qfm.factors import compute_factor, get_factor, list_factors, register_factor
from qfm.mining import run_mining
from qfm.pipeline import factor_report, generate_report
from qfm.pipeline.report import ic_chart, layer_chart

st.set_page_config(page_title="量化因子挖掘流水线", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

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
section = st.sidebar.radio("操作台", ["因子库", "因子检验", "自动挖掘", "自定义因子", "策略回测", "数据管理"],
                           label_visibility="collapsed")

# 自动挖掘子分支：批量挖掘（传统指标穷举）/ ML 合成（LightGBM walk-forward）
mine_mode = "批量挖掘"
if section == "自动挖掘":
    mine_mode = st.sidebar.radio("挖掘方式", ["批量挖掘", "ML 合成"], key="mine_mode",
                                 help="批量挖掘=指标×窗口×变换穷举；ML 合成=LightGBM 非线性合成新因子")

with st.sidebar.expander("数据参数", expanded=False):
    pool = st.selectbox("股票池", ["index800", "full"],
                        help="index800=沪深300+中证500（快）；full=全市场（很慢）")
    max_stocks = st.number_input("限制股票数（0=全部）", 0, 6000, 0)
    st.caption("首次拉取约 5-15 分钟，之后全部走本地缓存")

dl = DataLoader()
st.sidebar.caption(f"缓存状态：日线 {dl.status()['stocks_cached']} 只 · "
                   f"财务 {'✅' if dl.status()['indicators_cached'] else '❌'}")

# ---------------------------------------------------------------------------
# ① 因子库
# ---------------------------------------------------------------------------
def page_library():
    st.markdown('<div class="qfm-sig"><h1>因子库</h1>'
                '<div class="sub">31 个内置因子 · 7 大家族 · 点左侧「因子检验」逐个验证</div></div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns([1, 2])
    family = c1.selectbox("家族筛选", ["全部"] + [f for f in ["价值", "质量", "成长", "动量反转", "波动", "流动性", "规模"]])
    keyword = c2.text_input("关键词搜索", placeholder="如：动量 / roe / 换手")
    fs = list_factors(family if family != "全部" else None)
    if keyword:
        fs = [f for f in fs if keyword.lower() in f.name.lower() or keyword in f.description]
    st.caption(f"共 {len(fs)} 个因子")
    cols = st.columns(3)
    for i, f in enumerate(fs):
        with cols[i % 3].container(border=True):
            st.markdown(f"**{f.name}**"
                        f"<span class='qfm-badge amber'>{f.family}</span>"
                        f"<span class='qfm-badge cyan'>{'正向' if f.direction=='positive' else '负向'}</span>",
                        unsafe_allow_html=True)
            st.markdown(f"<div class='qfm-desc'>{f.description}</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# ② 因子检验
# ---------------------------------------------------------------------------
def page_test(panel):
    st.markdown('<div class="qfm-sig"><h1>因子检验</h1>'
                '<div class="sub">清洗 → IC / IC_IR / 分层 / 换手 / 衰减监控 → 一键导出 HTML 报告</div></div>',
                unsafe_allow_html=True)
    import plotly.graph_objects as go

    tab_single, tab_cmp, tab_ortho = st.tabs(["单因子检验", "多因子对比", "正交化"])

    # ---------- Tab 1：单因子检验 ----------
    with tab_single:
        c1, c2, c3 = st.columns([2, 1, 1])
        factor_names = [f.name for f in list_factors()]
        name = c1.selectbox("选择因子", factor_names, help="自定义因子注册后也会出现在这里")
        horizon = c2.selectbox("前瞻天数", [5, 10, 20, 60], index=2)
        run = c3.button("运行检验", use_container_width=True)

        f = get_factor(name)
        if run:
            with st.status("检验中…", expanded=False) as status:
                rep = factor_report(compute_factor(name, panel), panel.close,
                                    horizon=horizon, direction=f.direction)
                status.update(label=f"✅ 完成：{name} · 前瞻 {horizon} 日", state="complete")
            s = rep["ic_summary"]
            mono = rep["monotonicity"]
            verdict, vcolor = verdict_of(rep)

            st.markdown(f"<span style='color:{vcolor};font-weight:600;font-size:15px'>{verdict}</span>"
                        f"<span class='qfm-badge'>{f.family}</span>"
                        f"<span class='qfm-badge'>{'正向' if f.direction=='positive' else '负向'}</span>"
                        f"<div class='qfm-desc'>{f.description}</div>", unsafe_allow_html=True)
            k1, k2, k3, k4, k5, k6 = st.columns(6)
            k1.metric("平均 IC", f"{s['ic_mean']:+.4f}" if pd.notna(s["ic_mean"]) else "—")
            k2.metric("IC_IR", f"{s['ic_ir']:.2f}" if pd.notna(s["ic_ir"]) else "—")
            k3.metric("t 值", f"{s['ic_t']:.2f}" if pd.notna(s["ic_t"]) else "—")
            k4.metric("IC 正占比", f"{s['pos_ratio']:.0%}" if pd.notna(s["pos_ratio"]) else "—")
            k5.metric("组合换手", f"{rep['turnover']:.0%}" if pd.notna(rep["turnover"]) else "—")
            k6.metric("多空价差", f"{mono['spread']:+.4f}" if pd.notna(mono["spread"]) else "—")

            st.plotly_chart(ic_chart(rep), use_container_width=True)
            st.plotly_chart(layer_chart(rep, f.direction), use_container_width=True)

            with st.expander("分年 IC"):
                y = rep["ic_by_year"]
                st.dataframe(pd.DataFrame({"年份": y.index, "平均 IC": y.values.round(4)}), hide_index=True)

            with st.expander("IC 衰减监控"):
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=rep["ic_rolling"].index, y=rep["ic_rolling"].values,
                                          name="IC 120日均线", line=dict(color="#7C4DFF", width=2)))
                fig3.add_hline(y=0, line=dict(color="#C9B8F0", width=1, dash="dash"))
                fig3.update_layout(height=260, margin=dict(t=30), paper_bgcolor="rgba(0,0,0,0)",
                                   plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#3B2E5E"))
                st.plotly_chart(fig3, use_container_width=True)
                d = rep["ic_decay"]
                dc1, dc2 = st.columns(2)
                dc1.metric("全期 IC", f"{s['ic_mean']:+.4f}")
                dc2.metric("近 60 日 IC", f"{rep['ic_recent']:+.4f}" if pd.notna(rep["ic_recent"]) else "—")
                if pd.notna(d):
                    if d > 0.01:
                        state_txt, state_col = "📈 增强（近端强于全期）", "#2E9E6B"
                    elif d < -0.01:
                        state_txt, state_col = "📉 衰减（近端弱于全期，警惕失效）", "#E5484D"
                    else:
                        state_txt, state_col = "➡️ 稳定", "#C9B8F0"
                    st.markdown(f"<span style='color:{state_col};font-weight:600'>{state_txt}</span>"
                                f"<span class='qfm-desc'>　滚动 120 日均线持续下滑且近 60 日 IC 明显低于全期 → 因子正在失效，"
                                f"建议正交化、换窗口或移出因子池</span>", unsafe_allow_html=True)

            html = generate_report(rep, name, f.family, f.description, f.direction,
                                   os.path.join("reports", f"{name}_h{horizon}.html"))
            with open(html, encoding="utf-8") as fh:
                st.download_button("⬇ 下载 HTML 报告", fh.read(), file_name=os.path.basename(html),
                                   mime="text/html", use_container_width=True)
        else:
            st.info("选择因子和前瞻天数后，点「运行检验」。")
            if panel is not None:
                st.caption(f"当前面板：{panel.close.shape[0]} 个交易日 × {panel.close.shape[1]} 只股票")

    # ---------- Tab 2：多因子横向对比 ----------
    with tab_cmp:
        cmp_names = st.multiselect("对比因子（可多选）", [f.name for f in list_factors()],
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
                    "因子": nm, "家族": f.family, "方向": "正" if f.direction == "positive" else "负",
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
            st.caption("紫色=正向 IC，粉色=负向 IC。负向 IC 的因子（如 A 股反转）取反后即为有效信号。")

    # ---------- Tab 3：正交化（QuantSkills factor-orthogonalize 方法论） ----------
    with tab_ortho:
        st.markdown("逐日截面 OLS 正交化：剥离行业 / 市值 / 风格暴露 → 残差因子"
                    "（方法论源自 QuantSkills factor-orthogonalize）")
        c1, c2, c3 = st.columns([2, 1, 1])
        ortho_name = c1.selectbox("选择因子", [f.name for f in list_factors()], key="ortho_f")
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
    if mode != "批量挖掘":
        page_mine_ml(panel)
        return
    c1, c2 = st.columns([1, 1])
    horizon = c1.selectbox("前瞻天数", [5, 10, 20, 60], index=2, key="mine_h")
    max_c = c2.selectbox("候选数限制", [36, 72, 144, 288], index=0,
                         help="36=单窗口集；越大跑得越久（每候选约 2 秒）")
    save_trials = st.checkbox("保存试验矩阵（供策略回测页过拟合检验使用）", value=True,
                              help="逐候选计算月频 TOP-30 组合收益，落盘 data_cache/trials/")
    if st.button("开始挖掘", use_container_width=True):
        prog = st.progress(0.0, text="准备…")
        df, trials_meta = run_mining(panel, horizon=horizon, max_candidates=max_c,
                                     save_trials=save_trials,
                                     progress=lambda i, n, nm: prog.progress((i + 1) / n, text=f"{i+1}/{n} · {nm}"))
        st.success(f"挖掘完成：{len(df)} 个候选因子")
        st.dataframe(df.style.format({"IC": "{:+.4f}", "IC_IR": "{:.2f}", "t值": "{:.2f}",
                                      "正占比": "{:.0%}", "换手率": "{:.0%}"}),
                     use_container_width=True, height=420)
        if trials_meta:
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
    ml_names = c1.multiselect("特征因子", [f.name for f in list_factors()],
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

        register_factor(name="ml_synth", family="机器学习",
                        description=f"LightGBM walk-forward 合成（{ml_h}日前瞻，训练截止 {ml_cutoff}）",
                        direction="positive")(lambda d: pred)
        st.caption(f"✅ 已注册为因子 `ml_synth`，可直接在「策略回测」页选择。"
                   f"预测区间 {meta['pred_start']} ~ {meta['pred_end']}——回测起点请设在此区间内或之后。")


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


def page_custom(panel):
    st.markdown('<div class="qfm-sig"><h1>自定义因子</h1>'
                '<div class="sub">把你的主观交易经验写成因子，立即进入统一检验流程</div></div>',
                unsafe_allow_html=True)
    st.code("从 qfm.factors 导入后，装饰器会自动注册；模板如下（可直接改）：", language=None)
    code = st.text_area("因子代码", CUSTOM_TEMPLATE, height=260)
    c1, c2, c3 = st.columns([1, 1, 2])
    if c1.button("注册并检验", use_container_width=True):
        try:
            ns = {"pd": pd, "np": np, "register_factor": register_factor}
            from qfm.data.panel import DataPanel
            ns["DataPanel"] = DataPanel
            exec(code, ns)
            from qfm.pipeline.lookahead import scan_source
            chk = scan_source(code)
            if chk["leaks"]:
                st.error("⚠️ 检测到未来函数泄漏模式：" + "；".join(chk["leaks"]) +
                         "（因子将引用未来数据，检验结果不可信）")
            elif chk["warnings"]:
                st.warning("提示：" + "；".join(chk["warnings"]))
            st.success("注册成功，已自动进入检验流程")
            # 找出刚注册的因子（模板最后定义的函数名）
            new_names = [n for n in ns if n.startswith("my_") and callable(ns[n])]
            if new_names:
                name = new_names[-1].replace("def ", "")
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
    c1, c2, c3 = st.columns([2, 1, 1])
    names = c1.multiselect("选择因子", [f.name for f in list_factors()],
                           default=["ep_ttm", "roe", "rev_20"])
    mode = c2.selectbox("权重模式", ["equal", "ic", "icir"],
                        format_func=lambda m: {"equal": "等权", "ic": "IC 加权", "icir": "IC_IR 加权"}[m])
    horizon = c3.selectbox("前瞻天数", [5, 10, 20, 60], index=2, key="strategy_h")
    with st.expander("因子正交化（行业 / 市值 / 风格暴露剥离）"):
        ortho = st.checkbox("启用正交化", value=False,
                            help="逐日截面 OLS 残差化，消除所选暴露（方法论源自 QuantSkills factor-orthogonalize）")
        ortho_controls = st.multiselect("剥离暴露", ["industry", "size", "style"], default=["size"],
                                        format_func=lambda c: {"industry": "行业", "size": "市值",
                                                               "style": "风格(beta/波动率)"}[c])

    st.markdown("**② 组合回测**")
    c5, c6, c7 = st.columns([1, 1, 1])
    top_n = c5.slider("持仓数量", 10, 100, 30, step=5)
    start_d = c6.selectbox("回测起点", ["2021-01-01", "2022-01-01", "2023-01-01"], index=0)
    bench_mode = c7.selectbox("基准", ["equal", "mv"],
                              format_func=lambda m: {"equal": "全池等权", "mv": "市值加权"}[m])

    if st.button("运行策略回测", use_container_width=True):
        if not names:
            st.error("至少选择一个因子")
            return
        from qfm.portfolio import (factor_corr, factor_panel, perf_stats,
                                   run_backtest, synthesize, yearly_perf)
        with st.status("策略流水线运行中…", expanded=True) as status:
            score, weights = synthesize(panel, names, mode=mode, horizon=horizon,
                                        orthogonalize=ortho,
                                        ortho_controls=tuple(ortho_controls) if ortho else None)
            status.update(label="✅ 因子合成完成 · 开始组合回测…")
            bt = run_backtest(panel, score, top_n=top_n, start=start_d, bench_mode=bench_mode)
            status.update(label=f"✅ 完成：策略净值 {bt.nav.iloc[-1]:.2f} vs 基准 {bt.bench_nav.iloc[-1]:.2f}",
                          state="complete")

        # 权重 + 相关性
        wdf = pd.DataFrame({"因子": list(weights), "权重": list(weights.values())}).sort_values("权重", ascending=False)
        st.markdown("**因子权重 & 相关性**")
        cw, cc = st.columns([1, 2])
        cw.dataframe(wdf.style.format({"权重": "{:.1%}"}), hide_index=True, use_container_width=True)
        import plotly.graph_objects as go
        corr = factor_corr(factor_panel(panel, names))
        fig = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1,
            colorscale="Purples", text=corr.round(2), texttemplate="%{text}"))
        fig.update_layout(title="因子截面相关性", height=280, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#3B2E5E"))
        cc.plotly_chart(fig, use_container_width=True)

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
            last_expo = bt.industry_exposure.iloc[-1].sort_values(ascending=False)
            figx = go.Figure(go.Bar(
                x=last_expo.index, y=last_expo.values, marker_color="#7C4DFF",
                text=last_expo.round(3), texttemplate="%{text:.0%}", textposition="outside"))
            figx.update_layout(height=300, margin=dict(t=30), paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#3B2E5E"),
                               xaxis_tickangle=-30)
            st.plotly_chart(figx, use_container_width=True)
            hhi = float((last_expo ** 2).sum())
            st.caption(f"行业集中度 HHI：**{hhi:.3f}**（1.0=全押一个行业；越低越分散）。"
                       f"若某行业权重长期偏高，说明策略隐含行业赌注——面试可讲行业中性化作为下一步")

        # 绩效指标
        st.markdown("**绩效指标**")
        stats = perf_stats(bt.nav, bt.bench_nav)
        kcols = st.columns(6)
        kcols[0].metric("总收益", f"{stats.get('总收益', float('nan')):+.1%}")
        kcols[1].metric("年化收益", f"{stats.get('年化收益', float('nan')):+.1%}")
        kcols[2].metric("夏普比率", f"{stats.get('夏普比率', float('nan')):.2f}")
        kcols[3].metric("最大回撤", f"{stats.get('最大回撤', float('nan')):.1%}")
        kcols[4].metric("年化超额", f"{stats.get('年化超额', float('nan')):+.1%}")
        kcols[5].metric("年化换手", f"{bt.turnover:.1f}")

        yp = yearly_perf(bt.nav)
        st.markdown("**分年绩效**")
        st.dataframe(yp.style.format({"收益": "{:+.1%}", "最大回撤": "{:.1%}", "日胜率": "{:.0%}"}),
                     hide_index=True, use_container_width=True)

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
    st.divider()
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
if section == "因子库":
    page_library()
elif section == "数据管理":
    page_data()
else:
    if pool == "full" and max_stocks == 0:
        st.warning("全市场池需要几分钟到几十分钟，建议先设「限制股票数」或改用 index800。")
    panel = load_panel(pool, max_stocks if max_stocks > 0 else None)
    if section == "因子检验":
        page_test(panel)
    elif section == "自动挖掘":
        page_mine(panel, mine_mode)
    elif section == "自定义因子":
        page_custom(panel)
    elif section == "策略回测":
        page_strategy(panel)
