"""HTML 检验报告生成（plotly 自包含页面，离线可打开，极光玻璃配色）"""

from __future__ import annotations

import html
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 极光玻璃配色（与页面主题一致）：浅底 + 紫/粉/蓝
BG = "#F6F2FF"
CARD = "#FFFFFF"
BORDER = "#E5DCF5"
TEXT = "#3B2E5E"
AMBER = "#7C4DFF"  # 强调（紫）
CYAN = "#2F6FED"   # 数据（蓝）
RED = "#E5484D"
GREEN = "#2E9E6B"
MUTED = "#6E5C93"


def _fig_style(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template=None,
        paper_bgcolor=CARD,
        plot_bgcolor=BG,
        font=dict(color=TEXT, size=12, family="PingFang SC, Microsoft YaHei, sans-serif"),
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED)),
    )
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    return fig


def ic_chart(report: dict) -> go.Figure:
    ic = report["ic_series"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ic.index, y=ic.values, mode="lines", name="日度IC",
                             line=dict(color=CYAN, width=1), opacity=0.6))
    roll = ic.rolling(20, min_periods=1).mean()
    fig.add_trace(go.Scatter(x=roll.index, y=roll.values, mode="lines", name="IC 20日均值",
                             line=dict(color=AMBER, width=2)))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1, dash="dash"))
    fig.update_layout(title="IC 时间序列", height=340)
    return _fig_style(fig)


def layer_chart(report: dict, direction: str) -> go.Figure:
    layer = report["layer"]
    colors = [CYAN] * len(layer)
    fig = go.Figure(go.Bar(
        x=[f"L{i}" for i in layer.index], y=layer["mean_ret"].values,
        marker_color=colors, text=layer["mean_ret"].round(4),
        textposition="outside",
    ))
    fig.update_layout(title="分层收益（未来{}日，L1最低因子值~L{}最高）".format(
        report["horizon"], len(layer)), height=340, yaxis_title="平均收益")
    return _fig_style(fig)


def _year_ic_table(report: dict) -> str:
    y = report["ic_by_year"]
    if not len(y):
        return "<p class='muted'>无分年数据</p>"
    rows = "".join(
        f"<tr><td>{year}</td><td>{v:+.4f}</td></tr>" for year, v in y.items()
    )
    return f"<table><tr><th>年份</th><th>平均 IC</th></tr>{rows}</table>"


def _verdict(report: dict, direction: str) -> tuple[str, str]:
    """(结论文字, 颜色)"""
    s = report["ic_summary"]
    ic_mean = abs(s["ic_mean"]) if pd.notna(s["ic_mean"]) else 0
    ic_t = abs(s["ic_t"]) if pd.notna(s["ic_t"]) else 0
    if ic_mean >= 0.03 and ic_t >= 2 and report["monotonicity"].get("monotonic", False):
        return "✅ 通过核心检验（IC 显著 + 分层单调）", GREEN
    if ic_mean >= 0.02 or ic_t >= 2:
        return "⚠️ 边缘有效（单项达标，建议正交化/换窗口再验）", AMBER
    return "❌ 未通过（IC 弱或不稳定）", RED


def generate_report(report: dict, factor_name: str, family: str, description: str,
                    direction: str, out_path: str) -> str:
    """生成自包含 HTML 报告，返回文件路径"""
    s = report["ic_summary"]
    mono = report["monotonicity"]
    verdict, vcolor = _verdict(report, direction)

    kpi = [
        ("平均 IC", f"{s['ic_mean']:+.4f}" if pd.notna(s["ic_mean"]) else "—"),
        ("IC_IR", f"{s['ic_ir']:.2f}" if pd.notna(s["ic_ir"]) else "—"),
        ("IC t 值", f"{s['ic_t']:.2f}" if pd.notna(s["ic_t"]) else "—"),
        ("IC 正占比", f"{s['pos_ratio']:.1%}" if pd.notna(s["pos_ratio"]) else "—"),
        ("有效天数", str(s["n_days"])),
        ("组合换手", f"{report['turnover']:.1%}" if pd.notna(report["turnover"]) else "—"),
        ("多空价差", f"{mono['spread']:+.4f}" if pd.notna(mono["spread"]) else "—"),
        ("单调相关", f"{mono['corr']:.2f}" if pd.notna(mono.get("corr")) else "—"),
    ]
    kpi_html = "".join(
        f"<div class='kpi'><div class='kpi-label'>{k}</div><div class='kpi-val'>{v}</div></div>"
        for k, v in kpi
    )

    ic_fig = ic_chart(report).to_html(full_html=False, include_plotlyjs=False)
    layer_fig = layer_chart(report, direction).to_html(full_html=False, include_plotlyjs=False)

    html_doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>因子检验报告 · {html.escape(factor_name)}</title>
<style>
  body {{ background:{BG}; color:{TEXT}; font-family:'PingFang SC','Microsoft YaHei',sans-serif; margin:0; padding:24px; }}
  .wrap {{ max-width:1000px; margin:0 auto; }}
  h1 {{ font-size:22px; margin:0 0 4px; }}
  .sub {{ color:{MUTED}; font-size:13px; margin-bottom:18px; }}
  .kpis {{ display:flex; flex-wrap:wrap; gap:10px; margin-bottom:20px; }}
  .kpi {{ background:{CARD}; border:1px solid {BORDER}; border-radius:8px; padding:10px 16px; min-width:110px; }}
  .kpi-label {{ color:{MUTED}; font-size:12px; }}
  .kpi-val {{ font-size:18px; font-weight:600; margin-top:4px; color:{AMBER}; }}
  .card {{ background:{CARD}; border:1px solid {BORDER}; border-radius:10px; padding:14px; margin-bottom:20px; }}
  .verdict {{ display:inline-block; padding:6px 14px; border-radius:20px; font-weight:600; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  td,th {{ padding:6px 10px; border-bottom:1px solid {BORDER}; text-align:left; }}
  th {{ color:{MUTED}; font-weight:500; }}
  .muted {{ color:{MUTED}; }}
  footer {{ color:{MUTED}; font-size:12px; margin-top:24px; }}
</style></head><body><div class="wrap">
  <h1>因子检验报告 · {html.escape(factor_name)}</h1>
  <div class="sub">{html.escape(family)} 家族 · {html.escape(description)} · 方向:{"正向" if direction=="positive" else "负向"} · 前瞻 {report['horizon']} 日</div>
  <div class="kpis">{kpi_html}</div>
  <div class="card"><span class="verdict" style="background:{vcolor}22;color:{vcolor};">{verdict}</span></div>
  <div class="card">{ic_fig}</div>
  <div class="card">{layer_fig}</div>
  <div class="card"><h3 style="margin:4px 0 10px;">分年 IC</h3>{_year_ic_table(report)}</div>
  <footer>quant-factor-mine · 生成于 {datetime.now():%Y-%m-%d %H:%M} · 检验口径：截面去极值(MAD) + z-score，Spearman IC</footer>
</div></body></html>"""
    import os

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_doc)
    return out_path