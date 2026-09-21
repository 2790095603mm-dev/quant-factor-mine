"""供网页及离线下载复用的单因子模拟图表。"""
from __future__ import annotations

import html
import json

import pandas as pd
import plotly.graph_objects as go

from qfm.pipeline.report import _fig_style, ic_chart, layer_chart
from qfm.portfolio.performance import drawdown, perf_stats


def nav_chart(result):
    fig = go.Figure()
    for name, series, color, dash in (
        ("成本后", result.backtest.nav, "#7C4DFF", "solid"),
        ("无费用对照", result.gross.nav, "#2E9E6B", "dot"),
        ("全池等权基准", result.backtest.bench_nav, "#8590A6", "dash"),
    ):
        fig.add_trace(go.Scatter(x=series.index, y=series, name=name,
                                line={"color": color, "width": 2, "dash": dash}))
    fig.update_layout(title="累计净值", height=360, hovermode="x unified", yaxis_title="净值（起点 = 1）")
    return _fig_style(fig)


def drawdown_chart(result):
    values = drawdown(result.backtest.nav)
    fig = go.Figure(go.Scatter(x=values.index, y=values, name="回撤", fill="tozeroy",
                              line={"color": "#E5484D", "width": 1.4}))
    fig.update_layout(title="成本后回撤", height=230, yaxis_tickformat=".0%", hovermode="x unified")
    return _fig_style(fig)


def cost_comparison(result):
    rows = []
    for label, bt in (("成本后", result.backtest), ("无费用对照", result.gross)):
        stats = perf_stats(bt.nav, risk_free_rate=result.settings.risk_free_rate)
        rows.append({"模拟": label, **{key: stats.get(key) for key in ("年化收益", "夏普比率", "最大回撤")},
                     "累计成本": bt.cost_total})
    return pd.DataFrame(rows)


def export_html(result):
    stats = {key: result.stats.get(key) for key in ("年化收益", "夏普比率", "最大回撤", "日均换手", "Fitness")}
    stats["信号覆盖率"] = result.coverage
    metrics = pd.Series(stats, name="数值").to_frame().to_html(float_format=lambda x: f"{x:.6f}")
    charts = [nav_chart(result), drawdown_chart(result), ic_chart(result.report), layer_chart(result.report, "positive")]
    figures = "".join(fig.to_html(full_html=False, include_plotlyjs=True if i == 0 else False) for i, fig in enumerate(charts))
    config = html.escape(json.dumps(result.settings.to_dict(), ensure_ascii=False, indent=2))
    warnings = "".join(f"<p>{html.escape(warning)}</p>" for warning in result.warnings)
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>单因子模拟 · {html.escape(result.name)}</title>
<style>body{{font-family:system-ui,sans-serif;color:#3B2E5E;background:#F6F2FF;margin:30px auto;max-width:1100px;padding:0 20px}}table{{border-collapse:collapse;width:100%;background:white}}th,td{{padding:10px;border:1px solid #E5DCF5;text-align:right}}pre{{background:white;padding:20px;white-space:pre-wrap}}p{{line-height:1.7}}</style>
<h1>单因子检验与模拟 · {html.escape(result.name)}</h1>
<p>{result.backtest.nav.index.min():%Y-%m-%d} — {result.backtest.nav.index.max():%Y-%m-%d}；下一交易日开盘执行；已按因子方向统一为高分优先。各收益比例采用小数表示。</p>
{metrics}{warnings}{figures}<h2>成本前后</h2>{cost_comparison(result).to_html(index=False)}
<h2>分年表现</h2>{result.yearly.to_html(index=False)}
<h2>计算规则</h2><p>Sharpe = (平均日收益 − 日无风险利率) / 日收益标准差 × √252。日均换手 = 每日买卖成交额之和 / 交易前净资产，再对交易区间所有收益日取均值。</p>
<p>Fitness = Sharpe × √(|252 × 平均日收益| / max(日均换手, 0.125))，为本地研究口径，不能直接套用 BRAIN 提交门槛。无费用曲线是相同规则下独立模拟，费用造成的可用资金差异可能改变成交数量。分年夏普按本次无风险利率计算。</p>
<p>IC 使用未来 H 日收盘收益，末尾标签不完整的日期不参与检验；分层收益是预测统计。信号行业/市值中性化不保证选股后的组合行业权重中性。当前指数股票池不包含历史成分变动。</p>
<h2>运行设置</h2><pre>{config}</pre></html>'''
