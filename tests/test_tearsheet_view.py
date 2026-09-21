"""Tear Sheet 图表构造测试（不需要 Streamlit 运行时）。"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pytest

from qfm.pipeline.pipeline import PipelineConfig, run_pipeline
from qfm.pipeline.tearsheet import build_tear_sheet
from qfm.pipeline import tearsheet_view as view


@pytest.fixture(scope="module")
def sheet(wide_panel):
    signal = run_pipeline(wide_panel.close.pct_change(20), panel=wide_panel).signal
    return build_tear_sheet(wide_panel, signal, horizon=5, name="mom_20")


def _trace_names(figure: go.Figure) -> list[str]:
    return [str(trace.name) for trace in figure.data if getattr(trace, "name", None)]


def test_all_figures_are_plotly_figures(sheet):
    builders = [
        view.ic_figure, view.rolling_icir_figure, view.layer_stat_figure,
        view.layer_nav_figure, view.turnover_figure, view.distribution_figure,
        view.histogram_figure, view.industry_exposure_figure,
        view.size_exposure_figure, view.coverage_figure,
    ]

    for builder in builders:
        figure = builder(sheet)
        assert isinstance(figure, go.Figure), f"{builder.__name__} 未返回 Figure"
        assert figure.layout.title.text, f"{builder.__name__} 缺少标题"


def test_ic_figure_shows_rolling_and_cumulative(sheet):
    names = _trace_names(view.ic_figure(sheet))

    assert "日度 IC" in names
    assert "IC 120 日均值" in names
    assert "累计 IC" in names


def test_layer_nav_figure_marks_extremes_and_long_short(sheet):
    figure = view.layer_nav_figure(sheet)
    names = _trace_names(figure)

    assert "L1 净值" in names
    assert "L5 净值" in names
    assert any("多空净值" in name for name in names)
    # 极端层的线更粗，便于阅读
    widths = {trace.name: trace.line.width for trace in figure.data}
    assert widths["L1 净值"] > widths["L3 净值"]


def test_layer_stat_figure_colors_gain_and_loss_differently(sheet):
    figure = view.layer_stat_figure(sheet)
    colors = list(figure.data[0].marker.color)

    assert len(colors) == len(sheet.layer_stat)
    assert len(set(colors)) <= 2


def test_coverage_figure_lists_pipeline_stages(wide_panel):
    """带管线留痕时必须展示各阶段覆盖率；无留痕时至少报 signal（不伪造阶段值）。"""
    from qfm.pipeline.pipeline import run_pipeline
    from qfm.pipeline.tearsheet import build_tear_sheet

    pipeline = run_pipeline(wide_panel.close.pct_change(20), panel=wide_panel)
    enriched = build_tear_sheet(wide_panel, pipeline.signal, horizon=5, pipeline=pipeline)

    figure = view.coverage_figure(enriched)
    labels = list(figure.data[0].x)
    assert "raw" in labels and "signal" in labels
    assert tuple(figure.layout.yaxis.range) == (0, 1.05)

    # 降级路径：没有留痕时只报 signal，且图表仍然合法
    degraded = build_tear_sheet(wide_panel, pipeline.signal, horizon=5)
    assert set(degraded.coverage) == {"signal"}
    assert isinstance(view.coverage_figure(degraded), go.Figure)


def test_distribution_figure_draws_band_and_median(sheet):
    names = _trace_names(view.distribution_figure(sheet))

    assert "P95" in names and "P5" in names and "P50" in names


def test_figures_survive_degenerate_sheets(wide_panel):
    """分层无法计算时图表仍必须是合法 Figure，不能抛异常。"""
    from copy import deepcopy
    from qfm.pipeline.tearsheet import build_tear_sheet

    thin = deepcopy(wide_panel)
    for name in ("close", "open", "high", "low", "volume", "amount",
                 "turnover", "factor", "close_raw", "mv_float", "industry"):
        frame = getattr(thin, name, None)
        if frame is not None and hasattr(frame, "iloc"):
            setattr(thin, name, frame.iloc[:, :8])
    thin.fund = {k: v.iloc[:, :8] for k, v in thin.fund.items()}
    sheet = build_tear_sheet(thin, run_pipeline(thin.close.pct_change(20), panel=thin).signal, horizon=5)

    assert sheet.layer_nav.empty
    for builder in (view.layer_nav_figure, view.layer_stat_figure, view.histogram_figure):
        assert isinstance(builder(sheet), go.Figure)


def test_size_exposure_figure_marks_warning_band(sheet):
    figure = view.size_exposure_figure(sheet)
    horizontal = [shape for shape in figure.layout.shapes if shape.type == "line"]

    # 0 基准 + 均值 + ±0.3 预警 = 4 条水平线
    assert len(horizontal) >= 4
    assert np.isfinite(sheet.size_exposure_mean)


def test_ready_to_render_for_a_pure_size_factor(wide_panel):
    """市值暴露极高时也必须能正常出图（并带出告警文案）。"""
    log_mv = np.log(wide_panel.mv_float)
    signal = run_pipeline(log_mv, panel=wide_panel).signal
    sheet = build_tear_sheet(wide_panel, signal, horizon=5, name="ln_mv", exposure_sample=1)

    assert sheet.size_exposure_mean > 0.9
    assert isinstance(view.size_exposure_figure(sheet), go.Figure)
    assert any("规模暴露" in w for w in sheet.warnings)
