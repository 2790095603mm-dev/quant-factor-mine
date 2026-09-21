"""单因子 Tear Sheet 测试：指标完整性、双口径一致性、退化场景。

使用 `wide_panel`（160 只 × 6 行业）而非 60 只的 `panel`：
`layer_test` 要求每日至少 100 只有效股票（20 × 5 层），60 只无法产出分层统计。
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from qfm.pipeline.pipeline import PipelineConfig, run_pipeline
from qfm.pipeline.tearsheet import build_tear_sheet


def _signal(panel, config=None, raw=None):
    return run_pipeline(
        panel.close.pct_change(20) if raw is None else raw, panel=panel, config=config,
    ).signal


@pytest.fixture(scope="module")
def sheet(wide_panel):
    """动量因子的完整 Tear Sheet（复用默认管线信号）。"""
    return build_tear_sheet(
        wide_panel, _signal(wide_panel), horizon=5, name="mom_20", direction="positive",
    )


def test_tear_sheet_covers_every_requested_dimension(sheet):
    """第 3 项要求的全部维度都必须产出，不允许留空。"""
    assert len(sheet.ic) > 0                     # IC
    assert len(sheet.ic_summary) > 0             # RankIC 汇总（spearman 即秩相关）
    assert np.isfinite(sheet.ic_summary["ic_ir"])      # ICIR
    assert len(sheet.ic_rolling) > 0             # Rolling IC
    assert len(sheet.ic_rolling_ir) > 0          # Rolling ICIR
    assert len(sheet.ic_cumulative) > 0          # 累计 IC
    assert len(sheet.ic_by_year) > 0             # 分年 IC
    assert not sheet.layer_stat.empty            # 分组收益（统计口径）
    assert np.isfinite(sheet.long_short_spread)  # 多空收益（统计口径）
    assert not sheet.layer_nav.empty             # 分组累计收益（可交易）
    assert len(sheet.long_short_nav) > 0         # 多空累计收益（可交易）
    assert np.isfinite(sheet.factor_turnover)    # Turnover
    assert sheet.coverage.get("signal") is not None   # Coverage
    assert not sheet.distribution.empty          # 因子分布
    assert sheet.histogram                       # 最新截面直方图
    assert not sheet.industry_exposure.empty     # 行业暴露
    assert len(sheet.size_exposure) > 0          # 市值暴露
    assert np.isfinite(sheet.size_exposure_mean)


def test_kpis_are_complete_and_finite(sheet):
    kpis = sheet.kpis()

    assert set(kpis) == {
        "平均 IC", "IC 均值绝对值", "IC_IR", "IC t 值", "IC 正占比", "有效天数",
        "多空价差（统计）", "多空累计（可交易）", "组合换手", "信号覆盖率", "市值暴露",
    }
    for key, value in kpis.items():
        if key == "有效天数":
            assert value > 0
        else:
            assert np.isfinite(value), f"{key} 不应为空"


def test_statistical_layer_matches_factor_report(sheet, wide_panel):
    """统计口径必须与既有 factor_report 完全一致，不能各算一套。"""
    from qfm.pipeline.tests import factor_report

    report = factor_report(_signal(wide_panel), wide_panel.close, horizon=5, preprocessed=True)

    pd.testing.assert_frame_equal(sheet.layer_stat, report["layer"])
    assert sheet.long_short_spread == pytest.approx(report["monotonicity"]["spread"])
    pd.testing.assert_series_equal(sheet.ic, report["ic_series"])
    assert sheet.factor_turnover == pytest.approx(report["turnover"])


def test_reusing_a_report_avoids_recomputation(sheet, wide_panel):
    """传入已算好的 report 必须得到相同结果（证明复用路径正确）。"""
    from qfm.pipeline.tests import factor_report

    signal = _signal(wide_panel)
    report = factor_report(signal, wide_panel.close, horizon=5, preprocessed=True)

    reused = build_tear_sheet(wide_panel, signal, horizon=5, report=report)

    pd.testing.assert_series_equal(reused.ic, sheet.ic)
    pd.testing.assert_frame_equal(reused.layer_nav, sheet.layer_nav)


def test_layer_nav_is_compounded_and_ordered(sheet):
    """分层净值：层数正确、末值为复利结果、逐期收益与净值自洽。"""
    assert list(sheet.layer_nav.columns) == [1, 2, 3, 4, 5]
    assert (sheet.layer_nav > 0).all().all()
    expected = (1 + sheet.layer_period_returns).cumprod()
    pd.testing.assert_frame_equal(sheet.layer_nav, expected)
    assert sheet.layer_nav.index.is_monotonic_increasing


def test_long_short_nav_is_top_minus_bottom(sheet):
    """多空净值必须来自「最高分层 − 最低分层」的复利，而不是别的口径。"""
    spread = sheet.layer_period_returns.iloc[:, -1] - sheet.layer_period_returns.iloc[:, 0]
    expected = (1 + spread).cumprod()

    pd.testing.assert_series_equal(sheet.long_short_nav, expected.rename("多空净值"))


def test_tradable_and_statistical_views_can_disagree(wide_panel):
    """双口径确实不同：统计口径是重叠窗口均值，可交易口径是非重叠复利。

    只要求两者都存在且不相等，避免某天把两个口径合并成一个而丢失信息。
    """
    sheet = build_tear_sheet(wide_panel, _signal(wide_panel), horizon=5)

    tradable_last = sheet.layer_nav.iloc[-1, -1] - 1
    statistical = sheet.layer_stat["mean_ret"].iloc[-1]
    assert not np.isclose(tradable_last, statistical)


def test_turnover_series_uses_non_overlapping_grid(sheet):
    assert len(sheet.turnover_series) > 0
    assert ((sheet.turnover_series >= 0) & (sheet.turnover_series <= 1)).all()


def test_distribution_records_cross_section_shape(sheet):
    dist = sheet.distribution

    assert set(dist.columns) >= {"样本数", "均值", "标准差", "偏度", "峰度", "P5", "P50", "P95"}
    assert (dist["样本数"] > 0).all()
    assert (dist["P5"] <= dist["P50"]).all()
    assert (dist["P50"] <= dist["P95"]).all()


def test_histogram_reflects_latest_cross_section(sheet):
    histogram = sheet.histogram

    assert len(histogram["counts"]) == len(histogram["bin_edges"]) - 1
    assert sum(histogram["counts"]) == pytest.approx(len(sheet.latest_snapshot))
    assert histogram["date"] == sheet.distribution.index[-1]


def test_industry_exposure_is_reported_per_industry(wide_panel):
    sheet = build_tear_sheet(wide_panel, _signal(wide_panel), horizon=5)

    exposure = sheet.industry_exposure
    assert set(exposure.columns) == {"平均暴露", "暴露标准差", "平均成分数", "样本日数"}
    assert set(exposure.index) == set(wide_panel.industry.iloc[-1].unique())
    assert (exposure["平均成分数"] > 0).all()


def test_size_exposure_detects_a_pure_size_factor(wide_panel):
    """直接用 log(流通市值) 当因子时，市值暴露必须接近 1。"""
    log_mv = np.log(wide_panel.mv_float)
    signal = run_pipeline(log_mv, panel=wide_panel).signal

    sheet = build_tear_sheet(wide_panel, signal, horizon=5, exposure_sample=1)

    assert sheet.size_exposure_mean > 0.9
    assert any("规模暴露" in w for w in sheet.warnings)


def test_neutralizing_size_removes_size_exposure(wide_panel):
    """市值中性化后，市值暴露必须降到 0 附近（z-score 是线性变换，保持正交性）。"""
    log_mv = np.log(wide_panel.mv_float)
    raw_signal = _signal(wide_panel, raw=log_mv)
    neutral = _signal(wide_panel, config=PipelineConfig(neutralize=("size",)), raw=log_mv)

    before = build_tear_sheet(wide_panel, raw_signal, horizon=5, exposure_sample=1)
    after = build_tear_sheet(wide_panel, neutral, horizon=5, exposure_sample=1)

    assert abs(after.size_exposure_mean) < abs(before.size_exposure_mean)
    assert abs(after.size_exposure_mean) < 1e-8


def test_rank_standardization_breaks_linear_neutralization(wide_panel):
    """记录一个真实的阶段交互：rank 是非线性单调变换，会破坏中性化的线性正交性。

    同样做市值中性化，z-score 保持残差与控制变量的线性正交（暴露≈0），
    而 rank 之后市值暴露可高达 0.78。这是被文档化的已知行为（见
    qfm.pipeline.stages.apply_standardize），此处用测试锁定，防止被误当 bug 改坏。
    """
    log_mv = np.log(wide_panel.mv_float)

    def exposure(config):
        signal = _signal(wide_panel, config=config, raw=log_mv)
        return build_tear_sheet(
            wide_panel, signal, horizon=5, exposure_sample=1,
        ).size_exposure_mean

    linear = exposure(PipelineConfig(neutralize=("size",)))
    ranked = exposure(PipelineConfig(neutralize=("size",), standardize="rank"))

    assert abs(linear) < 1e-8, "z-score 是线性变换，中性化后的线性正交性必须保持"
    assert abs(ranked) > 0.1, (
        "rank 标准化会破坏线性正交性，市值暴露不应接近 0；此断言失败说明阶段语义变了"
    )


def test_negative_direction_is_recorded(wide_panel):
    sheet = build_tear_sheet(wide_panel, _signal(wide_panel), horizon=5, direction="negative")

    assert sheet.direction == "negative"
    assert sheet.horizon == 5


def test_rejects_empty_signal(wide_panel):
    with pytest.raises(ValueError, match="信号为空"):
        build_tear_sheet(wide_panel, pd.DataFrame())


@pytest.mark.parametrize("kwargs, message", [({"horizon": 0}, "前瞻天数"), ({"rebalance": 0}, "调仓间隔")])
def test_rejects_invalid_periods(wide_panel, kwargs, message):
    with pytest.raises(ValueError, match=message):
        build_tear_sheet(wide_panel, _signal(wide_panel), **kwargs)


def test_thin_cross_section_produces_warnings_not_crashes(wide_panel):
    """样本不足时必须给出警告而不是抛异常或返回误导性数字。"""
    thin = deepcopy(wide_panel)
    keep = 12
    for name in ("close", "open", "high", "low", "volume", "amount",
                 "turnover", "factor", "close_raw", "mv_float", "industry"):
        frame = getattr(thin, name, None)
        if isinstance(frame, pd.DataFrame):
            setattr(thin, name, frame.iloc[:, :keep])
    thin.fund = {key: frame.iloc[:, :keep] for key, frame in thin.fund.items()}

    sheet = build_tear_sheet(thin, _signal(thin), horizon=5)

    assert sheet.layer_nav.empty
    assert sheet.layer_stat.empty
    assert any("样本不足" in w for w in sheet.warnings)


def test_signal_is_realigned_to_panel_grid(wide_panel):
    """传入列子集时不允许静默错位，仍是完整网格（网格外为空）。"""
    signal = _signal(wide_panel).iloc[:, :20]

    sheet = build_tear_sheet(wide_panel, signal, horizon=5)

    assert np.isfinite(sheet.size_exposure_mean)
    assert set(sheet.industry_exposure.index) == set(wide_panel.industry.iloc[-1].unique())


def test_tearsheet_is_reproducible(wide_panel):
    """同一输入必须给出逐位相同的结果（实验可复现的前提）。"""
    signal = _signal(wide_panel)

    first = build_tear_sheet(wide_panel, signal, horizon=5)
    second = build_tear_sheet(wide_panel, signal, horizon=5)

    pd.testing.assert_series_equal(first.ic, second.ic)
    pd.testing.assert_frame_equal(first.layer_nav, second.layer_nav)
    pd.testing.assert_frame_equal(first.distribution, second.distribution)
    for key, value in first.kpis().items():
        if isinstance(value, (int, float, np.floating)) and np.isfinite(value):
            assert value == pytest.approx(second.kpis()[key])
