"""统一因子管线测试：阶段行为、等价性与「全项目只有一个信号」不变量。

核心要防的回归：历史上检验用 clean_factor、回测用 prepare_signal 内联 OLS、
正交化页用 orthogonalize，三套实现互不复用，导致看到的 IC 与实际持仓来自不同信号。
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from qfm.pipeline.clean import clean_factor
from qfm.pipeline.pipeline import PipelineConfig, run_pipeline
from qfm.pipeline.stages import (
    apply_decay,
    apply_direction,
    apply_lag,
    apply_missing,
    apply_standardize,
)
from qfm.pipeline.tests import compute_ic, factor_report, forward_returns
from qfm.simulation.engine import SimulationSettings, prepare_signal


# ---------- 默认配置必须与历史 clean_factor 等价 ----------

def test_default_pipeline_reproduces_historical_clean_factor(panel):
    """默认配置 = MAD 去极值 → z-score，与历史 clean_factor 逐值一致。"""
    raw = panel.close.pct_change(20)

    signal = run_pipeline(raw, panel=panel, direction="positive").signal
    expected = clean_factor(raw.reindex(index=panel.close.index, columns=panel.close.columns))

    pd.testing.assert_frame_equal(signal, expected)


def test_negative_direction_equals_negated_positive_signal(panel):
    raw = panel.close.pct_change(20)

    positive = run_pipeline(raw, panel=panel, direction="positive").signal
    negative = run_pipeline(raw, panel=panel, direction="negative").signal

    pd.testing.assert_frame_equal(negative, -positive)


def test_pipeline_signal_feeds_factor_report_identically(panel):
    """关键不变量：管线产物配 preprocessed=True 与原始因子配默认配置，IC 必须完全相同。

    这正是「检验用的信号 == 回测用的信号」的可执行表达。
    """
    raw = panel.close.pct_change(20)
    signal = run_pipeline(raw, panel=panel, direction="positive").signal

    through_pipeline = factor_report(raw, panel.close, horizon=5, panel=panel)
    precomputed = factor_report(signal, panel.close, horizon=5, preprocessed=True)

    pd.testing.assert_series_equal(through_pipeline["ic_series"], precomputed["ic_series"])
    assert through_pipeline["ic_summary"] == precomputed["ic_summary"]


def test_prepare_signal_is_exactly_the_pipeline_output(panel):
    """回测入口不得再有任何自有处理：prepare_signal 必须逐值等于 run_pipeline 的输出。"""
    raw = panel.close.pct_change(5)
    settings = SimulationSettings(start="2024-03-01", end="2024-12-31", horizon=5, decay=3)
    dates = panel.close.index
    start_pos = max(0, dates.searchsorted(pd.Timestamp(settings.start)) - settings.decay - settings.delay - 1)
    history = dates[start_pos:][dates[start_pos:] <= pd.Timestamp(settings.end)]
    from qfm.simulation.engine import slice_panel

    window = slice_panel(panel, history)
    expected = run_pipeline(
        raw, panel=window, config=PipelineConfig.from_settings(settings),
        direction="positive", close=window.close,
    ).signal

    pd.testing.assert_frame_equal(prepare_signal(panel, raw, "positive", settings), expected)


# ---------- 阶段语义 ----------

def test_stage_frames_are_recorded_in_pipeline_order(panel):
    result = run_pipeline(panel.close.pct_change(20), panel=panel)

    order = list(result.stages)
    assert order == [
        "raw", "missing", "winsorize", "neutralize", "standardize",
        "oriented", "decay", "lag", "signal",
    ]
    assert set(result.coverage) == set(order)


def test_missing_row_drop_masks_thin_cross_sections():
    frame = pd.DataFrame(
        {"a": [1.0, np.nan, 1.0], "b": [2.0, 2.0, np.nan], "c": [3.0, 3.0, np.nan]},
        index=pd.bdate_range("2024-01-02", periods=3),
    )

    dropped = apply_missing(frame, "row_drop", min_stocks=2)

    assert dropped.iloc[0].notna().sum() == 3
    assert dropped.iloc[1].notna().sum() == 2, "达到门槛应保留"
    assert dropped.iloc[2].isna().all(), "第 3 行只有 1 个有效值，低于门槛应整行置空"


def test_missing_cs_median_fills_from_cross_section():
    frame = pd.DataFrame(
        {"a": [1.0], "b": [np.nan], "c": [3.0]},
        index=pd.bdate_range("2024-01-02", periods=1),
    )

    assert apply_missing(frame, "cs_median").iloc[0]["b"] == pytest.approx(2.0)
    assert apply_missing(frame, "none").iloc[0]["b"] != apply_missing(frame, "none").iloc[0]["b"]


def test_missing_rejects_unknown_method():
    frame = pd.DataFrame({"a": [1.0]})
    with pytest.raises(ValueError, match="未知缺失处理方式"):
        apply_missing(frame, "magic")


def test_rank_standardization_is_bounded_and_centred(panel):
    ranked = apply_standardize(panel.close, "rank")

    assert ranked.min().min() >= -0.5
    assert ranked.max().max() <= 0.5
    # 百分位排名的截面均值恒为 (n+1)/(2n)，平移 0.5 后残差为 1/(2n)（n=60 → 1/120）
    n = panel.close.shape[1]
    assert ranked.mean(axis=1).abs().max() <= 1 / (2 * n) + 1e-12
    # 排名与 z-score 必须不同，否则说明 rank 分支没生效
    assert not ranked.equals(apply_standardize(panel.close, "zscore"))


def test_standardize_none_is_passthrough(panel):
    pd.testing.assert_frame_equal(apply_standardize(panel.close, "none"), panel.close)


def test_decay_weights_recent_days_highest(panel):
    """N=3 时权重比应为 3:2:1，且只有连续 3 天都有值才出值。"""
    index = pd.bdate_range("2024-01-02", periods=4)
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, np.nan]}, index=index)

    decayed = apply_decay(frame, 3)

    assert decayed.iloc[0, 0] != decayed.iloc[0, 0], "第 1 天没有 3 天历史，应为空"
    assert decayed.iloc[1, 0] != decayed.iloc[1, 0], "第 2 天只有 2 天历史，应为空"
    assert decayed.iloc[2, 0] == pytest.approx((1 * 1 + 2 * 2 + 3 * 3) / 6)
    assert decayed.iloc[3, 0] != decayed.iloc[3, 0], "含缺失值不应出值"


def test_decay_zero_and_one_are_passthrough(panel):
    pd.testing.assert_frame_equal(apply_decay(panel.close, 0), panel.close)
    pd.testing.assert_frame_equal(apply_decay(panel.close, 1), panel.close)


def test_lag_matches_engine_execution_convention(panel):
    """引擎已按 T 日信号 T+1 开盘执行，故 lag=1 不作位移，lag=2 等价 shift(1)。"""
    frame = panel.close

    pd.testing.assert_frame_equal(apply_lag(frame, 1), frame)
    pd.testing.assert_frame_equal(apply_lag(frame, 2), frame.shift(1))
    pd.testing.assert_frame_equal(apply_lag(frame, 3), frame.shift(2))


def test_direction_returns_sign(panel):
    oriented, sign = apply_direction(panel.close, "negative")
    assert sign == -1
    pd.testing.assert_frame_equal(oriented, -panel.close)

    _, sign_positive = apply_direction(panel.close, "positive")
    assert sign_positive == 1


# ---------- 配置校验 ----------

@pytest.mark.parametrize(
    "changes, message",
    [
        ({"missing": "bad"}, "未知缺失处理方式"),
        ({"winsorize": "bad"}, "未知去极值方式"),
        ({"standardize": "bad"}, "未知标准化方式"),
        ({"neutralize": ("sector",)}, "未知中性化控制变量"),
        ({"winsorize_n": 0}, "去极值倍数"),
        ({"neutralize_min_samples": 2}, "中性化最少样本数"),
        ({"decay": -1}, "Decay"),
        ({"decay": 121}, "Decay"),
        ({"lag": 0}, "Lag"),
        ({"lag": 61}, "Lag"),
        ({"min_stocks": -1}, "截面最少股票数"),
    ],
)
def test_pipeline_config_rejects_invalid_values(changes, message):
    config = PipelineConfig(**changes)
    with pytest.raises(ValueError, match=message):
        config.validate()


def test_config_round_trips_to_dict(panel):
    config = PipelineConfig(neutralize=("industry", "size"), decay=5)

    data = config.to_dict()

    assert data["neutralize"] == ["industry", "size"]
    assert data["decay"] == 5


def test_from_settings_maps_neutralization_options():
    assert PipelineConfig.from_settings(SimulationSettings(neutralization="none")).neutralize == ()
    assert PipelineConfig.from_settings(SimulationSettings(neutralization="industry")).neutralize == ("industry",)
    assert PipelineConfig.from_settings(
        SimulationSettings(neutralization="industry_size")
    ).neutralize == ("industry", "size")


def test_neutralize_without_panel_is_rejected(panel):
    with pytest.raises(ValueError, match="但未提供 panel"):
        run_pipeline(panel.close, close=panel.close, config=PipelineConfig(neutralize=("size",)))


def test_pipeline_requires_a_grid():
    with pytest.raises(ValueError, match="必须提供 panel 或 close"):
        run_pipeline(pd.DataFrame({"a": [1.0]}))


def test_pipeline_rejects_empty_factor(panel):
    with pytest.raises(ValueError, match="原始因子为空"):
        run_pipeline(pd.DataFrame(), panel=panel)


# ---------- 中性化走统一实现 ----------

def test_neutralize_removes_industry_and_size_exposure(panel):
    # 用无缺失的价格水平作因子，避免 pct_change 造成的预热空值干扰断言
    result = run_pipeline(
        panel.close, panel=panel,
        config=PipelineConfig(neutralize=("industry", "size")),
    )

    diag = result.neutralize_diag
    assert diag["controls"], "控制变量应包含行业/市值"
    assert diag["days_skipped"] == 0
    assert diag["days"] == len(panel.close)
    assert diag["exposure_before"] > diag["exposure_after"], "中性化后暴露 R² 必须下降"
    # 逐日残差的行业均值与市值相关性应为 0
    signal = result.signal
    for date in signal.index[:5]:
        row = signal.loc[date]
        labels = panel.industry.loc[date]
        for industry in labels.dropna().unique():
            members = labels[labels == industry].index
            assert abs(row[members].mean()) < 1e-8
        size = np.log(panel.mv_float.loc[date])
        assert abs(row.corr(size)) < 1e-8


def test_neutralize_skips_days_with_too_few_samples(panel):
    result = run_pipeline(
        panel.close.pct_change(20), panel=panel,
        config=PipelineConfig(neutralize=("size",), neutralize_min_samples=10_000),
    )

    assert result.neutralize_diag["days"] == 0
    assert result.signal.isna().all().all()
    assert any("中性化有" in w for w in result.warnings)


# ---------- 轴对齐加固 ----------

def test_shuffled_columns_are_realigned_by_label(panel):
    """列顺序被打乱时必须按标签重建横截面，而不是按位置错配股票。

    注意不能拿「列子集」来测：截面只有 20 只股票时 z-score 的统计量本就不同，
    那是正确行为。这里保持列集合不变、只打乱顺序。
    """
    raw = panel.close.pct_change(20)

    straight = run_pipeline(raw, panel=panel).signal
    shuffled = run_pipeline(raw[list(reversed(raw.columns))], panel=panel).signal

    pd.testing.assert_frame_equal(shuffled, straight)


def test_date_subset_is_realigned_and_blank_outside_the_provided_window(panel):
    """只覆盖部分日期时，网格外的日期必须为空，窗口内应与全量结果一致。"""
    raw = panel.close.pct_change(20)

    full = run_pipeline(raw, panel=panel).signal
    partial = run_pipeline(raw.iloc[10:50], panel=panel).signal

    assert partial.shape == panel.close.shape
    assert partial.index.equals(panel.close.index)
    assert partial.columns.equals(panel.close.columns)
    assert partial.iloc[:10].isna().all().all()
    assert partial.iloc[50:].isna().all().all()
    # 20 行之后 raw 本身有值（pct_change(20) 的预热），应与全量结果逐值一致
    pd.testing.assert_frame_equal(partial.iloc[20:50], full.iloc[20:50])


def test_subset_of_columns_gets_nan_outside_its_region(panel):
    """列子集虽然截面统计量不同，但网格外的股票必须是空值而非错位填充。"""
    raw = panel.close.pct_change(20)

    partial = run_pipeline(raw.iloc[10:50, :20], panel=panel).signal

    assert partial.shape == panel.close.shape
    assert partial.iloc[:, 20:].isna().all().all()
    assert partial.iloc[:10].isna().all().all()
    assert partial.iloc[20:50, :20].notna().any().any()


def test_compute_ic_aligns_by_label_not_position(panel):
    """fwd_ret 传入被打乱的列顺序时，IC 必须与正确顺序一致（历史上会静默错位）。"""
    factor = clean_factor(panel.close.pct_change(20))
    fwd = forward_returns(panel.close, 5)
    shuffled = fwd[list(reversed(fwd.columns))]

    straight = compute_ic(factor, fwd)
    scrambled = compute_ic(factor, shuffled)

    pd.testing.assert_series_equal(straight, scrambled)


def test_compute_ic_handles_subset_future_returns(panel):
    factor = clean_factor(panel.close.pct_change(20))
    fwd = forward_returns(panel.close, 5)

    subset = compute_ic(factor, fwd.iloc[:100])
    full = compute_ic(factor, fwd)

    pd.testing.assert_series_equal(subset.dropna(), full.iloc[:100].dropna())


def test_compute_ic_rejects_unknown_method(panel):
    with pytest.raises(ValueError, match="未知 IC 方法"):
        compute_ic(panel.close, panel.close, method="kendall")


def test_forward_returns_requires_positive_horizon(panel):
    for bad in (0, -1):
        with pytest.raises(ValueError, match="前瞻天数"):
            forward_returns(panel.close, bad)


def test_layer_test_requires_at_least_two_layers(panel):
    from qfm.pipeline.tests import layer_test

    with pytest.raises(ValueError, match="分层数至少为 2"):
        layer_test(panel.close, panel.close, n_layers=1)


# ---------- 覆盖率与诊断 ----------

def test_coverage_decreases_or_holds_along_the_pipeline(panel):
    sparse = panel.close.pct_change(60).copy()
    sparse.iloc[:30, :] = np.nan

    result = run_pipeline(sparse, panel=panel, config=PipelineConfig(decay=5))

    assert result.coverage["raw"] < 1.0
    assert result.coverage["signal"] <= result.coverage["standardize"]
    assert result.signal_coverage == result.coverage["signal"]


def test_finalize_signal_masks_thin_days_only_when_requested(panel):
    thin = panel.close.iloc[-10:]
    defaults = run_pipeline(thin, panel=None, close=thin).signal
    masked = run_pipeline(thin, panel=None, close=thin, config=PipelineConfig(min_stocks=100)).signal

    assert defaults.notna().any().any(), "默认不做截面掩码"
    assert masked.isna().all().all(), "只有 10 只股票却要求 100 只，应全部掩码"


def test_duplicate_panel_is_not_mutated_by_pipeline(panel):
    raw = panel.close.pct_change(20)
    original = deepcopy(raw)

    run_pipeline(raw, panel=panel, config=PipelineConfig(neutralize=("industry",)))

    pd.testing.assert_frame_equal(raw, original)
