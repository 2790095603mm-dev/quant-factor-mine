"""未来函数与日期对齐的运行时验证。

`qfm.pipeline.lookahead` 只做源码正则扫描（静态、可被绕过）。本文件用**变异法**做运行时验证：
篡改某日之后的全部行情/财务/行业/因子后，该日之前的信号与统计量必须不变；追加数据也不得改写历史。
这是「结构性无未来函数」的可执行证据。

容差说明：中性化路径走逐日 OLS，`controls_from_panel` 用**整个面板**构造行业哑变量，
因此未来新增行业标签会让历史的等价设计矩阵在浮点层面重排 —— 列空间不变、残差理论上相同，
但 `np.linalg.lstsq` 结果会有 1e-16 级差异。故中性化配置用极紧的相对容差断言，
纯时序路径（默认/Decay/Rank/Lag）则要求**逐位相同**。
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from qfm.pipeline.pipeline import PipelineConfig, run_pipeline
from qfm.pipeline.tests import compute_ic, factor_report, forward_returns, layer_test

NUMERIC_FIELDS = (
    "close", "open", "high", "low", "volume", "amount", "turnover",
    "factor", "close_raw", "mv_float",
)

PIPELINE_CONFIGS = {
    "default": PipelineConfig(),
    "decay5": PipelineConfig(decay=5),
    "rank": PipelineConfig(standardize="rank"),
    "cs_median": PipelineConfig(missing="cs_median"),
    "row_drop": PipelineConfig(missing="row_drop", neutralize_min_samples=10),
    "lag3": PipelineConfig(lag=3),
    "min_stocks": PipelineConfig(min_stocks=30),
    "industry_size": PipelineConfig(neutralize=("industry", "size")),
    "style": PipelineConfig(neutralize=("style",)),
}


def _requires_tolerance(config: PipelineConfig) -> bool:
    """中性化走 OLS，允许 1e-16 级浮点差异；其余路径必须逐位相同。"""
    return bool(config.neutralize)


def _mutate_after(panel, cutoff):
    """把 cutoff 之后的行情/财务/行业全部改成畸变值（因子由 close 派生，故一并变化）。"""
    changed = deepcopy(panel)
    future = changed.close.index > cutoff
    for name in NUMERIC_FIELDS:
        frame = getattr(changed, name, None)
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            frame.loc[future, :] = frame.loc[future, :] * 3.0 + 7.0
    if isinstance(changed.industry, pd.DataFrame) and not changed.industry.empty:
        changed.industry.loc[future, :] = "畸变行业"
    for key, frame in changed.fund.items():
        changed.fund[key] = frame.copy()
        changed.fund[key].loc[future, :] = frame.loc[future, :] * -5.0 + 100.0
    return changed


def _factor_of(candidate_panel):
    """因子由 close 派生，确保面板变异能传导到因子（否则变异测试会空转）。"""
    return candidate_panel.close.pct_change(20)


def _assert_same(left, right, config, message):
    if _requires_tolerance(config):
        pd.testing.assert_frame_equal(left, right, rtol=1e-12, atol=1e-14, obj=message)
    else:
        pd.testing.assert_frame_equal(left, right, check_exact=True, obj=message)


@pytest.mark.parametrize("label", list(PIPELINE_CONFIGS))
def test_signal_before_cutoff_is_invariant_to_future_data(panel, label):
    config = PIPELINE_CONFIGS[label]
    cutoff = panel.close.index[150]
    mutated_panel = _mutate_after(panel, cutoff)

    original = run_pipeline(_factor_of(panel), panel=panel, config=config).signal
    mutated = run_pipeline(_factor_of(mutated_panel), panel=mutated_panel, config=config).signal

    past = original.index <= cutoff
    _assert_same(
        original.loc[past], mutated.loc[past], config,
        f"{label}：cutoff 之前的信号被未来数据改写了",
    )
    # 变异必须真的传导到未来区间，否则本测试是空转
    assert not original.iloc[~past].equals(mutated.iloc[~past]), (
        f"{label}：变异未能传导到未来区间，测试无效"
    )


@pytest.mark.parametrize("label", list(PIPELINE_CONFIGS))
def test_ic_before_cutoff_is_invariant_to_future_data(panel, label):
    """IC 是回测结论的直接来源，必须同样不受未来数据影响。"""
    config = PIPELINE_CONFIGS[label]
    index = panel.close.index
    cutoff, horizon = index[150], 5
    # 标签要用**原始** close 计算；且末 horizon 天的标签必然用到 cutoff 之后的价格，需剔除
    compare_until = index[150 - horizon]
    mutated_panel = _mutate_after(panel, cutoff)
    fwd = forward_returns(panel.close, horizon)

    def ic_until(candidate_panel):
        signal = run_pipeline(_factor_of(candidate_panel), panel=candidate_panel, config=config).signal
        series = compute_ic(signal, fwd)
        return series.loc[series.index <= compare_until].dropna()

    base, changed = ic_until(panel), ic_until(mutated_panel)

    if _requires_tolerance(config):
        pd.testing.assert_series_equal(base, changed, rtol=1e-12, atol=1e-14)
    else:
        pd.testing.assert_series_equal(base, changed, check_exact=True)


def test_appending_history_does_not_rewrite_earlier_signal(panel):
    """数据变长后，已有区间的信号必须逐值不变（实验可复现的前提）。"""
    raw = _factor_of(panel)
    config = PipelineConfig(neutralize=("industry", "size"))
    keep = panel.close.index[:200]

    truncated = deepcopy(panel)
    for name in NUMERIC_FIELDS:
        frame = getattr(truncated, name, None)
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            setattr(truncated, name, frame.loc[keep])
    for key in list(truncated.fund):
        truncated.fund[key] = truncated.fund[key].loc[keep]

    short_signal = run_pipeline(raw.loc[keep], panel=truncated, config=config).signal
    long_signal = run_pipeline(raw, panel=panel, config=config).signal

    _assert_same(short_signal, long_signal.loc[keep], config, "截断后重算改写了已有区间")


def test_forward_returns_look_strictly_forward(panel):
    """未来收益必须只使用 t 之后的价格，且末尾 horizon 行必然为空。"""
    horizon = 5
    fwd = forward_returns(panel.close, horizon)

    assert fwd.iloc[-horizon:].isna().all().all(), "末尾缺少未来价格，应为空"
    assert fwd.iloc[:-horizon].notna().any().any()
    for position in (0, 50, 100):
        date = panel.close.index[position]
        expected = panel.close.iloc[position + horizon] / panel.close.iloc[position] - 1
        pd.testing.assert_series_equal(
            fwd.loc[date], expected.rename(date), check_names=False, check_exact=True,
        )


def test_future_price_change_does_not_touch_past_ic(panel):
    """只改未来价格（因子固定）时，cutoff 之前的 IC 序列必须完全不变。"""
    horizon = 5
    index = panel.close.index
    cutoff, compare_until = index[150], index[145]
    signal = run_pipeline(_factor_of(panel), panel=panel).signal

    base_ic = compute_ic(signal, forward_returns(panel.close, horizon))
    changed = deepcopy(panel)
    changed.close.loc[changed.close.index > cutoff, :] *= 10.0
    changed_ic = compute_ic(signal, forward_returns(changed.close, horizon))

    past = base_ic.index <= compare_until
    pd.testing.assert_series_equal(
        base_ic.loc[past], changed_ic.loc[past], check_exact=True,
        obj="未来价格变化不得影响历史 IC",
    )


def test_mutating_only_the_factor_leaves_history_bit_identical(panel):
    """对照：只改未来因子值时，历史区间必须逐位相同。"""
    config = PipelineConfig()
    raw = panel.close.pct_change(20)
    cutoff = panel.close.index[150]

    original = run_pipeline(raw, panel=panel, config=config).signal
    mutated_raw = raw.copy()
    mutated_raw.loc[mutated_raw.index > cutoff, :] = 123.0
    mutated = run_pipeline(mutated_raw, panel=panel, config=config).signal

    past = original.index <= cutoff
    pd.testing.assert_frame_equal(original.loc[past], mutated.loc[past], check_exact=True)
    # 常数截面的 MAD=0、标准差=0，z-score 后无有效排名，故未来区间应为空信号
    assert mutated.iloc[~past].isna().all().all()


def test_layer_statistics_sample_counts_only_shrink_when_factor_is_truncated(panel):
    """分层是「未来收益的统计」，允许被未来价格影响；这里验证样本数只减不增。"""
    raw = panel.close.pct_change(20)
    cutoff = panel.close.index[150]
    fwd = forward_returns(panel.close, 5)

    full = layer_test(raw, fwd)
    truncated_raw = raw.copy()
    truncated_raw.loc[truncated_raw.index > cutoff, :] = np.nan

    truncated = layer_test(truncated_raw, fwd)

    assert list(truncated.index) == list(full.index)
    assert (truncated["n_days"] <= full["n_days"]).all()


def test_factor_report_pipeline_trace_carries_coverage(panel):
    report = factor_report(panel.close.pct_change(20), panel.close, horizon=5, panel=panel)

    assert report["pipeline"] is not None
    assert report["coverage"]["signal"] == pytest.approx(report["pipeline"].signal_coverage)
    assert set(report["coverage"]) >= {"raw", "winsorize", "neutralize", "standardize", "signal"}


def test_preprocessed_report_has_no_pipeline_trace(panel):
    """调用方已处理过的信号不应再次进管线（否则会重复截尾、IC 与持仓排序不一致）。"""
    raw = panel.close.pct_change(20)
    signal = run_pipeline(raw, panel=panel).signal

    report = factor_report(signal, panel.close, horizon=5, preprocessed=True)

    assert report["pipeline"] is None
    pd.testing.assert_frame_equal(report["cleaned"], signal)
