"""合成数据夹具：带行业/规模效应的 DataPanel，全量测试使用（零网络）"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qfm.data.panel import DataPanel


@pytest.fixture(scope="session")
def panel() -> DataPanel:
    rng = np.random.default_rng(42)
    n_stocks, n_days = 60, 260
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    cols = [f"{600000 + i}" for i in range(n_stocks)]
    inds = ["银行", "白酒", "科技", "医药"]
    stock_ind = [inds[i % 4] for i in range(n_stocks)]
    ind_eff = np.array([0.0009, -0.0002, 0.0006, 0.0001])       # 行业漂移
    mv_base = rng.uniform(2e9, 5e11, n_stocks)
    size_eff = (np.log(mv_base) - np.log(mv_base).mean()) / np.log(mv_base).std()

    drift = ind_eff[[i % 4 for i in range(n_stocks)]] + 0.0004 * size_eff
    ret = rng.normal(0.0002 + drift, 0.02, (n_days, n_stocks))
    close = 100.0 * np.exp(np.cumsum(ret, axis=0))

    return DataPanel(
        close=pd.DataFrame(close, index=dates, columns=cols),
        open=pd.DataFrame(close * (1 + rng.normal(0, 0.005, (n_days, n_stocks))), index=dates, columns=cols),
        high=pd.DataFrame(close * (1 + np.abs(rng.normal(0.005, 0.01, (n_days, n_stocks)))), index=dates, columns=cols),
        low=pd.DataFrame(close * (1 - np.abs(rng.normal(0.005, 0.01, (n_days, n_stocks)))), index=dates, columns=cols),
        volume=pd.DataFrame(rng.integers(1e6, 5e7, (n_days, n_stocks)).astype(float), index=dates, columns=cols),
        amount=pd.DataFrame(rng.uniform(1e8, 5e9, (n_days, n_stocks)), index=dates, columns=cols),
        turnover=pd.DataFrame(rng.uniform(0.005, 0.05, (n_days, n_stocks)), index=dates, columns=cols),
        mv_float=pd.DataFrame(np.tile(mv_base, (n_days, 1)), index=dates, columns=cols),
        industry=pd.DataFrame(np.tile(np.array(stock_ind, dtype=object), (n_days, 1)), index=dates, columns=cols),
        fund={"roe": pd.DataFrame(rng.uniform(0.05, 0.25, (n_days, n_stocks)), index=dates, columns=cols),
              "eps_ttm": pd.DataFrame(rng.uniform(0.2, 3.0, (n_days, n_stocks)), index=dates, columns=cols),
              "gross_margin": pd.DataFrame(rng.uniform(0.1, 0.5, (n_days, n_stocks)), index=dates, columns=cols)},
        fund_names=["roe", "eps_ttm", "gross_margin"],
    )


@pytest.fixture(scope="session")
def wide_panel() -> DataPanel:
    """加宽夹具：160 只股票 × 6 行业，覆盖分层检验的样本门槛。

    `qfm.pipeline.tests.layer_test` 要求每日至少 100 只有效股票（20 × 5 层），
    60 只的 `panel` 无法产出分层统计，故单独提供本夹具。
    """
    rng = np.random.default_rng(7)
    n_stocks, n_days = 160, 260
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    cols = [f"{600000 + i}" for i in range(n_stocks)]
    inds = ["银行", "白酒", "科技", "医药", "地产", "能源"]
    stock_ind = [inds[i % len(inds)] for i in range(n_stocks)]
    ind_eff = np.array([0.0009, -0.0002, 0.0006, 0.0001, -0.0004, 0.0003])
    mv_base = rng.uniform(2e9, 5e11, n_stocks)
    log_mv = np.log(mv_base)
    size_eff = (log_mv - log_mv.mean()) / log_mv.std()

    drift = ind_eff[[i % len(inds) for i in range(n_stocks)]] + 0.0004 * size_eff
    ret = rng.normal(0.0002 + drift, 0.02, (n_days, n_stocks))
    close = 100.0 * np.exp(np.cumsum(ret, axis=0))

    return DataPanel(
        close=pd.DataFrame(close, index=dates, columns=cols),
        open=pd.DataFrame(close * (1 + rng.normal(0, 0.005, (n_days, n_stocks))), index=dates, columns=cols),
        high=pd.DataFrame(close * (1 + np.abs(rng.normal(0.005, 0.01, (n_days, n_stocks)))), index=dates, columns=cols),
        low=pd.DataFrame(close * (1 - np.abs(rng.normal(0.005, 0.01, (n_days, n_stocks)))), index=dates, columns=cols),
        volume=pd.DataFrame(rng.integers(1e6, 5e7, (n_days, n_stocks)).astype(float), index=dates, columns=cols),
        amount=pd.DataFrame(rng.uniform(1e8, 5e9, (n_days, n_stocks)), index=dates, columns=cols),
        turnover=pd.DataFrame(rng.uniform(0.005, 0.05, (n_days, n_stocks)), index=dates, columns=cols),
        factor=pd.DataFrame(np.ones((n_days, n_stocks)), index=dates, columns=cols),
        close_raw=pd.DataFrame(close, index=dates, columns=cols),
        mv_float=pd.DataFrame(np.tile(mv_base, (n_days, 1)), index=dates, columns=cols),
        industry=pd.DataFrame(np.tile(np.array(stock_ind, dtype=object), (n_days, 1)), index=dates, columns=cols),
        fund={
            "roe": pd.DataFrame(rng.uniform(0.05, 0.25, (n_days, n_stocks)), index=dates, columns=cols),
            "eps_ttm": pd.DataFrame(rng.uniform(0.2, 3.0, (n_days, n_stocks)), index=dates, columns=cols),
            "bvps": pd.DataFrame(rng.uniform(2.0, 20.0, (n_days, n_stocks)), index=dates, columns=cols),
        },
        fund_names=["roe", "eps_ttm", "bvps"],
    )


# ---------------------------------------------------------------------------
# Agent 层夹具：全部离线（合成面板 + 注入的休眠/时钟），零网络、零数据缓存依赖
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def synthetic_spec():
    """固定种子的合成面板参数：同种子必然同面板，保证测试可复现。"""
    from qfm.agent.synthetic import SyntheticSpec

    return SyntheticSpec(as_of="2026-09-18", n_days=620, seed=20260928)


@pytest.fixture(scope="session")
def synthetic_provider(synthetic_spec):
    from qfm.agent.providers import SyntheticProvider

    return SyntheticProvider(synthetic_spec)


@pytest.fixture(scope="session")
def bank_symbols(synthetic_provider):
    """合成面板里的银行股代码（42 只，与真实缓存里的数量一致）。"""
    from qfm.agent.providers import build_industry_series

    panel = synthetic_provider.load("synthetic").panel
    series = build_industry_series(panel)
    return sorted(series[series == "银行Ⅱ"].index.tolist())


@pytest.fixture()
def tool_context(tmp_path, synthetic_provider):
    """一个可直接调工具的上下文，产物写到 tmp_path。"""
    from pathlib import Path

    from qfm.agent.registry import ToolContext

    return ToolContext(
        workdir=tmp_path / "run",
        data_dir=Path("data_cache"),
        provider=synthetic_provider,
    )


@pytest.fixture()
def registry():
    from qfm.agent.tools import build_default_registry

    return build_default_registry()


@pytest.fixture()
def quick_runtime(tmp_path, synthetic_provider, registry):
    """不真实等待退避的运行时，用于端到端测试。"""
    from qfm.agent.planner import RulePlanner
    from qfm.agent.runtime import AgentConfig, AgentRuntime
    from qfm.agent.retry import RetryPolicy

    def build(**overrides):
        config = AgentConfig(
            retry=RetryPolicy(max_attempts=2, base_delay=0.0, jitter=0.0),
            **overrides,
        )
        return AgentRuntime(
            registry,
            RulePlanner(),
            config,
            provider=synthetic_provider,
            runs_root=tmp_path / "runs",
            sleep=lambda seconds: None,
        )

    return build
