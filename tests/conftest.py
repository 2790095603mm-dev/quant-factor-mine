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
