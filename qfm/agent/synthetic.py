"""离线合成面板：让仓库在「没有任何缓存数据」的机器上也能一键跑通。

## 为什么必须要有这个

`data_cache/` 在 `.gitignore` 里，所以面试官 clone 下来的仓库**没有任何行情
数据**。如果 Agent Demo 的第一条命令是「先去 akshare 拉 800 只股票 6 年数据」，
那这个 Demo 大概率死在网络/限流上——还没有人看到 Agent 做了什么。

因此提供一条零网络、确定性的合成数据通路：同一种子必然产出同一面板，
跑起来 1 秒内出结果，面试官能立刻看到计划、工具调用、重试、异常检查、
报告这整条链路。

## 与真实数据的一致性

合成面板不是「随便造点随机数」，而是刻意为对齐真实场景设计的：

- **行业标签用东财口径**（`银行Ⅱ`、`白酒Ⅱ`、`半导体` …），与
  `data_cache/indicators.parquet` 里的实际取值一致。真实缓存里 `银行Ⅱ`
  有 42 只，这里同样给 42 只 —— 于是「截面股票数不足 100 导致分层检验不可用」
  这个问题在两种数据源下表现一致，异常检查逻辑不会只在合成数据上才生效。
- **字段齐全**：`bvps / eps_ttm / roe / gross_margin / rev_yoy / profit_yoy /
  ocfps …` 全部生成，因此价值/质量/成长族因子都能算出非空结果。
- **复权口径正确**：同时给 `factor` 与 `close_raw`，于是
  `valuation_price(panel)` 走真实价、收益计算走前复权价 —— 与真实数据的
  量纲约定完全相同（估值类因子用前复权价当分母是经典错误）。
- **时点化行业标签**：`industry` 是 date × stock 矩阵，行业变更会在指定日期
  生效，因此按行业选股票池必须用「当期截面」而不是「最新标签」。

## 诚实声明

面板里**人为植入了银行行业内一个很弱的价值效应**（使 `bp` 因子有可观测但
不夸张的 IC），目的是让 Demo 的异常检查与结论生成有真实内容可讲。
这是合成数据，**不构成任何实证结论**。运行报告里会以显著横幅标注数据来源，
避免把演示结果误读为研究发现。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qfm.data.panel import DataPanel

__all__ = ["SyntheticSpec", "build_synthetic_panel", "SYNTHETIC_INDUSTRIES"]

#: 行业 → 股票数。银行 42 只与真实缓存一致（这是刻意对齐的关键数字）。
SYNTHETIC_INDUSTRIES: dict[str, int] = {
    "银行Ⅱ": 42,
    "白酒Ⅱ": 18,
    "半导体": 20,
    "化学制药": 16,
    "电网设备": 14,
    "软件开发": 14,
    "房地产开发": 12,
    "汽车零部件": 14,
    "食品加工": 10,
    "电力": 10,
}

#: 各行业的年化漂移（用于制造真实的行业分化，避免所有行业同涨同跌）。
_INDUSTRY_DRIFT: dict[str, float] = {
    "银行Ⅱ": 0.05,
    "白酒Ⅱ": -0.02,
    "半导体": 0.12,
    "化学制药": 0.01,
    "电网设备": 0.07,
    "软件开发": 0.03,
    "房地产开发": -0.09,
    "汽车零部件": 0.06,
    "食品加工": -0.01,
    "电力": 0.04,
}

_FUND_FIELDS: tuple[str, ...] = (
    "roe",
    "gross_margin",
    "ocfps",
    "rev_yoy",
    "profit_yoy",
    "rev_qoq",
    "profit_qoq",
)


class SyntheticSpec:
    """合成面板的生成参数。

    Args:
        as_of: 最后一个交易日。默认与本地缓存的最晚日期对齐，便于两种数据源对比。
        n_days: 交易日数量（默认 620 ≈ 2.5 年，覆盖「最近一年」「最近两年」等区间）。
        seed: 随机种子。同种子 → 完全相同面板（这是可复现性的前提）。
        value_effect: 银行行业内植入的价值效应强度。0 表示纯噪声面板，
            用于测试「无效应时 Agent 是否如实报告不显著」。
    """

    __slots__ = ("as_of", "n_days", "seed", "value_effect")

    def __init__(
        self,
        as_of: str = "2026-09-18",
        n_days: int = 620,
        seed: int = 20260928,
        value_effect: float = 0.16,
    ) -> None:
        if n_days < 260:
            raise ValueError(f"n_days 至少 260（约一年），得到 {n_days}")
        if value_effect < 0:
            raise ValueError("value_effect 不能为负")
        self.as_of = as_of
        self.n_days = n_days
        self.seed = seed
        self.value_effect = value_effect

    def to_dict(self) -> dict[str, object]:
        return {
            "as_of": self.as_of,
            "n_days": self.n_days,
            "seed": self.seed,
            "value_effect": self.value_effect,
            "industries": dict(SYNTHETIC_INDUSTRIES),
        }


def build_synthetic_panel(spec: SyntheticSpec | None = None) -> DataPanel:
    """生成确定性合成面板。纯 CPU、无网络、无 IO。"""
    cfg = spec or SyntheticSpec()
    rng = np.random.default_rng(cfg.seed)

    codes, industries = _symbols_and_industries()
    n_stocks = len(codes)
    dates = pd.bdate_range(end=pd.Timestamp(cfg.as_of), periods=cfg.n_days)

    industry_arr = np.array(industries, dtype=object)
    is_bank = industry_arr == "银行Ⅱ"

    # ---- 基本面：每只股票一条缓慢变化的每股净资产 / 盈利路径 ----
    bvps0 = rng.uniform(3.0, 26.0, n_stocks)
    eps0 = np.maximum(bvps0 * rng.uniform(0.05, 0.22, n_stocks), 0.03)
    roe0 = rng.uniform(0.03, 0.22, n_stocks)
    gross0 = rng.uniform(0.08, 0.55, n_stocks)
    # 基本面按季度阶梯推进（真实财报就是季度频率），再前向填充到日频。
    quarter_steps = max(2, cfg.n_days // 62)
    q_idx = np.linspace(0, quarter_steps - 1, cfg.n_days).astype(int)
    growth = rng.normal(0.012, 0.02, (quarter_steps, n_stocks))
    fund_scale = np.exp(np.cumsum(growth, axis=0))
    bvps = bvps0[None, :] * fund_scale[q_idx]
    eps_ttm = eps0[None, :] * fund_scale[q_idx] * rng.normal(1.0, 0.05, (cfg.n_days, n_stocks))
    eps_ttm = np.maximum(eps_ttm, 0.01)

    # ---- 价格：行业漂移 + 规模效应 + 特质波动 ----
    log_mv = rng.uniform(np.log(2e9), np.log(8e11), n_stocks)
    size_z = (log_mv - log_mv.mean()) / log_mv.std()
    drift = np.array([_INDUSTRY_DRIFT[ind] for ind in industries]) / 252.0
    daily_drift = drift + 0.00020 * size_z

    idio = rng.normal(0.0, 0.019, (cfg.n_days, n_stocks))
    # 银行股波动天然更低，也更同步（真实特征）
    idio[:, is_bank] *= 0.72

    intrinsic_ret = daily_drift[None, :] + idio
    intrinsic_close = 100.0 * np.exp(np.cumsum(intrinsic_ret, axis=0))

    # ---- 植入价值效应：银行行业内，低估值（高 bp）股票次日略微占优 ----
    # 用「内部价格」算 bp 以避免循环依赖；标签只用 t-1 及之前的数据，无未来函数。
    intrinsic_bp = bvps / intrinsic_close
    shock = np.zeros((cfg.n_days, n_stocks))
    if cfg.value_effect > 0:
        bank_idx = np.flatnonzero(is_bank)
        bank_bp = intrinsic_bp[:, bank_idx]
        # 截面 rank 标准化到 [-0.5, 0.5]
        ranks = pd.DataFrame(bank_bp).rank(axis=1, pct=True).to_numpy() - 0.5
        lagged = np.vstack([np.zeros((1, len(bank_idx))), ranks[:-1]])
        shock[:, bank_idx] = cfg.value_effect / 252.0 * lagged * 2.0

    ret = intrinsic_ret + shock
    close = 100.0 * np.exp(np.cumsum(ret, axis=0))

    # ---- 组装 DataPanel（复权因子与真实价分离，与真实数据量纲一致）----
    factor = np.cumprod(1.0 + rng.normal(0.0, 0.0002, (cfg.n_days, n_stocks)), axis=0)
    close_raw = close / factor

    def frame(values: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(values, index=dates, columns=codes)

    open_ = close * (1 + rng.normal(0, 0.004, (cfg.n_days, n_stocks)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0.004, 0.008, (cfg.n_days, n_stocks))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0.004, 0.008, (cfg.n_days, n_stocks))))
    volume = rng.integers(2e5, 8e7, (cfg.n_days, n_stocks)).astype(float)
    amount = volume * close * rng.uniform(0.9, 1.1, (cfg.n_days, n_stocks))

    mv_float = np.tile(np.exp(log_mv), (cfg.n_days, 1))
    # 流通股本 = 流通市值 / 真实价；换手率 = 成交量 / 流通股本
    outstanding = mv_float / close_raw
    turnover = volume / outstanding

    # 时点化行业标签：第 60% 处让 1 只股票换行业，用于验证「按当期截面选池」。
    industry_df = pd.DataFrame(
        np.tile(industry_arr, (cfg.n_days, 1)), index=dates, columns=codes
    )
    if cfg.n_days > 300:
        switch_at = int(cfg.n_days * 0.6)
        mover = next(i for i, ind in enumerate(industries) if ind == "食品加工")
        industry_df.iloc[switch_at:, mover] = "汽车零部件"

    fund: dict[str, pd.DataFrame] = {
        "bvps": frame(bvps),
        "eps_ttm": frame(eps_ttm),
        "eps": frame(eps_ttm * 0.7),
        "roe": frame(np.clip(roe0[None, :] * rng.normal(1.0, 0.06, (cfg.n_days, n_stocks)), 0.01, 0.4)),
        "gross_margin": frame(np.clip(gross0[None, :] * rng.normal(1.0, 0.05, (cfg.n_days, n_stocks)), 0.01, 0.9)),
    }
    for name in _FUND_FIELDS:
        if name in fund:
            continue
        if "yoy" in name:
            fund[name] = frame(rng.normal(0.10, 0.28, (cfg.n_days, n_stocks)))
        elif "qoq" in name:
            fund[name] = frame(rng.normal(0.03, 0.16, (cfg.n_days, n_stocks)))
        elif name == "ocfps":
            fund[name] = frame(np.abs(eps_ttm) * rng.uniform(0.6, 1.8, (cfg.n_days, n_stocks)))

    return DataPanel(
        close=frame(close),
        open=frame(open_),
        high=frame(high),
        low=frame(low),
        volume=frame(volume),
        amount=frame(amount),
        turnover=frame(turnover),
        factor=frame(factor),
        close_raw=frame(close_raw),
        mv_float=frame(mv_float),
        industry=industry_df,
        fund=fund,
        fund_names=list(fund),
    )


def _symbols_and_industries() -> tuple[list[str], list[str]]:
    """生成股票代码与行业标签。

    代码按沪深两市交替分配（沪市 60xxxx / 深市 00xxxx），并按行业顺序排列，
    使同一行业的代码集中，便于人工核对分组是否正确。
    """
    codes: list[str] = []
    industries: list[str] = []
    sh_next, sz_next = 600000, 1
    for industry, count in SYNTHETIC_INDUSTRIES.items():
        for _ in range(count):
            if len(codes) % 2 == 0:
                codes.append(f"{sh_next:06d}")
                sh_next += 1
            else:
                codes.append(f"{sz_next:06d}")
                sz_next += 1
            industries.append(industry)
    return codes, industries
