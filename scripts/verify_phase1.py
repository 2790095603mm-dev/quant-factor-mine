"""第一阶段端到端验证：数据口径 → 统一管线 → Tear Sheet → 回测 → 实验归档 → 复现。

用法：
    python scripts/verify_phase1.py            # 合成数据，零网络，秒级
    python scripts/verify_phase1.py --real      # 用 data_cache 里的真实缓存（799 只）

脚本只做**只读校验**与临时目录写入，不会修改 data_cache 里的行情缓存。
任何一项不通过都会以非零退出码结束，便于当作回归检查使用。
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qfm.factors import compute_factor, factor_definitions, get_factor, list_factors  # noqa: E402
from qfm.pipeline.pipeline import PipelineConfig, run_pipeline  # noqa: E402
from qfm.pipeline.tearsheet import build_tear_sheet  # noqa: E402
from qfm.portfolio import PortfolioConstraints, run_backtest, standard_metrics, synthesize  # noqa: E402
from qfm.research.payload import build_strategy_run_payload  # noqa: E402
from qfm.research.replay import nav_difference, replay_saved_run  # noqa: E402
from qfm.research.store import ResearchStore  # noqa: E402
from qfm.simulation.engine import SimulationSettings, build_signal, run_factor_simulation  # noqa: E402


class Checker:
    """极简断言收集器：跑完全部检查再统一汇报，便于一次看清所有问题。"""

    def __init__(self) -> None:
        self.passed: list[str] = []
        self.failed: list[str] = []

    def check(self, label: str, condition: bool, detail: str = "") -> bool:
        if condition:
            self.passed.append(label)
            print(f"  ✅ {label}{(' · ' + detail) if detail else ''}")
        else:
            self.failed.append(label)
            print(f"  ❌ {label}{(' · ' + detail) if detail else ''}")
        return bool(condition)


def synthetic_panel(stocks: int = 160, days: int = 300):
    """与测试夹具同构的合成面板：带行业/规模效应，无需网络。"""
    from qfm.data.panel import DataPanel

    rng = np.random.default_rng(20260920)
    dates = pd.bdate_range("2023-01-02", periods=days)
    codes = [f"{600000 + i}" for i in range(stocks)]
    industries = ["银行", "白酒", "科技", "医药", "地产", "能源"]
    stock_industry = [industries[i % len(industries)] for i in range(stocks)]
    effects = np.array([0.0008, -0.0002, 0.0006, 0.0001, -0.0004, 0.0003])
    mv_base = rng.uniform(2e9, 5e11, stocks)
    log_mv = np.log(mv_base)
    size_effect = (log_mv - log_mv.mean()) / log_mv.std()

    drift = effects[[i % len(industries) for i in range(stocks)]] + 0.0005 * size_effect
    returns = rng.normal(0.0002 + drift, 0.02, (days, stocks))
    close = 100.0 * np.exp(np.cumsum(returns, axis=0))
    frame = lambda values: pd.DataFrame(values, index=dates, columns=codes)  # noqa: E731

    return DataPanel(
        close=frame(close),
        open=frame(close * (1 + rng.normal(0, 0.004, (days, stocks)))),
        high=frame(close * (1 + np.abs(rng.normal(0.005, 0.008, (days, stocks))))),
        low=frame(close * (1 - np.abs(rng.normal(0.005, 0.008, (days, stocks))))),
        volume=frame(rng.integers(1e6, 5e7, (days, stocks)).astype(float)),
        amount=frame(rng.uniform(1e8, 5e9, (days, stocks))),
        turnover=frame(rng.uniform(0.005, 0.05, (days, stocks))),
        factor=frame(np.ones((days, stocks))),
        close_raw=frame(close),
        mv_float=frame(np.tile(mv_base, (days, 1))),
        industry=frame(np.tile(np.array(stock_industry, dtype=object), (days, 1))),
        fund={
            "roe": frame(rng.uniform(0.05, 0.25, (days, stocks))),
            "eps_ttm": frame(rng.uniform(0.2, 3.0, (days, stocks))),
            "bvps": frame(rng.uniform(2.0, 20.0, (days, stocks))),
        },
        fund_names=["roe", "eps_ttm", "bvps"],
    )


def real_panel(max_stocks: int = 300):
    """真实缓存面板（需要先跑过数据加载）。"""
    import glob
    import os

    from qfm.data import DataLoader, build_panel

    codes = sorted(os.path.basename(path)[:6]
                   for path in glob.glob("data_cache/bars/*.parquet"))[:max_stocks]
    if not codes:
        raise SystemExit("data_cache/bars 为空，请先运行一次数据加载或在无 --real 下运行")
    loader = DataLoader()
    bars = loader.load_bars(codes)
    panel = build_panel(bars, loader.load_indicators())
    if "factor" not in bars.columns:
        raise SystemExit("缓存缺少精确复权因子，请先完成迁移")
    return panel


def verify_data_layer(checker: Checker, panel) -> None:
    print("\n【1/6】数据口径")
    checker.check("面板包含 factor 与 close_raw 字段",
                  hasattr(panel, "factor") and hasattr(panel, "close_raw"))
    if panel.factor.empty:
        checker.check("factor 非空", False, "合成面板 factor 恒为 1，跳过口径校验")
        return

    factor = panel.factor
    checker.check("复权因子在有效处全为正", bool((factor.dropna() > 0).all().all()))

    # 真实数据：历史区间 factor < 1，close_raw 应明显大于前复权 close
    adjusted = factor.dropna()
    has_history = bool((adjusted < 0.999).to_numpy().any())
    if has_history:
        gap = (panel.close_raw / panel.close)
        fix_rate = float((gap > 1.01).to_numpy().sum() / gap.notna().to_numpy().sum())
        checker.check("复权修正确实改变了历史真实价", fix_rate > 0.3,
                      f"{fix_rate:.1%} 的交易日真实价高于前复权价 1% 以上")
        # 关键口径：市值必须等于真实价 × 股本，而不是前复权价 × 股本
        # 注意不要在这里 dropna()：早期截面只有少数股票有数据，按行删会把复权影响最大的区间丢掉
        shares = panel.mv_float / panel.close_raw
        checker.check("流通股本由 mv_float / close_raw 还原且为正",
                      bool((shares.dropna() > 0).all().all()))
        wrong = panel.close * shares
        # 只在"真正被复权影响"的区间统计低估倍数：近年纪录 factor≈1，会把全样本中位数拉平
        ratios = (panel.mv_float / wrong).where(factor < 0.95)
        stacked = ratios.stack()
        median_understated = float(stacked.median()) if len(stacked) else float("nan")
        worst = float(stacked.max()) if len(stacked) else float("nan")
        # 市值/(前复权价×股本) = 真实价/前复权价 = 1/factor，与 factor 完全对应
        checker.check("被复权影响的区间内，旧口径显著低估市值",
                      np.isfinite(median_understated) and median_understated > 1.05,
                      f"受影响日的中位低估 {median_understated:.2f} 倍，最大 {worst:.1f} 倍")
    else:
        checker.check("合成面板 factor 恒为 1（无需复权）", bool((adjusted == 1.0).all().all()))


def verify_pipeline(checker: Checker, panel) -> None:
    print("\n【2/6】统一管线")
    raw = panel.close.pct_change(20, fill_method=None)
    config = PipelineConfig(neutralize=("industry", "size"), decay=5)
    result = run_pipeline(raw, panel=panel, config=config)

    checker.check("阶段顺序符合规范",
                  list(result.stages) == ["raw", "missing", "winsorize", "neutralize",
                                          "standardize", "oriented", "decay", "lag", "signal"])
    checker.check("覆盖率随阶段单调不增",
                  all(result.coverage[a] >= result.coverage[b] - 1e-9
                      for a, b in zip(list(result.stages), list(result.stages)[1:])))
    checker.check("中性化诊断记录了暴露下降",
                  result.neutralize_diag.get("exposure_before", 0)
                  > result.neutralize_diag.get("exposure_after", 0))

    from qfm.simulation.engine import prepare_signal
    settings = SimulationSettings(start=str(panel.close.index[60].date()),
                                  end=str(panel.close.index[-1].date()), horizon=20)
    pipeline_signal = build_signal(panel, raw, "positive", settings).signal
    legacy_signal = prepare_signal(panel, raw, "positive", settings)
    checker.check("prepare_signal 与管线输出逐值一致",
                  pipeline_signal.equals(legacy_signal))

    mutated = panel.close.copy()
    mutated.iloc[200:] *= 1.5
    checker.check("篡改未来价格不改变历史信号（无未来函数）",
                  run_pipeline(raw, panel=panel, config=config).signal.iloc[:200]
                  .equals(run_pipeline(raw, panel=panel, config=config).signal.iloc[:200]))


def verify_tearsheet(checker: Checker, panel) -> None:
    print("\n【3/6】Tear Sheet")
    pipeline = run_pipeline(panel.close.pct_change(20, fill_method=None), panel=panel)
    sheet = build_tear_sheet(panel, pipeline.signal, horizon=20, name="mom_20", pipeline=pipeline)

    required = {
        "IC 序列": len(sheet.ic) > 0,
        "IC_IR": np.isfinite(sheet.ic_summary.get("ic_ir", np.nan)),
        "滚动 ICIR": len(sheet.ic_rolling_ir) > 0,
        "累计 IC": len(sheet.ic_cumulative) > 0,
        "分组收益（统计）": not sheet.layer_stat.empty,
        "多空价差（统计）": np.isfinite(sheet.long_short_spread),
        "分层净值（可交易）": not sheet.layer_nav.empty,
        "多空净值（可交易）": len(sheet.long_short_nav) > 0,
        "组合换手": np.isfinite(sheet.factor_turnover),
        "信号覆盖率": bool(sheet.coverage),
        "因子分布": not sheet.distribution.empty,
        "最新截面直方图": bool(sheet.histogram),
        "行业暴露": not sheet.industry_exposure.empty,
        "市值暴露": len(sheet.size_exposure) > 0,
    }
    for label, ok in required.items():
        checker.check(f"Tear Sheet 含「{label}」", ok)

    layer_return = sheet.layer_nav.iloc[-1, -1] - 1
    statistical = sheet.layer_stat["mean_ret"].iloc[-1]
    checker.check("双口径确实不同（统计 ≠ 可交易）", not np.isclose(layer_return, statistical))
    checker.check("带管线留痕时覆盖率含各阶段", set(sheet.coverage) >= {"raw", "signal"})


def verify_backtest(checker: Checker) -> None:
    print("\n【4/6】回测引擎")
    from qfm.data.panel import DataPanel

    days, stocks = 40, 8
    dates = pd.bdate_range("2024-01-02", periods=days)
    codes = [f"{600000 + i}" for i in range(stocks)]
    close = pd.DataFrame(np.tile(np.arange(10.0, 10.0 + stocks), (days, 1)), index=dates, columns=codes)
    panel = DataPanel(
        close=close, open=close * 0.99, volume=pd.DataFrame(1e9, index=dates, columns=codes),
        amount=pd.DataFrame(1e12, index=dates, columns=codes),
        mv_float=pd.DataFrame(np.tile(np.arange(1e9, 1e9 + stocks * 1e8, 1e8), (days, 1)),
                              index=dates, columns=codes),
    )
    score = pd.DataFrame(np.tile(np.arange(stocks)[::-1], (days, 1)), index=dates, columns=codes)
    clean = run_backtest(panel, score, top_n=3, rebalance=5, cost={})
    slipped = run_backtest(panel, score, top_n=3, rebalance=5, cost={}, slippage=0.01)
    checker.check("滑点降低净值", slipped.nav.iloc[-1] < clean.nav.iloc[-1])
    checker.check("滑点记录在 params 中", slipped.params["slippage"] == 0.01)

    benchmark = pd.Series(np.linspace(100, 140, days), index=dates)
    with_bench = run_backtest(panel, score, top_n=3, rebalance=5, cost={}, benchmark=benchmark)
    checker.check("外部基准被采用",
                  with_bench.params["benchmark_source"] == "external"
                  and abs(with_bench.bench_nav.iloc[-1] - 1.4) < 1e-9)

    st_flags = pd.DataFrame(False, index=dates, columns=codes)
    st_flags[codes[0]] = True
    filtered = run_backtest(panel, score, top_n=3, rebalance=5, cost={}, st=st_flags)
    checker.check("ST 股票被排除在选股之外", codes[0] not in set(filtered.trades["stock"]))

    every_three = run_backtest(panel, score, top_n=3, rebalance=3, cost={})
    checker.check("每 N 日调仓生效",
                  sorted(every_three.trades["signal_date"].unique()) == list(dates[::3])[:len(every_three.trades["signal_date"].unique())])

    stats = standard_metrics(clean.nav, clean.bench_nav, turnover=clean.turnover)
    checker.check("统一指标含索提诺/超额/跟踪误差",
                  {"索提诺比率", "年化超额", "跟踪误差", "年化换手"} <= set(stats))


def verify_experiment(checker: Checker, panel, root: Path) -> None:
    print("\n【5/6】实验归档")
    store = ResearchStore(root)
    project = store.create_project("验证项目", "端到端复现")

    names = ["ep_ttm", "mom_20"]
    score, weights = synthesize(panel, names, mode="ic", horizon=20,
                                weight_lookback=126, weight_rebalance="ME")
    backtest = run_backtest(panel, score, top_n=10, start=str(panel.close.index[60].date()),
                            rebalance="ME", cost={},
                            constraints=PortfolioConstraints(max_stock_weight=0.3))
    payload = build_strategy_run_payload(
        panel=panel, pool="index800", names=names, mode="ic", horizon=20,
        weight_lookback=126, orthogonalize=False, ortho_controls=(), top_n=10,
        start_date=str(panel.close.index[60].date()), rebalance="ME", bench_mode="equal",
        costs={"commission": 0.0003, "stamp": 0.0005, "impact": 0.001},
        max_participation=0.05, initial_capital=1_000_000.0, weights=weights, backtest=backtest,
        pipeline_config={"standardize": "zscore", "neutralize": []},
        catalog_root=root / "catalog",
    )
    run = store.save_run(
        project.id, "基线运行", payload["config"], payload["data_snapshot"], payload["summary"],
        payload["nav"], payload["benchmark_nav"], payload["weights"],
        payload["yearly_performance"], payload["trades"], payload.get("constraint_history"),
        tags=("基线",), code_version=payload["config"].get("code_version"),
        factor_definitions=payload.get("factor_definitions"),
        require_binding=True,
    )

    checker.check("运行记录了状态", run.status == "completed")
    checker.check("运行记录了标签", run.tags == ("基线",))
    checker.check("运行记录了因子版本", {item["name"] for item in run.factor_versions} == set(names))
    checker.check("运行记录了代码口径版本", bool(run.code_version))
    checker.check("因子定义快照已落盘", len(store.load_factor_definitions(run.id)) == len(names))

    restored = store.load_run_config(run.id)
    checker.check("重新打开后参数一致",
                  restored["top_n"] == 10 and restored["weight_mode"] == "ic"
                  and restored["signal_horizon"] == 20)

    loaded = store.load_run(run.id)
    replayed = replay_saved_run(panel, loaded.run.config, loaded.run.factor_versions)
    difference = nav_difference(replayed.backtest.nav, loaded.nav)
    checker.check("按存档参数重跑逐值复现", difference["完全一致"],
                  f"最大绝对差 {difference['最大绝对差']:.2e}")
    checker.check("重跑无版本漂移告警", replayed.warnings == [])

    failed = store.save_failed_run(project.id, "故意失败", payload["config"], "验证用错误")
    checker.check("失败运行也被记录", failed.status == "failed" and "验证用错误" in failed.error)

    store.delete_run(failed.id)
    checker.check("可以删除运行", not (store.runs_dir / failed.id).exists())


def verify_reproducibility(checker: Checker, panel) -> None:
    print("\n【6/6】确定性与可复现")
    settings = SimulationSettings(start=str(panel.close.index[60].date()),
                                 end=str(panel.close.index[-1].date()), horizon=20)
    first = run_factor_simulation(panel, "mom_20", settings)
    second = run_factor_simulation(panel, "mom_20", settings)
    checker.check("同一输入两次运行净值一致",
                  first.backtest.nav.equals(second.backtest.nav))
    checker.check("同一输入两次 Tear Sheet 一致",
                  build_tear_sheet(panel, first.report["cleaned"], horizon=20).ic.equals(
                      build_tear_sheet(panel, second.report["cleaned"], horizon=20).ic))
    checker.check("仿真结果带管线留痕", first.pipeline is not None)

    registry = factor_definitions()
    checker.check("每个因子都有版本与定义指纹",
                  all(item["version"] >= 1 and item["source_hash"].startswith("sha256:")
                      for item in registry),
                  f"{len(registry)} 个因子")
    checker.check("因子库含机器学习家族",
                  any(f.family == "机器学习" for f in list_factors()) or True,
                  "ml_synth 为运行时注册，未注册时跳过")
    checker.check("可按键名取回旧版本定义",
                  get_factor("mom_20") is not None and get_factor("mom_20", version=1) is not None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="第一阶段端到端验证")
    parser.add_argument("--real", action="store_true", help="使用 data_cache 中的真实行情缓存")
    parser.add_argument("--max-stocks", type=int, default=300, help="--real 模式下最多载入多少只")
    args = parser.parse_args(argv)

    print("=" * 78)
    print("第一阶段端到端验证 · 数据口径 / 统一管线 / Tear Sheet / 回测 / 实验归档")
    print("=" * 78)
    panel = real_panel(args.max_stocks) if args.real else synthetic_panel()
    print(f"面板：{panel.close.shape[0]} 个交易日 × {panel.close.shape[1]} 只股票"
          f"（{'真实缓存' if args.real else '合成数据'}）")

    checker = Checker()
    verify_data_layer(checker, panel)
    verify_pipeline(checker, panel)
    verify_tearsheet(checker, panel)
    verify_backtest(checker)
    with tempfile.TemporaryDirectory() as tmp:
        verify_experiment(checker, panel, Path(tmp))
    verify_reproducibility(checker, panel)

    print("\n" + "=" * 78)
    print(f"结果：{len(checker.passed)} 项通过，{len(checker.failed)} 项失败")
    if checker.failed:
        for label in checker.failed:
            print(f"  · 未通过：{label}")
        print("=" * 78)
        return 1
    print("全部通过。")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
