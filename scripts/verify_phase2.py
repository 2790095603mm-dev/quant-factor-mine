"""第二阶段离线端到端验证：五项新增能力，全程使用合成数据与临时目录。"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qfm.analysis import compare_factors, compare_strategies, high_correlation_pairs  # noqa: E402
from qfm.data import DataCatalog  # noqa: E402
from qfm.factors import FACTORS, FACTOR_HISTORY, compute_factor, get_factor  # noqa: E402
from qfm.jobs import JobService  # noqa: E402
from qfm.multifactor import CompositeDefinition, CompositeRegistry  # noqa: E402
from qfm.portfolio import run_backtest, synthesize  # noqa: E402
from qfm.research.payload import build_strategy_run_payload  # noqa: E402
from qfm.research.store import ResearchStore  # noqa: E402
from scripts.verify_phase1 import Checker, synthetic_panel  # noqa: E402


def _save(store, project, payload, name):
    return store.save_run(
        project.id, name, payload["config"], payload["data_snapshot"], payload["summary"],
        payload["nav"], payload["benchmark_nav"], payload["weights"],
        payload["yearly_performance"], payload["trades"], payload.get("constraint_history"),
        code_version=payload["config"].get("code_version"),
        factor_definitions=payload.get("factor_definitions"), require_binding=True,
    )


def main(argv: list[str] | None = None) -> int:
    del argv
    checker = Checker()
    panel = synthetic_panel(stocks=160, days=300)
    factors_snapshot = dict(FACTORS)
    history_snapshot = {name: list(items) for name, items in FACTOR_HISTORY.items()}
    print("=" * 72)
    print("第二阶段端到端验证 · Job/Cache / Dataset/Universe / Compare / Multi-Factor")
    print("=" * 72)
    try:
        with tempfile.TemporaryDirectory(prefix="qfm-phase2-") as temp:
            root = Path(temp)

            print("\n【1/5】Job/Cache")
            service = JobService(root / "jobs")
            request = {
                "factor_versions": [{"name": "mom_20", "version": get_factor("mom_20").version}],
                "dataset_version": "dataset_verify", "universe": "cn_hs300",
                "universe_version": "universe_verify", "date_range": ["2024-01-01", "2024-12-31"],
                "pipeline_config": {}, "backtest_config": {},
            }
            calls: list[int] = []
            first = service.run("FACTOR_COMPUTE", request, lambda: calls.append(1) or {"ok": True})
            second = service.run("FACTOR_COMPUTE", request, lambda: calls.append(2) or {"ok": False})
            checker.check("Job/Cache 相同请求只计算一次", calls == [1])
            checker.check("Job/Cache 命中被记录", not first.job.cache_hit and second.job.cache_hit)

            print("\n【2/5】Dataset/Universe")
            catalog = DataCatalog(root / "catalog")
            custom = catalog.create_custom_universe("验证池", ["1", "600000", "000001"])
            checker.check("Dataset/Universe 自定义代码规范化",
                          custom.symbols == ("000001", "600000"))

            names = ["ep_ttm", "roe"]
            score, weights = synthesize(panel, names, mode="equal", horizon=20)
            backtest = run_backtest(panel, score, top_n=20, start=str(panel.close.index[60].date()))
            payload = build_strategy_run_payload(
                panel=panel, pool="cn_hs300", names=names, mode="equal", horizon=20,
                weight_lookback=126, orthogonalize=False, ortho_controls=(), top_n=20,
                start_date=str(panel.close.index[60].date()), rebalance="ME", bench_mode="equal",
                costs={"commission": 0.0003, "stamp": 0.0005, "impact": 0.001},
                max_participation=0.05, initial_capital=1_000_000,
                weights=weights, backtest=backtest, catalog_root=root / "catalog",
            )
            checker.check("Dataset/Universe 新实验绑定四字段",
                          all(payload["data_snapshot"].get(key) for key in
                              ("dataset_id", "dataset_version", "universe_id", "universe_version")))

            print("\n【3/5】Factor Compare")
            compared = compare_factors(panel, ["mom_20", "rev_20", "ep_ttm"], horizon=20)
            checker.check("Factor Compare 含八项指标", {
                "ic", "rank_ic", "icir", "long_short_return", "turnover", "coverage", "stability"
            } <= set(compared.metrics.columns))
            checker.check("Factor Compare 生成两套相关矩阵",
                          compared.pearson_corr.shape == (3, 3) and compared.spearman_corr.shape == (3, 3))
            checker.check("Factor Compare 能识别高相关因子",
                          not high_correlation_pairs(compared.spearman_corr, 0.7).empty)

            print("\n【4/5】Multi-Factor")
            for mode in ("equal", "ic", "icir", "ic_x_ir"):
                composite, latest = synthesize(panel, names, mode=mode, horizon=20, weight_lookback=126)
                checker.check(f"Multi-Factor {mode}",
                              composite.notna().any().any() and abs(sum(latest.values()) - 1) < 1e-9)
            registry = CompositeRegistry(root / "factors")
            saved = registry.save(CompositeDefinition.create("phase2_verify_combo", names, mode="icir"))
            warnings = registry.register_all()
            checker.check("Multi-Factor 综合因子可持久化重载",
                          not warnings and get_factor(saved.name) is not None
                          and compute_factor(saved.name, panel).notna().any().any())

            print("\n【5/5】Strategy Compare")
            store = ResearchStore(root / "research")
            project_a = store.create_project("验证 A")
            project_b = store.create_project("验证 B")
            run_a = _save(store, project_a, payload, "等权")
            score_b, weights_b = synthesize(panel, names, mode="ic", horizon=20, weight_lookback=126)
            backtest_b = run_backtest(panel, score_b, top_n=15, start=str(panel.close.index[60].date()))
            payload_b = build_strategy_run_payload(
                panel=panel, pool="cn_hs300", names=names, mode="ic", horizon=20,
                weight_lookback=126, orthogonalize=False, ortho_controls=(), top_n=15,
                start_date=str(panel.close.index[60].date()), rebalance="ME", bench_mode="equal",
                costs={"commission": 0.0003, "stamp": 0.0005, "impact": 0.001},
                max_participation=0.05, initial_capital=1_000_000,
                weights=weights_b, backtest=backtest_b, catalog_root=root / "catalog",
            )
            run_b = _save(store, project_b, payload_b, "IC 加权")
            strategies = compare_strategies([store.load_run(run_a.id), store.load_run(run_b.id)])
            checker.check("Strategy Compare 指标与四类曲线齐全",
                          len(strategies.metrics) == 2 and not strategies.nav.empty
                          and not strategies.benchmark.empty and not strategies.excess.empty
                          and not strategies.drawdown.empty)
    finally:
        FACTORS.clear()
        FACTORS.update(factors_snapshot)
        FACTOR_HISTORY.clear()
        FACTOR_HISTORY.update(history_snapshot)

    print("\n" + "=" * 72)
    print(f"验证完成：{len(checker.passed)} 项通过，{len(checker.failed)} 项失败")
    if checker.failed:
        print("失败项：" + "、".join(checker.failed))
        return 1
    print("0 项失败")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
