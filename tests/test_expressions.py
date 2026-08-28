"""表达式挖掘 v1（WQ 方法论吸收）：操作符求值 / 无未来函数 / 字段扩充 / Fitness / 去重 / 集成"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qfm.mining.expressions import (OPS, EXPR_OPS, expr_candidate_total, generate_candidates_expr,
                                    parse_candidate_name, apply_op)
from qfm.mining.fields import tech_fields
from qfm.mining.engine import run_mining_expr, fitness_score, dedup_candidates


# ── 小合成面板（手算对拍用）──────────────────────────────────────
@pytest.fixture()
def small():
    idx = pd.bdate_range("2024-01-02", periods=8)
    cols = ["a", "b", "c"]
    s = pd.DataFrame([[1.0, 4.0, 7.0],
                      [2.0, 3.0, 8.0],
                      [3.0, 2.0, 6.0],
                      [4.0, 5.0, 5.0],
                      [5.0, 6.0, 4.0],
                      [6.0, 1.0, 3.0],
                      [7.0, 8.0, 2.0],
                      [8.0, 7.0, 1.0]], index=idx, columns=cols)
    return s


# ── 操作符求值对拍 ───────────────────────────────────────────────
class TestOps:
    def test_rank_cross_section(self, small):
        out = apply_op("rank", small, None)
        assert out.iloc[0].to_dict() == {"a": 1.0, "b": 2.0, "c": 3.0}

    def test_ts_mean(self, small):
        out = apply_op("ts_mean", small, 3)
        assert out.iloc[2].to_dict() == pytest.approx({"a": 2.0, "b": 3.0, "c": 7.0})
        assert out.iloc[0].isna().all()

    def test_delta(self, small):
        out = apply_op("delta", small, 2)
        assert out.iloc[2].to_dict() == {"a": 2.0, "b": -2.0, "c": -1.0}
        assert out.iloc[1].isna().all()

    def test_abs(self, small):
        neg = -small.copy()
        out = apply_op("abs", neg, None)
        assert (out == small).all().all()

    def test_ts_std(self, small):
        out = apply_op("ts_std", small, 3)
        assert out.iloc[2].to_dict() == pytest.approx(
            {"a": np.std([1, 2, 3], ddof=1), "b": np.std([4, 3, 2], ddof=1),
             "c": np.std([7, 8, 6], ddof=1)})

    def test_ts_zscore(self, small):
        out = apply_op("ts_zscore", small, 3)
        row = out.iloc[2]
        assert row["a"] == pytest.approx((3 - 2) / np.std([1, 2, 3], ddof=1))
        assert row["c"] == pytest.approx((6 - 7) / np.std([7, 8, 6], ddof=1))

    def test_ts_rank(self, small):
        out = apply_op("ts_rank", small, 3)
        # a 列前 3 个值 1,2,3 → 最新值 3 是窗口内最大 → pct rank=3/3=1.0
        assert out.iloc[2]["a"] == pytest.approx(1.0)
        # c 列前 3 个值 7,8,6 → 最新值 6 是窗口内最小 → pct rank=1/3
        assert out.iloc[2]["c"] == pytest.approx(1 / 3)

    def test_decay_linear(self, small):
        out = apply_op("decay_linear", small, 3)
        # a: (1*1 + 2*2 + 3*3) / 6 = 14/6
        assert out.iloc[2]["a"] == pytest.approx(14 / 6)
        # c: (7*1 + 8*2 + 6*3) / 6 = 41/6
        assert out.iloc[2]["c"] == pytest.approx(41 / 6)

    def test_unknown_op(self, small):
        with pytest.raises(KeyError):
            apply_op("not_an_op", small, None)


# ── 无未来函数：篡改未来行，当前行输出必须不变 ───────────────────
class TestNoLookahead:
    @pytest.mark.parametrize("op", EXPR_OPS)
    def test_future_values_do_not_leak(self, small, op):
        n = {"ts_rank": 3, "ts_mean": 3, "ts_zscore": 3, "ts_std": 3,
             "decay_linear": 3, "delta": 2}.get(op)
        base = apply_op(op, small, n)
        poisoned = small.copy()
        poisoned.iloc[-2:] = np.nan * 0 + 999.0  # 篡改未来两行
        out = apply_op(op, poisoned, n)
        for dt in small.index[:-2]:
            pd.testing.assert_series_equal(out.loc[dt], base.loc[dt],
                                           check_names=False, rtol=1e-9)


# ── 候选生成器 ───────────────────────────────────────────────────
class TestGenerator:
    def test_candidate_naming(self, panel):
        names = []
        for name, _ in generate_candidates_expr(
                panel, fields=["mom_20"], ops=["rank", "ts_mean"], n_list=[5]):
            names.append(name)
            if len(names) >= 4:
                break
        assert "rank__mom_20" in names
        assert "ts_mean_5__mom_20" in names

    def test_candidate_total(self):
        n = expr_candidate_total(["mom_20", "bp"],
                                 ops=["rank", "abs", "ts_mean"], n_list=[5])
        # rank/abs 无参各 2 个 + ts_mean_5 2 个 = 6
        assert n == 6

    def test_generator_streams_shape(self, panel):
        gen = generate_candidates_expr(panel, fields=["mom_20"], ops=["rank"], n_list=[])
        name, fdf = next(gen)
        assert name == "rank__mom_20"
        assert fdf.shape == panel.close.shape
        assert fdf.index.equals(panel.close.index)

    def test_parse_candidate_name(self):
        assert parse_candidate_name("ts_mean_5__mom_20") == ("ts_mean", 5, "mom_20")
        assert parse_candidate_name("rank__mom_20") == ("rank", None, "mom_20")


# ── 技术指标字段扩充 ─────────────────────────────────────────────
class TestTechFields:
    def test_fields_available(self, panel):
        f = tech_fields(panel)
        assert len(f) >= 10
        assert "rsi_14" in f and "bias_5" in f and "vol_ratio_5" in f

    def test_rsi_hand_calc(self):
        # 15 天 → 14 个日变化：+1×10, -1, -1, +3, +1 → gains 均值 1.0，losses 均值 2/14
        idx = pd.bdate_range("2024-01-02", periods=15)
        close = pd.DataFrame({"a": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                    19, 18, 21, 22]}, index=idx)
        p = type("P", (), {"close": close})()
        out = tech_fields(p)["rsi_14"]
        expect = 100 - 100 / (1 + 1.0 / (2 / 14))   # RS = 1.0 / (2/14) = 7
        assert out.iloc[-1]["a"] == pytest.approx(expect)

    def test_bias(self, panel):
        f = tech_fields(panel)
        ma5 = panel.close.rolling(5).mean()
        expect = (panel.close - ma5) / ma5
        pd.testing.assert_frame_equal(f["bias_5"], expect)

    def test_vol_ratio(self, panel):
        f = tech_fields(panel)
        expect = panel.volume / panel.volume.rolling(5).mean()
        pd.testing.assert_frame_equal(f["vol_ratio_5"], expect)

    def test_close_pos(self, panel):
        f = tech_fields(panel)
        lo = panel.low.rolling(20).min()
        hi = panel.high.rolling(20).max()
        expect = (panel.close - lo) / (hi - lo)
        pd.testing.assert_frame_equal(f["close_pos_20"], expect)

    @pytest.mark.parametrize("field", ["rsi_14", "atr_14", "amp_ratio", "money_flow"])
    def test_tech_fields_no_lookahead(self, panel, field):
        f = tech_fields(panel)
        base = f[field]
        p2 = type("P", (), {k: v.copy() for k, v in panel.__dict__.items()})()
        for k in ("close", "open", "high", "low", "volume", "amount", "turnover"):
            df = getattr(p2, k)
            df.iloc[-2:] = 999.0
        out = tech_fields(p2)[field]
        for dt in base.index[:-2]:
            pd.testing.assert_series_equal(out.loc[dt], base.loc[dt],
                                           check_names=False, rtol=1e-6)


# ── Fitness ──────────────────────────────────────────────────────
class TestFitness:
    def test_fitness_hand_calc(self):
        rng = np.random.default_rng(1)
        rets = pd.Series(rng.normal(0.01, 0.05, 36))
        turnover = 0.3
        sr = rets.mean() / rets.std() * np.sqrt(12)
        ann = (1 + rets.mean()) ** 12 - 1
        expect = sr * np.sqrt(abs(ann) / max(turnover, 0.125))
        assert fitness_score(rets, turnover) == pytest.approx(expect)

    def test_fitness_turnover_floor(self):
        rets = pd.Series(np.full(12, 0.01))
        # 换手极低时按 0.125 下限
        assert fitness_score(rets, 0.001) == pytest.approx(
            fitness_score(rets, 0.125))


# ── 相关性去重 ──────────────────────────────────────────────────
class TestDedup:
    def test_greedy_dedup(self):
        rng = np.random.default_rng(2)
        n = 24
        x = rng.normal(size=n)
        a = pd.Series(x + 0.1 * rng.normal(size=n))
        b = pd.Series(x + 0.1 * rng.normal(size=n))   # 与 a 高相关
        c = pd.Series(rng.normal(size=n))              # 独立
        m = pd.DataFrame({"A": a, "B": b, "C": c})
        scores = pd.Series({"A": 1.0, "B": 0.9, "C": 0.8})
        keep, groups = dedup_candidates(m, scores, corr_th=0.7)
        assert set(keep) == {"A", "C"}                 # B 与 A 相关被剔除
        assert groups["B"] == groups["A"]              # 同组标注


# ── 集成：run_mining_expr ───────────────────────────────────────
class TestIntegration:
    def test_run_mining_expr_leaderboard(self, panel):
        lb, meta = run_mining_expr(panel, horizon=5, max_candidates=20,
                                   fields=["mom_20", "bp", "rsi_14"],
                                   ops=["rank", "abs"], n_list=[],
                                   save_trials=False)
        assert len(lb) <= 6                             # 3 字段 × 2 无参操作符
        assert "Fitness" in lb.columns and "去重组" in lb.columns
        assert {"因子", "IC", "t值", "变换"}.issubset(lb.columns)

    def test_run_mining_expr_with_trials(self, panel):
        lb, meta = run_mining_expr(panel, horizon=5, max_candidates=10,
                                   fields=["mom_20"], ops=["rank", "abs"],
                                   n_list=[], save_trials=True)
        assert meta is not None and "n_trials" in meta
        assert meta["n_trials"] == len(lb)
