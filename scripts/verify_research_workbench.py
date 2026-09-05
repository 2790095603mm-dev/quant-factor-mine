"""零网络冒烟：研究账本可创建、写入并读取一条完整运行。"""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qfm.research import ResearchStore


def main() -> None:
    with TemporaryDirectory() as temporary:
        store = ResearchStore(Path(temporary) / "research")
        project = store.create_project("smoke")
        dates = pd.date_range("2026-01-01", periods=2, freq="D", name="date")
        nav = pd.Series([1.0, 1.01], index=dates)
        benchmark_nav = pd.Series([1.0, 1.005], index=dates)
        weights = pd.DataFrame({"weight": [1.0]}, index=pd.Index(["ep_ttm"], name="factor"))
        yearly = pd.DataFrame({"年份": [2026], "收益": [0.01]})
        trades = pd.DataFrame({"date": [dates[1]], "stock": ["600000"], "side": ["BUY"]})
        constraint_history = pd.DataFrame(
            {
                "signal_date": [dates[0]],
                "execution_date": [dates[1]],
                "target_cash": [0.0],
                "actual_cash": [0.25],
            }
        )
        data_snapshot = {
            "snapshot_version": 2,
            "pool": "index800",
            "stocks": 1,
            "trading_days": 2,
            "coverage": {"close": 1.0, "amount": 1.0, "fundamentals": {}},
            "fingerprints": {"market": "sha256:test", "fundamentals": "sha256:test"},
            "quality_warnings": [],
        }
        run = store.save_run(
            project.id,
            "smoke",
            {"top_n": 30},
            data_snapshot,
            {"夏普比率": 1.0},
            nav,
            benchmark_nav,
            weights,
            yearly,
            trades,
            constraint_history,
        )
        loaded = store.load_run(run.id)
        assert loaded.run.id == run.id
        assert loaded.run.data_snapshot["snapshot_version"] == 2
        assert loaded.trades.loc[0, "stock"] == "600000"
        assert loaded.constraint_history.loc[0, "actual_cash"] == 0.25
    print("research workbench smoke: ok")


if __name__ == "__main__":
    main()
