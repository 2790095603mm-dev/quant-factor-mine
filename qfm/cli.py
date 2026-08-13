"""命令行入口：python -m qfm.cli

用法：
  python -m qfm.cli --factor mom_20 --horizon 20 [--pool index800] [--max-stocks 100]
  python -m qfm.cli --mine --max-candidates 200
"""

from __future__ import annotations

import argparse
import time

from qfm.data import DataLoader, build_panel, get_universe
from qfm.factors import compute_factor, get_factor
from qfm.mining import run_mining
from qfm.pipeline import factor_report, generate_report


def _load_panel(pool: str, max_stocks: int | None):
    t0 = time.time()
    print(f"▶ 加载股票池: {pool} ...")
    codes = get_universe(pool)
    if max_stocks:
        codes = codes[:max_stocks]
    print(f"▶ 拉取 {len(codes)} 只股票日线 ...")
    dl = DataLoader()
    bars = dl.load_bars(codes)
    ind = dl.load_indicators()
    panel = build_panel(bars, ind)
    print(f"▶ 数据就绪: 面板 {panel.close.shape}, 耗时 {time.time()-t0:.0f}s")
    return panel


def cmd_factor(args):
    f = get_factor(args.factor)
    if f is None:
        raise SystemExit(f"未知因子: {args.factor}（可用: python -m qfm.cli --list）")
    panel = _load_panel(args.pool, args.max_stocks)
    t0 = time.time()
    rep = factor_report(compute_factor(args.factor, panel), panel.close,
                        horizon=args.horizon, direction=f.direction)
    s = rep["ic_summary"]
    print(f"✅ {args.factor}: IC={s['ic_mean']:+.4f} IC_IR={s['ic_ir']:.2f} "
          f"t={s['ic_t']:.2f} 正占比={s['pos_ratio']:.0%} 换手={rep['turnover']:.0%}")
    path = generate_report(rep, args.factor, f.family, f.description, f.direction,
                           f"reports/{args.factor}.html")
    print(f"✅ 报告: {path} (耗时 {time.time()-t0:.0f}s)")


def cmd_mine(args):
    panel = _load_panel(args.pool, args.max_stocks)
    t0 = time.time()

    def progress(i, n, name):
        if (i + 1) % 10 == 0 or i == n - 1:
            print(f"  ⏳ 挖掘中 {i+1}/{n} ({name})")

    df = run_mining(panel, horizon=args.horizon, max_candidates=args.max_candidates,
                    progress=progress)
    print(f"\n🏆 TOP 10 因子排行榜 (前瞻 {args.horizon} 日, 总耗时 {time.time()-t0:.0f}s):")
    print(df.head(10).to_string(index=False))


def cmd_list(_args):
    from qfm.factors import list_factors
    from collections import Counter
    fs = list_factors()
    print(f"共 {len(fs)} 个因子:")
    for fam, group in Counter(f.family for f in fs).items():
        print(f"  {fam} ({group}): " + ", ".join(f.name for f in fs if f.family == fam))


def main():
    ap = argparse.ArgumentParser(description="量化因子挖掘流水线 CLI")
    ap.add_argument("--factor", help="单因子检验，如 mom_20")
    ap.add_argument("--mine", action="store_true", help="运行自动挖掘")
    ap.add_argument("--list", action="store_true", help="列出所有因子")
    ap.add_argument("--horizon", type=int, default=20, help="前瞻天数（默认 20）")
    ap.add_argument("--pool", default="index800", help="股票池: index800/full")
    ap.add_argument("--max-stocks", type=int, default=None, help="限制股票数（调试用）")
    ap.add_argument("--max-candidates", type=int, default=None, help="限制候选因子数")
    args = ap.parse_args()

    if args.list:
        cmd_list(args)
    elif args.factor:
        cmd_factor(args)
    elif args.mine:
        cmd_mine(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
