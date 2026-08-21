"""全量 ML 合成演示：799 只 × 全部因子特征 → LightGBM walk-forward → IC 报告

用法：
  python3 scripts/ml_synth_full.py --smoke     # 小样本冒烟（5 因子 × 300 天 × 2 fold）
  python3 scripts/ml_synth_full.py             # 全量（49 因子 × 8700 日 × 4 fold）

内存优化：特征/目标转 float32（长表 ~2.7GB → ~1.4GB），LightGBM 直接吃 float32。
"""

import os
import socket
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

ROOT = Path("/Users/zxt/.zcode/workspace/default/quant-factor-mine")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from qfm.data import DataLoader, build_panel, get_universe
from qfm.ml_synthesizer import walk_forward_train
from qfm.pipeline.tests import factor_report
from qfm.factors import list_factors, ZH_NAMES

socket.setdefaulttimeout(20)

SMOKE = "--smoke" in sys.argv
HORIZON = 20
CUTOFF = "2023-12-31"
FOLDS = 2 if SMOKE else 4

t0 = time.time()
say = lambda m: print(m, flush=True)  # noqa: E731

# ── 数据 ─────────────────────────────────────────────────────────
dl = DataLoader(cache_dir=str(ROOT / "data_cache"))
codes = get_universe("index800")
cached = [c for c in codes if (ROOT / "data_cache" / "bars" / f"{c}.parquet").exists()]
bars = dl.load_bars(cached)
ind = dl.load_indicators()
panel = build_panel(bars, ind)
say(f"[data] 面板 {panel.close.shape[0]} 日 × {panel.close.shape[1]} 只（{time.time()-t0:.0f}s）")

if SMOKE:
    names = [f.name for f in list_factors()][:5]
    CUTOFF = "2024-06-30"
    say(f"[smoke] 5 因子 × {FOLDS} fold，cutoff {CUTOFF}（不切日期，验证全链路）")
else:
    names = [f.name for f in list_factors()]
    say(f"[full] {len(names)} 因子 × 4 fold，目标前瞻 {HORIZON} 日，cutoff {CUTOFF}")

# ── 训练 ─────────────────────────────────────────────────────────
t1 = time.time()
try:
    pred, meta = walk_forward_train(
        panel, names, horizon=HORIZON, train_cutoff=CUTOFF, n_folds=FOLDS,
        progress=lambda i, n, msg: say(f"[fold] {i+1}/{n} {msg}"))
except Exception as e:  # noqa: BLE001
    say(f"[FAIL] {type(e).__name__}: {e}")
    sys.exit(1)
say(f"[train] 完成 {time.time()-t1:.0f}s，样本外 {len(pred)} 日 × {pred.shape[1]} 只")

# ── 检验 ─────────────────────────────────────────────────────────
rep = factor_report(pred, panel.close, horizon=HORIZON, direction="positive")
s = rep["ic_summary"]
say(f"[IC] OOS IC={s['ic_mean']:+.4f}  IC_IR={s['ic_ir']:.2f}  t={s['ic_t']:.1f}  "
    f"正占比={s['pos_ratio']:.2f}  有效天数={s['n_days']}")
say(f"[turnover] 换手率={rep['turnover']:.3f}")

imp = meta["importance_top"]
say("\n[特征重要性 TOP10]")
for k, v in imp.items():
    label = ZH_NAMES.get(k.replace("f_", ""), k.replace("f_", ""))
    say(f"  {label:　<8} {v:.1f}")

# ── 落盘 ─────────────────────────────────────────────────────────
out = ROOT / "data_cache" / f"ml_pred_full_{'smoke' if SMOKE else 'all'}.parquet"
pred.to_parquet(out)
say(f"\n📄 预测已存: {out}")
say(f"⏱ 总耗时 {time.time()-t0:.0f}s")
