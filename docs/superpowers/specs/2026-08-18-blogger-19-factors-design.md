# 博主「每天一个因子」19 因子接入设计文档

日期：2026-08-18
状态：已获用户批准

## 1. 目标

将博主公开整理的 19 个因子接入因子库。盘点后：**10 个新增实现**，**9 个跳过**（与现库重复或缺数据源）。

## 2. 盘点结论

| 处置 | 因子 | 说明 |
|---|---|---|
| ✅ 新增 | bb_break_20 / amplitude_3 / turnover_heat / rav_4 / gm_yoy / sentiment_20 / alpha144_191 / rev_5 / mom_250 / vol_ratio_20 | 见 3.1 |
| ⏭️ 跳过 | 20日动量=现有 mom_20；PIV=现有 bp；PITTM=现有 ep_ttm；20日波动=现有 vol_20；低波动=vol_20(negative)；规模=现有 ln_mv_float；Amihud=现有 amihud_20（成交额口径）；多空组合=分层检验"多空价差"指标；大单净流入=缺 Level2 数据源（东财资金流被环境拦截，二期） | README 注明 |

## 3. 实现

### 3.1 核心改动：DataPanel 增加 open/high/low

`qfm/data/panel.py`：DataPanel 新增 `open/high/low` 三个 DataFrame 字段（默认空）；`build_panel` 补透视（bars 缓存本就含 OHLC 列）。向后兼容（backtest 等现有代码只用 close）。`tests/conftest.py` 合成面板补 OHLC + fund 补 gross_margin。

### 3.2 新文件 qfm/factors/blogger.py（10 因子，@register_factor）

- `bb_break_20`（动量反转, positive）：close > MA20+2σ → 1/0（个股版；全市场宽度=按列求均值）
- `amplitude_3`（波动, negative）：(high-low)/前收
- `turnover_heat`（流动性, positive）：换手 20 日均值 / 250 日均值
- `rav_4`（动量反转, positive）：(RSI4 - RSI4[-4])/RSI4[-4]；RSI=简单均值口径
- `gm_yoy`（成长, positive）：gross_margin / gross_margin.shift(250) - 1（公告对齐后近似同比，描述注明）
- `sentiment_20`（流动性, positive）：(换手/换手[-19]) × (收盘/收盘[-19])
- `alpha144_191`（流动性, positive）：20 日下跌日 |ret|/volume 累计
- `rev_5`（动量反转, positive）：close[-5]/close - 1（= -mom_5）
- `mom_250`（动量反转, positive）：close/close[-250] - 1
- `vol_ratio_20`（流动性, positive）：volume/volume[-19]

### 3.3 测试 tests/test_blogger_factors.py

- 全部 10 因子在合成面板可计算：形状==close 形状、含有限值、NaN 占比合理（250 日窗口因子 warmup 后有效）
- 定点校验：rev_5 ≈ -mom_5；amplitude_3 = (high-low)/前收；bb_break_20 ∈ {0,1}
- 无未来函数：scan_source 结构性 pass（滚动/shift 均为正窗口）

### 3.4 页面接入

注册后自动出现在因子库/检验/策略回测，零页面改动。

## 4. 验收

1. pytest 全绿（既有 36 + 新增）
2. 真实缓存数据跑 5 个新因子 IC 冒烟（bb_break_20 / sentiment_20 / rav_4 / gm_yoy / alpha144_191）
3. streamlit 健康；README/交接 更新（来源注明 + 跳过清单）
