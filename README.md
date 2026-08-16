# 量化因子挖掘流水线 (quant-factor-mine)

A 股多因子研究与挖掘工具：内置 30+ 经典因子，支持单因子检验、自定义因子、自动挖掘，附深色金融终端风 Streamlit 操作台。

## 快速开始

```bash
# 1. 启动页面（首次会自动拉取数据，约 5-10 分钟，之后秒级）
streamlit run app.py

# 2. 命令行跑单因子检验
python -m qfm.cli --factor momentum_20 --horizon 20

# 3. 命令行跑自动挖掘
python -m qfm.cli --mine --max-candidates 200
```

## 架构

```
app.py              Streamlit 操作台（深色金融终端风）
qfm/
├── data/           数据层：akshare 拉取 + parquet 本地缓存
│   ├── loader.py   日线行情 / 财务指标 / 股票池
│   └── universe.py 股票池过滤（默认 沪深300+中证500）
├── factors/        因子库：注册表 + 5 大家族 30+ 因子
│   ├── base.py     @register_factor 装饰器（一行注册一个因子）
│   ├── value.py    价值族：EP / BP / SP / 股息率
│   ├── quality.py  质量族：ROE / 毛利率 / 资产负债率
│   ├── momentum.py 动量反转族：5/20/60/120 日
│   ├── volatility.py 波动族：20/60日波动 / 下行波动 / 偏度
│   ├── liquidity.py  流动性族：换手率 / Amihud / 成交额
│   ├── size.py     规模族：总市值 / 流通市值
│   └── growth.py   成长族：营收 / 利润同比增速
├── pipeline/       检验流水线（自研，不依赖 alphalens）
│   ├── clean.py    去极值(MAD/分位数) + z-score
│   ├── tests.py    IC / IC_IR / 分层回测 / 单调性 / 换手率
│   └── report.py   HTML 检验报告（plotly 自包含页面）
├── mining/         自动挖掘引擎
│   └── engine.py   基础指标×窗口×变换 → 批量生成 → 批量检验 → TOP 排行
└── cli.py          命令行入口
```

## 自定义因子（页面 / 代码两入口）

```python
from qfm.factors import register_factor

@register_factor(name="my_idea", family="自定义", description="我的主观想法")
def my_idea(d: DataPanel) -> pd.DataFrame:
    # d.close / d.volume / d.turnover 是 日期×股票 的透视表
    return d.close.pct_change(20) / d.turnover.rolling(20).mean()
```

## 数据说明

- 默认股票池：沪深300 + 中证500（约 800 只，首次拉取 5-10 分钟）
- 全市场可选（约 5400 只，首次约 30-60 分钟，需耐心）
- 数据缓存在 `data_cache/`，二次加载秒级；日线为前复权
- 财务指标按"公告日期"对齐，避免未来函数

## 方法论出处（QuantSkills 三技能融入）

| 功能 | 方法论来源 | 独立实现依据 |
|---|---|---|
| 逐日截面 OLS 正交化 | QuantSkills `skill-factor-orthogonalize`（GPL-3.0） | 公开标准截面回归；行业/市值/风格暴露剥离 |
| 回测过拟合检验 DSR/PBO/Haircut/MinTRL | QuantSkills `skill-backtest-overfit`（GPL-3.0） | Bailey & López de Prado (2012/2014)、Bailey et al. (2017)、Harvey & Liu (2015) 文献公式 |
| 无未来函数检查 | QuantSkills `skill-quant-factor-skill-factory`（GPL-3.0） | 静态泄漏模式扫描 + 挖掘候选结构性判定 |

本项目未拷贝上述仓库源码，算法按文献公式独立实现。

## 机器学习合成因子

因子检验页「ML 合成」Tab：LightGBM walk-forward 滚动训练把现有因子库非线性合成为一个新因子（仅样本外预测，结构性无未来函数）。合成后自动注册为 `ml_synth` 因子，可直接进入策略回测页参与合成与回测。
