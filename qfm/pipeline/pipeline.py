"""统一因子管线：全项目唯一的「原始因子 → 交易信号」路径。

设计目标（解决问题）：历史上存在三套并行的信号处理实现——`pipeline.clean.clean_factor`
（检验用）、`simulation.engine.prepare_signal` 的内联 OLS（回测用）、`orthogonalize`（正交化页），
彼此不复用，导致**看到的 IC 与实际持仓来自不同的信号对象**。本模块用一个显式的阶段
序列产出唯一的 `SignalResult.signal`，检验、回测、Tear Sheet、实验归档全部只消费它。

阶段顺序（固定）：

    缺失处理 → 去极值 → 方向 → 中性化 → 标准化 → Decay 平滑 → Lag → Signal

对齐约定：原始因子先按 panel.close 的日期/股票轴重索引，之后所有阶段都在同一
date×stock 网格上运算，不做隐式位置对齐。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

import numpy as np
import pandas as pd

from qfm.pipeline import stages

# 中性化控制变量（对应 qfm.orthogonalize.controls_from_panel 的 controls）
NEUTRALIZE_CONTROLS = ("industry", "size", "style")


@dataclass(frozen=True)
class PipelineConfig:
    """管线配置。默认值等价于历史 `clean_factor`（MAD 去极值 → z-score），不引入额外变换。"""

    missing: str = "none"                 # none | row_drop | cs_median
    winsorize: str = "mad"                # none | mad | quantile
    winsorize_n: float = 3.0
    neutralize: tuple[str, ...] = ()      # () 或 ("industry",) / ("size",) / ("industry","size") / ("style",)
    neutralize_min_samples: int = 30
    standardize: str = "zscore"           # none | zscore | rank
    decay: int = 0                        # 0/1 不平滑
    lag: int = 1                          # 1 = T 日信号 T+1 开盘执行（引擎已提供，故不作位移）
    min_stocks: int = 0                   # >0 时剔除截面过薄的交易日

    def validate(self) -> None:
        if self.missing not in stages.MISSING_METHODS:
            raise ValueError(f"未知缺失处理方式: {self.missing}")
        if self.winsorize not in stages.WINSORIZE_METHODS:
            raise ValueError(f"未知去极值方式: {self.winsorize}")
        if self.standardize not in stages.STANDARDIZE_METHODS:
            raise ValueError(f"未知标准化方式: {self.standardize}")
        unknown = [c for c in self.neutralize if c not in NEUTRALIZE_CONTROLS]
        if unknown:
            raise ValueError(f"未知中性化控制变量: {unknown}")
        if self.winsorize_n <= 0:
            raise ValueError("去极值倍数必须为正数")
        if self.neutralize_min_samples < 3:
            raise ValueError("中性化最少样本数不得小于 3")
        if not 0 <= self.decay <= 120:
            raise ValueError("Decay 必须在 0 到 120 之间")
        if not 1 <= self.lag <= 60:
            raise ValueError("Lag 必须在 1 到 60 之间")
        if self.min_stocks < 0:
            raise ValueError("截面最少股票数不能为负")

    def to_dict(self) -> dict:
        data = asdict(self)
        data["neutralize"] = list(self.neutralize)
        return data

    @classmethod
    def from_settings(cls, settings) -> "PipelineConfig":
        """从 SimulationSettings 构造（保持既有默认行为不变）。"""
        neutral = () if settings.neutralization == "none" else tuple(settings.neutralization.split("_"))
        return cls(
            neutralize=neutral,
            decay=settings.decay,
            lag=settings.delay,
        )

    def with_(self, **changes) -> "PipelineConfig":
        return replace(self, **changes)


@dataclass
class SignalResult:
    """管线输出：唯一信号 + 逐阶段留痕（供 Tear Sheet、实验归档与排错）。"""

    signal: pd.DataFrame
    config: PipelineConfig
    direction_sign: int = 1
    stages: dict[str, pd.DataFrame] = field(default_factory=dict)
    coverage: dict[str, float] = field(default_factory=dict)
    neutralize_diag: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def signal_coverage(self) -> float:
        return self.coverage.get("signal", float("nan"))

    def stage_frames(self) -> dict[str, pd.DataFrame]:
        return dict(self.stages)


def run_pipeline(raw: pd.DataFrame, panel=None, config: PipelineConfig | None = None,
                 direction: str = "positive", close: pd.DataFrame | None = None) -> SignalResult:
    """把原始因子跑成交易信号。

    raw     : date×stock 原始因子值（可只覆盖部分日期/股票，会重索引到目标网格）
    panel   : DataPanel，提供中性化所需的行业/市值/风格控制变量；也可只用于确定价格网格
    config  : PipelineConfig，默认等价于 clean_factor(MAD 3.0 → zscore)
    direction: 因子方向；输出恒为"值越大预期收益越高"
    close   : 无 panel 时（例如只做 IC 检验）用它确定日期/股票轴
    """
    config = config or PipelineConfig()
    config.validate()

    grid = getattr(panel, "close", None) if panel is not None else None
    if not isinstance(grid, pd.DataFrame) or grid.empty:
        grid = close
    if not isinstance(grid, pd.DataFrame) or grid.empty:
        raise ValueError("必须提供 panel 或 close，用于确定信号的日期/股票轴")
    if config.neutralize and panel is None:
        raise ValueError(f"配置要求中性化 {config.neutralize}，但未提供 panel（无法构造控制变量）")

    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise ValueError("原始因子为空，无法构建信号")
    # 唯一的对齐点：缺失的日期/股票补 NaN，多余的轴丢弃
    if not raw.index.equals(grid.index) or not raw.columns.equals(grid.columns):
        raw = raw.reindex(index=grid.index, columns=grid.columns)
    frame = raw.replace([float("inf"), float("-inf")], np.nan).astype(float)

    result = SignalResult(signal=frame.copy(), config=config)
    result.stages["raw"] = frame.copy()
    result.coverage["raw"] = stages.coverage_of(frame)

    frame = stages.apply_missing(frame, config.missing, config.neutralize_min_samples)
    result.stages["missing"] = frame.copy()
    result.coverage["missing"] = stages.coverage_of(frame)

    frame = stages.apply_winsorize(frame, config.winsorize, config.winsorize_n)
    result.stages["winsorize"] = frame.copy()
    result.coverage["winsorize"] = stages.coverage_of(frame)

    frame, diag = stages.apply_neutralize(
        frame, panel, config.neutralize, config.neutralize_min_samples
    )
    result.neutralize_diag = diag
    result.stages["neutralize"] = frame.copy()
    result.coverage["neutralize"] = stages.coverage_of(frame)
    if config.neutralize and diag.get("days_skipped", 0):
        result.warnings.append(
            f"中性化有 {diag['days_skipped']} 个交易日因截面样本不足被跳过，这些日期信号为空。"
        )

    frame = stages.apply_standardize(frame, config.standardize)
    result.stages["standardize"] = frame.copy()
    result.coverage["standardize"] = stages.coverage_of(frame)

    # 方向统一放在中性化/标准化之后：这两个阶段对符号不敏感（OLS 残差与 z-score
    # 都只是整体变号），因此放在前面或后面得到完全相同的最终信号，而放在这里可以
    # 保留"标准化后、翻向前"的中间帧，与历史报告口径一致。
    frame, sign = stages.apply_direction(frame, direction)
    result.direction_sign = sign
    result.stages["oriented"] = frame.copy()
    result.coverage["oriented"] = stages.coverage_of(frame)

    frame = stages.apply_decay(frame, config.decay)
    result.stages["decay"] = frame.copy()
    result.coverage["decay"] = stages.coverage_of(frame)

    frame = stages.apply_lag(frame, config.lag)
    result.stages["lag"] = frame.copy()
    result.coverage["lag"] = stages.coverage_of(frame)

    frame = stages.finalize_signal(frame, config.min_stocks)
    result.signal = frame
    result.stages["signal"] = frame.copy()
    result.coverage["signal"] = stages.coverage_of(frame)

    if not frame.notna().any().any():
        result.warnings.append("该区间没有有效信号值，请检查财务覆盖、Decay 窗口或中性化设置。")
    return result
