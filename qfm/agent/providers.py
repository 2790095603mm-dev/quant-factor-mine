"""数据提供层：把「真实本地缓存」和「离线合成面板」统一成一个接口。

Agent 的工具层不应该关心数据从哪来，但**报告必须说清数据从哪来**——
否则合成数据上的 IC 会被误读成实证结论。所以每个 provider 都会返回带
`source` 与 `note` 的 `PanelBundle`，这两个字段一路传到运行报告里。

三种模式（`resolve_provider`）：

- ``cache``     强制用本地 `data_cache/`，缺失即失败（保证结果来自真实数据）；
- ``synthetic`` 强制用合成面板（演示、CI、无网环境）；
- ``auto``      **默认**。优先本地缓存，不可用则回落到合成面板并在报告里显著标注。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol

import pandas as pd

from qfm.agent.errors import PermanentError
from qfm.agent.synthetic import SyntheticSpec, build_synthetic_panel

__all__ = [
    "PanelBundle",
    "DataProvider",
    "LocalCacheProvider",
    "SyntheticProvider",
    "resolve_provider",
    "DEFAULT_AS_OF",
]

#: 合成面板的默认结束日，与本地缓存的最晚日期接近，便于两种数据源互相印证。
DEFAULT_AS_OF = "2026-09-18"


@dataclass
class PanelBundle:
    """一份可直接用于因子计算的面板及其出处说明。"""

    panel: Any
    pool: str
    source: str            # "cache" | "synthetic"
    symbols: list[str]
    note: str = ""
    as_of: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_symbols(self) -> int:
        return len(self.symbols)

    @property
    def is_synthetic(self) -> bool:
        return self.source == "synthetic"

    def describe(self) -> str:
        return (
            f"股票池 {self.pool} · {self.n_symbols} 只 · "
            f"数据来源 {self.source} · 截至 {self.as_of}"
        )


class DataProvider(Protocol):
    """数据源接口。"""

    name: str

    def available(self) -> bool: ...

    def describe(self) -> str:
        """一句话说明数据来源，用于运行报告。"""
        ...

    def as_of(self) -> str:
        """数据可用区间的终点（YYYY-MM-DD）。

        「最近一年」这类相对区间必须先知道数据到哪天，才能落成具体日期，
        所以规划阶段需要这个值。实现必须是**廉价**的——不能为拿一个日期
        先加载整张面板。
        """
        ...

    def symbols(self, pool: str) -> list[str]:
        """只取股票池的代码列表，**不加载行情数据**。

        解析股票池时不该顺带把几百只股票的面板读进内存——实测在真实缓存上
        一次多余的面板加载要 19 秒，而代码列表是毫秒级的。
        """
        ...

    def load(self, pool: str, symbols: Iterable[str] | None = None) -> PanelBundle: ...

    def industry_map(self, as_of: str | None = None) -> pd.Series: ...

    def industry_labels(self) -> list[str]: ...


class LocalCacheProvider:
    """本地 parquet 缓存（真实数据）。走 `qfm.data` 的既有三段式加载。"""

    name = "cache"

    def __init__(self, cache_dir: Path | str = "data_cache") -> None:
        self.cache_dir = Path(cache_dir)
        self._as_of: str | None = None

    def available(self) -> bool:
        bars = self.cache_dir / "bars"
        if not bars.is_dir():
            return False
        return any(bars.glob("*.parquet"))

    def describe(self) -> str:
        return f"本地 parquet 缓存（{self.cache_dir}）"

    def symbols(self, pool: str) -> list[str]:
        from qfm.data import get_universe

        return [str(code) for code in get_universe(pool)]

    def as_of(self) -> str:
        """数据最晚交易日。

        只读 parquet 的 `date` 列，不载入价格数据。结果在实例内缓存，
        避免规划阶段与执行阶段重复扫描。
        """
        if self._as_of is not None:
            return self._as_of
        frames = sorted((self.cache_dir / "bars").glob("*.parquet"))
        if not frames:
            raise PermanentError(f"{self.cache_dir}/bars 下没有缓存文件")
        latest = None
        for path in frames:
            try:
                column = pd.read_parquet(path, columns=["date"])["date"]
            except Exception:  # noqa: BLE001 - 单个文件损坏不影响整体判断
                continue
            if column.empty:
                continue
            candidate = column.max()
            latest = candidate if latest is None else max(latest, candidate)
        if latest is None:
            raise PermanentError("所有缓存文件都无法读取日期列")
        self._as_of = str(pd.Timestamp(latest).date())
        return self._as_of

    def _loader(self):
        from qfm.data import DataLoader

        return DataLoader(str(self.cache_dir))

    def load(self, pool: str, symbols: Iterable[str] | None = None) -> PanelBundle:
        from qfm.data import build_panel, get_universe

        codes = list(symbols) if symbols is not None else list(get_universe(pool))
        if not codes:
            raise PermanentError(f"股票池 {pool} 解析为空，无法取数")
        loader = self._loader()
        bars = loader.load_bars(codes)
        if bars is None or len(bars) == 0:
            raise PermanentError(
                f"本地缓存中 {pool}（{len(codes)} 只）没有任何日线数据；"
                "请先运行 python -m qfm.cli --list 确认缓存，或改用 synthetic 数据源"
            )
        indicators = loader.load_indicators()
        panel = build_panel(bars, indicators)

        # loader 的失败语义是「静默丢股票」而不是抛异常，所以这里主动核对覆盖率。
        loaded = [str(code) for code in panel.close.columns]
        missing = sorted(set(codes) - set(loaded))
        as_of = str(panel.close.index.max().date())

        note = f"本地 parquet 缓存（{self.cache_dir}）"
        if missing:
            note += f"；{len(missing)} 只无缓存（如 {', '.join(missing[:3])}），已按实际可用标的计算"
        return PanelBundle(
            panel=panel,
            pool=pool,
            source=self.name,
            symbols=loaded,
            note=note,
            as_of=as_of,
            meta={
                "requested": len(codes),
                "loaded": len(loaded),
                "missing": missing[:20],
                "coverage": len(loaded) / len(codes) if codes else 0.0,
                "cache_dir": str(self.cache_dir),
            },
        )

    def industry_map(self, as_of: str | None = None) -> pd.Series:
        """股票 → 行业。直接读 indicators.parquet，不构建整个面板（秒级）。"""
        path = self.cache_dir / "indicators.parquet"
        if not path.exists():
            raise PermanentError(f"缺少 {path}，无法解析行业标签")
        frame = pd.read_parquet(path, columns=["stock", "industry", "ann_date"])
        frame = frame.dropna(subset=["industry"])
        if as_of:
            cutoff = pd.Timestamp(as_of)
            frame = frame[frame["ann_date"] <= cutoff]
        if frame.empty:
            raise PermanentError(f"indicators.parquet 中在 {as_of or '全部日期'} 之前没有行业数据")
        latest = frame.sort_values("ann_date").groupby("stock")["industry"].last()
        latest.index = latest.index.astype(str)
        return latest

    def industry_labels(self) -> list[str]:
        return sorted(self.industry_map().unique().tolist())


class SyntheticProvider:
    """离线合成面板。零网络、确定性，用于演示与 CI。"""

    name = "synthetic"

    def __init__(self, spec: SyntheticSpec | None = None) -> None:
        self.spec = spec or SyntheticSpec(as_of=DEFAULT_AS_OF)
        self._cache: PanelBundle | None = None

    def available(self) -> bool:
        return True

    def describe(self) -> str:
        return f"离线合成面板（种子 {self.spec.seed}，截至 {self.spec.as_of}）"

    def symbols(self, pool: str) -> list[str]:
        if self._cache is None:
            self.load("synthetic")
        return list(self._cache.symbols)

    def as_of(self) -> str:
        return self.spec.as_of

    def load(self, pool: str, symbols: Iterable[str] | None = None) -> PanelBundle:
        if self._cache is None:
            panel = build_synthetic_panel(self.spec)
            symbols_all = [str(code) for code in panel.close.columns]
            self._cache = PanelBundle(
                panel=panel,
                pool="synthetic",
                source=self.name,
                symbols=symbols_all,
                note=(
                    "合成面板（确定性、零网络）。含人为植入的弱价值效应，"
                    "仅用于演示 Agent 流程，不构成任何实证结论"
                ),
                as_of=str(panel.close.index.max().date()),
                meta={"spec": self.spec.to_dict(), "requested": len(symbols_all)},
            )

        base = self._cache
        selected = list(symbols) if symbols is not None else list(base.symbols)
        selected = [code for code in selected if code in set(base.symbols)]
        if not selected:
            raise PermanentError(
                f"合成数据源中没有 {pool} 对应的标的；可用行业: "
                f"{', '.join(sorted(set(build_industry_series(base.panel).unique())))}"
            )
        return PanelBundle(
            panel=_select(selected, base.panel),
            pool=pool,
            source=self.name,
            symbols=selected,
            note=f"{base.note}；本次取 {len(selected)}/{base.n_symbols} 只",
            as_of=base.as_of,
            meta={**base.meta, "loaded": len(selected)},
        )

    def industry_map(self, as_of: str | None = None) -> pd.Series:
        if self._cache is None:
            self.load("synthetic")
        panel = self._cache.panel
        return build_industry_series(panel, as_of)

    def industry_labels(self) -> list[str]:
        return sorted(self.industry_map().unique().tolist())


def build_industry_series(panel: Any, as_of: str | None = None) -> pd.Series:
    """从面板取某个时点的行业标签（默认取最后一期）。

    真实数据的行业标签是**时点化**的（东财按公告日对齐），所以「银行股」
    在 t 时点的成分与最新一期可能不同。要复现历史研究就必须用当期截面，
    直接取 `panel.industry` 的最后一行是最新口径 —— 这是有偏的，
    因此这里的默认是「取最后一行」但调用方可以指定 `as_of` 取历史截面。
    """
    industry = getattr(panel, "industry", None)
    if industry is None or getattr(industry, "empty", True):
        return pd.Series(dtype=object)
    if as_of:
        cutoff = pd.Timestamp(as_of)
        subset = industry.loc[industry.index <= cutoff]
        row = subset.iloc[-1] if not subset.empty else industry.iloc[-1]
    else:
        row = industry.iloc[-1]
    series = row.dropna()
    series.index = series.index.astype(str)
    return series


def _select(symbols: list[str], panel: Any) -> Any:
    """从面板中挑出指定标的，返回新的 DataPanel（保持全部字段对齐）。"""
    from qfm.data.panel import DataPanel

    kwargs: dict[str, Any] = {}
    for name in ("close", "open", "high", "low", "volume", "amount", "turnover",
                 "factor", "close_raw", "mv_float", "industry"):
        frame = getattr(panel, name, None)
        if isinstance(frame, pd.DataFrame):
            kwargs[name] = frame.reindex(columns=symbols)
    kwargs["fund"] = {
        key: value.reindex(columns=symbols)
        for key, value in getattr(panel, "fund", {}).items()
    }
    kwargs["fund_names"] = list(getattr(panel, "fund_names", []))
    return DataPanel(**kwargs)


def resolve_provider(
    mode: str = "auto",
    *,
    cache_dir: Path | str = "data_cache",
    spec: SyntheticSpec | None = None,
) -> DataProvider:
    """按模式挑选数据源。

    Args:
        mode: ``auto`` / ``cache`` / ``synthetic``。
        cache_dir: 本地缓存目录。
        spec: 合成面板参数（仅 synthetic 模式生效）。

    Raises:
        PermanentError: `mode="cache"` 但本地缓存不可用（不静默降级——
            指定了真实数据就必须给真实数据）。
    """
    normalized = (mode or "auto").strip().lower()
    if normalized not in ("auto", "cache", "synthetic"):
        raise PermanentError(f"未知数据源模式 {mode!r}；可选 auto / cache / synthetic")

    if normalized == "synthetic":
        return SyntheticProvider(spec)
    local = LocalCacheProvider(cache_dir)
    if normalized == "cache":
        if not local.available():
            raise PermanentError(
                f"指定了 cache 数据源，但 {cache_dir}/bars 下没有 parquet 缓存；"
                "请改用 --data auto 或 --data synthetic"
            )
        return local
    return local if local.available() else SyntheticProvider(spec)
