"""数据类工具：解析股票池、加载面板、数据体检。

这三个工具对应研究流程的头三步，也是「一家公司到底能不能用 Agent 做研究」
的第一道门槛——真实数据不会像玩具数据那样干净：

- 上游 `DataLoader.load_bars` 在单只股票拉取失败时只 `print` 一行然后
  **静默丢掉**，最终返回一个看起来成功的面板。所以 `load_panel` 必须主动
  核对「请求 N 只、实际 M 只」，并在覆盖率过低时抛 `DegradedResultError`
  触发重试 —— 这是把静默降级接进重试机制的关键落点。
- 真实行业标签是**时点化**的（东财按公告日对齐），`银行Ⅱ` 在 2020 年与
  2026 年的成分不完全相同。所以按行业选池必须用「分析区间的当期截面」，
  而不是「最新标签」，否则历史研究会带成分漂移。
"""

from __future__ import annotations

import pandas as pd

from qfm.agent.errors import DegradedResultError, InvalidArgumentsError, PermanentError
from qfm.agent.providers import PanelBundle, build_industry_series
from qfm.agent.models import ToolKind
from qfm.agent.registry import ToolContext
from qfm.agent.vocab import match_industry, resolve_pool

__all__ = ["register_data_tools", "MIN_UNIVERSE_COVERAGE"]

#: 面板实际覆盖低于此比例即视为静默降级，触发重试（仅真实数据源启用）。
MIN_UNIVERSE_COVERAGE = 0.85

#: 因子计算的最低覆盖率；财务字段缺失会让因子大面积 NaN。
MIN_FACTOR_COVERAGE = 0.30

#: 默认预热交易日数。要覆盖最长的因子回看窗口（mom_250 / vol_120 /
#: 多因子权重回看 252 日），否则窗口首日会有大量因子取不到值。
DEFAULT_WARMUP_DAYS = 320


def register_data_tools(registry) -> None:
    """把数据类工具注册进 `registry`（`qfm.agent.registry.ToolRegistry`）。"""

    @registry.register(
        "resolve_universe",
        kind=ToolKind.READ,
        description="把自然语言股票池说法解析成具体标的列表：既支持行业关键词（银行股、白酒、科技板块），也支持指数池（沪深300、中证500、全市场）",
        parameters={
            "keyword": "str，行业/主题关键词，如「银行股」「科技板块」",
            "pool": "str，指数池标识或自然语言，如 cn_hs300 / 全市场",
            "as_of": "str，取行业标签的时点截面（YYYY-MM-DD），默认用数据最新一期",
            "max_stocks": "int，截断标的数量（调试/加速用）",
        },
        required=(),
        returns="payload: {pool, symbols, n_symbols, matched_labels, method, as_of, source, coverage_note}",
        cost=0.5,
        tags=("data", "universe"),
    )
    def resolve_universe(
        context: ToolContext,
        keyword: str | None = None,
        pool: str | None = None,
        as_of: str | None = None,
        max_stocks: int | None = None,
    ):
        if not keyword and not pool:
            preset = context.get("preset_universe")
            if preset:
                # 调用方（CLI / 页面）已显式指定标的：直接采用，不猜关键词。
                symbols = [str(code) for code in preset.get("symbols") or []]
                if not symbols:
                    raise InvalidArgumentsError("预设股票池为空")
                context.put("universe", {
                    "pool": preset.get("pool", "explicit"),
                    "n_symbols": len(symbols),
                    "symbols": symbols,
                    "matched_labels": [],
                    "method": "preset",
                    "as_of": preset.get("as_of", ""),
                    "source": getattr(context.provider, "name", "unknown"),
                    "coverage_note": "调用方显式指定的标的列表",
                    "note": "",
                })
                return context.get("universe"), (
                    f"股票池 {preset.get('pool', 'explicit')}：{len(symbols)} 只（调用方指定）"
                )
            raise InvalidArgumentsError(
                "resolve_universe 至少需要 keyword（行业关键词）或 pool（指数池）之一"
            )
        provider = context.provider
        if provider is None:
            raise PermanentError("运行上下文未配置数据源 provider")

        method = "pool"
        matched_labels: list[str] = []
        note = ""

        if keyword:
            labels = provider.industry_labels()
            matched_labels, method = match_industry(keyword, labels)
            if not matched_labels:
                raise PermanentError(
                    f"行业关键词 {keyword!r} 未匹配到任何行业标签；"
                    f"数据中可用的行业共 {len(labels)} 个，例如: {', '.join(labels[:12])}"
                )

        # 行业池优先于指数池：用户说「银行股」时不该落到全市场。
        if matched_labels:
            industry_series = build_industry_series(_industry_source(context), as_of)
            symbols = sorted(industry_series[industry_series.isin(matched_labels)].index.tolist())
            pool_id = "industry:" + "+".join(matched_labels)
            as_of_used = _industry_as_of(context, as_of)
            if method in ("alias", "substring"):
                note = (
                    f"关键词 {keyword!r} 经 {method} 匹配到 {matched_labels}；"
                    "东财行业分类与申万等口径不同，如需严格口径请核对"
                )
            coverage_note = f"{as_of_used} 截面行业标签"
        else:
            resolved = resolve_pool(pool or "") or (pool or "")
            # 只要代码列表；加载面板是 load_panel 的职责，这里加载纯属浪费。
            symbols = provider.symbols(resolved)
            pool_id = resolved
            as_of_used = provider.as_of()
            coverage_note = provider.describe()

        if max_stocks is not None:
            if max_stocks < 1:
                raise InvalidArgumentsError(f"max_stocks 必须 >= 1，得到 {max_stocks}")
            if len(symbols) > max_stocks:
                note = (note + "；" if note else "") + f"已按 max_stocks 截断到 {max_stocks} 只"
                symbols = symbols[:max_stocks]

        if not symbols:
            raise DegradedResultError(f"股票池 {pool_id} 解析出 0 只标的")

        payload = {
            "pool": pool_id,
            "n_symbols": len(symbols),
            "symbols": symbols,
            "matched_labels": matched_labels,
            "method": method,
            "as_of": as_of_used,
            "source": getattr(provider, "name", "unknown"),
            "coverage_note": coverage_note,
            "note": note,
        }
        context.put("universe", payload)
        summary = (
            f"股票池 {pool_id}：{len(symbols)} 只"
            + (f"，命中行业 {matched_labels}（{method}）" if matched_labels else "")
            + f"，截面 {as_of_used}"
        )
        return payload, summary

    @registry.register(
        "load_panel",
        kind=ToolKind.COMPUTE,
        description="按已解析的股票池加载面板数据（行情+财务+行业），裁剪到分析窗口（含预热段）并核对覆盖率；覆盖率过低会判定为上游静默丢数据并重试",
        parameters={
            "pool": "str，股票池标识；省略则沿用上一步 resolve_universe 的结果",
            "symbols": "list，显式指定标的（一般不用，由 resolve_universe 提供）",
            "max_stocks": "int，截断标的数量",
            "start": "str，分析起始日期 YYYY-MM-DD；给出则裁剪到 start-预热 至 end",
            "end": "str，分析结束日期 YYYY-MM-DD",
            "warmup_days": "int，预热交易日数（默认 320，覆盖 mom_250 等长窗口因子）",
            "min_coverage": "float，覆盖率下限，低于则重试（默认 0.85）",
        },
        required=(),
        returns="payload: {pool, n_symbols, n_days, start, end, source, coverage, coverage_ratio, note, requested}",
        cost=2.0,
        tags=("data",),
    )
    def load_panel(
        context: ToolContext,
        pool: str | None = None,
        symbols: list | None = None,
        max_stocks: int | None = None,
        start: str | None = None,
        end: str | None = None,
        warmup_days: int = DEFAULT_WARMUP_DAYS,
        min_coverage: float | None = None,
    ):
        universe = context.get("universe") or {}
        target_pool = pool or universe.get("pool") or "index800"
        target_symbols = list(symbols) if symbols else universe.get("symbols")
        if max_stocks and target_symbols:
            target_symbols = list(target_symbols)[:max_stocks]

        provider = context.provider
        if provider is None:
            raise PermanentError("运行上下文未配置数据源 provider")
        bundle: PanelBundle = provider.load(target_pool, symbols=target_symbols)
        panel = bundle.panel
        if panel is None or getattr(panel.close, "empty", True):
            raise DegradedResultError(f"股票池 {target_pool} 加载后为空面板")

        requested = len(target_symbols) if target_symbols else bundle.n_symbols
        loaded = len(panel.close.columns)
        ratio = loaded / requested if requested else 1.0

        # 裁剪到分析窗口。真实缓存里单只银行股可能有 30 年历史（8604 个交易日），
        # 而财务数据只有近几年——不裁剪会让估值类因子的覆盖率被历史区间严重稀释
        # （实测 15%），把一个「窗口内其实完整」的因子误判为数据缺失。
        # 预热段是必需的：mom_250 这类因子在窗口首日需要 250 天历史才能出值。
        window = _slice_window(panel, start, end, warmup_days)
        if window is not None:
            panel, window_note = window
        else:
            window_note = ""

        # 真实数据源走严格校验；合成数据是确定性的，重试没有意义。
        if not bundle.is_synthetic:
            floor = MIN_UNIVERSE_COVERAGE if min_coverage is None else float(min_coverage)
            if floor > 0 and ratio < floor:
                raise DegradedResultError(
                    f"面板覆盖率 {ratio:.1%}（请求 {requested} 只，实际 {loaded} 只）"
                    f"低于下限 {floor:.0%}；上游 loader 对失败的股票是静默丢弃，"
                    "本结果不可信，需重试"
                )

        context.panel = panel
        context.panel_meta = {
            "pool": target_pool,
            "symbols": [str(code) for code in panel.close.columns],
            "n_symbols": loaded,
            "n_days": int(panel.close.shape[0]),
            "start": str(panel.close.index.min().date()),
            "end": str(panel.close.index.max().date()),
            "analysis_window": [start, end] if (start or end) else None,
            "warmup_days": int(warmup_days),
            "source": bundle.source,
            "note": bundle.note,
            "window_note": window_note,
        }
        payload = {
            **context.panel_meta,
            "requested": requested,
            "coverage_ratio": round(ratio, 4),
        }
        summary = (
            f"面板就绪：{loaded} 只 × {payload['n_days']} 交易日"
            f"（{payload['start']} ~ {payload['end']}），来源 {bundle.source}"
            + (f"，覆盖 {ratio:.0%}" if not bundle.is_synthetic else "")
            + (f"；{window_note}" if window_note else "")
        )
        if bundle.is_synthetic:
            payload["is_synthetic"] = True
            payload["warnings"] = ["数据为合成面板，非真实行情；结论仅供流程演示"]
        return payload, summary

    @registry.register(
        "describe_panel",
        kind=ToolKind.READ,
        description="数据体检：截面宽度、行业分布、财务字段可用性、缺失情况。用于在计算前发现数据问题",
        parameters={
            "top_industries": "int，返回前 N 个行业（默认 8）",
        },
        required=(),
        returns="payload: {n_symbols, n_days, avg_cross_section, min_cross_section, max_cross_section, industries, fund_fields, fund_coverage, warnings}",
        cost=1.0,
        tags=("data", "quality"),
    )
    def describe_panel(context: ToolContext, top_industries: int = 8):
        panel = context.panel
        if panel is None or getattr(panel.close, "empty", True):
            raise InvalidArgumentsError("尚未加载面板；请先执行 load_panel")

        close = panel.close
        valid_per_day = close.notna().sum(axis=1)
        industry_series = build_industry_series(panel)
        counts = industry_series.value_counts()
        fund_coverage = {
            name: round(float(frame.notna().to_numpy().mean()), 4)
            for name, frame in (getattr(panel, "fund", {}) or {}).items()
        }
        warnings: list[str] = []
        avg_cs = float(valid_per_day.mean())
        if avg_cs < 100:
            warnings.append(
                f"平均截面仅 {avg_cs:.0f} 只股票，低于分层检验所需的 100 只；"
                "分层/单调性统计将不可用（IC 仍有效但统计功效受限）"
            )
        thin = [name for name, value in fund_coverage.items() if value < 0.5]
        if thin:
            warnings.append(f"财务字段覆盖率偏低: {', '.join(thin)}；相关因子可能大面积缺失")
        if not fund_coverage:
            warnings.append("面板没有任何财务字段；价值/质量/成长族因子会全部为空")

        payload = {
            "n_symbols": int(close.shape[1]),
            "n_days": int(close.shape[0]),
            "start": str(close.index.min().date()),
            "end": str(close.index.max().date()),
            "avg_cross_section": round(avg_cs, 1),
            "min_cross_section": int(valid_per_day.min()),
            "max_cross_section": int(valid_per_day.max()),
            "industries": {str(k): int(v) for k, v in counts.head(int(top_industries)).items()},
            "n_industries": int(counts.size),
            "fund_fields": sorted(fund_coverage),
            "fund_coverage": fund_coverage,
            "layer_test_available": bool(avg_cs >= 100),
            "warnings": warnings,
        }
        summary = (
            f"{payload['n_symbols']} 只 × {payload['n_days']} 日，"
            f"平均截面 {payload['avg_cross_section']}，"
            f"{payload['n_industries']} 个行业，"
            f"财务字段 {len(fund_coverage)} 个"
            + (f"；{len(warnings)} 条数据告警" if warnings else "")
        )
        return payload, summary


def _slice_window(panel, start: str | None, end: str | None, warmup_days: int):
    """把面板裁剪到 [start - 预热, end]。

    Returns:
        ``(裁剪后的面板, 说明文字)``；未指定区间时返回 ``None``（保持原面板）。
    """
    if not start and not end:
        return None
    from qfm.simulation.engine import slice_panel

    index = panel.close.index
    try:
        end_ts = pd.Timestamp(end) if end else index.max()
        start_ts = pd.Timestamp(start) if start else index.min()
    except Exception as exc:  # noqa: BLE001
        raise InvalidArgumentsError(f"日期解析失败: {exc}") from exc

    # 先显式判断无交集：否则下面的预热回溯会把 anchor 拉到数据区间内，
    # 把「请求 2030 年的数据」这种错误悄悄变成「给你一段 2024 年的数据」。
    if start_ts > index.max() or end_ts < index.min():
        raise InvalidArgumentsError(
            f"分析窗口 {start_ts.date()} ~ {end_ts.date()} 与数据区间 "
            f"{index.min().date()} ~ {index.max().date()} 没有交集"
        )

    hold = max(0, int(warmup_days))
    if hold:
        positions = index.searchsorted(start_ts)
        warm_start = max(0, int(positions) - hold)
        anchor = index[warm_start]
    else:
        anchor = start_ts

    dates = index[(index >= anchor) & (index <= end_ts)]
    if len(dates) == 0:
        raise InvalidArgumentsError(
            f"分析窗口 {start_ts.date()} ~ {end_ts.date()} 与数据区间 "
            f"{index.min().date()} ~ {index.max().date()} 没有交集"
        )
    trimmed = slice_panel(panel, dates)
    note = (
        f"已裁剪到 {dates[0].date()} ~ {dates[-1].date()}"
        f"（含 {hold} 日预热）"
    )
    return trimmed, note


def _industry_source(context: ToolContext):
    """取得可用于行业筛选的对象：优先已加载面板，否则回退到 provider 的行业映射。

    回退路径很重要：`resolve_universe` 通常跑在 `load_panel` 之前，
    此时若为了拿行业标签而先加载整张面板，真实数据下是几十秒的额外开销。
    provider 的 `industry_map()` 只读一个 parquet 列，秒级返回。
    """
    if context.panel is not None:
        return context.panel
    provider = context.provider
    if provider is None:
        raise PermanentError("运行上下文未配置数据源 provider")
    return _SeriesPanel(provider.industry_map())


def _industry_as_of(context: ToolContext, as_of: str | None) -> str:
    """回报行业标签实际取用的时点，避免展示成占位日期。"""
    if as_of:
        return str(pd.Timestamp(as_of).date())
    if context.panel is not None and not getattr(context.panel.industry, "empty", True):
        return str(context.panel.industry.index[-1].date())
    meta = context.panel_meta or {}
    return str(meta.get("end") or "最新可得截面")


class _SeriesPanel:
    """只有 `industry` 的轻量替身，让 `build_industry_series` 可复用于未加载面板的场景。"""

    def __init__(self, series: pd.Series) -> None:
        self.industry = pd.DataFrame([series], index=[pd.Timestamp("1970-01-01")])
