"""上下文管理：让 Agent 在「6 年 × 800 只股票」的数据规模下不撑爆上下文。

## 问题

一次量化研究里，工具返回的东西体量极不对等：

- `list_factors` 返回 49 个因子的元数据 —— 几 KB，值得全量保留；
- `compute_factor` 返回 2000 × 800 的浮点矩阵 —— 内存里 12 MB 量级，
  序列化进提示词则直接超出任何模型的上下文窗口；
- `run_experiment` 返回 IC 序列、分层收益、净值曲线、交易明细 —— 同样巨大。

如果实现方式是「把工具返回值都塞进消息历史」，那么这个 Agent 只能跑玩具
数据，一碰真实面板就崩。这是大多数 Agent Demo 与生产实现的分水岭。

## 三层策略

**① 摘要代替原文。** 工具返回 `(payload, summary)`；上下文只收 `summary`
（默认上限 600 字符），完整 `payload` 留在 `ToolContext` 供后续步骤使用。
`summary` 由工具作者手写，因此能只保留「判断下一步所需要的信息」。

**② 超限内容外置。** 仍超过 `max_item_chars` 的条目，完整内容写到
`artifacts/` 下的 JSON 文件，上下文里只留「摘要 + 文件路径」。这样事后能
完整复盘，运行时的上下文却是恒定的。

**③ 超预算时折叠最旧的非置顶条目。** 总字符数超过 `max_chars` 时，
从最旧的开始折叠（不是删除——折叠成一句占位说明，保留「曾经发生过什么」，
避免模型误以为这一步没执行过）。`pinned=True` 的条目（原始请求、计划、
关键结论）永不折叠。

折叠是**从最旧开始**而不是「按大小裁剪」，因为 Agent 的对话具有因果次序：
最新观察是决策依据，最旧观察大多已被后续结论吸收。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from qfm.agent.models import new_id

__all__ = [
    "ContextBudget",
    "ContextItem",
    "ContextStats",
    "ContextManager",
]


@dataclass(frozen=True)
class ContextBudget:
    """上下文预算。

    Args:
        max_chars: 渲染后的总字符上限。
        max_item_chars: 单条目内联上限；超出则外置到 artifact。
        keep_recent: 折叠时至少保留最新的 N 条（无论字符数）。
        keep_pinned: 置顶条目是否豁免折叠。
    """

    max_chars: int = 24000
    max_item_chars: int = 2000
    keep_recent: int = 8
    keep_pinned: bool = True

    def __post_init__(self) -> None:
        if self.max_chars < 500:
            raise ValueError(f"max_chars 过小，无法容纳任何有效上下文：{self.max_chars}")
        if self.max_item_chars < 100:
            raise ValueError(f"max_item_chars 过小：{self.max_item_chars}")
        if self.keep_recent < 0:
            raise ValueError("keep_recent 不能为负")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_chars": self.max_chars,
            "max_item_chars": self.max_item_chars,
            "keep_recent": self.keep_recent,
            "keep_pinned": self.keep_pinned,
        }


@dataclass
class ContextItem:
    """上下文中的一条。`artifact` 非空表示正文已外置，`content` 只是摘要。"""

    role: str
    title: str
    content: str
    pinned: bool = False
    artifact: str = ""
    offloaded: bool = False
    folded: bool = False
    item_id: str = ""
    original_chars: int = 0

    def __post_init__(self) -> None:
        if not self.item_id:
            self.item_id = new_id("ctx")
        if not self.original_chars:
            self.original_chars = len(self.content)

    @property
    def chars(self) -> int:
        return len(self.content)

    def render(self) -> str:
        flags = []
        if self.pinned:
            flags.append("置顶")
        if self.artifact:
            flags.append("已外置")
        if self.folded:
            flags.append("已折叠")
        suffix = f" [{'/'.join(flags)}]" if flags else ""
        pointer = f"\n  ↳ 完整内容: {self.artifact}" if self.artifact else ""
        return f"### {self.title}{suffix}\n{self.content}{pointer}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "role": self.role,
            "title": self.title,
            "chars": self.chars,
            "original_chars": self.original_chars,
            "pinned": self.pinned,
            "artifact": self.artifact,
            "offloaded": self.offloaded,
            "folded": self.folded,
        }


@dataclass
class ContextStats:
    """一次压缩后的统计，直接写进运行报告——上下文管理必须可观测。"""

    items: int = 0
    chars: int = 0
    original_chars: int = 0
    offloaded_items: int = 0
    folded_items: int = 0
    budget: dict[str, Any] = field(default_factory=dict)

    @property
    def compression_ratio(self) -> float:
        if not self.original_chars:
            return 1.0
        return self.chars / self.original_chars

    def to_dict(self) -> dict[str, Any]:
        return {
            "items": self.items,
            "chars": self.chars,
            "original_chars": self.original_chars,
            "offloaded_items": self.offloaded_items,
            "folded_items": self.folded_items,
            "compression_ratio": round(self.compression_ratio, 3),
            "budget": dict(self.budget),
        }


_FOLD_TEMPLATE = "（此条已被折叠以节省上下文；原始 {chars} 字符，结论已并入后续步骤）"


class ContextManager:
    """管理 Agent 上下文的增删与压缩。

    Args:
        budget: 预算配置。
        artifact_dir: 外置内容的落盘目录。
    """

    def __init__(self, budget: ContextBudget | None = None, artifact_dir: Path | str = "artifacts") -> None:
        self.budget = budget or ContextBudget()
        self.artifact_dir = Path(artifact_dir)
        self._items: list[ContextItem] = []
        self._counter = 0
        self.stats = ContextStats(budget=self.budget.to_dict())

    # ---- 读写 -------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._items)

    @property
    def items(self) -> tuple[ContextItem, ...]:
        return tuple(self._items)

    def add(
        self,
        role: str,
        title: str,
        content: Any,
        *,
        pinned: bool = False,
        offload_name: str | None = None,
        artifact_payload: Any = None,
    ) -> ContextItem:
        """加入一条。超长内容自动外置。

        Args:
            content: 文本内容；非字符串会先做紧凑化。
            artifact_payload: 若给出，则把这份完整结构写入 artifact，
                `content` 只作为摘要展示（用于「摘要 + 完整文件」分离）。
        """
        text = content if isinstance(content, str) else _stringify(content)
        item = ContextItem(role=role, title=title, content=text, pinned=pinned)

        should_offload = (
            artifact_payload is not None
            or len(text) > self.budget.max_item_chars
        )
        if should_offload:
            payload = artifact_payload if artifact_payload is not None else text
            name = offload_name or f"{self._counter:02d}_{_slug(title)}.json"
            item.artifact = str(self._offload(name, payload))
            if len(text) > self.budget.max_item_chars:
                keep = max(1, self.budget.max_item_chars - 120)
                item.content = _truncate(text, keep)
            item.offloaded = True

        self._counter += 1
        self._items.append(item)
        return item

    def pin(self, item: ContextItem) -> ContextItem:
        item.pinned = True
        return item

    def compact(self) -> ContextStats:
        """按预算压缩，返回统计。幂等：多次调用结果一致。"""
        offloaded = sum(1 for item in self._items if item.offloaded)
        folded = sum(1 for item in self._items if item.folded)

        total = sum(item.chars for item in self._items)
        if total > self.budget.max_chars:
            folded += self._fold_oldest(total)

        self.stats = ContextStats(
            items=len(self._items),
            chars=sum(item.chars for item in self._items),
            original_chars=sum(item.original_chars for item in self._items),
            offloaded_items=offloaded,
            folded_items=folded,
            budget=self.budget.to_dict(),
        )
        return self.stats

    def _fold_oldest(self, total: int) -> int:
        """从最旧开始折叠，直到回到预算内。返回新折叠的条数。"""
        protected = set()
        if self.budget.keep_pinned:
            protected |= {id(item) for item in self._items if item.pinned}
        protected |= {id(item) for item in self._items[-self.budget.keep_recent:]}

        newly = 0
        for item in self._items:
            if total <= self.budget.max_chars:
                break
            if id(item) in protected or item.folded:
                continue
            before = item.chars
            item.content = _FOLD_TEMPLATE.format(chars=item.original_chars)
            item.folded = True
            total -= before - item.chars
            newly += 1
        return newly

    # ---- 渲染 -------------------------------------------------------------
    def render(self, *, max_chars: int | None = None) -> str:
        """渲染成文本（供 LLM 提示词或报告使用）。"""
        limit = max_chars or self.budget.max_chars
        chunks: list[str] = []
        used = 0
        for item in self._items:
            block = item.render()
            if used + len(block) > limit:
                break
            chunks.append(block)
            used += len(block)
        return "\n\n".join(chunks)

    def render_messages(self) -> list[dict[str, str]]:
        """渲染成 OpenAI 风格的消息列表。"""
        role_map = {
            "user": "user",
            "system": "system",
            "plan": "assistant",
            "tool": "user",
            "observation": "user",
            "finding": "assistant",
        }
        return [
            {"role": role_map.get(item.role, "user"), "content": item.render()}
            for item in self._items
        ]

    def digest(self, limit: int = 12) -> str:
        """只列标题的紧凑目录，便于一眼看清「上下文里都有什么」。"""
        lines = []
        for item in self._items[-limit:]:
            mark = "📌" if item.pinned else ("🗂" if item.artifact else "·")
            lines.append(f"{mark} {item.title} ({item.chars}字)")
        return "\n".join(lines)

    # ---- 内部 -------------------------------------------------------------
    def _offload(self, name: str, payload: Any) -> Path:
        target = self.artifact_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        text = payload if isinstance(payload, str) else json.dumps(
            _jsonable(payload), ensure_ascii=False, indent=2, default=str
        )
        target.write_text(text, encoding="utf-8")
        return target


def _stringify(value: Any) -> str:
    if value is None:
        return "（无）"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_jsonable(value), ensure_ascii=False, default=str)
    return str(value)


def _jsonable(value: Any) -> Any:
    """把任意对象转成可 JSON 序列化的结构（DataFrame 只留形状与列名）。"""
    from qfm.agent.models import _json_safe

    return _json_safe(value)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n…（已截断，原文 {len(text)} 字符）"


def _slug(text: str) -> str:
    cleaned: Iterable[str] = (ch if ch.isalnum() or ch in "-_" else "_" for ch in text)
    slug = "".join(cleaned).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return (slug or "item")[:48]
