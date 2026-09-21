"""综合因子 JSON 注册表，以及重启后的 Factor Library 恢复。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from qfm.factors.base import FACTORS, get_factor, register_factor
from qfm.multifactor.models import CompositeDefinition


class CompositeRegistry:
    def __init__(self, root: Path | str | None = None):
        configured = root or os.environ.get("QFM_COMPOSITE_REGISTRY") or "data_cache/factors"
        path = Path(configured)
        self.path = path if path.suffix == ".json" else path / "composites.json"

    def _read_rows(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        rows = payload.get("definitions") if isinstance(payload, dict) else None
        return [dict(item) for item in rows if isinstance(item, dict)] if isinstance(rows, list) else []

    def _write(self, definitions: list[CompositeDefinition]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        try:
            temporary.write_text(
                json.dumps(
                    {"definitions": [item.to_dict() for item in definitions]},
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
            temporary.replace(self.path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def list_definitions(self, name: str | None = None) -> list[CompositeDefinition]:
        definitions: list[CompositeDefinition] = []
        for row in self._read_rows():
            try:
                item = CompositeDefinition.from_dict(row)
            except (TypeError, ValueError):
                continue
            if name is None or item.name == name:
                definitions.append(item)
        return sorted(definitions, key=lambda item: (item.name, item.version))

    def latest_definitions(self) -> dict[str, CompositeDefinition]:
        latest: dict[str, CompositeDefinition] = {}
        for item in self.list_definitions():
            if item.name not in latest or item.version > latest[item.name].version:
                latest[item.name] = item
        return latest

    @staticmethod
    def _assert_acyclic(definitions: dict[str, CompositeDefinition]) -> None:
        graph = {
            name: [str(component.get("name")) for component in item.components
                   if str(component.get("name")) in definitions]
            for name, item in definitions.items()
        }
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node: str) -> None:
            if node in visiting:
                raise ValueError(f"综合因子存在循环依赖: {node}")
            if node in visited:
                return
            visiting.add(node)
            for child in graph.get(node, []):
                visit(child)
            visiting.remove(node)
            visited.add(node)

        for name in graph:
            visit(name)

    def save(self, definition: CompositeDefinition) -> CompositeDefinition:
        existing = self.list_definitions()
        # 内置因子不能被综合因子静默覆盖；已登记的综合因子可以新增版本。
        if definition.name in FACTORS and definition.name not in {item.name for item in existing}:
            raise ValueError(f"因子名称已被占用: {definition.name}")
        same_name = [item for item in existing if item.name == definition.name]
        digest = definition.definition_hash()
        for item in same_name:
            if item.source_hash == digest:
                return item
        version = max((item.version for item in same_name), default=0) + 1
        saved = definition.with_persistence(version)
        latest = self.latest_definitions()
        latest[saved.name] = saved
        self._assert_acyclic(latest)
        self._write(existing + [saved])
        return saved

    @staticmethod
    def _register(definition: CompositeDefinition) -> None:
        names = [str(item["name"]) for item in definition.components]
        versions = {str(item["name"]): int(item["version"]) for item in definition.components}

        @register_factor(
            definition.name,
            "多因子",
            definition.description or "持久化综合因子",
            direction="positive",
            tags=("合成", "多因子"),
            version=definition.version,
            source_key=json.dumps(definition.identity(), ensure_ascii=False, sort_keys=True),
            params={"mode": definition.mode, "horizon": definition.horizon},
        )
        def composite(panel):
            from qfm.portfolio.synthesis import synthesize

            score, _ = synthesize(
                panel,
                names,
                mode=definition.mode,
                horizon=definition.horizon,
                orthogonalize=definition.orthogonalize,
                ortho_controls=definition.ortho_controls,
                weight_lookback=definition.weight_lookback,
                weight_rebalance=definition.weight_rebalance,
                factor_versions=versions,
            )
            return score

        assert composite is not None

    def register_all(self) -> list[str]:
        """按依赖顺序恢复全部版本；不安全定义跳过并返回可展示告警。"""
        pending = self.list_definitions()
        warnings: list[str] = []
        while pending:
            progressed = False
            for definition in list(pending):
                missing = []
                drifted = []
                for component in definition.components:
                    name = str(component.get("name") or "")
                    version = component.get("version")
                    factor = get_factor(name, int(version)) if version is not None else None
                    if factor is None:
                        missing.append(f"{name} v{version}")
                    elif component.get("source_hash") and factor.source_hash != component.get("source_hash"):
                        drifted.append(f"{name} v{version}")
                if missing:
                    continue
                if drifted:
                    warnings.append(
                        f"综合因子 {definition.name} v{definition.version} 成分定义已漂移：{'、'.join(drifted)}"
                    )
                    pending.remove(definition)
                    progressed = True
                    continue
                self._register(definition)
                pending.remove(definition)
                progressed = True
            if not progressed:
                for definition in pending:
                    names = "、".join(str(item.get("name")) for item in definition.components)
                    warnings.append(
                        f"综合因子 {definition.name} v{definition.version} 缺少成分或依赖未就绪：{names}"
                    )
                break
        return warnings
