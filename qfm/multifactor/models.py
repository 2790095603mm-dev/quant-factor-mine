"""可持久化综合因子定义。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Mapping

from qfm.factors.base import get_factor


WEIGHT_MODES = ("equal", "ic", "icir", "ic_x_ir")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class CompositeDefinition:
    name: str
    description: str
    components: tuple[dict[str, Any], ...]
    mode: str = "equal"
    horizon: int = 20
    weight_lookback: int = 252
    weight_rebalance: str = "ME"
    orthogonalize: bool = False
    ortho_controls: tuple[str, ...] = ()
    version: int = 0
    source_hash: str = ""
    created_at: str = ""

    @classmethod
    def create(
        cls,
        name: str,
        component_names: list[str] | tuple[str, ...],
        *,
        description: str = "",
        mode: str = "equal",
        horizon: int = 20,
        weight_lookback: int = 252,
        weight_rebalance: str = "ME",
        orthogonalize: bool = False,
        ortho_controls: tuple[str, ...] = (),
        validate_components: bool = True,
    ) -> "CompositeDefinition":
        cleaned_name = name.strip()
        names = list(dict.fromkeys(str(item).strip() for item in component_names if str(item).strip()))
        if not cleaned_name:
            raise ValueError("综合因子名称不能为空")
        if len(names) < 1:
            raise ValueError("至少选择 1 个成分因子")
        if mode not in WEIGHT_MODES:
            raise ValueError(f"未知权重模式: {mode}")
        if horizon < 1 or weight_lookback < 1:
            raise ValueError("horizon 和 weight_lookback 必须为正整数")
        components: list[dict[str, Any]] = []
        for component_name in names:
            factor = get_factor(component_name)
            if factor is None and validate_components:
                raise ValueError(f"成分因子不存在: {component_name}")
            components.append({
                "name": component_name,
                "version": factor.version if factor else None,
                "source_hash": factor.source_hash if factor else "",
            })
        return cls(
            name=cleaned_name,
            description=description.strip(),
            components=tuple(components),
            mode=mode,
            horizon=int(horizon),
            weight_lookback=int(weight_lookback),
            weight_rebalance=str(weight_rebalance),
            orthogonalize=bool(orthogonalize),
            ortho_controls=tuple(ortho_controls),
        )

    def identity(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "components": [dict(item) for item in self.components],
            "mode": self.mode,
            "horizon": self.horizon,
            "weight_lookback": self.weight_lookback,
            "weight_rebalance": self.weight_rebalance,
            "orthogonalize": self.orthogonalize,
            "ortho_controls": list(self.ortho_controls),
        }

    def definition_hash(self) -> str:
        raw = json.dumps(self.identity(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return "sha256:" + sha256(raw.encode("utf-8")).hexdigest()[:32]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["components"] = [dict(item) for item in self.components]
        data["ortho_controls"] = list(self.ortho_controls)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CompositeDefinition":
        components = data.get("components")
        if not data.get("name") or not isinstance(components, list):
            raise ValueError("综合因子定义缺少名称或成分")
        mode = str(data.get("mode") or "equal")
        if mode not in WEIGHT_MODES:
            raise ValueError(f"未知权重模式: {mode}")
        return cls(
            name=str(data["name"]),
            description=str(data.get("description") or ""),
            components=tuple(dict(item) for item in components if isinstance(item, dict)),
            mode=mode,
            horizon=int(data.get("horizon") or 20),
            weight_lookback=int(data.get("weight_lookback") or 252),
            weight_rebalance=str(data.get("weight_rebalance") or "ME"),
            orthogonalize=bool(data.get("orthogonalize", False)),
            ortho_controls=tuple(str(item) for item in data.get("ortho_controls") or ()),
            version=int(data.get("version") or 0),
            source_hash=str(data.get("source_hash") or ""),
            created_at=str(data.get("created_at") or ""),
        )

    def with_persistence(self, version: int) -> "CompositeDefinition":
        return replace(
            self,
            version=version,
            source_hash=self.definition_hash(),
            created_at=_now(),
        )
