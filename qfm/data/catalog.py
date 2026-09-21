"""版本化 Dataset / Universe 注册表。"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields as dataclass_fields
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


BUILTIN_UNIVERSES: dict[str, dict[str, str]] = {
    "cn_hs300": {"name": "沪深300", "source": "csindex:000300"},
    "cn_zz500": {"name": "中证500", "source": "csindex:000905"},
    "cn_zz1000": {"name": "中证1000", "source": "csindex:000852"},
    "cn_all_a": {"name": "全A", "source": "akshare:eastmoney-yjbb"},
    "index800": {"name": "沪深300 + 中证500（兼容）", "source": "csindex:000300+000905"},
    "full": {"name": "全A（兼容）", "source": "akshare:eastmoney-yjbb"},
}

REQUIRED_BINDING_FIELDS = ("dataset_id", "dataset_version", "universe_id", "universe_version")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _digest(prefix: str, payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{prefix}_" + sha256(raw.encode("utf-8")).hexdigest()[:16]


def normalise_symbols(values: Iterable[object]) -> tuple[str, ...]:
    symbols: set[str] = set()
    for value in values:
        if value is None:
            raise ValueError("股票代码不能为空")
        text = str(value).strip()
        if text.endswith(".0") and text[:-2].isdigit():
            text = text[:-2]
        if not text.isdigit() or len(text) > 6:
            raise ValueError(f"非法 A 股代码: {value!r}")
        symbols.add(text.zfill(6))
    if not symbols:
        raise ValueError("股票池至少需要一个有效代码")
    return tuple(sorted(symbols))


@dataclass(frozen=True)
class DatasetVersion:
    dataset_id: str
    dataset_version: str
    source: str
    start_date: str
    end_date: str
    last_update: str
    symbols: tuple[str, ...]
    fields: tuple[str, ...]
    market_fingerprint: str = ""
    fundamental_fingerprint: str = ""
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["symbols"] = list(self.symbols)
        data["fields"] = list(self.fields)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DatasetVersion":
        required = ("dataset_id", "dataset_version", "source", "start_date", "end_date", "last_update")
        if any(not data.get(key) for key in required):
            raise ValueError("Dataset 记录缺少必需字段")
        return cls(
            dataset_id=str(data["dataset_id"]),
            dataset_version=str(data["dataset_version"]),
            source=str(data["source"]),
            start_date=str(data["start_date"]),
            end_date=str(data["end_date"]),
            last_update=str(data["last_update"]),
            symbols=normalise_symbols(data.get("symbols") or ()),
            fields=tuple(sorted(str(item) for item in data.get("fields") or ())),
            market_fingerprint=str(data.get("market_fingerprint") or ""),
            fundamental_fingerprint=str(data.get("fundamental_fingerprint") or ""),
            created_at=str(data.get("created_at") or data["last_update"]),
        )


@dataclass(frozen=True)
class UniverseDefinition:
    universe_id: str
    universe_version: str
    name: str
    source: str
    symbols: tuple[str, ...]
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["symbols"] = list(self.symbols)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "UniverseDefinition":
        required = ("universe_id", "universe_version", "name", "source", "created_at", "updated_at")
        if any(not data.get(key) for key in required):
            raise ValueError("Universe 记录缺少必需字段")
        return cls(
            universe_id=str(data["universe_id"]),
            universe_version=str(data["universe_version"]),
            name=str(data["name"]),
            source=str(data["source"]),
            symbols=normalise_symbols(data.get("symbols") or ()),
            created_at=str(data["created_at"]),
            updated_at=str(data["updated_at"]),
        )


class DataCatalog:
    def __init__(self, root: Path | str | None = None):
        configured = root or os.environ.get("QFM_CATALOG_ROOT") or "data_cache/catalog"
        self.root = Path(configured)

    @property
    def datasets_path(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root / "datasets.json"

    @property
    def universes_path(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root / "universes.json"

    @staticmethod
    def _read(path: Path, key: str) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        rows = data.get(key) if isinstance(data, dict) else None
        return [dict(item) for item in rows if isinstance(item, dict)] if isinstance(rows, list) else []

    @staticmethod
    def _write(path: Path, key: str, rows: list[dict[str, Any]]) -> None:
        target = path
        temporary = target.with_suffix(".tmp")
        try:
            temporary.write_text(
                json.dumps({key: rows}, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False),
                encoding="utf-8",
            )
            temporary.replace(target)
        finally:
            if temporary.exists():
                temporary.unlink()

    def list_datasets(self) -> list[DatasetVersion]:
        result = []
        for row in self._read(self.datasets_path, "datasets"):
            try:
                result.append(DatasetVersion.from_dict(row))
            except ValueError:
                continue
        return sorted(result, key=lambda item: item.last_update, reverse=True)

    def register_panel(self, dataset_id: str, panel, snapshot: Mapping[str, Any],
                       source: str) -> DatasetVersion:
        if not dataset_id.strip() or not source.strip():
            raise ValueError("dataset_id 和 source 不能为空")
        symbols = normalise_symbols(panel.close.columns)
        available_fields: list[str] = []
        for field in dataclass_fields(panel):
            value = getattr(panel, field.name)
            if isinstance(value, pd.DataFrame) and not value.empty:
                available_fields.append(field.name)
            elif field.name == "fund" and isinstance(value, dict):
                available_fields.extend(f"fund.{name}" for name, frame in value.items()
                                        if isinstance(frame, pd.DataFrame) and not frame.empty)
        field_names = tuple(sorted(set(available_fields)))
        fingerprints = snapshot.get("fingerprints") if isinstance(snapshot.get("fingerprints"), dict) else {}
        identity = {
            "dataset_id": dataset_id.strip(),
            "source": source.strip(),
            "start_date": str(snapshot.get("data_start") or ""),
            "end_date": str(snapshot.get("data_end") or ""),
            "symbols": list(symbols),
            "fields": list(field_names),
            "market_fingerprint": str(fingerprints.get("market") or ""),
            "fundamental_fingerprint": str(fingerprints.get("fundamentals") or ""),
        }
        version = _digest("dataset", identity)
        existing = next((item for item in self.list_datasets() if item.dataset_version == version), None)
        if existing is not None:
            return existing
        now = _now()
        dataset = DatasetVersion(
            dataset_id=identity["dataset_id"],
            dataset_version=version,
            source=identity["source"],
            start_date=identity["start_date"],
            end_date=identity["end_date"],
            last_update=now,
            symbols=symbols,
            fields=field_names,
            market_fingerprint=identity["market_fingerprint"],
            fundamental_fingerprint=identity["fundamental_fingerprint"],
            created_at=now,
        )
        rows = [item.to_dict() for item in self.list_datasets()] + [dataset.to_dict()]
        self._write(self.datasets_path, "datasets", rows)
        return dataset

    def list_universes(self) -> list[UniverseDefinition]:
        result = []
        for row in self._read(self.universes_path, "universes"):
            try:
                result.append(UniverseDefinition.from_dict(row))
            except ValueError:
                continue
        return sorted(result, key=lambda item: (item.name, item.universe_id))

    def register_universe(self, universe_id: str, symbols: Iterable[object], *,
                          name: str | None = None, source: str | None = None) -> UniverseDefinition:
        normalised = normalise_symbols(symbols)
        built_in = BUILTIN_UNIVERSES.get(universe_id, {})
        resolved_name = (name or built_in.get("name") or universe_id).strip()
        resolved_source = (source or built_in.get("source") or "custom").strip()
        if not universe_id.strip() or not resolved_name:
            raise ValueError("universe_id 和名称不能为空")
        version = _digest("universe", {
            "universe_id": universe_id.strip(), "source": resolved_source, "symbols": list(normalised),
        })
        existing = next((item for item in self.list_universes()
                         if item.universe_id == universe_id and item.universe_version == version), None)
        if existing is not None:
            return existing
        now = _now()
        universe = UniverseDefinition(
            universe_id=universe_id.strip(), universe_version=version,
            name=resolved_name, source=resolved_source, symbols=normalised,
            created_at=now, updated_at=now,
        )
        rows = [item.to_dict() for item in self.list_universes()] + [universe.to_dict()]
        self._write(self.universes_path, "universes", rows)
        return universe

    def create_custom_universe(self, name: str, symbols: Iterable[object], *,
                               source: str = "user-defined") -> UniverseDefinition:
        cleaned = name.strip()
        if not cleaned:
            raise ValueError("自定义股票池名称不能为空")
        if any(item.name == cleaned for item in self.list_universes()):
            raise ValueError(f"股票池名称已存在: {cleaned}")
        universe_id = "custom_" + sha256(cleaned.encode("utf-8")).hexdigest()[:12]
        return self.register_universe(universe_id, symbols, name=cleaned, source=source)

    def get_universe(self, universe_id: str, universe_version: str | None = None) -> UniverseDefinition:
        candidates = [item for item in self.list_universes() if item.universe_id == universe_id]
        if universe_version is not None:
            candidates = [item for item in candidates if item.universe_version == universe_version]
        if not candidates:
            raise KeyError(f"Universe 不存在: {universe_id}")
        return sorted(candidates, key=lambda item: item.updated_at, reverse=True)[0]


def register_panel_dataset(panel, universe_id: str, snapshot: Mapping[str, Any], *,
                           root: Path | str | None = None,
                           universe_symbols: Iterable[object] | None = None) -> dict[str, str]:
    """注册当前面板及股票池，返回 Experiment/Job 共用的四字段绑定。"""
    catalog = DataCatalog(root)
    sources = snapshot.get("sources") if isinstance(snapshot.get("sources"), dict) else {}
    source = "+".join(str(value) for value in sources.values() if value) or "unknown"
    dataset = catalog.register_panel("cn_equity_daily", panel, snapshot, source=source)
    symbols = universe_symbols if universe_symbols is not None else panel.close.columns
    universe = catalog.register_universe(universe_id, symbols)
    return {
        "dataset_id": dataset.dataset_id,
        "dataset_version": dataset.dataset_version,
        "universe_id": universe.universe_id,
        "universe_version": universe.universe_version,
    }

