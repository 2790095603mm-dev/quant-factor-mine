"""Dataset / Universe 版本目录。"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from qfm.data.catalog import (
    BUILTIN_UNIVERSES,
    DataCatalog,
    normalise_symbols,
    register_panel_dataset,
)
from qfm.data.universe import get_universe
from qfm.research.snapshot import build_data_snapshot


def test_normalise_symbols_sorts_deduplicates_and_pads_codes():
    assert normalise_symbols(["1", " 600000 ", "000001", 300750]) == ("000001", "300750", "600000")


@pytest.mark.parametrize("bad", [[], ["abc"], ["1234567"], [None]])
def test_normalise_symbols_rejects_empty_or_invalid_codes(bad):
    with pytest.raises(ValueError):
        normalise_symbols(bad)


def test_dataset_version_is_content_stable(tmp_path, panel):
    catalog = DataCatalog(tmp_path)
    snapshot = build_data_snapshot(panel, "index800")

    first = catalog.register_panel("cn_equity_daily", panel, snapshot, source="akshare")
    second = catalog.register_panel("cn_equity_daily", panel, snapshot, source="akshare")

    assert first == second
    assert first.dataset_version.startswith("dataset_")
    assert set(first.to_dict()) >= {
        "dataset_id", "dataset_version", "source", "start_date", "end_date",
        "last_update", "symbols", "fields",
    }
    assert len(catalog.list_datasets()) == 1


def test_dataset_version_changes_when_content_fingerprint_changes(tmp_path, panel):
    snapshot = build_data_snapshot(panel, "index800")
    catalog = DataCatalog(tmp_path)
    first = catalog.register_panel("cn_equity_daily", panel, snapshot, source="akshare")
    changed = dict(snapshot)
    changed["fingerprints"] = {**snapshot["fingerprints"], "market": "sha256:changed"}
    second = catalog.register_panel("cn_equity_daily", panel, changed, source="akshare")

    assert first.dataset_version != second.dataset_version
    assert len(catalog.list_datasets()) == 2


def test_register_panel_dataset_returns_binding_dict(tmp_path, panel):
    snapshot = build_data_snapshot(panel, "cn_hs300")

    binding = register_panel_dataset(
        panel,
        universe_id="cn_hs300",
        snapshot=snapshot,
        root=tmp_path,
        universe_symbols=panel.close.columns,
    )

    assert binding["dataset_id"] == "cn_equity_daily"
    assert binding["dataset_version"].startswith("dataset_")
    assert binding["universe_id"] == "cn_hs300"
    assert binding["universe_version"].startswith("universe_")


def test_custom_universe_normalises_and_versions_symbols(tmp_path):
    catalog = DataCatalog(tmp_path)

    universe = catalog.create_custom_universe("自选池", ["1", "600000", "000001"])

    assert universe.symbols == ("000001", "600000")
    assert universe.universe_id.startswith("custom_")
    assert universe.universe_version.startswith("universe_")
    assert catalog.get_universe(universe.universe_id) == universe


def test_custom_universe_name_must_be_unique(tmp_path):
    catalog = DataCatalog(tmp_path)
    catalog.create_custom_universe("自选池", ["000001"])

    with pytest.raises(ValueError, match="已存在"):
        catalog.create_custom_universe("自选池", ["600000"])


def test_builtin_universe_can_be_registered_without_duplication(tmp_path):
    catalog = DataCatalog(tmp_path)
    first = catalog.register_universe("cn_hs300", ["000001", "600000"])
    second = catalog.register_universe("cn_hs300", ["600000", "000001"])

    assert first == second
    assert first.name == BUILTIN_UNIVERSES["cn_hs300"]["name"]
    assert len(catalog.list_universes()) == 1


def test_catalog_skips_corrupt_individual_records(tmp_path):
    root = tmp_path
    root.mkdir(exist_ok=True)
    (root / "universes.json").write_text(
        json.dumps({"universes": [{"bad": "record"}]}), encoding="utf-8"
    )

    assert DataCatalog(root).list_universes() == []


def test_standard_universe_ids_resolve_through_existing_fetchers(monkeypatch):
    monkeypatch.setattr("qfm.data.universe._load_cache", lambda: {})
    monkeypatch.setattr("qfm.data.universe._save_cache", lambda _pools: None)
    monkeypatch.setattr(
        "qfm.data.universe.get_index_cons",
        lambda symbol: {"000300": ["600000"], "000905": ["000001"], "000852": ["300001"]}[symbol],
    )

    assert get_universe("cn_hs300") == ["600000"]
    assert get_universe("cn_zz500") == ["000001"]
    assert get_universe("cn_zz1000") == ["300001"]
    assert get_universe("index800") == ["000001", "600000"]
