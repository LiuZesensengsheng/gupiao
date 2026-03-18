from __future__ import annotations

import json
from pathlib import Path

from src.domain.entities import Security
from src.infrastructure.discovery import build_candidate_universe
from src.infrastructure.security_metadata import enrich_securities_with_metadata


def test_enrich_securities_with_metadata_replaces_placeholder_sector(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.infrastructure.security_metadata._load_tushare_stock_basic",
        lambda: {
            "603607.SH": {"name": "京华激光", "industry": "轻工制造"},
            "600879.SH": {"name": "航天电子", "industry": "航空装备"},
            "600968.SH": {"name": "海油发展", "industry": "石油石化"},
            "300308.SZ": {"name": "中际旭创", "industry": "通信设备"},
        },
    )
    monkeypatch.setattr(
        "src.infrastructure.security_metadata._load_symbol_concepts",
        lambda symbols: {},
    )
    rows = [
        Security(symbol="603607.SH", name="bad", sector="其他"),
        Security(symbol="600879.SH", name="bad", sector="其他"),
        Security(symbol="600968.SH", name="bad", sector="其他"),
        Security(symbol="300308.SZ", name="bad", sector="其他"),
    ]

    out = enrich_securities_with_metadata(rows)

    sector_map = {row.symbol: row.sector for row in out}
    name_map = {row.symbol: row.name for row in out}
    assert sector_map["603607.SH"] == "轻工制造"
    assert sector_map["600879.SH"] == "航天军工"
    assert sector_map["600968.SH"] == "能源石油"
    assert sector_map["300308.SZ"] == "光模块"
    assert name_map["603607.SH"] == "京华激光"


def test_build_candidate_universe_auto_repairs_broken_sector_metadata(tmp_path: Path, monkeypatch) -> None:
    universe_path = tmp_path / "universe.json"
    universe_path.write_text(
        json.dumps(
            {
                "stocks": [
                    {"symbol": "603607.SH", "name": "broken", "sector": "其他"},
                    {"symbol": "600879.SH", "name": "broken", "sector": "其他"},
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "src.infrastructure.security_metadata._load_tushare_stock_basic",
        lambda: {
            "603607.SH": {"name": "京华激光", "industry": "轻工制造"},
            "600879.SH": {"name": "航天电子", "industry": "航空装备"},
        },
    )
    monkeypatch.setattr(
        "src.infrastructure.security_metadata._load_symbol_concepts",
        lambda symbols: {},
    )

    universe = build_candidate_universe(
        source="auto",
        data_dir=str(tmp_path),
        universe_file=str(universe_path),
        candidate_limit=10,
        exclude_symbols=[],
    )

    assert [row.name for row in universe.rows] == ["京华激光", "航天电子"]
    assert [row.sector for row in universe.rows] == ["轻工制造", "航天军工"]


def test_enrich_securities_with_metadata_uses_symbol_fallback_when_tushare_empty(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.infrastructure.security_metadata._load_tushare_stock_basic",
        lambda: {},
    )
    monkeypatch.setattr(
        "src.infrastructure.security_metadata._load_symbol_metadata_with_fallback",
        lambda symbols: {
            "600968.SH": {"name": "海油发展", "industry": "石油石化"},
            "300308.SZ": {"name": "中际旭创", "industry": "通信设备"},
        },
    )
    monkeypatch.setattr(
        "src.infrastructure.security_metadata._load_symbol_concepts",
        lambda symbols: {},
    )

    out = enrich_securities_with_metadata(
        [
            Security(symbol="600968.SH", name="bad", sector="其他"),
            Security(symbol="300308.SZ", name="bad", sector="其他"),
        ]
    )

    assert [row.sector for row in out] == ["能源石油", "光模块"]


def test_concepts_override_broad_industry_labels(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.infrastructure.security_metadata._load_tushare_stock_basic",
        lambda: {
            "300308.SZ": {"name": "中际旭创", "industry": "通信设备"},
            "601898.SH": {"name": "中煤能源", "industry": "煤炭开采"},
        },
    )
    monkeypatch.setattr(
        "src.infrastructure.security_metadata._load_symbol_concepts",
        lambda symbols: {
            "300308.SZ": ["光纤", "CPO"],
            "601898.SH": ["煤化工", "焦煤"],
        },
    )

    out = enrich_securities_with_metadata(
        [
            Security(symbol="300308.SZ", name="bad", sector="其他"),
            Security(symbol="601898.SH", name="bad", sector="其他"),
        ]
    )

    assert [row.sector for row in out] == ["光模块", "煤化工"]
