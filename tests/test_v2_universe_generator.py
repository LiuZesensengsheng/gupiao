from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.application.v2_universe_generator import generate_dynamic_universe


def _write_daily(
    root: Path,
    symbol: str,
    *,
    periods: int,
    amount: float,
    slope: float,
) -> None:
    dates = pd.date_range("2023-01-02", periods=periods, freq="B")
    base = 10.0 + pd.Series(range(periods), dtype=float) * slope
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": base,
            "high": base * 1.01,
            "low": base * 0.99,
            "close": base,
            "volume": 1000000,
            "amount": amount,
        }
    )
    frame.to_csv(root / f"{symbol}.csv", index=False)


def test_dynamic_universe_generator_filters_and_applies_theme_caps(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    universe_file = tmp_path / "universe.json"
    universe_file.write_text(
        json.dumps(
            {
                "stocks": [
                    {"symbol": "000001.SZ", "name": "LowHistA", "sector": "资源"},
                    {"symbol": "000002.SZ", "name": "LowAmtB", "sector": "资源"},
                    {"symbol": "000003.SZ", "name": "EnergyA", "sector": "能源石油"},
                    {"symbol": "000004.SZ", "name": "EnergyB", "sector": "能源石油"},
                    {"symbol": "000005.SZ", "name": "EnergyC", "sector": "能源石油"},
                    {"symbol": "000006.SZ", "name": "ChipA", "sector": "半导体"},
                    {"symbol": "000007.SZ", "name": "ChipB", "sector": "半导体"},
                    {"symbol": "000008.SZ", "name": "PowerA", "sector": "新能源电力"},
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_daily(data_dir, "000001.SZ", periods=120, amount=8.0e7, slope=0.03)
    _write_daily(data_dir, "000002.SZ", periods=520, amount=5.0e6, slope=0.03)
    _write_daily(data_dir, "000003.SZ", periods=520, amount=8.0e7, slope=0.06)
    _write_daily(data_dir, "000004.SZ", periods=520, amount=7.5e7, slope=0.055)
    _write_daily(data_dir, "000005.SZ", periods=520, amount=7.0e7, slope=0.05)
    _write_daily(data_dir, "000006.SZ", periods=520, amount=6.5e7, slope=0.05)
    _write_daily(data_dir, "000007.SZ", periods=520, amount=6.2e7, slope=0.048)
    _write_daily(data_dir, "000008.SZ", periods=520, amount=6.8e7, slope=0.045)

    monkeypatch.setattr("src.application.v2_universe_generator._load_tushare_stock_basic", lambda: {})
    monkeypatch.setattr("src.application.v2_universe_generator._load_symbol_concepts", lambda symbols: {})

    result = generate_dynamic_universe(
        universe_file=str(universe_file),
        data_dir=str(data_dir),
        cache_root=str(tmp_path / "cache"),
        target_size=4,
        coarse_size=6,
        theme_aware=True,
        use_concepts=False,
        end_date="2026-03-11",
        min_history_days=480,
        min_recent_amount=2.0e7,
        theme_cap_ratio=0.50,
        theme_floor_count=1,
        turnover_quality_weight=0.25,
        theme_weight=0.20,
        refresh_cache=True,
    )

    selected_symbols = [item["symbol"] for item in result.selected_300]
    assert len(selected_symbols) == 4
    assert "000001.SZ" not in selected_symbols
    assert "000002.SZ" not in selected_symbols
    assert selected_symbols != ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"]
    selected_themes = [item["sector"] for item in result.selected_300]
    assert max(selected_themes.count(theme) for theme in set(selected_themes)) <= 2
    assert result.generator_manifest.selected_pool_size == 4
    assert Path(result.generator_manifest.manifest_path).exists()


def test_dynamic_universe_generator_uses_cache(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    universe_file = tmp_path / "universe.json"
    universe_file.write_text(
        json.dumps(
            {"stocks": [{"symbol": "000001.SZ", "name": "EnergyA", "sector": "能源石油"}]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_daily(data_dir, "000001.SZ", periods=520, amount=8.0e7, slope=0.05)
    monkeypatch.setattr("src.application.v2_universe_generator._load_tushare_stock_basic", lambda: {})
    monkeypatch.setattr("src.application.v2_universe_generator._load_symbol_concepts", lambda symbols: {})

    first = generate_dynamic_universe(
        universe_file=str(universe_file),
        data_dir=str(data_dir),
        cache_root=str(tmp_path / "cache"),
        target_size=1,
        coarse_size=4,
        theme_aware=True,
        use_concepts=False,
        end_date="2026-03-11",
        min_history_days=480,
        min_recent_amount=2.0e7,
        theme_cap_ratio=1.0,
        theme_floor_count=1,
        turnover_quality_weight=0.25,
        theme_weight=0.20,
        refresh_cache=True,
    )
    monkeypatch.setattr(
        "src.application.v2_universe_generator._safe_read_local_daily",
        lambda _: (_ for _ in ()).throw(AssertionError("cache should be used")),
    )
    second = generate_dynamic_universe(
        universe_file=str(universe_file),
        data_dir=str(data_dir),
        cache_root=str(tmp_path / "cache"),
        target_size=1,
        coarse_size=4,
        theme_aware=True,
        use_concepts=False,
        end_date="2026-03-11",
        min_history_days=480,
        min_recent_amount=2.0e7,
        theme_cap_ratio=1.0,
        theme_floor_count=1,
        turnover_quality_weight=0.25,
        theme_weight=0.20,
        refresh_cache=False,
    )

    assert second.generator_manifest.generator_hash == first.generator_manifest.generator_hash
    assert second.selected_300 == first.selected_300
