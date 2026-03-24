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


def _write_compound_daily(
    root: Path,
    symbol: str,
    *,
    periods: int,
    amount: float,
    daily_return: float,
    amount_ratio20: float = 1.0,
) -> None:
    dates = pd.date_range("2023-01-02", periods=periods, freq="B")
    close = 10.0 * ((1.0 + daily_return) ** pd.Series(range(periods), dtype=float))
    amount_series = pd.Series([amount] * periods, dtype=float)
    if periods >= 20:
        amount_series.iloc[-20:] = float(amount) * float(amount_ratio20)
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1000000,
            "amount": amount_series,
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


def test_dynamic_universe_generator_prefers_liquid_leaders(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    universe_file = tmp_path / "universe.json"
    universe_file.write_text(
        json.dumps(
            {
                "stocks": [
                    {"symbol": "600001.SH", "name": "LeaderA", "sector": "科技"},
                    {"symbol": "600002.SH", "name": "LeaderB", "sector": "科技"},
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_daily(data_dir, "600001.SH", periods=520, amount=1.2e8, slope=0.055)
    _write_daily(data_dir, "600002.SH", periods=520, amount=2.2e7, slope=0.058)
    monkeypatch.setattr("src.application.v2_universe_generator._load_tushare_stock_basic", lambda: {})
    monkeypatch.setattr("src.application.v2_universe_generator._load_symbol_concepts", lambda symbols: {})

    result = generate_dynamic_universe(
        universe_file=str(universe_file),
        data_dir=str(data_dir),
        cache_root=str(tmp_path / "cache"),
        target_size=1,
        coarse_size=2,
        theme_aware=False,
        use_concepts=False,
        end_date="2026-03-11",
        min_history_days=480,
        min_recent_amount=2.0e7,
        theme_cap_ratio=1.0,
        theme_floor_count=0,
        turnover_quality_weight=0.25,
        theme_weight=0.20,
        refresh_cache=True,
    )

    assert result.selected_300[0]["symbol"] == "600001.SH"
    assert result.selected_300[0]["leadership_score"] >= result.selected_300[0]["quality_score"] * 0.5
    assert "leadership_score" in result.coarse_pool[0]
    for field in ("close", "ma20", "ma60", "ret20", "ret60", "breakout_pos_120", "volatility20", "tradeability"):
        assert field in result.selected_300[0]


def test_dynamic_universe_generator_can_limit_to_main_board(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    universe_file = tmp_path / "universe.json"
    universe_file.write_text(
        json.dumps(
            {
                "stocks": [
                    {"symbol": "002409.SZ", "name": "MainBoardA", "sector": "煤化工"},
                    {"symbol": "300001.SZ", "name": "ChiNextA", "sector": "电气设备"},
                    {"symbol": "688001.SH", "name": "StarA", "sector": "半导体"},
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_daily(data_dir, "002409.SZ", periods=520, amount=8.0e7, slope=0.05)
    _write_daily(data_dir, "300001.SZ", periods=520, amount=8.0e7, slope=0.06)
    _write_daily(data_dir, "688001.SH", periods=520, amount=8.0e7, slope=0.07)
    monkeypatch.setattr("src.application.v2_universe_generator._load_tushare_stock_basic", lambda: {})
    monkeypatch.setattr("src.application.v2_universe_generator._load_symbol_concepts", lambda symbols: {})

    result = generate_dynamic_universe(
        universe_file=str(universe_file),
        data_dir=str(data_dir),
        cache_root=str(tmp_path / "cache"),
        target_size=3,
        coarse_size=3,
        theme_aware=False,
        use_concepts=False,
        end_date="2026-03-11",
        min_history_days=480,
        min_recent_amount=2.0e7,
        theme_cap_ratio=1.0,
        theme_floor_count=0,
        turnover_quality_weight=0.25,
        theme_weight=0.20,
        main_board_only=True,
        refresh_cache=True,
    )

    assert [item["symbol"] for item in result.selected_300] == ["002409.SZ"]
    assert result.generator_manifest.source_universe_size == 1


def test_dynamic_universe_generator_prefers_leaders_while_exporting_fresh_pool_metadata(
    monkeypatch,
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    universe_file = tmp_path / "universe.json"
    universe_file.write_text(
        json.dumps(
            {
                "stocks": [
                    {"symbol": "600010.SH", "name": "FreshA", "sector": "科技"},
                    {"symbol": "600011.SH", "name": "HotA", "sector": "科技"},
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_compound_daily(data_dir, "600010.SH", periods=520, amount=9.0e7, daily_return=0.0022, amount_ratio20=1.12)
    _write_compound_daily(data_dir, "600011.SH", periods=520, amount=9.0e7, daily_return=0.0120, amount_ratio20=1.55)
    monkeypatch.setattr("src.application.v2_universe_generator._load_tushare_stock_basic", lambda: {})
    monkeypatch.setattr("src.application.v2_universe_generator._load_symbol_concepts", lambda symbols: {})

    result = generate_dynamic_universe(
        universe_file=str(universe_file),
        data_dir=str(data_dir),
        cache_root=str(tmp_path / "cache"),
        target_size=1,
        coarse_size=2,
        theme_aware=False,
        use_concepts=False,
        end_date="2026-03-11",
        min_history_days=480,
        min_recent_amount=2.0e7,
        theme_cap_ratio=1.0,
        theme_floor_count=0,
        turnover_quality_weight=0.25,
        theme_weight=0.20,
        refresh_cache=True,
    )

    assert result.selected_300[0]["symbol"] == "600011.SH"
    assert "fresh_pool_score" in result.selected_300[0]
    assert "fresh_pool_pass" in result.selected_300[0]
    assert any(item["fresh_pool_pass"] is True for item in result.coarse_pool)
    assert result.generator_manifest.config["fresh_pool_pass_count"] >= 1
    assert result.generator_manifest.config["fresh_pool_funnel"][-1]["count"] >= 1
