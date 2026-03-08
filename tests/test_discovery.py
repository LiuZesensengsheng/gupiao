from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.infrastructure.discovery import build_predefined_universe


def _write_daily(path: Path, *, amount: float, rows: int = 520) -> None:
    dates = pd.date_range("2022-01-01", periods=rows, freq="B")
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": [10.0] * rows,
            "high": [10.2] * rows,
            "low": [9.8] * rows,
            "close": [10.0] * rows,
            "volume": [1000000] * rows,
            "amount": [amount] * rows,
            "symbol": [path.stem] * rows,
        }
    )
    frame.to_csv(path, index=False)


def test_build_predefined_universe_generated_is_stable_and_records_degrade_reason(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    _write_daily(data_dir / "000001.SZ.csv", amount=8.0e7)
    _write_daily(data_dir / "000002.SZ.csv", amount=5.0e7)
    _write_daily(data_dir / "000003.SZ.csv", amount=3.0e7)

    base_universe = tmp_path / "base.json"
    base_universe.write_text(
        json.dumps(
            {
                "stocks": [
                    {"symbol": "000001.SZ", "name": "A", "sector": "其他"},
                    {"symbol": "000002.SZ", "name": "B", "sector": "其他"},
                    {"symbol": "000003.SZ", "name": "C", "sector": "其他"},
                ]
            }
        ),
        encoding="utf-8",
    )
    favorites = tmp_path / "favorites.json"
    favorites.write_text(json.dumps({"stocks": []}), encoding="utf-8")

    manifest_path = tmp_path / "generated_80.json"
    first = build_predefined_universe(
        tier_id="generated_80",
        data_dir=str(data_dir),
        favorites_file=str(favorites),
        generated_base_file=str(base_universe),
        output_path=manifest_path,
    )
    second = build_predefined_universe(
        tier_id="generated_80",
        data_dir=str(data_dir),
        favorites_file=str(favorites),
        generated_base_file=str(base_universe),
        output_path=manifest_path,
    )

    assert [item.symbol for item in first.rows] == ["000001.SZ", "000002.SZ", "000003.SZ"]
    assert [item.symbol for item in first.rows] == [item.symbol for item in second.rows]
    assert any("generated universe degraded" in warning for warning in first.warnings)

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["universe_id"] == "generated_80"
    assert payload["symbol_count"] == 3
    assert payload["symbols"] == ["000001.SZ", "000002.SZ", "000003.SZ"]
