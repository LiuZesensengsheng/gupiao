from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.domain.entities import Security
from src.infrastructure.margin_sync import MarginSyncResult, sync_margin_data
from src.interfaces.cli import run_api_cli


def _market_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-03-10"]),
            "finance_balance": [1.0],
        }
    )


def _stock_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-03-10"]),
            "symbol": ["600000.SH"],
            "finance_balance": [2.0],
        }
    )


def test_sync_margin_data_can_mix_market_and_stock_sources(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fake_try_source(
        source: str,
        *,
        symbols,
        start: str,
        end: str,
        tushare_token: str,
        sleep_ms: int,
    ):
        del symbols, start, end, tushare_token, sleep_ms
        if source == "akshare":
            return _market_frame(), pd.DataFrame(columns=["date", "symbol"]), ["akshare stock detail empty"]
        if source == "tushare":
            return pd.DataFrame(columns=["date"]), _stock_frame(), []
        raise AssertionError(f"unexpected source {source}")

    monkeypatch.setattr("src.infrastructure.margin_sync._try_source", fake_try_source)

    result = sync_margin_data(
        source="auto",
        symbols=["600000.SH"],
        start="2026-02-01",
        end="2026-03-12",
        market_out=str(tmp_path / "margin_market.csv"),
        stock_out=str(tmp_path / "margin_stock.csv"),
    )

    assert result.source_used == "akshare+tushare"
    assert result.market_rows == 1
    assert result.stock_rows == 1
    assert "market margin sourced from akshare; stock detail sourced from tushare." in result.notes
    assert pd.read_csv(tmp_path / "margin_stock.csv")["symbol"].tolist() == ["600000.SH"]


def test_run_sync_margin_uses_universe_file_when_symbols_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        run_api_cli,
        "_load_discovery_universe_file",
        lambda path, enrich_metadata=False: [
            Security(symbol="600000.SH", name="A", sector="银行"),
            Security(symbol="000001.SZ", name="B", sector="银行"),
        ],
    )
    monkeypatch.setattr(
        run_api_cli,
        "load_watchlist",
        lambda path: (_ for _ in ()).throw(AssertionError("watchlist should not be used")),
    )

    captured: dict[str, object] = {}

    def fake_sync_margin_data(**kwargs) -> MarginSyncResult:
        captured.update(kwargs)
        return MarginSyncResult(
            source_used="tushare",
            market_rows=1,
            stock_rows=2,
            market_path=str(tmp_path / "margin_market.csv"),
            stock_path=str(tmp_path / "margin_stock.csv"),
            notes=[],
        )

    monkeypatch.setattr(run_api_cli, "sync_margin_data", fake_sync_margin_data)

    exit_code = run_api_cli.run_sync_margin(
        {
            "tushare_token": "",
            "symbols": "",
            "universe_file": str(tmp_path / "universe.json"),
            "universe_limit": 1,
            "watchlist": str(tmp_path / "watchlist.json"),
            "source": "tushare",
            "start": "2026-02-01",
            "end": "2026-03-12",
            "margin_market_file": str(tmp_path / "margin_market.csv"),
            "margin_stock_file": str(tmp_path / "margin_stock.csv"),
            "sleep_ms": 0,
        }
    )

    assert exit_code == 0
    assert captured["symbols"] == ["600000.SH"]
