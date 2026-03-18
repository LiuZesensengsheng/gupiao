from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.domain.entities import Security
from src.infrastructure import data_sync, market_data
from src.interfaces.cli import run_api_cli


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2026-03-02"],
            "open": [10.0],
            "high": [10.2],
            "low": [9.9],
            "close": [10.1],
            "volume": [1000.0],
        }
    )


def test_sync_market_data_prints_progress_and_cleans_checkpoint_on_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rows = [
        Security(symbol="600000.SH", name="浦发银行", sector="银行"),
        Security(symbol="000001.SZ", name="平安银行", sector="银行"),
    ]

    monkeypatch.setattr(
        data_sync,
        "_prepare_universe",
        lambda **_: (rows, "file:test"),
    )
    monkeypatch.setattr(data_sync, "load_symbol_daily", lambda **_: _sample_frame())
    monkeypatch.setattr(data_sync.time, "sleep", lambda *_: None)

    result = data_sync.sync_market_data(
        source="local",
        data_dir=str(tmp_path),
        start="2018-01-01",
        end="2026-03-02",
        universe_size=2,
        include_indices=False,
        force_refresh=True,
        sleep_ms=0,
        max_failures=5,
    )

    output = capsys.readouterr().out
    assert "[SYNC] Start sync" in output
    assert "[SYNC] 1/2 600000.SH" in output
    assert "[SYNC] 2/2 000001.SZ" in output
    assert result.downloaded == 2
    assert result.attempted == 2
    assert result.resumed is False
    assert result.checkpoint_file == ""
    assert not (tmp_path / ".sync_state").exists() or not any((tmp_path / ".sync_state").iterdir())


def test_sync_market_data_resumes_after_keyboard_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rows = [
        Security(symbol="600000.SH", name="浦发银行", sector="银行"),
        Security(symbol="000001.SZ", name="平安银行", sector="银行"),
    ]

    monkeypatch.setattr(
        data_sync,
        "_prepare_universe",
        lambda **_: (rows, "file:test"),
    )
    monkeypatch.setattr(data_sync.time, "sleep", lambda *_: None)

    call_order: list[str] = []

    def interrupting_loader(*, symbol: str, **_: object) -> pd.DataFrame:
        call_order.append(symbol)
        if symbol == "000001.SZ":
            raise KeyboardInterrupt()
        return _sample_frame()

    monkeypatch.setattr(data_sync, "load_symbol_daily", interrupting_loader)

    with pytest.raises(KeyboardInterrupt):
        data_sync.sync_market_data(
            source="baostock",
            data_dir=str(tmp_path),
            start="2018-01-01",
            end="2026-03-02",
            universe_size=2,
            include_indices=False,
            force_refresh=True,
            sleep_ms=0,
            max_failures=5,
        )

    checkpoint_files = list((tmp_path / ".sync_state").glob("sync_*.json"))
    assert len(checkpoint_files) == 1
    checkpoint_payload = json.loads(checkpoint_files[0].read_text(encoding="utf-8"))
    assert checkpoint_payload["completed_symbols"] == ["600000.SH"]

    resumed_calls: list[str] = []

    def resumed_loader(*, symbol: str, **_: object) -> pd.DataFrame:
        resumed_calls.append(symbol)
        return _sample_frame()

    monkeypatch.setattr(data_sync, "load_symbol_daily", resumed_loader)

    result = data_sync.sync_market_data(
        source="baostock",
        data_dir=str(tmp_path),
        start="2018-01-01",
        end="2026-03-02",
        universe_size=2,
        include_indices=False,
        force_refresh=True,
        sleep_ms=0,
        max_failures=5,
    )

    output = capsys.readouterr().out
    assert "Resume checkpoint detected" in output
    assert result.resumed is True
    assert result.resume_completed == 1
    assert result.downloaded == 1
    assert result.attempted == 1
    assert resumed_calls == ["000001.SZ"]
    assert result.checkpoint_file == ""
    assert not checkpoint_files[0].exists()


def test_sync_market_data_supports_parallel_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rows = [
        Security(symbol="600000.SH", name="浦发银行", sector="银行"),
        Security(symbol="000001.SZ", name="平安银行", sector="银行"),
        Security(symbol="600519.SH", name="贵州茅台", sector="消费"),
    ]

    monkeypatch.setattr(
        data_sync,
        "_prepare_universe",
        lambda **_: (rows, "file:test"),
    )
    monkeypatch.setattr(data_sync.time, "sleep", lambda *_: None)

    def loader(*, symbol: str, **_: object) -> pd.DataFrame:
        return _sample_frame().assign(symbol=symbol)

    monkeypatch.setattr(data_sync, "load_symbol_daily", loader)

    result = data_sync.sync_market_data(
        source="akshare",
        data_dir=str(tmp_path),
        start="2021-03-12",
        end="2026-03-12",
        universe_size=3,
        include_indices=False,
        force_refresh=True,
        sleep_ms=0,
        parallel_workers=3,
        max_failures=5,
    )

    output = capsys.readouterr().out
    assert "Parallel download enabled workers=3" in output
    assert result.downloaded == 3
    assert result.attempted == 3
    assert result.failed == 0


def test_sync_market_data_uses_tushare_batch_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rows = [
        Security(symbol="600000.SH", name="浦发银行", sector="银行"),
        Security(symbol="000001.SZ", name="平安银行", sector="银行"),
        Security(symbol="600519.SH", name="贵州茅台", sector="消费"),
        Security(symbol="000333.SZ", name="美的集团", sector="家电"),
    ]

    monkeypatch.setattr(
        data_sync,
        "_prepare_universe",
        lambda **_: (rows, "file:test"),
    )
    monkeypatch.setattr(
        data_sync,
        "probe_tushare_official_daily_availability",
        lambda: "tushare official probe passed last_open_date=20260317 rows=5000",
    )
    monkeypatch.setattr(data_sync.time, "sleep", lambda *_: None)

    batch_calls: list[list[str]] = []

    def fake_batch_loader(*, symbols: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
        batch_calls.append(list(symbols))
        return {
            symbol: _sample_frame().assign(symbol=symbol)
            for symbol in symbols
        }

    monkeypatch.setattr(data_sync, "fetch_tushare_daily_batch", fake_batch_loader)

    result = data_sync.sync_market_data(
        source="tushare",
        data_dir=str(tmp_path),
        start="2025-01-01",
        end="2026-03-12",
        universe_size=4,
        include_indices=False,
        force_refresh=True,
        sleep_ms=0,
        parallel_workers=2,
        max_failures=5,
    )

    output = capsys.readouterr().out
    assert "Tushare batch mode enabled" in output
    assert result.downloaded == 4
    assert result.attempted == 4
    assert result.failed == 0
    assert batch_calls
    assert sum(len(batch) for batch in batch_calls) == 4
    assert (tmp_path / "600000.SH.csv").exists()
    assert (tmp_path / "000001.SZ.csv").exists()


def test_sync_market_data_uses_tushare_trade_date_fast_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rows = [
        Security(symbol="600000.SH", name="浦发银行", sector="测试"),
        Security(symbol="000001.SZ", name="平安银行", sector="测试"),
    ]

    monkeypatch.setattr(
        data_sync,
        "_prepare_universe",
        lambda **_: (rows, "file:test"),
    )
    monkeypatch.setattr(
        data_sync,
        "probe_tushare_official_daily_availability",
        lambda: "tushare official probe passed last_open_date=20260317 rows=5000",
    )
    monkeypatch.setattr(data_sync.time, "sleep", lambda *_: None)

    fast_calls: list[list[str]] = []

    def fake_trade_date_loader(*, symbols: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
        fast_calls.append(list(symbols))
        return {symbol: _sample_frame().assign(symbol=symbol) for symbol in symbols}

    monkeypatch.setattr(data_sync, "fetch_tushare_daily_by_trade_dates", fake_trade_date_loader)
    monkeypatch.setattr(
        data_sync,
        "fetch_tushare_daily_batch",
        lambda **_: pytest.fail("symbol batch loader should not be used when trade-date fast path is active"),
    )

    result = data_sync.sync_market_data(
        source="tushare",
        data_dir=str(tmp_path),
        start="2026-03-10",
        end="2026-03-18",
        universe_size=2,
        include_indices=False,
        force_refresh=True,
        sleep_ms=0,
        parallel_workers=3,
        max_failures=5,
    )

    output = capsys.readouterr().out
    assert "Tushare trade-date fast path enabled" in output
    assert result.downloaded == 2
    assert result.attempted == 2
    assert result.failed == 0
    assert fast_calls


def test_sync_market_data_merges_existing_csv_on_incremental_update(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [Security(symbol="600000.SH", name="浦发银行", sector="银行")]
    monkeypatch.setattr(data_sync, "_prepare_universe", lambda **_: (rows, "file:test"))
    monkeypatch.setattr(data_sync.time, "sleep", lambda *_: None)

    existing = pd.DataFrame(
        {
            "date": ["2026-03-14", "2026-03-17"],
            "open": [10.0, 10.2],
            "high": [10.1, 10.3],
            "low": [9.9, 10.1],
            "close": [10.05, 10.25],
            "volume": [1000.0, 1100.0],
            "amount": [10050.0, 11275.0],
            "symbol": ["600000.SH", "600000.SH"],
        }
    )
    existing.to_csv(tmp_path / "600000.SH.csv", index=False)

    update = pd.DataFrame(
        {
            "date": ["2026-03-18"],
            "open": [10.3],
            "high": [10.4],
            "low": [10.2],
            "close": [10.35],
            "volume": [1200.0],
            "amount": [12420.0],
            "symbol": ["600000.SH"],
        }
    )
    monkeypatch.setattr(data_sync, "load_symbol_daily", lambda **_: update.copy())

    result = data_sync.sync_market_data(
        source="local",
        data_dir=str(tmp_path),
        start="2026-03-18",
        end="2026-03-18",
        universe_size=1,
        include_indices=False,
        force_refresh=True,
        sleep_ms=0,
        max_failures=5,
    )

    merged = pd.read_csv(tmp_path / "600000.SH.csv")
    assert result.downloaded == 1
    assert len(merged) == 3
    assert merged["date"].tolist() == ["2026-03-14", "2026-03-17", "2026-03-18"]


def test_suggest_tushare_batch_size_for_long_ranges() -> None:
    assert data_sync._suggest_tushare_batch_size("2018-01-01", "2026-03-13") == 4


def test_sync_market_data_falls_back_to_single_symbol_when_batch_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rows = [
        Security(symbol="600000.SH", name="浦发银行", sector="银行"),
        Security(symbol="000001.SZ", name="平安银行", sector="银行"),
    ]

    monkeypatch.setattr(data_sync, "_prepare_universe", lambda **_: (rows, "file:test"))
    monkeypatch.setattr(
        data_sync,
        "probe_tushare_official_daily_availability",
        lambda: "tushare official probe passed last_open_date=20260317 rows=5000",
    )
    monkeypatch.setattr(data_sync.time, "sleep", lambda *_: None)

    def failing_batch(*, symbols: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
        raise data_sync.DataError("batch empty")

    fallback_calls: list[str] = []

    def single_loader(*, symbol: str, source: str, **_: object) -> pd.DataFrame:
        fallback_calls.append(f"{source}:{symbol}")
        return _sample_frame().assign(symbol=symbol)

    monkeypatch.setattr(data_sync, "fetch_tushare_daily_batch", failing_batch)
    monkeypatch.setattr(data_sync, "load_symbol_daily", single_loader)

    result = data_sync.sync_market_data(
        source="tushare",
        data_dir=str(tmp_path),
        start="2025-01-01",
        end="2026-03-12",
        universe_size=2,
        include_indices=False,
        force_refresh=True,
        sleep_ms=0,
        parallel_workers=2,
        max_failures=5,
    )

    output = capsys.readouterr().out
    assert "Batch fallback to single-symbol requests" in output
    assert result.downloaded == 2
    assert result.failed == 0
    assert fallback_calls == ["tushare:600000.SH", "tushare:000001.SZ"]


def test_sync_market_data_fails_fast_when_tushare_official_probe_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        Security(symbol="600000.SH", name="浦发银行", sector="银行"),
        Security(symbol="000001.SZ", name="平安银行", sector="银行"),
    ]

    monkeypatch.setattr(data_sync, "_prepare_universe", lambda **_: (rows, "file:test"))
    monkeypatch.setattr(
        data_sync,
        "probe_tushare_official_daily_availability",
        lambda: (_ for _ in ()).throw(data_sync.DataError("tushare official gateway returned HTTP 502 for trade_cal")),
    )
    monkeypatch.setattr(
        data_sync,
        "fetch_tushare_daily_batch",
        lambda **_: pytest.fail("batch loader should not run when official probe fails"),
    )
    monkeypatch.setattr(
        data_sync,
        "load_symbol_daily",
        lambda **_: pytest.fail("single-symbol fallback should not run when official probe fails"),
    )

    with pytest.raises(data_sync.DataError, match="HTTP 502"):
        data_sync.sync_market_data(
            source="tushare",
            data_dir=str(tmp_path),
            start="2026-03-10",
            end="2026-03-18",
            universe_size=2,
            include_indices=False,
            force_refresh=True,
            sleep_ms=0,
            parallel_workers=2,
            max_failures=5,
        )


def test_probe_tushare_official_daily_availability_raises_on_http_502(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    market_data._TUSHARE_PROBE_CACHE.update({"checked_at": 0.0, "detail": "", "ok": False})
    monkeypatch.setattr(market_data, "_resolve_tushare_token", lambda: "token")

    class FakeResponse:
        status_code = 502
        text = "Bad Gateway"

        def json(self) -> dict[str, object]:
            return {}

    monkeypatch.setattr(market_data._HTTP_SESSION, "post", lambda *args, **kwargs: FakeResponse())

    with pytest.raises(market_data.DataError, match="HTTP 502"):
        market_data.probe_tushare_official_daily_availability(force_refresh=True, timeout=1)


def test_resolve_settings_includes_parallel_workers() -> None:
    args = run_api_cli.build_parser().parse_args(
        ["sync-data", "--parallel-workers", "12", "--config", "config/api.json"]
    )
    settings = run_api_cli._resolve_settings(args, {})
    assert settings["parallel_workers"] == 12

