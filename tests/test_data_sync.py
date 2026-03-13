from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.domain.entities import Security
from src.infrastructure import data_sync


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
        source="tushare",
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

