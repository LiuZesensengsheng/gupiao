from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from src.interfaces.cli import run_api_cli


def test_parser_only_exposes_sync_tasks() -> None:
    parser = run_api_cli.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["daily"])

    parsed = parser.parse_args(["sync-data"])
    assert parsed.task == "sync-data"
    parsed_info = parser.parse_args(["sync-info"])
    assert parsed_info.task == "sync-info"


def test_main_dispatches_through_task_handlers(monkeypatch) -> None:
    parser = run_api_cli.build_parser()
    parsed = argparse.Namespace(
        task="sync-data",
        config="config/api.json",
        print_effective_config=False,
    )

    monkeypatch.setattr(parser, "parse_args", lambda: parsed)
    monkeypatch.setattr(run_api_cli, "build_parser", lambda: parser)
    monkeypatch.setattr(run_api_cli, "_read_json_config", lambda path: {})
    monkeypatch.setattr(run_api_cli, "_resolve_settings", lambda args, payload: {"task": args.task})
    monkeypatch.setattr(run_api_cli, "run_sync_data", lambda settings: 9)

    assert run_api_cli.main() == 9


def test_run_sync_info_delegates_to_sync_layer(monkeypatch, capsys) -> None:
    monkeypatch.setattr(run_api_cli, "set_tushare_token", lambda *_: None)
    monkeypatch.setattr(
        run_api_cli,
        "sync_info_data",
        lambda **_: SimpleNamespace(
            out_dir="D:/gupiao/input/info_parts",
            symbol_count=12,
            market_news_rows=8,
            announcement_rows=21,
            research_rows=13,
            notes=["market_news fetch returned 0 rows"],
        ),
    )

    rc = run_api_cli.run_sync_info(
        {
            "tushare_token": "",
            "info_dir": "input/info_parts",
            "start": "2026-01-01",
            "end": "2026-03-17",
            "watchlist": "config/watchlist.json",
            "universe_file": "",
            "universe_limit": 0,
            "symbols": "",
            "sleep_ms": 120,
            "max_retries": 3,
            "timeout": 20.0,
        }
    )

    captured = capsys.readouterr()
    assert rc == 0
    assert "Info output" in captured.out
    assert "Announcement rows: 21" in captured.out
