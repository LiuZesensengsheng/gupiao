from __future__ import annotations

import argparse

import pytest

from src.interfaces.cli import run_api_cli


def test_parser_only_exposes_sync_tasks() -> None:
    parser = run_api_cli.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["daily"])

    parsed = parser.parse_args(["sync-data"])
    assert parsed.task == "sync-data"


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
