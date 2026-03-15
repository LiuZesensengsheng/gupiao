from __future__ import annotations

import argparse

from src.interfaces.cli import run_api_cli


def test_run_daily_delegates_with_module_dependencies(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_daily(settings, *, dependencies):
        captured["settings"] = settings
        captured["dependencies"] = dependencies
        return 11

    monkeypatch.setattr(run_api_cli._legacy_cli_tasks, "run_daily", fake_run_daily)

    result = run_api_cli.run_daily({"task": "daily"})

    assert result == 11
    assert captured["settings"] == {"task": "daily"}
    assert captured["dependencies"] is run_api_cli


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
