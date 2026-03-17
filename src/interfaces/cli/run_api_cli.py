from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.application.watchlist import load_watchlist
from src.domain.symbols import SymbolError, normalize_symbol
from src.infrastructure.data_sync import sync_market_data
from src.infrastructure.discovery import _load_universe_file as _load_discovery_universe_file
from src.infrastructure.info_sync import sync_info_data
from src.infrastructure.margin_sync import sync_margin_data
from src.infrastructure.market_data import DataError, set_tushare_token

DEFAULT_COMMON: dict[str, Any] = {
    "source": "auto",
    "tushare_token": "",
    "watchlist": "config/watchlist.json",
    "data_dir": "data",
    "start": "2018-01-01",
    "end": "2099-12-31",
    "margin_market_file": "input/margin_market.csv",
    "margin_stock_file": "input/margin_stock.csv",
}

DEFAULT_TASK: dict[str, dict[str, Any]] = {
    "sync-data": {
        "universe_size": 500,
        "universe_file": "",
        "universe_min_amount": 50000000.0,
        "universe_exclude_st": True,
        "include_indices": True,
        "force_refresh": False,
        "sleep_ms": 80,
        "parallel_workers": 1,
        "max_failures": 100,
        "write_universe_file": "config/universe_auto.json",
    },
    "sync-margin": {
        "symbols": "",
        "universe_file": "",
        "universe_limit": 0,
        "sleep_ms": 80,
    },
    "sync-info": {
        "symbols": "",
        "universe_file": "",
        "universe_limit": 0,
        "info_dir": "input/info_parts",
        "sleep_ms": 120,
        "max_retries": 3,
        "timeout": 20.0,
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A-share sync CLI")
    config_parent = argparse.ArgumentParser(add_help=False)
    config_parent.add_argument("--config", default="config/api.json", help="Path to unified API JSON config")
    config_parent.add_argument(
        "--tushare-token",
        dest="tushare_token",
        default=None,
        help="Tushare token (or set env TUSHARE_TOKEN)",
    )
    config_parent.add_argument(
        "--print-effective-config",
        action="store_true",
        help="Print merged runtime config and exit",
    )

    sub = parser.add_subparsers(dest="task", required=True)

    sync = sub.add_parser("sync-data", parents=[config_parent], help="Sync local A-share universe data")
    sync.add_argument(
        "--source",
        default=None,
        help="Data source chain used for bar download: eastmoney/tushare/akshare/baostock/local/auto",
    )
    sync.add_argument("--data-dir", dest="data_dir", default=None, help="Local data directory")
    sync.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    sync.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    sync.add_argument("--universe-size", dest="universe_size", type=int, default=None, help="Universe size target")
    sync.add_argument("--universe-file", dest="universe_file", default=None, help="Optional universe file (csv/json)")
    sync.add_argument(
        "--universe-min-amount",
        dest="universe_min_amount",
        type=float,
        default=None,
        help="Minimum turnover filter for auto universe fetch; set 0 for full A-share coverage",
    )
    sync.add_argument(
        "--universe-exclude-st",
        dest="universe_exclude_st",
        choices=["true", "false"],
        default=None,
        help="Whether to exclude ST names in auto universe fetch",
    )
    sync.add_argument(
        "--include-indices",
        dest="include_indices",
        choices=["true", "false"],
        default=None,
        help="Also sync main index files (000300/000001/399001/399006)",
    )
    sync.add_argument(
        "--force-refresh",
        dest="force_refresh",
        choices=["true", "false"],
        default=None,
        help="Refresh files even if local data is recent",
    )
    sync.add_argument("--sleep-ms", dest="sleep_ms", type=int, default=None, help="Sleep between requests in ms")
    sync.add_argument(
        "--parallel-workers",
        dest="parallel_workers",
        type=int,
        default=None,
        help="Parallel download workers for remote sync",
    )
    sync.add_argument("--max-failures", dest="max_failures", type=int, default=None, help="Stop after N failures")
    sync.add_argument(
        "--write-universe-file",
        dest="write_universe_file",
        default=None,
        help="Write discovered universe json for later reuse",
    )

    sync_margin = sub.add_parser("sync-margin", parents=[config_parent], help="Sync margin financing data")
    sync_margin.add_argument(
        "--source",
        default=None,
        help="Margin source chain: akshare/tushare/auto or comma chain",
    )
    sync_margin.add_argument("--watchlist", default=None, help="Watchlist JSON path when --symbols is not provided")
    sync_margin.add_argument("--universe-file", dest="universe_file", default=None, help="Universe file (csv/json)")
    sync_margin.add_argument(
        "--universe-limit",
        dest="universe_limit",
        type=int,
        default=None,
        help="Optional limit on symbols loaded from universe file",
    )
    sync_margin.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    sync_margin.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    sync_margin.add_argument(
        "--symbols",
        default=None,
        help="Comma separated symbols, e.g. 600160.SH,000630.SZ",
    )
    sync_margin.add_argument("--sleep-ms", dest="sleep_ms", type=int, default=None, help="Sleep between requests in ms")

    sync_info = sub.add_parser("sync-info", parents=[config_parent], help="Sync structured info parts for V2 info layer")
    sync_info.add_argument("--watchlist", default=None, help="Watchlist JSON path when --symbols is not provided")
    sync_info.add_argument("--universe-file", dest="universe_file", default=None, help="Universe file (csv/json)")
    sync_info.add_argument(
        "--universe-limit",
        dest="universe_limit",
        type=int,
        default=None,
        help="Optional limit on symbols loaded from universe file",
    )
    sync_info.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    sync_info.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    sync_info.add_argument(
        "--symbols",
        default=None,
        help="Comma separated symbols, e.g. 600160.SH,000630.SZ",
    )
    sync_info.add_argument("--info-dir", dest="info_dir", default=None, help="Structured info output directory")
    sync_info.add_argument("--sleep-ms", dest="sleep_ms", type=int, default=None, help="Sleep between announcement pages in ms")
    sync_info.add_argument("--max-retries", dest="max_retries", type=int, default=None, help="Max retries per remote request")
    sync_info.add_argument("--timeout", dest="timeout", type=float, default=None, help="HTTP timeout in seconds")

    return parser


def _read_json_config(path: str) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be an object: {config_path}")
    return payload


def _read_config_section(payload: dict[str, Any], section_name: str) -> dict[str, Any]:
    section = payload.get(section_name, {})
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError(f"Config section `{section_name}` must be an object")
    return section


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def _parse_symbol_list(value: Any) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []

    symbols: list[str] = []
    for token in text.replace("\n", ",").replace(";", ",").split(","):
        code = token.strip()
        if not code:
            continue
        try:
            symbols.append(normalize_symbol(code).symbol)
        except SymbolError:
            continue
    return sorted(set(symbols))


def _resolve_settings(args: argparse.Namespace, payload: dict[str, Any]) -> dict[str, Any]:
    common_cfg = _read_config_section(payload, "common")
    task_cfg = _read_config_section(payload, args.task)
    defaults = DEFAULT_TASK[args.task]
    resolved: dict[str, Any] = {}

    for key, default in DEFAULT_COMMON.items():
        resolved[key] = _coalesce(
            getattr(args, key, None),
            task_cfg.get(key),
            common_cfg.get(key),
            default,
        )

    for key, default in defaults.items():
        resolved[key] = _coalesce(
            getattr(args, key, None),
            task_cfg.get(key),
            common_cfg.get(key),
            default,
        )

    return resolved


def _masked_settings(settings: dict[str, Any]) -> dict[str, Any]:
    masked = dict(settings)
    if str(masked.get("tushare_token", "")).strip():
        masked["tushare_token"] = "***"
    return masked


def run_sync_data(settings: dict[str, Any]) -> int:
    set_tushare_token(settings.get("tushare_token", ""))
    result = sync_market_data(
        source=settings["source"],
        data_dir=settings["data_dir"],
        start=settings["start"],
        end=settings["end"],
        universe_size=int(settings["universe_size"]),
        universe_file=settings["universe_file"],
        include_indices=_parse_bool(settings["include_indices"]),
        force_refresh=_parse_bool(settings["force_refresh"]),
        sleep_ms=int(settings["sleep_ms"]),
        parallel_workers=max(1, int(settings.get("parallel_workers", 1) or 1)),
        max_failures=int(settings["max_failures"]),
        write_universe_file=settings["write_universe_file"],
        universe_min_amount=max(0.0, float(settings["universe_min_amount"])),
        universe_exclude_st=_parse_bool(settings["universe_exclude_st"]),
    )
    if result.resumed:
        print(f"[OK] Resumed from checkpoint: {result.resume_completed} symbols already completed")
    print(f"[OK] Universe source: {result.universe_source}")
    print(f"[OK] Universe size: {result.universe_size} (requested {result.requested_universe_size})")
    print(f"[OK] Downloaded: {result.downloaded}, skipped: {result.skipped}, failed: {result.failed}, attempted: {result.attempted}")
    if result.checkpoint_file:
        print(f"[OK] Resume checkpoint saved: {Path(result.checkpoint_file).resolve()}")
    if result.universe_file:
        print(f"[OK] Universe file written: {Path(result.universe_file).resolve()}")
    if result.failed_symbols:
        print(f"[WARN] Failed symbols (first 20): {', '.join(result.failed_symbols)}")
    if result.downloaded <= 0 and result.skipped <= 0:
        print("[ERROR] No symbols were synced.")
        return 2
    return 0


def run_sync_margin(settings: dict[str, Any]) -> int:
    set_tushare_token(settings.get("tushare_token", ""))

    symbols = _parse_symbol_list(settings.get("symbols", ""))
    if not symbols:
        universe_file = str(settings.get("universe_file", "")).strip()
        if universe_file:
            rows = _load_discovery_universe_file(universe_file, enrich_metadata=False)
            limit = int(settings.get("universe_limit", 0) or 0)
            if limit > 0:
                rows = rows[:limit]
            symbols = [normalize_symbol(security.symbol).symbol for security in rows]
    if not symbols:
        _, stocks, _ = load_watchlist(settings["watchlist"])
        symbols = [normalize_symbol(security.symbol).symbol for security in stocks]
    if not symbols:
        print("[ERROR] No valid symbols for margin sync. Set --symbols, --universe-file, or check watchlist.")
        return 2

    result = sync_margin_data(
        source=settings["source"],
        symbols=symbols,
        start=settings["start"],
        end=settings["end"],
        market_out=settings["margin_market_file"],
        stock_out=settings["margin_stock_file"],
        tushare_token=str(settings.get("tushare_token", "")),
        sleep_ms=int(settings["sleep_ms"]),
    )
    print(f"[OK] Margin source used: {result.source_used}")
    print(f"[OK] Symbols: {len(symbols)}")
    print(f"[OK] Market rows: {result.market_rows} -> {Path(result.market_path).resolve()}")
    print(f"[OK] Stock rows: {result.stock_rows} -> {Path(result.stock_path).resolve()}")
    for note in result.notes:
        print(f"[WARN] {note}")
    return 0


def run_sync_info(settings: dict[str, Any]) -> int:
    set_tushare_token(settings.get("tushare_token", ""))
    symbols = _parse_symbol_list(settings.get("symbols", ""))
    result = sync_info_data(
        out_dir=str(settings["info_dir"]),
        start=str(settings["start"]),
        end=str(settings["end"]),
        watchlist=str(settings["watchlist"]),
        universe_file=str(settings.get("universe_file", "")),
        universe_limit=int(settings.get("universe_limit", 0) or 0),
        symbols=symbols,
        tushare_token=str(settings.get("tushare_token", "")),
        sleep_ms=int(settings["sleep_ms"]),
        max_retries=int(settings["max_retries"]),
        timeout=float(settings["timeout"]),
    )
    print(f"[OK] Info output: {result.out_dir}")
    print(f"[OK] Symbols resolved: {result.symbol_count}")
    print(f"[OK] Market news rows: {result.market_news_rows}")
    print(f"[OK] Announcement rows: {result.announcement_rows}")
    print(f"[OK] Research rows: {result.research_rows}")
    for note in result.notes:
        print(f"[WARN] {note}")
    if max(result.market_news_rows, result.announcement_rows, result.research_rows) <= 0:
        print("[ERROR] No info rows were synced.")
        return 2
    return 0


def _task_handlers() -> dict[str, Any]:
    return {
        "sync-data": run_sync_data,
        "sync-margin": run_sync_margin,
        "sync-info": run_sync_info,
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        payload = _read_json_config(args.config)
        settings = _resolve_settings(args, payload)
        if args.print_effective_config:
            print(json.dumps({"task": args.task, "settings": _masked_settings(settings)}, indent=2, ensure_ascii=False))
            return 0
        handler = _task_handlers().get(args.task)
        if handler is None:
            parser.error(f"Unknown task: {args.task}")
        return int(handler(settings))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"[ERROR] Config failure: {exc}")
        return 4
    except DataError as exc:
        print(f"[ERROR] {exc}")
        print("Hint: try `--source auto` or chain fallback like `--source eastmoney,tushare,akshare,baostock`.")
        return 2
    except Exception as exc:
        print(f"[ERROR] Unexpected failure: {exc}")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
