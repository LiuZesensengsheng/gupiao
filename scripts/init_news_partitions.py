#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.domain.symbols import normalize_symbol

HEADER = "date,target_type,target,horizon,direction,strength,confidence,source_weight,title\n"


def _load_watchlist_symbols(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    stocks = payload.get("stocks", [])
    out: list[str] = []
    for item in stocks:
        symbol = normalize_symbol(str(item.get("symbol", ""))).symbol
        if symbol:
            out.append(symbol)
    return sorted(set(out))


def _parse_args() -> argparse.Namespace:
    now_year = int(datetime.now().year)
    parser = argparse.ArgumentParser(description="Initialize stock/year news partition CSV files.")
    parser.add_argument("--watchlist", default="config/watchlist.json", help="Watchlist JSON path")
    parser.add_argument("--out-dir", default="input/news_parts", help="Output directory for partition CSV files")
    parser.add_argument("--start-year", type=int, default=2022, help="Start year (inclusive)")
    parser.add_argument("--end-year", type=int, default=now_year, help="End year (inclusive)")
    parser.add_argument("--no-market", action="store_true", help="Do not create MARKET yearly files")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files with header only")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    start_year = int(args.start_year)
    end_year = int(args.end_year)
    if end_year < start_year:
        raise ValueError(f"Invalid year range: {start_year}..{end_year}")

    symbols = _load_watchlist_symbols(Path(args.watchlist))
    targets = list(symbols)
    if not args.no_market:
        targets.insert(0, "MARKET")

    created = 0
    skipped = 0
    out_root = Path(args.out_dir)
    for target in targets:
        target_dir = out_root / target
        target_dir.mkdir(parents=True, exist_ok=True)
        for year in range(start_year, end_year + 1):
            file_path = target_dir / f"{year}.csv"
            if file_path.exists() and not args.overwrite:
                skipped += 1
                continue
            file_path.write_text(HEADER, encoding="utf-8")
            created += 1

    print(f"[OK] Initialized news partitions at {out_root.resolve()}")
    print(f"[OK] Files created: {created}, skipped existing: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
