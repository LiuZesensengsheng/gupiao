#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.domain.info_clock import derive_publish_timestamp_from_source_url, parse_timestamp


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill publish_datetime for info CSV files.")
    parser.add_argument("path", nargs="?", default="input/info_parts", help="CSV file or directory to backfill")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing files")
    return parser.parse_args()


def _csv_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(item for item in path.rglob("*.csv") if item.is_file())


def _derive_publish_datetime(row: pd.Series) -> str:
    current = parse_timestamp(row.get("publish_datetime", ""))
    if current is not None:
        return str(current.isoformat())
    derived = derive_publish_timestamp_from_source_url(row.get("source_url", ""))
    if derived is None:
        return ""
    return str(derived.isoformat())


def main() -> int:
    args = _parse_args()
    path = Path(args.path)
    files = _csv_files(path)
    if not files:
        print(f"[WARN] no csv files found under {path}")
        return 1

    updated_files = 0
    updated_rows = 0
    for file in files:
        try:
            frame = pd.read_csv(file)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] skipped {file}: {exc}")
            continue
        if frame.empty:
            continue
        if "publish_datetime" not in frame.columns:
            frame["publish_datetime"] = ""
        filled = frame["publish_datetime"].fillna("").astype(str)
        derived = frame.apply(_derive_publish_datetime, axis=1).astype(str)
        changed_mask = filled != derived
        if not bool(changed_mask.any()):
            continue
        frame["publish_datetime"] = derived
        if not args.dry_run:
            frame.to_csv(file, index=False, encoding="utf-8")
        updated_files += 1
        updated_rows += int(changed_mask.sum())
        print(f"[OK] {file}: updated_rows={int(changed_mask.sum())}")

    print(f"[DONE] files={updated_files}, rows={updated_rows}, dry_run={bool(args.dry_run)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
