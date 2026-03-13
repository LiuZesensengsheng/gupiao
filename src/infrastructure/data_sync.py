from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import requests

from src.domain.entities import Security
from src.domain.symbols import SymbolError, normalize_symbol
from src.infrastructure.market_data import DataError, fetch_tushare_daily_batch, load_symbol_daily


EASTMONEY_CLIST_URL = "https://push2.eastmoney.com/api/qt/clist/get"
_HTTP_SESSION = requests.Session()
_HTTP_SESSION.trust_env = False
INDEX_SECURITIES: tuple[Security, ...] = (
    Security(symbol="000300.SH", name="沪深300", sector="指数"),
    Security(symbol="000001.SH", name="上证指数", sector="指数"),
    Security(symbol="399001.SZ", name="深证成指", sector="指数"),
    Security(symbol="399006.SZ", name="创业板指", sector="指数"),
)
_SYMBOL_FILE_PATTERN = re.compile(r"^(\d{6}\.(SH|SZ))\.csv$", re.IGNORECASE)
_INDEX_SYMBOL_SET = {x.symbol for x in INDEX_SECURITIES} | {"000905.SH", "000852.SH"}
_SYNC_STATE_DIRNAME = ".sync_state"


@dataclass(frozen=True)
class DataSyncResult:
    requested_universe_size: int
    universe_size: int
    attempted: int
    downloaded: int
    skipped: int
    failed: int
    failed_symbols: list[str]
    universe_source: str
    universe_file: str
    resumed: bool = False
    resume_completed: int = 0
    checkpoint_file: str = ""


def _safe_symbol(value: str) -> str | None:
    try:
        return normalize_symbol(value).symbol
    except SymbolError:
        return None


def _normalize_exchange_code(code: str, market_id: int | None) -> str | None:
    text = str(code).strip()
    if not text.isdigit() or len(text) != 6:
        return None
    if market_id == 1:
        return f"{text}.SH"
    if market_id == 0:
        return f"{text}.SZ"
    symbol = _safe_symbol(text)
    return symbol


def _is_a_share(symbol: str) -> bool:
    try:
        info = normalize_symbol(symbol)
    except SymbolError:
        return False
    if info.exchange == "SH" and info.code.startswith("900"):
        return False
    if info.exchange == "SZ" and info.code.startswith("200"):
        return False
    return True


def _dedupe_rows(rows: Sequence[Security]) -> list[Security]:
    seen: set[str] = set()
    out: list[Security] = []
    for row in rows:
        symbol = _safe_symbol(row.symbol)
        if symbol is None or symbol in seen:
            continue
        seen.add(symbol)
        out.append(Security(symbol=symbol, name=row.name or symbol, sector=row.sector or "其他"))
    return out


def fetch_eastmoney_universe(limit: int, *, min_amount: float = 5e7, exclude_st: bool = True) -> list[Security]:
    limit = max(1, int(limit))
    out: list[Security] = []
    page = 1
    page_size = 200
    while len(out) < limit:
        params = {
            "pn": str(page),
            "pz": str(page_size),
            "po": "1",
            "np": "1",
            "fltt": "2",
            "invt": "2",
            "fid": "f3",
            "fs": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23",
            "fields": "f12,f13,f14,f2,f3,f6",
            "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        }
        resp = _HTTP_SESSION.get(EASTMONEY_CLIST_URL, params=params, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data") if isinstance(payload, dict) else None
        rows = data.get("diff") if isinstance(data, dict) else None
        if not rows:
            break
        for item in rows:
            if not isinstance(item, dict):
                continue
            symbol = _normalize_exchange_code(item.get("f12", ""), item.get("f13", None))
            if symbol is None or not _is_a_share(symbol) or symbol in _INDEX_SYMBOL_SET:
                continue
            name = str(item.get("f14", "")).strip() or symbol
            if exclude_st and "ST" in name.upper():
                continue
            amount = pd.to_numeric(item.get("f6", 0.0), errors="coerce")
            if pd.notna(amount) and float(amount) < float(min_amount):
                continue
            out.append(Security(symbol=symbol, name=name, sector="其他"))
            if len(out) >= limit:
                break
        page += 1
    return _dedupe_rows(out)[:limit]


def _load_universe_file(path: str | Path) -> list[Security]:
    file_path = Path(path)
    if not str(path).strip() or not file_path.exists():
        return []
    if file_path.suffix.lower() == ".json":
        with file_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        rows: list[dict[str, object]]
        if isinstance(payload, list):
            rows = [x for x in payload if isinstance(x, dict)]
        elif isinstance(payload, dict):
            if isinstance(payload.get("stocks"), list):
                rows = [x for x in payload["stocks"] if isinstance(x, dict)]
            else:
                rows = []
                if isinstance(payload.get("market_index"), dict):
                    rows.append(payload["market_index"])
        else:
            rows = []
        out: list[Security] = []
        for row in rows:
            symbol = _safe_symbol(str(row.get("symbol", "")))
            if symbol is None or not _is_a_share(symbol) or symbol in _INDEX_SYMBOL_SET:
                continue
            out.append(
                Security(
                    symbol=symbol,
                    name=str(row.get("name", symbol)),
                    sector=str(row.get("sector", "其他")),
                )
            )
        return _dedupe_rows(out)

    raw = pd.read_csv(file_path)
    if raw.empty:
        return []
    lower = {c.lower(): c for c in raw.columns}
    symbol_col = lower.get("symbol")
    if symbol_col is None:
        raise ValueError(f"Universe file missing column `symbol`: {file_path}")
    name_col = lower.get("name")
    sector_col = lower.get("sector")
    out: list[Security] = []
    for _, row in raw.iterrows():
        symbol = _safe_symbol(str(row[symbol_col]))
        if symbol is None or not _is_a_share(symbol) or symbol in _INDEX_SYMBOL_SET:
            continue
        out.append(
            Security(
                symbol=symbol,
                name=str(row[name_col]) if name_col is not None else symbol,
                sector=str(row[sector_col]) if sector_col is not None else "其他",
            )
        )
    return _dedupe_rows(out)


def _prepare_universe(
    *,
    universe_size: int,
    universe_file: str,
    data_dir: str,
    universe_min_amount: float,
    universe_exclude_st: bool,
) -> tuple[list[Security], str]:
    if str(universe_file).strip():
        rows = _load_universe_file(universe_file)
        if rows:
            return rows[: max(1, int(universe_size))], f"file:{universe_file}"

    last_error = ""
    try:
        rows = fetch_eastmoney_universe(
            limit=max(1, int(universe_size)),
            min_amount=float(universe_min_amount),
            exclude_st=bool(universe_exclude_st),
        )
        if rows:
            return rows, "eastmoney"
    except Exception as exc:
        last_error = str(exc)

    local_rows: list[Security] = []
    for path in sorted(Path(data_dir).glob("*.csv")):
        matched = _SYMBOL_FILE_PATTERN.match(path.name)
        if not matched:
            continue
        symbol = _safe_symbol(matched.group(1))
        if symbol is None or not _is_a_share(symbol) or symbol in _INDEX_SYMBOL_SET:
            continue
        local_rows.append(Security(symbol=symbol, name=symbol, sector="其他"))
        if len(local_rows) >= max(1, int(universe_size)):
            break
    local_rows = _dedupe_rows(local_rows)
    if local_rows:
        return local_rows, "data_dir_fallback"
    if last_error:
        raise RuntimeError(f"failed to fetch universe from eastmoney: {last_error}")
    raise RuntimeError("failed to build universe: empty result")


def _write_universe_file(path: str | Path, rows: Sequence[Security], source: str) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "source": source,
        "stocks": [{"symbol": x.symbol, "name": x.name, "sector": x.sector} for x in rows],
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(target)


def _is_fresh_enough(path: Path, target_end: pd.Timestamp) -> bool:
    if not path.exists():
        return False
    try:
        raw = pd.read_csv(path, usecols=["date"])
    except Exception:
        return False
    if raw.empty:
        return False
    max_date = pd.to_datetime(raw["date"], errors="coerce").dropna()
    if max_date.empty:
        return False
    latest = pd.Timestamp(max_date.max()).normalize()
    return latest >= (target_end - pd.Timedelta(days=5))


def _sync_checkpoint_path(
    *,
    data_dir: str,
    source: str,
    start: str,
    end: str,
    target_end: pd.Timestamp,
    force_refresh: bool,
    rows: Sequence[Security],
) -> Path:
    signature_payload = {
        "source": str(source).strip(),
        "start": str(start).strip(),
        "end": str(end).strip(),
        "target_end": target_end.strftime("%Y-%m-%d"),
        "force_refresh": bool(force_refresh),
        "symbols": [normalize_symbol(sec.symbol).symbol for sec in rows],
    }
    payload = json.dumps(signature_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    digest = hashlib.sha1(payload).hexdigest()[:16]
    return Path(data_dir) / _SYNC_STATE_DIRNAME / f"sync_{digest}.json"


def _load_sync_checkpoint(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    if not isinstance(payload, dict):
        return set()
    raw = payload.get("completed_symbols", [])
    if not isinstance(raw, list):
        return set()
    out: set[str] = set()
    for item in raw:
        symbol = _safe_symbol(str(item))
        if symbol is not None:
            out.add(symbol)
    return out


def _write_sync_checkpoint(
    path: Path,
    *,
    completed_symbols: set[str],
    total: int,
    source: str,
    target_end: pd.Timestamp,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": pd.Timestamp.now().isoformat(),
        "source": str(source).strip(),
        "target_end": target_end.strftime("%Y-%m-%d"),
        "total": int(total),
        "completed_symbols": sorted(completed_symbols),
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _should_print_progress(*, idx: int, total: int, progress_every: int, force: bool = False) -> bool:
    if force:
        return True
    if idx <= 1 or idx >= total:
        return True
    if progress_every <= 1:
        return True
    return idx % progress_every == 0


def _load_symbol_daily_task(
    *,
    symbol: str,
    source: str,
    data_dir: str,
    start: str,
    end: str,
) -> tuple[str, pd.DataFrame]:
    frame = load_symbol_daily(
        symbol=symbol,
        source=source,
        data_dir=data_dir,
        start=start,
        end=end,
    )
    return symbol, frame


def _estimate_trading_days(start: str, end: str) -> int:
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    days = max(1, int((end_ts - start_ts).days) + 1)
    return max(1, int(days * 245 / 365))


def _suggest_tushare_batch_size(start: str, end: str) -> int:
    trading_days = _estimate_trading_days(start, end)
    if trading_days >= 900:
        return 4
    if trading_days >= 500:
        return 6
    return 8


def _load_tushare_batch_task(
    *,
    symbols: Sequence[str],
    start: str,
    end: str,
) -> dict[str, pd.DataFrame]:
    return fetch_tushare_daily_batch(symbols=symbols, start=start, end=end)


def _load_single_tushare_fallback(
    *,
    symbol: str,
    data_dir: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    return load_symbol_daily(
        symbol=symbol,
        source="tushare",
        data_dir=data_dir,
        start=start,
        end=end,
    )


def sync_market_data(
    *,
    source: str,
    data_dir: str,
    start: str,
    end: str,
    universe_size: int,
    universe_file: str = "",
    include_indices: bool = True,
    force_refresh: bool = False,
    sleep_ms: int = 80,
    max_failures: int = 100,
    parallel_workers: int = 1,
    write_universe_file: str = "",
    universe_min_amount: float = 5e7,
    universe_exclude_st: bool = True,
) -> DataSyncResult:
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    rows, universe_source = _prepare_universe(
        universe_size=universe_size,
        universe_file=universe_file,
        data_dir=data_dir,
        universe_min_amount=float(universe_min_amount),
        universe_exclude_st=bool(universe_exclude_st),
    )
    rows = rows[: max(1, int(universe_size))]
    if include_indices:
        rows = _dedupe_rows(list(rows) + list(INDEX_SECURITIES))
    if not rows:
        raise ValueError("No symbols found for sync.")

    today = pd.Timestamp.today().normalize()
    end_ts = pd.Timestamp(end).normalize()
    target_end = min(today, end_ts)
    total = len(rows)
    progress_every = 1 if total <= 20 else 10
    checkpoint_path = _sync_checkpoint_path(
        data_dir=data_dir,
        source=source,
        start=start,
        end=end,
        target_end=target_end,
        force_refresh=force_refresh,
        rows=rows,
    )
    completed_symbols = _load_sync_checkpoint(checkpoint_path)
    resume_completed = len(completed_symbols)
    resumed = resume_completed > 0

    print(
        f"[SYNC] 开始同步: total={total}, source={str(source).strip() or 'auto'}, "
        f"target_end={target_end.strftime('%Y-%m-%d')}"
    )
    if resumed:
        print(
            f"[SYNC] 检测到断点续传检查点: 已完成 {resume_completed}/{total}, "
            f"将跳过已完成标的 -> {checkpoint_path.resolve()}"
        )

    downloaded = 0
    skipped = 0
    failed = 0
    attempts = 0
    failed_symbols: list[str] = []
    for idx, sec in enumerate(rows, start=1):
        symbol = normalize_symbol(sec.symbol).symbol
        if symbol in completed_symbols:
            continue
        path = Path(data_dir) / f"{symbol}.csv"
        if not force_refresh and _is_fresh_enough(path, target_end=target_end):
            skipped += 1
            completed_symbols.add(symbol)
            _write_sync_checkpoint(
                checkpoint_path,
                completed_symbols=completed_symbols,
                total=total,
                source=source,
                target_end=target_end,
            )
            if _should_print_progress(idx=idx, total=total, progress_every=progress_every):
                print(
                    f"[SYNC] {idx}/{total} {symbol} | downloaded={downloaded} skipped={skipped} "
                    f"failed={failed} done={len(completed_symbols)}/{total}"
                )
            continue
        attempts += 1
        try:
            df = load_symbol_daily(
                symbol=symbol,
                source=source,
                data_dir=data_dir,
                start=start,
                end=end,
            )
            if df.empty:
                raise DataError(f"{symbol}: empty dataframe after load")
            df.to_csv(path, index=False)
            downloaded += 1
            completed_symbols.add(symbol)
            _write_sync_checkpoint(
                checkpoint_path,
                completed_symbols=completed_symbols,
                total=total,
                source=source,
                target_end=target_end,
            )
            if _should_print_progress(idx=idx, total=total, progress_every=progress_every):
                print(
                    f"[SYNC] {idx}/{total} {symbol} | downloaded={downloaded} skipped={skipped} "
                    f"failed={failed} done={len(completed_symbols)}/{total}"
                )
        except Exception as exc:
            failed += 1
            failed_symbols.append(symbol)
            print(
                f"[WARN] [SYNC] {idx}/{total} {symbol} 同步失败: {exc} | "
                f"downloaded={downloaded} skipped={skipped} failed={failed} done={len(completed_symbols)}/{total}"
            )
            if failed >= max(1, int(max_failures)):
                print(
                    f"[WARN] [SYNC] 已达到失败上限 {max(1, int(max_failures))}，"
                    f"断点已保留，可直接重跑继续 -> {checkpoint_path.resolve()}"
                )
                break
        if int(sleep_ms) > 0:
            time.sleep(float(sleep_ms) / 1000.0)

    checkpoint_file = ""
    if len(completed_symbols) >= total:
        if checkpoint_path.exists():
            checkpoint_path.unlink()
    else:
        checkpoint_file = str(checkpoint_path)
        print(
            f"[SYNC] 当前已完成 {len(completed_symbols)}/{total}，保留断点供下次继续 -> {checkpoint_path.resolve()}"
        )

    universe_path = ""
    if str(write_universe_file).strip():
        universe_rows = [x for x in rows if x.sector != "指数"]
        universe_path = _write_universe_file(write_universe_file, universe_rows, source=universe_source)

    return DataSyncResult(
        requested_universe_size=int(universe_size),
        universe_size=len(rows),
        attempted=attempts,
        downloaded=downloaded,
        skipped=skipped,
        failed=failed,
        failed_symbols=failed_symbols[:20],
        universe_source=universe_source,
        universe_file=universe_path,
        resumed=resumed,
        resume_completed=resume_completed,
        checkpoint_file=checkpoint_file,
    )


def sync_market_data(
    *,
    source: str,
    data_dir: str,
    start: str,
    end: str,
    universe_size: int,
    universe_file: str = "",
    include_indices: bool = True,
    force_refresh: bool = False,
    sleep_ms: int = 80,
    max_failures: int = 100,
    parallel_workers: int = 1,
    write_universe_file: str = "",
    universe_min_amount: float = 5e7,
    universe_exclude_st: bool = True,
) -> DataSyncResult:
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    rows, universe_source = _prepare_universe(
        universe_size=universe_size,
        universe_file=universe_file,
        data_dir=data_dir,
        universe_min_amount=float(universe_min_amount),
        universe_exclude_st=bool(universe_exclude_st),
    )
    rows = rows[: max(1, int(universe_size))]
    if include_indices:
        rows = _dedupe_rows(list(rows) + list(INDEX_SECURITIES))
    if not rows:
        raise ValueError("No symbols found for sync.")

    today = pd.Timestamp.today().normalize()
    end_ts = pd.Timestamp(end).normalize()
    target_end = min(today, end_ts)
    total = len(rows)
    progress_every = 1 if total <= 20 else 10
    checkpoint_path = _sync_checkpoint_path(
        data_dir=data_dir,
        source=source,
        start=start,
        end=end,
        target_end=target_end,
        force_refresh=force_refresh,
        rows=rows,
    )
    completed_symbols = _load_sync_checkpoint(checkpoint_path)
    resume_completed = len(completed_symbols)
    resumed = resume_completed > 0

    print(
        f"[SYNC] Start sync total={total}, source={str(source).strip() or 'auto'}, "
        f"target_end={target_end.strftime('%Y-%m-%d')}"
    )
    if resumed:
        print(
            f"[SYNC] Resume checkpoint detected: completed={resume_completed}/{total}, "
            f"checkpoint={checkpoint_path.resolve()}"
        )

    downloaded = 0
    skipped = 0
    failed = 0
    attempts = 0
    failed_symbols: list[str] = []
    pending_rows: list[tuple[int, str, Path]] = []
    for idx, sec in enumerate(rows, start=1):
        symbol = normalize_symbol(sec.symbol).symbol
        if symbol in completed_symbols:
            continue
        path = Path(data_dir) / f"{symbol}.csv"
        if not force_refresh and _is_fresh_enough(path, target_end=target_end):
            skipped += 1
            completed_symbols.add(symbol)
            _write_sync_checkpoint(
                checkpoint_path,
                completed_symbols=completed_symbols,
                total=total,
                source=source,
                target_end=target_end,
            )
            if _should_print_progress(idx=idx, total=total, progress_every=progress_every):
                print(
                    f"[SYNC] {idx}/{total} {symbol} | downloaded={downloaded} skipped={skipped} "
                    f"failed={failed} done={len(completed_symbols)}/{total}"
                )
            continue
        pending_rows.append((idx, symbol, path))

    def _handle_success(*, idx: int, symbol: str, path: Path, df: pd.DataFrame) -> None:
        nonlocal downloaded
        if df.empty:
            raise DataError(f"{symbol}: empty dataframe after load")
        df.to_csv(path, index=False)
        downloaded += 1
        completed_symbols.add(symbol)
        _write_sync_checkpoint(
            checkpoint_path,
            completed_symbols=completed_symbols,
            total=total,
            source=source,
            target_end=target_end,
        )
        if _should_print_progress(idx=idx, total=total, progress_every=progress_every):
            print(
                f"[SYNC] {idx}/{total} {symbol} | downloaded={downloaded} skipped={skipped} "
                f"failed={failed} done={len(completed_symbols)}/{total}"
            )

    max_workers = max(1, int(parallel_workers))
    batch_source = str(source).strip().lower()
    if batch_source == "tushare" and pending_rows:
        batch_size = _suggest_tushare_batch_size(start, target_end.strftime("%Y-%m-%d"))
        batches: list[list[tuple[int, str, Path]]] = [
            pending_rows[i : i + batch_size] for i in range(0, len(pending_rows), batch_size)
        ]
        print(
            f"[SYNC] Tushare batch mode enabled workers={min(max_workers, len(batches))} "
            f"batch_size={batch_size} batches={len(batches)}"
        )
        executor = ThreadPoolExecutor(max_workers=min(max_workers, len(batches)))
        stop_early = False
        try:
            in_flight: dict[Future[dict[str, pd.DataFrame]], list[tuple[int, str, Path]]] = {}
            pending_iter = iter(batches)

            def _submit_next_batch() -> bool:
                try:
                    batch = next(pending_iter)
                except StopIteration:
                    return False
                future = executor.submit(
                    _load_tushare_batch_task,
                    symbols=[symbol for _, symbol, _ in batch],
                    start=start,
                    end=end,
                )
                in_flight[future] = batch
                return True

            for _ in range(min(max_workers, len(batches))):
                _submit_next_batch()

            while in_flight and not stop_early:
                done, _ = wait(tuple(in_flight.keys()), return_when=FIRST_COMPLETED)
                for future in done:
                    batch = in_flight.pop(future)
                    attempts += len(batch)
                    batch_frames: dict[str, pd.DataFrame] = {}
                    batch_error: Exception | None = None
                    try:
                        batch_frames = future.result()
                    except KeyboardInterrupt:
                        for other in in_flight:
                            other.cancel()
                        raise
                    except Exception as exc:
                        batch_error = exc

                    if batch_error is not None:
                        symbols_text = ", ".join(symbol for _, symbol, _ in batch[:4])
                        print(f"[WARN] [SYNC] Batch fallback to single-symbol requests: {symbols_text} | reason: {batch_error}")

                    for idx, symbol, path in batch:
                        if stop_early:
                            break
                        df = batch_frames.get(symbol)
                        try:
                            if df is None or df.empty:
                                df = _load_single_tushare_fallback(
                                    symbol=symbol,
                                    data_dir=data_dir,
                                    start=start,
                                    end=end,
                                )
                            _handle_success(idx=idx, symbol=symbol, path=path, df=df)
                        except KeyboardInterrupt:
                            for other in in_flight:
                                other.cancel()
                            raise
                        except Exception as exc:
                            failed += 1
                            failed_symbols.append(symbol)
                            print(
                                f"[WARN] [SYNC] {idx}/{total} {symbol} sync failed: {exc} | "
                                f"downloaded={downloaded} skipped={skipped} failed={failed} done={len(completed_symbols)}/{total}"
                            )
                            if failed >= max(1, int(max_failures)):
                                print(
                                    f"[WARN] [SYNC] Reached failure limit {max(1, int(max_failures))}, "
                                    f"resume checkpoint retained -> {checkpoint_path.resolve()}"
                                )
                                for other in in_flight:
                                    other.cancel()
                                stop_early = True
                                break
                    if not stop_early:
                        _submit_next_batch()
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
    elif max_workers > 1 and pending_rows:
        print(f"[SYNC] Parallel download enabled workers={min(max_workers, len(pending_rows))}")
        executor = ThreadPoolExecutor(max_workers=min(max_workers, len(pending_rows)))
        stop_early = False
        try:
            in_flight: dict[Future[tuple[str, pd.DataFrame]], tuple[int, str, Path]] = {}
            pending_iter = iter(pending_rows)

            def _submit_next() -> bool:
                try:
                    next_idx, next_symbol, next_path = next(pending_iter)
                except StopIteration:
                    return False
                future = executor.submit(
                    _load_symbol_daily_task,
                    symbol=next_symbol,
                    source=source,
                    data_dir=data_dir,
                    start=start,
                    end=end,
                )
                in_flight[future] = (next_idx, next_symbol, next_path)
                return True

            for _ in range(min(max_workers, len(pending_rows))):
                _submit_next()

            while in_flight and not stop_early:
                done, _ = wait(tuple(in_flight.keys()), return_when=FIRST_COMPLETED)
                for future in done:
                    idx, symbol, path = in_flight.pop(future)
                    attempts += 1
                    try:
                        _, df = future.result()
                        _handle_success(idx=idx, symbol=symbol, path=path, df=df)
                    except KeyboardInterrupt:
                        for other in in_flight:
                            other.cancel()
                        raise
                    except Exception as exc:
                        failed += 1
                        failed_symbols.append(symbol)
                        print(
                            f"[WARN] [SYNC] {idx}/{total} {symbol} sync failed: {exc} | "
                            f"downloaded={downloaded} skipped={skipped} failed={failed} done={len(completed_symbols)}/{total}"
                        )
                        if failed >= max(1, int(max_failures)):
                            print(
                                f"[WARN] [SYNC] Reached failure limit {max(1, int(max_failures))}, "
                                f"resume checkpoint retained -> {checkpoint_path.resolve()}"
                            )
                            for other in in_flight:
                                other.cancel()
                            stop_early = True
                            break
                    if not stop_early:
                        _submit_next()
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
    else:
        for idx, symbol, path in pending_rows:
            attempts += 1
            try:
                df = load_symbol_daily(
                    symbol=symbol,
                    source=source,
                    data_dir=data_dir,
                    start=start,
                    end=end,
                )
                _handle_success(idx=idx, symbol=symbol, path=path, df=df)
            except Exception as exc:
                failed += 1
                failed_symbols.append(symbol)
                print(
                    f"[WARN] [SYNC] {idx}/{total} {symbol} sync failed: {exc} | "
                    f"downloaded={downloaded} skipped={skipped} failed={failed} done={len(completed_symbols)}/{total}"
                )
                if failed >= max(1, int(max_failures)):
                    print(
                        f"[WARN] [SYNC] Reached failure limit {max(1, int(max_failures))}, "
                        f"resume checkpoint retained -> {checkpoint_path.resolve()}"
                    )
                    break
            if int(sleep_ms) > 0:
                time.sleep(float(sleep_ms) / 1000.0)

    checkpoint_file = ""
    if len(completed_symbols) >= total:
        if checkpoint_path.exists():
            checkpoint_path.unlink()
    else:
        checkpoint_file = str(checkpoint_path)
        print(
            f"[SYNC] Progress saved {len(completed_symbols)}/{total}, resume checkpoint -> {checkpoint_path.resolve()}"
        )

    universe_path = ""
    if str(write_universe_file).strip():
        universe_rows = [x for x in rows if x.sector != "鎸囨暟"]
        universe_path = _write_universe_file(write_universe_file, universe_rows, source=universe_source)

    return DataSyncResult(
        requested_universe_size=int(universe_size),
        universe_size=len(rows),
        attempted=attempts,
        downloaded=downloaded,
        skipped=skipped,
        failed=failed,
        failed_symbols=failed_symbols[:20],
        universe_source=universe_source,
        universe_file=universe_path,
        resumed=resumed,
        resume_completed=resume_completed,
        checkpoint_file=checkpoint_file,
    )
